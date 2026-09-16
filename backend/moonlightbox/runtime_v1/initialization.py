"""Director 起点任务：独立请求、检查点和工具缓存，只通过最终状态连接正常 Runtime。"""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime import AgentSpec, RunScope
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest
from moonlightbox.agent_runtime.persistence import checkpoint_path
from moonlightbox.agent_runtime.policy import DIRECTOR_INITIALIZATION_POLICY, controller_budget

from .agent_support import prompt_hash
from .branch_models import Branch, BranchMessage
from .clock import create_clock
from .collaboration.plans import local_time, shared_plan_window
from .context import ContextAssembler
from .context_views import director_context_payload
from .conversation_history import ConversationHistory
from .db_models import (
    RuntimeClockRow,
    RuntimeInitializationRow,
    RuntimeLifeStateRow,
    RuntimeSnapshotRow,
)
from .director_contracts import InitialState
from .initialization_context import InitializationToolbox, preload_history
from .prompting import assemble_prompt
from .tools.initialization import build_initialization_tools


def _fingerprint(now, plans, snapshot, state):
    return prompt_hash(
        json.dumps(
            {
                "protocol": "director-initialization-v1",
                "snapshot": snapshot.id,
                "profile": snapshot.profile,
                "cutoff": str(snapshot.cutoff_at),
                "virtual_now": now.isoformat(),
                "plans": plans,
                "state_id": state.id,
                "state_version": state.version,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )


def initialize_director(
    agent,
    session,
    *,
    project_id,
    branch_id,
    bootstrap,
    cancellation_requested=None,
):
    """覆盖材料装配和模型执行的失败边界，失败状态不能一直停留在 running。"""
    try:
        return _initialize_director(
            agent,
            session,
            project_id=project_id,
            branch_id=branch_id,
            bootstrap=bootstrap,
            cancellation_requested=cancellation_requested,
        )
    except Exception as error:
        session.rollback()
        row = session.get(RuntimeInitializationRow, branch_id)
        if row is not None and row.status != "ready":
            row.status, row.error_code = "failed", getattr(error, "code", type(error).__name__)
            row.updated_at = datetime.now(UTC)
            session.commit()
        raise


def _initialize_director(
    agent, session, *, project_id, branch_id, bootstrap, cancellation_requested=None
):
    """调用者持有分支写锁。成功后原子提交；重试不沿用普通 Director 的执行作用域。"""
    from .service import RuntimeModelExecutionError

    row = session.get(RuntimeInitializationRow, branch_id)
    if row and row.status == "ready":
        return
    branch = session.get(Branch, branch_id)
    if branch.lifecycle_status not in {"preparing", "prepare_failed"} or session.scalar(
        select(BranchMessage.id).where(BranchMessage.branch_id == branch_id).limit(1)
    ):
        raise RuntimeModelExecutionError(
            "initialization_not_allowed", "已有运行中的分支不能重置起点"
        )
    if agent.model is None:
        raise RuntimeModelExecutionError("model_unavailable", "起点理解需要 Director 模型")
    snapshot = bootstrap["snapshot"]
    packet = ContextAssembler(session).assemble(
        branch_id=branch_id,
        trigger={"type": "initialize"},
        clock=bootstrap["clock"],
        snapshot=snapshot,
    )
    state = session.scalar(
        select(RuntimeLifeStateRow).where(
            RuntimeLifeStateRow.branch_id == branch_id,
            RuntimeLifeStateRow.is_current.is_(True),
        )
    )
    expected = _fingerprint(packet.virtual_now, packet.current["day_plans"], snapshot, state)
    if row is None:
        row = RuntimeInitializationRow(branch_id=branch_id, input_hash=expected, work={})
        session.add(row)
    elif row.input_hash != expected:
        # 输入版本变化开启新调查，旧执行仍保留在其独立检查点里。
        row.input_hash, row.work = expected, {}
    row.status, row.error_code = "running", None
    session.commit()

    def current_input():
        with Session(session.get_bind()) as check:
            current_branch = check.get(Branch, branch_id)
            latest_snapshot = check.get(RuntimeSnapshotRow, snapshot.id)
            latest_state = check.scalar(
                select(RuntimeLifeStateRow).where(
                    RuntimeLifeStateRow.branch_id == branch_id,
                    RuntimeLifeStateRow.is_current.is_(True),
                )
            )
            clock_row = check.get(RuntimeClockRow, branch_id)
            if not all((current_branch, latest_snapshot, latest_state, clock_row)):
                return None
            if current_branch.lifecycle_status not in {"preparing", "prepare_failed"}:
                return None
            # 版本守卫只读实际依赖，不再重建聊天窗口、历史摘要和完整上下文。
            clock = create_clock(
                branch_id,
                clock_row.virtual_anchor,
                wall_anchor=clock_row.wall_anchor,
                timezone=clock_row.timezone,
            ).model_copy(update={"status": clock_row.status, "time_scale": clock_row.time_scale})
            now = local_time(clock.now(), clock_row.timezone)
            plans = shared_plan_window(check, branch_id, now, clock_row.timezone)
            unchanged = _fingerprint(now, plans, latest_snapshot, latest_state) == expected
            unchanged = unchanged and not check.scalar(
                select(BranchMessage.id).where(BranchMessage.branch_id == branch_id).limit(1)
            )
            return 1 if unchanged else None

    history = ConversationHistory(session, branch_id, snapshot, packet.virtual_now)
    allowed = set(history.references()[0])
    base = director_context_payload(packet)
    base["branch"] = {}  # 本次自行预读更长历史，不重复普通聊天窗口。
    base.pop("collaboration", None)
    base["history_navigation"] = {
        "total_messages": len(allowed),
        "cutoff_at": str(snapshot.cutoff_at),
    }
    box = InitializationToolbox(base, lambda: deepcopy(row.work))

    tools = build_initialization_tools(session, packet, snapshot, row, box, allowed)
    prompt_file = Path(__file__).with_name("prompts") / "director" / "initialization.md"
    assembly = assemble_prompt(
        prompt_file.read_text(encoding="utf-8"),
        InitialState,
        submission_tool_name="submit_initial_state",
        tools=tools,
        source=str(prompt_file),
    )
    budget = controller_budget(DIRECTOR_INITIALIZATION_POLICY)
    raw = preload_history(
        history, base, assembly, tools, budget, len(allowed), cancellation_requested
    )
    box.project(raw)  # 即使预读正文日后被压缩，也能通过稳定引用恢复。
    messages = (
        assembly.message(),
        HumanMessage(
            content=json.dumps(
                {
                    "initialization_context": base,
                    "recent_conversation": raw,
                    "investigation_work": row.work,
                },
                ensure_ascii=False,
                default=str,
            )
        ),
    )
    session.commit()
    outcome = agent.controller.run(
        spec=AgentSpec(
            name="director_initialization",
            prompt_version=prompt_hash(assembly.text),
            submission_tool_name="submit_initial_state",
            tools=tools,
            budget=budget,
            context_compactor=box.compact,
            resume_messages=box.resume_messages,
            restore_tool_results=box.restore,
        ),
        request=AgentExecutionRequest(
            owner_type="director_initialization",
            owner_id=f"{branch_id}:{expected}",
            project_id=project_id,
            messages=messages,
            checkpoint_path=checkpoint_path(session),
            scope=RunScope(
                project_id=project_id,
                branch_id=branch_id,
                allowed_source_snapshot_id=snapshot.id,
                permissions=frozenset({"runtime.read_memory"}),
            ),
            input_revision_resolver=current_input,
            cancellation_requested=cancellation_requested,
        ),
        model=agent.model,
    )
    if outcome.terminal_reason != "success":
        raise RuntimeModelExecutionError(
            outcome.terminal_reason, "Director 起点理解未完成，可恢复重试"
        )
    if (cancellation_requested and cancellation_requested()) or current_input() != 1:
        raise RuntimeModelExecutionError(
            "initialization_stale_or_cancelled", "起点输入已变化或任务被取消"
        )
    from .executor import RuntimeExecutor

    RuntimeExecutor(session).commit_initial_state(
        branch_id, state.id, outcome.value, allowed, packet.virtual_now
    )
    row.status, row.error_code, row.updated_at = "ready", None, datetime.now(UTC)
    session.commit()
