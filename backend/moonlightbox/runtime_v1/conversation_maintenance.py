"""独立后台历史维护任务；消息事务只写 Job outbox，不等待模型或向量推理。"""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field
from sqlalchemy import select

from moonlightbox.agent_runtime import AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest
from moonlightbox.agent_runtime.persistence import checkpoint_path
from moonlightbox.agent_runtime.policy import DIRECTOR_RUNTIME_POLICY, controller_budget
from moonlightbox.agent_runtime.submission import result_submission_tool
from moonlightbox.jobs.service import JobService

from .agent_support import prompt_hash
from .branch_models import BranchMessage
from .db_models import RuntimeContextSummaryRow
from .director_contracts import Contract
from .prompting import assemble_prompt
from .prompts.conversation_summary import SYSTEM_PROMPT


class Summary(Contract):
    summary: str = Field(
        min_length=1,
        description="压缩本批已处理对话，保留人物、时间与具体进展；不添加材料中没有的事实",
    )
    unresolved_items: list[str] = Field(
        default_factory=list,
        description="本批仍未结束的话题或承诺，供后续回查；没有则 []，不把沉默当作结束",
    )


def enqueue_maintenance(session, branch_id, key):
    return JobService(session).enqueue_unique(
        "runtime-v1-cycle",
        {"branch_id": branch_id, "conversation_maintenance": True},
        dedupe_key=f"conversation-maintenance:{branch_id}:{key}",
        commit=False,
    )


def summarize(session, branch_id, project_id, model):
    """按连续来源分批压缩；成功持久化之前不减少已加载的旧原文。"""
    rows = list(
        session.scalars(
            select(BranchMessage)
            .where(
                BranchMessage.branch_id == branch_id,
                BranchMessage.generation_status != "failed",
            )
            .order_by(BranchMessage.sequence)
        )
    )
    covered = {
        ref
        for row in session.scalars(
            select(RuntimeContextSummaryRow).where(
                RuntimeContextSummaryRow.branch_id == branch_id,
                RuntimeContextSummaryRow.created_by == "director-summary-v2",
            )
        )
        for ref in row.source_message_ids
    }
    batch = []
    for row in rows[:-60]:
        if row.id in covered:
            continue
        if (row.generation_metadata or {}).get("input_status") in {"pending", "awaiting_response"}:
            break
        batch.append(row)
        if len(batch) >= 40:
            break
    if not batch:
        return False
    submission = result_submission_tool("submit_summary", Summary)
    prompt = assemble_prompt(
        SYSTEM_PROMPT, Summary, tools=(submission,), submission_tool_name="submit_summary"
    ).text
    result = AgentLoopController().run(
        spec=AgentSpec(
            name="conversation_summary",
            prompt_version=prompt_hash(prompt),
            submission_tool_name="submit_summary",
            budget=controller_budget(DIRECTOR_RUNTIME_POLICY),
            tools=(submission,),
        ),
        request=AgentExecutionRequest(
            checkpoint_path=checkpoint_path(session),
            owner_type="conversation_summary",
            owner_id=batch[-1].id,
            project_id=project_id,
            input_revision=1,
            scope=RunScope(project_id=project_id, branch_id=branch_id),
            messages=(
                SystemMessage(content=prompt),
                HumanMessage(
                    content=json.dumps(
                        [
                            {
                                "source_ref": row.id,
                                "role": row.role,
                                "content": row.content,
                                "occurred_at": (row.observed_at or row.created_at).isoformat(),
                            }
                            for row in batch
                        ],
                        ensure_ascii=False,
                    )
                ),
            ),
        ),
        model=model,
    )
    if not isinstance(result.value, Summary):
        raise RuntimeError("conversation_summary_failed")
    session.add(
        RuntimeContextSummaryRow(
            branch_id=branch_id,
            covered_until=batch[-1].observed_at or batch[-1].created_at,
            source_message_ids=[row.id for row in batch],
            text=result.value.summary,
            unresolved_items=result.value.unresolved_items,
            created_by="director-summary-v2",
        )
    )
    session.commit()
    return True
