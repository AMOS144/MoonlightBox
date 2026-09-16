"""恢复已生成但未落库的单个 Runtime Cycle。

默认只读检查；--apply 才会调用模型修正计划并提交。只复用同一输入版本的 Phoenix
产物；不伪造工具调用，不将回复重放计成云端生成。新的输入或活跃任务会阻止恢复。
环境变量由调用方提供，不在脚本中保存密钥、项目 ID 或数据库默认路径。
"""

import argparse
import json
import sqlite3
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage
from moonlightbox.agent_runtime import AgentBudgetPolicy, AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest
from moonlightbox.agent_runtime.submission import result_submission_tool, submission_instruction
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.observability import initialize_phoenix, shutdown_phoenix
from moonlightbox.runtime_v1.agent_support import prompt_hash
from moonlightbox.runtime_v1.branch_models import Branch
from moonlightbox.runtime_v1.cloud_models import RuntimeCloudClient, create_cloud_models
from moonlightbox.runtime_v1.db_models import (
    RuntimeCycleTraceRow,
    RuntimeDayPlanRow,
    RuntimeEventRow,
    RuntimeLifeEventRow,
    RuntimeLifeStateRow,
    RuntimeWakeupRow,
)
from moonlightbox.runtime_v1.executor import RuntimeExecutor
from moonlightbox.runtime_v1.prompts.day_plan_recovery import DAY_PLAN_RECOVERY_PROMPT
from moonlightbox.runtime_v1.schemas import ActorMessage, DayPlanProposal, LifeDecision
from moonlightbox.runtime_v1.service import RuntimeService
from opentelemetry import trace as otel_trace
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session


def load_outputs(path, cycle_id):
    outputs = {}
    traces = set()
    revision = None
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        for trace_id, raw in db.execute(
            "select trace_rowid,json(attributes) from spans where name='moonlightbox.agent.run'"
        ):
            data = json.loads(raw)
            mb = data.get("moonlightbox", {})
            if mb.get("owner", {}).get("id") != cycle_id:
                continue
            agent = mb.get("agent", {}).get("name")
            if agent == "day_planner":
                traces.add(trace_id)
            if agent in {"director", "persona_actor"}:
                assert mb["agent"]["status"] == "succeeded", f"{agent} 没有合格产物"
                current_revision = mb["input_revision"]
                assert revision in {None, current_revision}, "产物输入版本不一致"
                revision = current_revision
                outputs[agent] = json.loads(data["output"]["value"])
        for trace_id in traces:
            for (raw,) in db.execute(
                "select json(attributes) from spans where trace_rowid=? "
                "and name='moonlightbox.provider.chat_completion' order by start_time",
                (trace_id,),
            ):
                data = json.loads(raw)
                try:
                    response = json.loads(data["output"]["value"])
                    plan = json.loads(response["content"])
                except (KeyError, ValueError):
                    continue
                if "blocks" in plan:
                    outputs["plan"] = plan
    assert set(outputs) == {"director", "persona_actor", "plan"}, "Phoenix 产物不完整"
    return outputs, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phoenix-db", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    outputs, revision = load_outputs(args.phoenix_db, args.cycle_id)
    settings = Settings()
    database = Database(settings.database_url)
    with Session(database.engine) as session:
        old = session.get(RuntimeCycleTraceRow, args.cycle_id)
        assert old is not None
        branch = session.get(Branch, old.branch_id)

        def check_revision():
            return session.scalar(
                select(Branch.runtime_input_revision).where(Branch.id == branch.id)
            )

        def check_idle():
            assert check_revision() == revision, "有新输入，不能重放旧回复"
            assert not any(
                j.payload.get("branch_id") == branch.id
                for j in session.scalars(select(Job).where(Job.status.in_(["queued", "running"])))
            ), "存在活跃分支任务"
            assert not list(
                session.scalars(
                    select(RuntimeEventRow).where(
                        RuntimeEventRow.branch_id == branch.id,
                        RuntimeEventRow.status.in_(["queued", "claimed"]),
                        RuntimeEventRow.id.not_in(old.trigger_event_ids),
                    )
                )
            ), "存在额外事件，需重新决策"

        existing = session.scalar(
            select(RuntimeLifeEventRow).where(RuntimeLifeEventRow.idempotency_key == old.cycle_key)
        )
        if existing:
            print("该 Cycle 已提交，不重复回复")
            return
        check_idle()
        try:
            proposal = DayPlanProposal.model_validate_json(json.dumps(outputs["plan"]))
            invalid = set()
        except ValidationError as error:
            invalid = {
                e["loc"][1] for e in error.errors() if len(e["loc"]) > 1 and e["loc"][0] == "blocks"
            }
            print("计划需修正的块（从 0 开始）", sorted(invalid), flush=True)
            proposal = None
        print("恢复输入版本", revision, "计划块数", len(outputs["plan"]["blocks"]), flush=True)
        if not args.apply:
            return
        initialize_phoenix(settings, service_name="runtime-recovery")
        executor = RuntimeExecutor(session)
        virtual_now = old.virtual_now.replace(tzinfo=UTC)
        plan_row = session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch.id,
                RuntimeDayPlanRow.plan_date == virtual_now.date().isoformat(),
            )
        )
        assert plan_row is not None
        already_saved = plan_row.generation_metadata.get("recovered_from_cycle") == old.id
        assert already_saved or plan_row.generation_metadata.get("status") == "pending"
        if already_saved:
            # 上次恢复可能只成功提交了计划。重复执行只恢复回复，不再次花费模型调用。
            proposal = DayPlanProposal.model_validate_json(
                json.dumps(
                    {
                        "plan_date": plan_row.plan_date,
                        "blocks": [
                            {k: v for k, v in block.items() if k != "id"}
                            for block in plan_row.blocks
                        ],
                        "assumptions": plan_row.generation_metadata.get("assumptions", []),
                        "private_reason": plan_row.generation_metadata.get("reason", ""),
                    }
                )
            )
        if proposal is None:
            prompt = (
                DAY_PLAN_RECOVERY_PROMPT + "\n\n" + submission_instruction("submit_recovered_plan")
            )

            def validate(value, _context):
                original = outputs["plan"]
                if len(value.blocks) != len(original["blocks"]):
                    return "不得增加或删除原计划块"
                for i, block in enumerate(value.blocks):
                    raw = block.model_dump(mode="json", exclude_none=True)
                    before = original["blocks"][i]
                    if any(raw.get(k) != before.get(k) for k in ("start", "end", "activity")):
                        return f"第 {i} 块时间和活动不能改变"
                    if i not in invalid:
                        expected = (
                            DayPlanProposal.model_validate_json(
                                json.dumps({**original, "blocks": [before]})
                            )
                            .blocks[0]
                            .model_dump(mode="json", exclude_none=True)
                        )
                        if raw != expected:
                            return f"第 {i} 块无需修正，必须保留"
                    elif block.evidence_ids != before.get("evidence_ids", []):
                        return f"第 {i} 块不能补造来源"
                try:
                    executor._validate_day_plan_proposal(
                        branch_id=branch.id,
                        plan=plan_row,
                        proposal=value,
                        virtual_now=virtual_now,
                        mode="initial",
                    )
                except ValueError as error:
                    return str(error)
                return None

            client = RuntimeCloudClient.from_settings(settings)
            try:
                model, _actor = create_cloud_models(client)
                result = AgentLoopController().run(
                    spec=AgentSpec(
                        name="day_plan_recovery",
                        prompt_version=prompt_hash(prompt),
                        submission_tool_name="submit_recovered_plan",
                        tools=(
                            result_submission_tool(
                                "submit_recovered_plan", DayPlanProposal, validate
                            ),
                        ),
                        budget=AgentBudgetPolicy(
                            max_wall_seconds=180,
                            emergency_max_model_steps=8,
                            max_tool_result_chars=1000,
                        ),
                    ),
                    request=AgentExecutionRequest(
                        owner_type="day_plan",
                        owner_id=old.id,
                        project_id=old.project_id,
                        input_revision=revision,
                        scope=RunScope(project_id=old.project_id, branch_id=branch.id),
                        messages=(
                            SystemMessage(content=prompt),
                            HumanMessage(
                                content=json.dumps(
                                    {
                                        "invalid_block_indices": sorted(invalid),
                                        "draft": outputs["plan"],
                                    },
                                    ensure_ascii=False,
                                )
                            ),
                        ),
                        input_revision_resolver=check_revision,
                    ),
                    model=model,
                )
                assert result.value is not None, f"修正失败：{result.terminal_reason}"
                proposal = result.value
            finally:
                client.close()
        check_idle()
        with otel_trace.get_tracer(__name__).start_as_current_span(
            "moonlightbox.runtime.recovery"
        ) as span:
            span.set_attribute("moonlightbox.cycle.id", old.id)
            span.set_attribute("moonlightbox.branch.id", branch.id)
            span.set_attribute("recovery.reuses_director_actor", True)
            committed = executor.commit_day_plan_proposal(
                branch_id=branch.id, proposal=proposal, virtual_now=virtual_now, mode="initial"
            )
            committed.generation_metadata = {
                **committed.generation_metadata,
                "recovered_from_cycle": old.id,
                "original_draft": outputs["plan"],
            }
            session.commit()
            # 计划先独立提交，回复恢复再次检查输入版本与幂等键。
            check_idle()
            service = RuntimeService(session)
            service._apply_plan_boundary(branch.id, virtual_now, committed.blocks)
            service._expire_state(branch.id, virtual_now, committed.blocks)
            state = session.scalar(
                select(RuntimeLifeStateRow).where(
                    RuntimeLifeStateRow.branch_id == branch.id,
                    RuntimeLifeStateRow.is_current.is_(True),
                )
            )
            result = executor.commit(
                project_id=old.project_id,
                branch_id=branch.id,
                decision=LifeDecision.model_validate_json(json.dumps(outputs["director"])),
                actor_message=ActorMessage.model_validate(outputs["persona_actor"]),
                expected_version=state.version,
                virtual_now=virtual_now,
                trigger_event_ids=old.trigger_event_ids,
                idempotency_key=old.cycle_key,
            )
            for event_id in old.trigger_event_ids:
                row = session.get(RuntimeEventRow, event_id)
                if row is not None:
                    row.status = "completed"
                    row.completed_at = datetime.now(UTC)
            for wakeup_id in old.wakeup_ids:
                row = session.get(RuntimeWakeupRow, wakeup_id)
                if row is not None:
                    row.status = "completed"
                    row.executing_at = None
            old.status = "recovered"
            old.stage = "committed"
            old.completed_at = datetime.now(UTC)
            old.error_code = "NoReferencedTableError"
            old.error_message = "原 Worker 缺少媒体 ORM 注册；原失败保留，使用 Phoenix 产物恢复"
            old.planner_proposal = proposal.model_dump(mode="json")
            old.director_decision = outputs["director"]
            old.committed_decision = outputs["director"]
            old.outcome = {
                **old.outcome,
                "recovered": True,
                "message_id": result["message"].id,
                "life_event_id": result["event"].id,
                "original_job_kept_failed": True,
            }
            session.commit()
            print("已恢复计划和回复", result["message"].id, result["message"].content, flush=True)
    shutdown_phoenix()
    database.close()


if __name__ == "__main__":
    main()
