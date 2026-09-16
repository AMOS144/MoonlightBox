"""同级协作冒烟：真实图、真实 SQLite、离线模型，不依赖 GPU 或供应商。"""

import json
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.runtime_v1.branch_models import Branch
from moonlightbox.runtime_v1.collaboration.messages import merge_shared_messages
from moonlightbox.runtime_v1.collaboration.plans import shared_plan_window
from moonlightbox.runtime_v1.db_models import RuntimeDayPlanRow, RuntimeSnapshotRow
from moonlightbox.runtime_v1.service import RuntimeService
from moonlightbox.training.models import ModelVersion
from sqlalchemy import select
from sqlalchemy.orm import Session


@pytest.fixture
def session(tmp_path):
    db = Database(f"sqlite:///{tmp_path / 'runtime.db'}")
    db.create_schema()
    with Session(db.engine) as s:
        s.add(Project(id="p", name="test"))
        s.flush()
        s.add(
            ModelVersion(
                id="m",
                project_id="p",
                base_model="test",
                adapter_path="none",
                dataset_hash="h",
                metrics={},
            )
        )
        s.add(
            EventNode(
                id="e",
                project_id="p",
                type="origin",
                start_message_id="1",
                end_message_id="1",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        s.flush()
        s.add(
            Branch(
                id="b",
                project_id="p",
                origin_event_id="e",
                model_version_id="m",
                title="test",
                origin_time=datetime.now(UTC),
            )
        )
        s.flush()
        s.add(
            RuntimeSnapshotRow(
                id="snapshot",
                branch_id="b",
                cutoff_at=datetime.now(UTC),
                timezone="UTC",
                snapshot_mode="latest_profile",
                source_message_ids=[],
                profile={},
                routine_profile={},
                compiler_version="test",
            )
        )
        s.commit()
        yield s


def with_pending_inputs(output, messages):
    """旧冒烟替身也明确遵守新增的输入处理回执契约。"""
    if "text" in output and "action" not in output:
        return {
            "status": "ready",
            "messages": [
                {"kind": "text", "text": text} for text in output.get("bubbles") or [output["text"]]
            ],
        }
    if output.get("action") == "speak" and not (
        output.get("plan_request") or output.get("peer_reply")
    ):
        for message in messages:
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.startswith("<runtime_context>"):
                packet = json.loads(
                    content.removeprefix("<runtime_context>").removesuffix("</runtime_context>")
                )
                pending = packet["branch"]["working_window"].get("pending_message_refs", [])
                output = {
                    **output,
                    "input_resolutions": [
                        {"message_ref": ref, "status": "completed"} for ref in pending
                    ],
                }
    return output


class Model:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.inputs = []

    def bind_tools(self, tools, **kwargs):
        # LangChain 同时接受 BaseTool 与已转换的原生工具定义（分工具 strict 策略）。
        names = [
            tool["function"]["name"] if isinstance(tool, dict) else tool.name for tool in tools
        ]
        self.submission_name = next((name for name in names if name.startswith("submit_")), None)
        return self

    def invoke(self, messages):
        from submission_helpers import submission_message

        self.inputs.append(messages)
        return submission_message(
            content=json.dumps(with_pending_inputs(self.outputs.pop(0), messages)),
            name=getattr(self, "submission_name", None),
        )


def test_plan_preparation_retry_does_not_destroy_committed_blocks(session):
    from datetime import timedelta

    from moonlightbox.runtime_v1.plan_status import can_schedule, preparation, set_preparation

    now = datetime.now(UTC)
    row = RuntimeDayPlanRow(
        branch_id="b",
        plan_date=now.date().isoformat(),
        version=7,
        blocks=[{"activity": "已确认安排"}],
        generation_metadata={"status": "agent"},
    )
    session.add(row)
    session.flush()
    assert preparation(row)["status"] == "ready"
    for attempt in range(1, 5):
        set_preparation(session, "b", now.date(), "running", now=now)
        set_preparation(session, "b", now.date(), "retryable_failed", error_code="network", now=now)
        assert preparation(row)["attempt"] == attempt
        assert preparation(row)["retry_at"] is None  # 退避时间等待 Job 回填。
        assert row.version == 7 and row.blocks == [{"activity": "已确认安排"}]
        assert can_schedule(row, now + timedelta(minutes=10))
    # 准备对象不再单独按次数终止；由 Job 的预算终态同步为 blocked。
    set_preparation(session, "b", now.date(), "blocked", error_code="retry_exhausted", now=now)
    assert preparation(row)["status"] == "blocked"
    assert not can_schedule(row, now + timedelta(days=1))


def test_scheduler_resumes_one_failed_plan_job_after_backoff(session):
    from datetime import timedelta

    from moonlightbox.jobs.models import Job
    from moonlightbox.jobs.service import JobService
    from moonlightbox.runtime_v1.jobs import RUNTIME_CYCLE_JOB_KIND, enqueue_due_runtime_cycles
    from moonlightbox.runtime_v1.plan_status import set_preparation

    bootstrap = RuntimeService(session).bootstrap("p", "b")
    day = bootstrap["plan"].plan_date
    now = datetime.now(UTC)
    job = JobService(session).enqueue_unique(
        RUNTIME_CYCLE_JOB_KIND,
        {
            "project_id": "p",
            "branch_id": "b",
            "plan_date": day,
            "input_revision": session.get(Branch, "b").runtime_input_revision,
        },
        dedupe_key="plan-test",
    )
    job.status = "failed"
    set_preparation(
        session, "b", datetime.fromisoformat(day).date(), "running", job_id=job.id, now=now
    )
    set_preparation(session, "b", datetime.fromisoformat(day).date(), "retryable_failed", now=now)
    session.commit()
    database = Database(str(session.get_bind().url))
    enqueue_due_runtime_cycles(database, now=now)
    session.expire_all()
    assert session.get(Job, job.id).status == "failed"
    enqueue_due_runtime_cycles(database, now=now + timedelta(seconds=31))
    session.expire_all()
    assert session.get(Job, job.id).status == "queued"
    assert (
        len([item for item in session.scalars(select(Job)) if item.payload.get("plan_date") == day])
        == 1
    )
    database.close()


def test_planning_work_restores_and_survives_compaction():
    from langchain_core.messages import SystemMessage
    from moonlightbox.agent_runtime.policy import DAY_PLAN_RUNTIME_POLICY
    from moonlightbox.runtime_v1.tools.plan_work import PlanningToolbox

    first = PlanningToolbox(DAY_PLAN_RUNTIME_POLICY)
    result = first.work_tool().invoke(
        {
            "established_arrangements": ["保留已经确认的工作安排"],
            "open_questions": ["晚餐地点"],
            "draft_blocks": [{"start": "18:00", "activity": "晚餐候选"}],
            "next_steps": ["只补晚餐地点，不再重复调查工作时间"],
        }
    )
    restored = PlanningToolbox(DAY_PLAN_RUNTIME_POLICY)
    restored.restore([result])
    assert restored.work == first.work
    messages, _ = restored.compact(
        [SystemMessage(content="planner"), HumanMessage(content="task")],
        source_refs=[],
        unresolved=[],
    )
    assert "晚餐候选" in str(messages)
    assert "不再重复调查工作时间" in str(messages)


def test_explicit_background_binding_uses_publication_and_is_idempotent(session):
    from moonlightbox.imports.models import ImportSource, Participant
    from moonlightbox.runtime_v1.db_models import RuntimeLifeEventRow
    from moonlightbox.world.models import PersonWorldProfile, WorldGraphVersion, WorldPublication

    session.add(
        ImportSource(
            id="import",
            project_id="p",
            preview_id="preview",
            source_path="test",
            message_count=0,
            confirmed_at=datetime.now(UTC),
        )
    )
    session.add(Participant(id="target", project_id="p", name="目标", role="target"))
    session.flush()
    session.add(
        WorldGraphVersion(
            id="graph",
            project_id="p",
            trigger_import_id="import",
            workspace_key="test",
            status="ready",
            source_fingerprint="s",
            config_fingerprint="c",
            source_import_ids=[],
            compiler_version="test",
        )
    )
    session.flush()
    profile = PersonWorldProfile(
        id="profile",
        project_id="p",
        subject_person_id="target",
        graph_version_id="graph",
        profile_schema_version="v3",
        compiler_version="test",
        profile_v3={
            "life_context": {"overview": "正在工作"},
            "practices": {"overview": "弹性作息"},
        },
        identity={},
        work_and_education=[],
        places=[],
        social_relationships=[],
        preferences=[],
        recurring_activities=[],
        routine_summary={},
        life_phases=[],
        relationship_with_user={},
        important_events=[],
        unresolved_candidates=[],
        source_message_ids=[],
        retrieval_manifest=[],
    )
    session.add(profile)
    session.flush()
    session.add(
        WorldPublication(id="pub", project_id="p", graph_version_id="graph", profile_id="profile")
    )
    session.commit()
    service = RuntimeService(session)
    binding = service.refresh_published_background("p", "b")
    snapshot = session.get(RuntimeSnapshotRow, "snapshot")
    assert binding["profile_id"] == "profile"
    assert snapshot.profile["profile_schema_version"] == "v3"
    assert snapshot.profile["life_context"]["overview"] == "正在工作"
    assert snapshot.profile["practices"]["overview"] == "弹性作息"
    revision = session.get(Branch, "b").runtime_input_revision
    assert service.refresh_published_background("p", "b") == binding
    assert session.get(Branch, "b").runtime_input_revision == revision
    events = list(
        session.scalars(
            select(RuntimeLifeEventRow).where(
                RuntimeLifeEventRow.event_type == "background_rebound"
            )
        )
    )
    assert len(events) == 1
    assert events[0].payload["previous"]["profile"] == {}


def proposal(day):
    return {
        "plan_date": day,
        "blocks": [
            {
                "start": "00:00",
                "end": "24:00",
                "activity": "测试安排",
                "basis": "fallback",
                "confidence": "fallback",
                "evidence_ids": [],
            }
        ],
    }


def speak():
    return {
        "action": "speak",
        "speech_mode": "reply",
        "reply": {"status": "ready", "messages": [{"kind": "text", "text": "你好"}]},
    }


def test_greeting_does_not_generate_missing_plan(session):
    director, planner = Model([speak()]), Model([])
    service = RuntimeService(
        session,
        director_model=director,
        planner_model=planner,
        actor_model=Model([{"text": "你好"}]),
    )
    service.submit_user_message(
        project_id="p", branch_id="b", content="你好", idempotency_key="one"
    )
    result = service.process_next(project_id="p", branch_id="b")
    assert result["message"].content == "你好"
    from moonlightbox.runtime_v1.branch_models import BranchMessage

    users = session.scalars(select(BranchMessage).where(BranchMessage.role == "user")).all()
    assert all(m.generation_metadata["input_status"] == "completed" for m in users)
    assert not planner.inputs
    assert len(result["packet"].current["day_plans"]) == 3


def test_new_message_does_not_invalidate_current_reply(session):
    """最终决定返回期间新增消息，不作废本轮，也不把新消息误标为已处理。"""
    from moonlightbox.runtime_v1.branch_models import BranchMessage
    from moonlightbox.runtime_v1.db_models import RuntimeEventRow

    arrivals = []

    class ArrivingModel(Model):
        def invoke(self, messages):
            if not arrivals:
                with Session(session.get_bind()) as incoming:
                    result = RuntimeService(incoming).submit_user_message(
                        project_id="p", branch_id="b", content="我刚到家", idempotency_key="later"
                    )
                    arrivals.append((result["message"].id, result["event"].id))
            return super().invoke(messages)

    director = ArrivingModel([speak()])
    actor = Model([])
    service = RuntimeService(
        session, director_model=director, actor_model=actor, planner_model=Model([])
    )
    generation = session.get(Branch, "b").runtime_input_revision
    service.submit_user_message(
        project_id="p",
        branch_id="b",
        content="你好",
        idempotency_key="first",
        occurred_at=session.get(Branch, "b").origin_time,
    )
    result = service.process_next(project_id="p", branch_id="b")
    session.expire_all()
    assert result["message"].content == "你好"
    assert len(director.inputs) == 1
    assert not actor.inputs  # 本版表达已经内化到 Director 的 speaking skill。
    assert session.get(Branch, "b").runtime_input_revision == generation
    message_id, event_id = arrivals[0]
    assert session.get(BranchMessage, message_id).generation_metadata["input_status"] == "pending"
    assert session.get(RuntimeEventRow, event_id).status == "queued"


def test_prepare_then_modify_through_shared_receipt(session):
    day = datetime.now(UTC).date().isoformat()
    planner = Model(
        [
            proposal(day),
            {"reply": {"kind": "clarification", "content": "是否保留活动？"}},
            proposal(day),
        ]
    )
    director = Model(
        [
            {"action": "wait", "plan_request": {"reason": "调整安排但保留已开始活动"}},
            {"action": "wait", "peer_reply": {"kind": "answer", "content": "保留"}},
            speak(),
        ]
    )
    service = RuntimeService(
        session,
        director_model=director,
        planner_model=planner,
        actor_model=Model([{"text": "已处理"}]),
    )
    prepared = service.process_next(project_id="p", branch_id="b", prepare_branch=True)
    assert prepared["message"] is None
    assert not director.inputs
    service.submit_user_message(
        project_id="p", branch_id="b", content="调整安排", idempotency_key="one"
    )
    result = service.process_next(project_id="p", branch_id="b")
    assert result["message"] is None
    from moonlightbox.jobs.models import Job
    from moonlightbox.runtime_v1.collaboration.planner_tasks import run_planner_task

    def run_pending_plan():
        job = next(
            j
            for j in session.scalars(select(Job).where(Job.status == "queued"))
            if j.payload.get("planner_task")
        )
        job.status, job.worker_token = "running", "test"
        from datetime import timedelta

        job.lease_expires_at = datetime.now(UTC) + timedelta(hours=1)
        session.commit()
        run_planner_task(service, job)
        job.status, job.worker_token, job.lease_expires_at = "succeeded", None, None
        session.commit()

    run_pending_plan()  # 持久化问题并退出，不占住 Director。
    from moonlightbox.runtime_v1.plan_status import can_schedule, preparation

    waiting_plan = service.get_plan("p", "b")
    assert preparation(waiting_plan)["status"] == "blocked"
    assert not can_schedule(waiting_plan, datetime.now(UTC))
    assert service.process_next(project_id="p", branch_id="b")["message"] is None
    run_pending_plan()  # 回答后继续已有规划，提交并通知。
    result = service.process_next(project_id="p", branch_id="b")
    assert result["message"].content == "你好"
    assert len(director.inputs) == 3
    assert "commit_result" in str(director.inputs[-1])
    assert "是否保留活动" in str(director.inputs[-1])
    assert session.scalar(select(RuntimeDayPlanRow)).version == 3


def test_shared_window_uses_local_day_and_missing_is_not_available(session):
    now = datetime(2026, 5, 9, 17, tzinfo=UTC)
    window = shared_plan_window(session, "b", now, "Asia/Shanghai")
    assert list(window) == ["2026-05-09", "2026-05-10", "2026-05-11"]
    assert all(slot["status"] == "missing" for slot in window.values())
    assert window["2026-05-09"]["read_only"]


def test_shared_messages_cannot_be_overwritten():
    old = HumanMessage(id="one", content="原消息")
    assert len(merge_shared_messages([old], [old])) == 1
    with pytest.raises(ValueError, match="不可覆盖"):
        merge_shared_messages([old], [HumanMessage(id="one", content="篡改")])


def test_next_day_maintenance_never_calls_director(session):
    from datetime import timedelta

    from moonlightbox.runtime_v1.db_models import RuntimeLifeStateRow

    today = datetime.now(UTC).date()
    planner = Model(
        [proposal(today.isoformat()), proposal((today + timedelta(days=1)).isoformat())]
    )
    director = Model([])
    service = RuntimeService(session, director_model=director, planner_model=planner)
    service.process_next(project_id="p", branch_id="b", prepare_branch=True)
    before = session.scalar(
        select(RuntimeLifeStateRow).where(RuntimeLifeStateRow.is_current.is_(True))
    ).version
    service.process_next(
        project_id="p", branch_id="b", plan_date=(today + timedelta(days=1)).isoformat()
    )
    assert len(list(session.scalars(select(RuntimeDayPlanRow)))) == 2
    assert (
        session.scalar(
            select(RuntimeLifeStateRow).where(RuntimeLifeStateRow.is_current.is_(True))
        ).version
        == before
    )
    assert not director.inputs
    from moonlightbox.runtime_v1.db_models import RuntimeEventRow

    notices = list(
        session.scalars(
            select(RuntimeEventRow).where(RuntimeEventRow.event_type == "agent_message")
        )
    )
    assert len(notices) == 2
    assert all(n.payload["discussion"][0]["payload"]["status"] == "committed" for n in notices)


def test_shared_message_tool_batches_and_completes_without_ping_pong(session):
    """真实工具→持久队列→同一 Planner 图→无修改完成；不调用任何云端。"""
    from datetime import timedelta

    from moonlightbox.agent_runtime.tool_errors import ToolInputError
    from moonlightbox.jobs.models import Job
    from moonlightbox.runtime_v1.collaboration.planner_tasks import run_planner_task
    from moonlightbox.runtime_v1.collaboration.transport import peer_history, schedule_planner_inbox
    from moonlightbox.runtime_v1.db_models import RuntimeEventRow
    from moonlightbox.runtime_v1.tools.send_agent_message import build_send_agent_message_tool

    planner = Model(
        [proposal(datetime.now(UTC).date().isoformat()), {"completion": "已理解，现有计划无需调整"}]
    )
    runtime = RuntimeService(session, planner_model=planner)
    runtime.process_next(project_id="p", branch_id="b", prepare_branch=True)
    tool = build_send_agent_message_tool(session, branch_id="b", sender="director", task_id="test")
    with pytest.raises(ToolInputError):
        tool.invoke({"recipient": "director", "content": "不能给自己"})
    args = {"recipient": "day_planner", "content": "今天不想再接额外任务，先保留原计划"}
    first = tool.invoke(args)
    assert tool.invoke(args)["message_id"] == first["message_id"]
    second = tool.invoke({"recipient": "day_planner", "content": "只是同步感受，不需要重新排程"})
    assert len(peer_history(session, "b", recipient="day_planner")) == 0
    assert len(peer_history(session, "b", recipient="director")) == 2
    now = runtime.get_clock("p", "b").now()
    assert schedule_planner_inbox(session, "b", now)
    session.commit()
    assert not schedule_planner_inbox(session, "b", now)
    job = next(j for j in session.scalars(select(Job)) if j.payload.get("planner_task"))
    assert set(job.payload["planner_task"]["input_ids"]) == {
        first["message_id"],
        second["message_id"],
    }
    before = runtime.get_plan("p", "b").version
    job.status, job.worker_token = "running", "test"
    job.lease_expires_at = datetime.now(UTC) + timedelta(hours=1)
    session.commit()
    run_planner_task(runtime, job)
    assert not job.checkpoint["awaiting_answer"]
    assert runtime.get_plan("p", "b").version == before
    assert session.get(RuntimeEventRow, first["message_id"]).status == "completed"
    assert len(peer_history(session, "b", recipient="day_planner")) == 2
    assert not list(
        session.scalars(
            select(RuntimeEventRow).where(RuntimeEventRow.event_type == "planner_notification")
        )
    )
    assert "今天不想再接额外任务" in str(planner.inputs[-1])
    run_planner_task(runtime, job)  # 崩溃恢复复用已完成检查点，不重新请求模型。
    assert len(planner.inputs) == 2
    back = build_send_agent_message_tool(
        session, branch_id="b", sender="day_planner", task_id="reply-test"
    ).invoke({"recipient": "director", "content": "原计划留有独处时间，无需调整"})
    receiver = Model([{"action": "wait", "private_reason": "知悉，无需告诉用户"}])
    RuntimeService(session, director_model=receiver).process_next(project_id="p", branch_id="b")
    session.expire_all()
    assert session.get(RuntimeEventRow, back["message_id"]).status == "completed"
    assert "原计划留有独处时间" in str(receiver.inputs[-1])


def test_peer_outbox_rollback_and_cancel(session):
    from moonlightbox.agent_runtime.resilience import ExecutionInterrupted
    from moonlightbox.runtime_v1.collaboration.transport import enqueue_peer_message
    from moonlightbox.runtime_v1.db_models import RuntimeEventRow
    from moonlightbox.runtime_v1.tools.send_agent_message import build_send_agent_message_tool

    enqueue_peer_message(
        session,
        branch=session.get(Branch, "b"),
        sender="executor",
        recipient="director",
        content="未提交的计划不能通知",
        task_id="rollback",
        occurred_at=datetime.now(UTC),
    )
    session.rollback()
    assert session.scalar(select(RuntimeEventRow.id)) is None
    tool = build_send_agent_message_tool(
        session,
        branch_id="b",
        sender="director",
        task_id="cancel",
        cancellation_requested=lambda: True,
    )
    with pytest.raises(ExecutionInterrupted):
        tool.invoke({"recipient": "day_planner", "content": "不能发出"})
    assert session.scalar(select(RuntimeEventRow.id)) is None


def test_director_calls_shared_tool_without_consuming_planner_inbox(session):
    from moonlightbox.runtime_v1.db_models import RuntimeEventRow

    class SendingDirector(Model):
        def invoke(self, messages):
            if not self.inputs:
                self.inputs.append(messages)
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "send-one",
                            "name": "send_agent_message",
                            "args": {"recipient": "day_planner", "content": "今天留些独处时间"},
                        }
                    ],
                )
            return super().invoke(messages)

    director = SendingDirector([{"action": "wait", "private_reason": "等候计划伙伴意见"}])
    service = RuntimeService(
        session,
        director_model=director,
        planner_model=Model([proposal(datetime.now(UTC).date().isoformat())]),
    )
    service.process_next(project_id="p", branch_id="b", prepare_branch=True)
    service.process_next(project_id="p", branch_id="b")
    assert len(director.inputs) == 2
    row = session.scalar(
        select(RuntimeEventRow).where(RuntimeEventRow.event_type == "planner_inbox")
    )
    assert row.status == "queued"
    assert row.payload["discussion"][0]["sender"] == "director"
    assert "message_id" in str(director.inputs[-1])


def test_plan_commit_receipt_is_idempotent_and_version_checked(session):
    from moonlightbox.runtime_v1.executor import RuntimeExecutor
    from moonlightbox.runtime_v1.schemas import DayPlanProposal

    service = RuntimeService(session)
    bootstrap = service.bootstrap("p", "b")
    value = DayPlanProposal.model_validate_json(
        json.dumps(proposal(datetime.now(UTC).date().isoformat()))
    )
    executor = RuntimeExecutor(session)
    kwargs = dict(
        branch_id="b",
        proposal=value,
        virtual_now=bootstrap["clock"].now(),
        mode="initial",
        expected_version=1,
    )
    first = executor.commit_day_plan_proposal(**kwargs, idempotency_key="once")
    session.commit()
    assert first.version == 2
    again = executor.commit_day_plan_proposal(**kwargs, idempotency_key="once")
    assert again.version == 2
    with pytest.raises(ValueError, match="版本冲突"):
        executor.commit_day_plan_proposal(**kwargs, idempotency_key="different")


def test_planner_retry_restores_completed_tool_result(session):
    from moonlightbox.runtime_v1.service import RuntimeModelExecutionError

    class Interrupted:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "constraint-call", "name": "get_plan_constraints", "args": {}},
                        {
                            "id": "work-call",
                            "name": "update_plan_work",
                            "args": {
                                "established_arrangements": ["保留已确认安排"],
                                "open_questions": [],
                                "draft_blocks": [{"activity": "恢复草稿"}],
                                "next_steps": ["完成提案"],
                            },
                        },
                    ],
                )
            raise RuntimeError("offline interrupted provider")

    interrupted = RuntimeService(session, planner_model=Interrupted())
    with pytest.raises(RuntimeModelExecutionError):
        interrupted.process_next(project_id="p", branch_id="b", prepare_branch=True)
    resumed = Model([proposal(datetime.now(UTC).date().isoformat())])
    service = RuntimeService(session, planner_model=resumed)
    result = service.process_next(project_id="p", branch_id="b", prepare_branch=True)
    assert result["trace"].status == "succeeded"
    assert len(resumed.inputs) == 1
    assert any(
        getattr(message, "tool_call_id", None) == "constraint-call" for message in resumed.inputs[0]
    )
    assert "恢复草稿" in str(resumed.inputs[0])
