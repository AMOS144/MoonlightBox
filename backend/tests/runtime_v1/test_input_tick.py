"""三秒节拍的确定性边界冒烟，不等待真实三秒、不调用供应商。"""

# pytest fixture 按名称注入。
# ruff: noqa: F811
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier

from langchain_core.messages import AIMessage, ToolMessage
from moonlightbox.jobs.models import Job
from moonlightbox.runtime_v1.branch_models import Branch
from moonlightbox.runtime_v1.db_models import RuntimeEventRow, RuntimeWakeupRow
from moonlightbox.runtime_v1.event_queue import RuntimeEventQueue
from moonlightbox.runtime_v1.executor import RuntimeExecutor
from moonlightbox.runtime_v1.jobs import enqueue_input_tick
from moonlightbox.runtime_v1.service import RuntimeService
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_peer_collaboration import Model, session, speak  # noqa: F401


def test_more_than_16_inputs_and_idle_wait(session):
    class WaitModel(Model):
        def invoke(self, messages):
            packet = json.loads(
                next(
                    m.content
                    for m in messages
                    if isinstance(m.content, str) and m.content.startswith("<runtime_context>")
                )
                .removeprefix("<runtime_context>")
                .removesuffix("</runtime_context>")
            )
            refs = packet["branch"]["working_window"]["pending_message_refs"]
            assert len(packet["trigger"]["input_events"]) == len(refs) == 25
            self.outputs = [
                {
                    "action": "wait",
                    "input_resolutions": [
                        {"message_ref": ref, "status": "no_response_needed"} for ref in refs
                    ],
                }
            ]
            return super().invoke(messages)

    model = WaitModel([])
    service = RuntimeService(session, director_model=model)
    for i in range(25):
        service.submit_user_message(
            project_id="p",
            branch_id="b",
            content=str(i),
            idempotency_key=str(i),
            occurred_at=session.get(Branch, "b").origin_time - timedelta(seconds=1),
        )
    assert all(j.payload.get("conversation_maintenance") for j in session.scalars(select(Job)))
    assert service.process_next(project_id="p", branch_id="b")["message"] is None
    assert all(e.status == "completed" for e in session.scalars(select(RuntimeEventRow)))
    assert service.process_next(project_id="p", branch_id="b") is None
    assert len(model.inputs) == 1


def test_next_thought_collects_user_peer_and_wakeup_with_skill_result(session):
    class Continuation(Model):
        def invoke(self, messages):
            if not self.inputs:
                self.inputs.append(messages)
                with Session(session.get_bind()) as incoming:
                    runtime = RuntimeService(incoming)
                    runtime.submit_user_message(
                        project_id="p",
                        branch_id="b",
                        content="还有，刚刚到家了",
                        idempotency_key="two",
                    )
                    now = runtime.get_clock("p", "b").now()
                    RuntimeEventQueue(incoming).enqueue(
                        project_id="p",
                        branch_id="b",
                        event_type="planner_notification",
                        payload={"content": "计划已调整"},
                        idempotency_key="peer",
                        occurred_at=now - timedelta(days=1),  # 迟到的发生时刻不能跳过。
                    )
                    incoming.add(
                        RuntimeWakeupRow(
                            branch_id="b",
                            wake_at=now,
                            reason="休息结束",
                            trigger_type="plan_transition",
                            idempotency_key="wake",
                        )
                    )
                    incoming.commit()
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "skill", "name": "read_skill", "args": {"skill": "speaking"}}
                    ],
                )
            assert any(isinstance(m, ToolMessage) and m.tool_call_id == "skill" for m in messages)
            context = next(
                m.content
                for m in messages
                if isinstance(m.content, str) and m.content.startswith("<runtime_context>")
            )
            packet = json.loads(
                context.removeprefix("<runtime_context>").removesuffix("</runtime_context>")
            )
            events = packet["trigger"]["input_events"]
            assert [e["type"] for e in events] == [
                "user_message",
                "user_message",
                "planner_notification",
                "plan_transition",
            ]
            assert all(e["source_event_id"] for e in events)
            assert "刚刚到家了" in context and "计划已调整" in context
            return super().invoke(messages)

    model = Continuation([speak()])
    runtime = RuntimeService(session, director_model=model)
    runtime.submit_user_message(
        project_id="p", branch_id="b", content="你好", idempotency_key="one"
    )
    assert runtime.process_next(project_id="p", branch_id="b")["message"].content == "你好"
    assert len(model.inputs) == 2
    assert all(e.status == "completed" for e in session.scalars(select(RuntimeEventRow)))
    assert all(
        w.status == "completed"
        for w in session.scalars(select(RuntimeWakeupRow))
        if w.idempotency_key == "wake"
    )


def test_scanners_atomically_enqueue_one_director_and_respect_failure(session):
    RuntimeService(session).bootstrap("p", "b")
    barrier = Barrier(2)
    bind = session.get_bind()
    session.commit()

    def scan(ident):
        with Session(bind) as worker:
            barrier.wait()
            result = enqueue_input_tick(worker, "b", ident)
            worker.commit()
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(scan, ["first", "second"])) == [False, True]
    job = session.scalar(select(Job).where(Job.kind == "runtime-v1-cycle"))
    job.status = "failed"
    session.commit()
    assert not enqueue_input_tick(session, "b", "later")  # 不绕过失败退避和总重试上限。
    session.rollback()


def test_accepted_decision_recovers_original_inputs_and_commit_is_idempotent(session, monkeypatch):
    import pytest

    job = Job(
        kind="runtime-v1-cycle",
        status="running",
        worker_token="test",
        lease_expires_at=datetime.now(UTC) + timedelta(hours=1),
        payload={"branch_id": "b", "input_tick": True},
    )
    session.add(job)
    session.commit()
    model = Model([speak()])
    runtime = RuntimeService(session, director_model=model)
    first = runtime.submit_user_message(
        project_id="p", branch_id="b", content="第一句", idempotency_key="first"
    )
    first_id = first["event"].id
    original = RuntimeExecutor.commit
    crash = [True]

    def commit(executor, **kwargs):
        if crash.pop() if crash else False:
            raise RuntimeError("模拟接受决定之后、事务之前重启")
        return original(executor, **kwargs)

    monkeypatch.setattr(RuntimeExecutor, "commit", commit)
    with pytest.raises(RuntimeError, match="模拟"):
        runtime.process_next(project_id="p", branch_id="b", job_id=job.id)
    later = runtime.submit_user_message(
        project_id="p", branch_id="b", content="第二句", idempotency_key="later"
    )
    later_id = later["event"].id
    # 新的门面对象模拟重启，复用 Job 与磁盘检查点，而不是 Python 闭包。
    resumed = RuntimeService(session, director_model=model)
    result = resumed.process_next(project_id="p", branch_id="b", job_id=job.id)
    assert result["message"].content == "你好" and len(model.inputs) == 1
    assert session.get(RuntimeEventRow, first_id).status == "completed"
    assert session.get(RuntimeEventRow, later_id).status == "queued"
    assert resumed.process_next(project_id="p", branch_id="b", job_id=job.id)["replayed"]
    assert len(model.inputs) == 1


def test_wait_does_not_retrigger_until_due_wakeup(session):
    from moonlightbox.runtime_v1.db_models import RuntimeClockRow

    class Deferred(Model):
        def invoke(self, messages):
            if not self.inputs:
                packet = json.loads(
                    next(
                        m.content
                        for m in messages
                        if isinstance(m.content, str) and m.content.startswith("<runtime_context>")
                    )
                    .removeprefix("<runtime_context>")
                    .removesuffix("</runtime_context>")
                )
                self.outputs.insert(
                    0,
                    {
                        "action": "wait",
                        "next_wakeup_at": (
                            datetime.fromisoformat(packet["virtual_now"]) + timedelta(minutes=1)
                        ).isoformat(),
                        "input_resolutions": [
                            {"message_ref": ref, "status": "awaiting_response"}
                            for ref in packet["branch"]["working_window"]["pending_message_refs"]
                        ],
                    },
                )
            return super().invoke(messages)

    model = Deferred([speak()])
    runtime = RuntimeService(session, director_model=model)
    runtime.submit_user_message(
        project_id="p", branch_id="b", content="等你忙完", idempotency_key="wait"
    )
    assert runtime.process_next(project_id="p", branch_id="b")["message"] is None
    assert runtime.process_next(project_id="p", branch_id="b") is None
    assert len(model.inputs) == 1
    clock = session.get(RuntimeClockRow, "b")
    clock.virtual_anchor += timedelta(minutes=2)
    session.commit()
    assert runtime.process_next(project_id="p", branch_id="b")["message"].content == "你好"
    assert len(model.inputs) == 2


def test_local_timezone_does_not_deliver_future_inputs_early(session):
    from datetime import timezone

    from moonlightbox.runtime_v1.db_models import RuntimeClockRow

    model = Model([])
    runtime = RuntimeService(session, director_model=model)
    runtime.bootstrap("p", "b")
    clock = session.get(RuntimeClockRow, "b")
    now = datetime(2026, 9, 13, 2, tzinfo=UTC)
    clock.virtual_anchor, clock.wall_anchor, clock.timezone = (
        now,
        datetime.now(UTC),
        "Asia/Shanghai",
    )
    future = now + timedelta(hours=2)
    event = RuntimeEventQueue(session).enqueue(
        project_id="p",
        branch_id="b",
        event_type="system",
        occurred_at=future,
        payload={},
        idempotency_key="future",
    )
    wakeup = RuntimeExecutor(session).schedule_wakeup(
        branch_id="b",
        wake_at=future.astimezone(timezone(timedelta(hours=8))),
        reason="未来边界",
        trigger_type="plan_transition",
        idempotency_key="future-wake",
    )
    session.commit()
    assert runtime.process_next(project_id="p", branch_id="b") is None
    assert not model.inputs
    assert event.status == "queued" and wakeup.status == "scheduled"
    assert wakeup.wake_at.hour == 4  # 数据库保存 UTC，不是本地中午 12 点。
