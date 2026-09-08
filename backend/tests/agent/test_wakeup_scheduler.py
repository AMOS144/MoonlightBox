from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Lock, Thread

import pytest
from moonlightbox.agent.models import (
    AgentWakeup,
    CognitiveCycle,
    MentalStateVersion,
    PerceptionEvent,
)
from moonlightbox.branches.models import Branch
from moonlightbox.db import Base, Database
from moonlightbox.events.models import EventNode
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy import func, select
from sqlalchemy.orm import Session

NOW = datetime(2026, 7, 23, 8, 0, tzinfo=UTC)


def _seed_branch(session: Session) -> None:
    session.add(Project(id="project-1", name="project-1"))
    session.flush()
    session.add(
        ModelVersion(
            id="model-1",
            project_id="project-1",
            base_model="test",
            adapter_path="test",
            dataset_hash="model-1",
            metrics={},
        )
    )
    session.add(
        EventNode(
            id="origin-1",
            project_id="project-1",
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
    session.flush()
    session.add(
        Branch(
            id="branch-1",
            project_id="project-1",
            origin_event_id="origin-1",
            model_version_id="model-1",
            title="branch-1",
            origin_time=NOW,
            state_snapshot={},
        )
    )
    session.commit()


def _schedule(
    session: Session,
    *,
    key: str,
    wake_at: datetime,
) -> AgentWakeup:
    from moonlightbox.agent.service import ShadowCognitionService

    return ShadowCognitionService(session).schedule_wakeup(
        project_id="project-1",
        branch_id="branch-1",
        wake_at=wake_at,
        reason=f"唤醒原因-{key}",
        idempotency_key=key,
        model_version_id="model-1",
    )


def test_due_wakeup_creates_idempotent_event_state_cycle_and_job(
    tmp_path: Path,
) -> None:
    from moonlightbox.agent.jobs import COGNITIVE_CYCLE_JOB_KIND
    from moonlightbox.agent.service import process_due_wakeups

    database = Database(f"sqlite:///{tmp_path / 'due.db'}")
    Base.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _seed_branch(session)
        due = _schedule(session, key="due", wake_at=NOW)
        future = _schedule(session, key="future", wake_at=NOW + timedelta(minutes=1))

        assert process_due_wakeups(session, now=NOW, limit=10) == 1
        assert process_due_wakeups(session, now=NOW, limit=10) == 0

        session.expire_all()
        assert session.get(AgentWakeup, due.id).status == "completed"
        assert session.get(AgentWakeup, future.id).status == "scheduled"
        event = session.scalar(select(PerceptionEvent))
        assert event is not None
        assert event.event_type == "elapsed_time"
        assert event.idempotency_key == f"wakeup:{due.id}"
        assert event.evidence == {
            "wakeup_id": due.id,
            "reason": "唤醒原因-due",
        }
        state = session.scalar(select(MentalStateVersion))
        assert state is not None
        assert state.is_current is True
        cycle = session.scalar(select(CognitiveCycle))
        assert cycle is not None
        assert cycle.trigger_event_id == event.id
        job = session.scalar(select(Job))
        assert job is not None
        assert job.kind == COGNITIVE_CYCLE_JOB_KIND
        assert job.payload == {"cycle_id": cycle.id}
        assert job.dedupe_key == f"subject-cognitive-cycle:{cycle.id}"


def test_two_concurrent_scanners_do_not_duplicate_wakeup_consumption(
    tmp_path: Path,
) -> None:
    from moonlightbox.agent.service import process_due_wakeups

    database_url = f"sqlite:///{tmp_path / 'race.db'}"
    setup = Database(database_url)
    Base.metadata.create_all(setup.engine)
    with Session(setup.engine) as session:
        _seed_branch(session)
        wakeup_id = _schedule(session, key="race", wake_at=NOW).id
    setup.close()

    barrier = Barrier(2)
    result_lock = Lock()
    results: list[int] = []
    errors: list[Exception] = []

    def scan() -> None:
        database = Database(database_url)
        try:
            with Session(database.engine) as session:
                barrier.wait()
                result = process_due_wakeups(session, now=NOW, limit=1)
            with result_lock:
                results.append(result)
        except Exception as error:
            with result_lock:
                errors.append(error)
        finally:
            database.close()

    threads = [Thread(target=scan), Thread(target=scan)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(results) == [0, 1]
    verify = Database(database_url)
    with Session(verify.engine) as session:
        assert session.get(AgentWakeup, wakeup_id).status == "completed"
        assert session.scalar(select(func.count()).select_from(PerceptionEvent)) == 1
        assert session.scalar(select(func.count()).select_from(CognitiveCycle)) == 1
        assert session.scalar(select(func.count()).select_from(Job)) == 1
    verify.close()


def test_failed_wakeup_transaction_remains_scheduled_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.agent.service import process_due_wakeups

    database = Database(f"sqlite:///{tmp_path / 'recovery.db'}")
    Base.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        _seed_branch(session)
        wakeup = _schedule(session, key="recover", wake_at=NOW)

        original = JobService.enqueue_unique

        def fail_enqueue(*_args: object, **_kwargs: object) -> Job:
            raise RuntimeError("模拟任务入队失败")

        monkeypatch.setattr(JobService, "enqueue_unique", fail_enqueue)
        with pytest.raises(RuntimeError, match="模拟任务入队失败"):
            process_due_wakeups(session, now=NOW, limit=1)

        session.expire_all()
        assert session.get(AgentWakeup, wakeup.id).status == "scheduled"
        assert session.scalar(select(func.count()).select_from(PerceptionEvent)) == 0
        assert session.scalar(select(func.count()).select_from(CognitiveCycle)) == 0
        assert session.scalar(select(func.count()).select_from(Job)) == 0

        monkeypatch.setattr(JobService, "enqueue_unique", original)
        assert process_due_wakeups(session, now=NOW, limit=1) == 1
        assert session.get(AgentWakeup, wakeup.id).status == "completed"
