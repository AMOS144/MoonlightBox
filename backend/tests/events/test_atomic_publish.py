from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Thread

import pytest
from moonlightbox.db import Database
from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.events.ranking import RankableCandidate, score_candidate
from moonlightbox.events.reviewer import EventCandidate, EventCandidateReview
from moonlightbox.events.runs import AnalysisRunService
from moonlightbox.events.schemas import ReviewedEvent
from moonlightbox.events.service import EventService
from moonlightbox.imports.models import ImportSource
from moonlightbox.jobs.service import (
    InvalidJobTransitionError,
    JobLeaseLostError,
    JobService,
)
from moonlightbox.projects.models import Project
from sqlalchemy import func, select
from sqlalchemy.orm import Session


def _session_with_running_run(
    tmp_path: Path,
) -> tuple[Database, Session, AnalysisRun]:
    database = Database(f"sqlite:///{tmp_path / 'publish.db'}")
    AnalysisRun.metadata.create_all(database.engine)
    session = Session(database.engine)
    session.add(Project(id="project-1", name="原子发布测试"))
    session.flush()
    session.add(
        ImportSource(
            id="import-1",
            project_id="project-1",
            preview_id="preview-1",
            source_path="/tmp/chat.json",
            message_count=3,
            confirmed_at=datetime.now(UTC),
        )
    )
    session.commit()
    service = AnalysisRunService(session)
    run = service.get_or_create(
        project_id="project-1",
        import_id="import-1",
        analysis_version="hybrid-v2",
        prompt_version="prompt-v2",
        model="cloud-v2",
        config={"threshold": 0.72, "maximum_nodes": 25},
        window_ids=[],
    )
    service.start(run.id)
    session.commit()
    return database, session, run


def _reviewed_event(*, topic: str = "旧事件") -> ReviewedEvent:
    return ReviewedEvent(
        type="long_pause",
        start_message_id="old-1",
        end_message_id="old-2",
        before_state="中断",
        after_state="恢复",
        emotion_labels=["间隔"],
        topic=topic,
        conflict_level=1,
        importance=0.8,
        reason="旧启发式事件",
        evidence_ids=["old-1", "old-2"],
    )


def _ranked_event(key: str = "candidate-1"):
    candidate = EventCandidate(
        candidate_key=key,
        type="conflict",
        start_message_id="m1",
        end_message_id="m2",
        before_state="正常交流",
        after_state="关系紧张",
        emotion_labels=["生气"],
        topic="边界冲突",
        conflict_level=4,
        state_change_strength=0.9,
        model_confidence=0.9,
        reason="冲突持续",
        evidence_ids=["m1", "m2"],
    )
    return score_candidate(
        RankableCandidate(
            candidate=candidate,
            review=EventCandidateReview(
                accepted=True,
                type_supported=True,
                evidence_alignment=0.9,
                decisive_event=False,
                persistence=0.9,
                evidence_ids=["m3"],
                reason="后续仍有影响",
            ),
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
            ended_at=datetime(2026, 1, 2, tzinfo=UTC),
            valid_follow_up_ids=("m3",),
        )
    )


def _running_job(
    session: Session,
    *,
    import_id: str = "import-1",
) -> tuple[str, str]:
    job = JobService(session).enqueue(
        "event_analysis_v2",
        {"project_id": "project-1", "import_id": import_id},
    )
    claimed = JobService(session).claim_next_queued(
        worker_token="worker-token",
        lease_duration=timedelta(minutes=2),
    )
    assert claimed is not None
    return job.id, "worker-token"


def test_cancel_wins_before_publish_and_rolls_back_all_publish_writes(
    tmp_path: Path,
) -> None:
    database, setup_session, run = _session_with_running_run(tmp_path)
    job_id, token = _running_job(setup_session)
    run_id = run.id
    setup_session.close()
    gate = Barrier(2, timeout=1)
    publish_errors: list[Exception] = []

    def cancel_first() -> None:
        with Session(database.engine) as session:
            JobService(session).cancel(job_id)
        gate.wait()

    def publish_second() -> None:
        gate.wait()
        with Session(database.engine) as session:
            try:
                EventService(session).publish_v2(
                    project_id="project-1",
                    run_id=run_id,
                    candidates=[_ranked_event()],
                    job_id=job_id,
                    job_worker_token=token,
                )
            except Exception as error:
                publish_errors.append(error)

    threads = [Thread(target=cancel_first), Thread(target=publish_second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], JobLeaseLostError)
    with Session(database.engine) as session:
        assert JobService(session).get(job_id).status == "cancelled"
        assert session.get(AnalysisRun, run_id).status == "running"  # type: ignore[union-attr]
        assert session.scalar(select(func.count()).select_from(EventNode)) == 0
    database.close()


def test_publish_wins_before_cancel_and_commits_job_run_nodes_together(
    tmp_path: Path,
) -> None:
    database, setup_session, run = _session_with_running_run(tmp_path)
    job_id, token = _running_job(setup_session)
    run_id = run.id
    setup_session.close()
    publish_has_write_lock = Barrier(2, timeout=1)
    cancel_errors: list[Exception] = []

    def publish_first() -> None:
        with Session(database.engine) as session:
            service = EventService(session)
            original_create = service._create_published_event

            def create_then_release_cancel(*args: object, **kwargs: object) -> EventNode:
                created = original_create(*args, **kwargs)  # type: ignore[arg-type]
                publish_has_write_lock.wait()
                return created

            service._create_published_event = create_then_release_cancel  # type: ignore[method-assign]
            service.publish_v2(
                project_id="project-1",
                run_id=run_id,
                candidates=[_ranked_event()],
                job_id=job_id,
                job_worker_token=token,
            )

    def cancel_second() -> None:
        publish_has_write_lock.wait()
        with Session(database.engine) as session:
            try:
                JobService(session).cancel(job_id)
            except Exception as error:
                cancel_errors.append(error)

    threads = [Thread(target=publish_first), Thread(target=cancel_second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(cancel_errors) == 1
    assert isinstance(cancel_errors[0], InvalidJobTransitionError)
    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        assert job.status == "succeeded"
        assert job.progress == 1.0
        assert session.get(AnalysisRun, run_id).status == "succeeded"  # type: ignore[union-attr]
        assert session.scalar(select(func.count()).select_from(EventNode)) == 1
    database.close()


def test_publish_v2_creates_revision_metadata_and_completes_run(
    tmp_path: Path,
) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        created = EventService(session).publish_v2(
            project_id="project-1",
            run_id=run.id,
            candidates=[_ranked_event()],
        )

        revision = session.scalar(
            select(AnalysisRevision).where(AnalysisRevision.event_id == created[0].id)
        )
        session.refresh(run)
        assert len(created) == 1
        assert revision is not None
        assert revision.analysis_version == "hybrid-v2"
        assert revision.prompt_version == "prompt-v2"
        assert revision.model == "cloud-v2"
        assert revision.run_id == run.id
        assert run.status == "succeeded"
        assert run.completed_at is not None
    finally:
        session.close()
        database.close()


def test_publish_supersedes_unedited_heuristic_node(tmp_path: Path) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        old = EventService(session).create(
            "project-1",
            _reviewed_event(),
            analysis_version="heuristic-v1",
            prompt_version="none",
        )

        EventService(session).publish_v2(
            project_id="project-1",
            run_id=run.id,
            candidates=[_ranked_event()],
        )

        session.refresh(old)
        revisions = EventService(session).revisions(old.id)
        assert old.status == "superseded"
        assert revisions[-1].snapshot["status"] == "superseded"
    finally:
        session.close()
        database.close()


def test_publish_preserves_manually_revised_node(tmp_path: Path) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        service = EventService(session)
        old = service.create(
            "project-1",
            _reviewed_event(),
            analysis_version="heuristic-v1",
            prompt_version="none",
        )
        service.revise(old.id, {"topic": "人工确认保留"}, "人工修订")

        service.publish_v2(
            project_id="project-1",
            run_id=run.id,
            candidates=[_ranked_event()],
        )

        session.refresh(old)
        assert old.status == "active"
        assert old.topic == "人工确认保留"
    finally:
        session.close()
        database.close()


def test_publish_supersedes_unedited_hybrid_v2_node(tmp_path: Path) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        service = EventService(session)
        old = service.create(
            "project-1",
            _reviewed_event(topic="旧 V2 自动节点"),
            analysis_version="hybrid-v2",
            prompt_version="prompt-v1",
        )

        service.publish_v2(
            project_id="project-1",
            run_id=run.id,
            candidates=[_ranked_event()],
        )

        session.refresh(old)
        assert old.status == "superseded"
        assert [revision.revision_number for revision in service.revisions(old.id)] == [
            1,
            2,
        ]
    finally:
        session.close()
        database.close()


def test_publish_preserves_manually_revised_hybrid_v2_node(tmp_path: Path) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        service = EventService(session)
        old = service.create(
            "project-1",
            _reviewed_event(topic="旧 V2 自动节点"),
            analysis_version="hybrid-v2",
            prompt_version="prompt-v1",
        )
        service.revise(old.id, {"topic": "人工确认的 V2 节点"}, "人工修订")

        service.publish_v2(
            project_id="project-1",
            run_id=run.id,
            candidates=[_ranked_event()],
        )

        session.refresh(old)
        assert old.status == "active"
        assert old.topic == "人工确认的 V2 节点"
        assert len(service.revisions(old.id)) == 2
    finally:
        session.close()
        database.close()


def test_repeated_publish_is_idempotent(tmp_path: Path) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        service = EventService(session)
        first = service.publish_v2(
            project_id="project-1",
            run_id=run.id,
            candidates=[_ranked_event()],
        )
        second = service.publish_v2(
            project_id="project-1",
            run_id=run.id,
            candidates=[_ranked_event()],
        )

        assert [event.id for event in second] == [event.id for event in first]
        assert session.scalar(select(func.count()).select_from(EventNode)) == 1
    finally:
        session.close()
        database.close()


def test_publish_failure_rolls_back_old_and_new_nodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        service = EventService(session)
        old = service.create(
            "project-1",
            _reviewed_event(),
            analysis_version="heuristic-v1",
            prompt_version="none",
        )

        def fail_creation(*_: object, **__: object) -> EventNode:
            raise RuntimeError("模拟发布写入失败")

        monkeypatch.setattr(service, "_create_published_event", fail_creation)
        with pytest.raises(RuntimeError, match="模拟发布写入失败"):
            service.publish_v2(
                project_id="project-1",
                run_id=run.id,
                candidates=[_ranked_event()],
            )

        session.expire_all()
        assert session.get(EventNode, old.id).status == "active"  # type: ignore[union-attr]
        assert session.scalar(select(func.count()).select_from(EventNode)) == 1
        assert session.get(AnalysisRun, run.id).status == "running"  # type: ignore[union-attr]
    finally:
        session.close()
        database.close()


def test_second_event_failure_rolls_back_flushed_first_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        service = EventService(session)
        old = service.create(
            "project-1",
            _reviewed_event(),
            analysis_version="heuristic-v1",
            prompt_version="none",
        )
        original_create = service._create_published_event
        first_new_event_id: str | None = None
        call_count = 0

        def fail_after_first_flush(*args: object, **kwargs: object) -> EventNode:
            nonlocal call_count, first_new_event_id
            call_count += 1
            if call_count == 2:
                assert first_new_event_id is not None
                assert session.get(EventNode, first_new_event_id) is not None
                raise RuntimeError("第二条节点写入失败")
            created = original_create(*args, **kwargs)  # type: ignore[arg-type]
            first_new_event_id = created.id
            return created

        monkeypatch.setattr(
            service,
            "_create_published_event",
            fail_after_first_flush,
        )
        with pytest.raises(RuntimeError, match="第二条节点写入失败"):
            service.publish_v2(
                project_id="project-1",
                run_id=run.id,
                candidates=[
                    _ranked_event("candidate-1"),
                    _ranked_event("candidate-2"),
                ],
            )

        session.expire_all()
        remaining = list(session.scalars(select(EventNode)))
        assert call_count == 2
        assert [event.id for event in remaining] == [old.id]
        assert remaining[0].status == "active"
        assert session.get(AnalysisRun, run.id).status == "running"  # type: ignore[union-attr]
    finally:
        session.close()
        database.close()


def test_publish_does_not_refresh_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        original_refresh = session.refresh

        def reject_post_commit_refresh(*args: object, **kwargs: object) -> None:
            if not session.in_transaction():
                raise RuntimeError("commit 后禁止 refresh")
            original_refresh(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(session, "refresh", reject_post_commit_refresh)

        created = EventService(session).publish_v2(
            project_id="project-1",
            run_id=run.id,
            candidates=[_ranked_event()],
        )

        assert len(created) == 1
        assert created[0].status == "active"
    finally:
        session.close()
        database.close()


def test_commit_failure_rolls_back_entire_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, session, run = _session_with_running_run(tmp_path)
    try:
        service = EventService(session)
        old = service.create(
            "project-1",
            _reviewed_event(),
            analysis_version="heuristic-v1",
            prompt_version="none",
        )

        def fail_commit() -> None:
            raise RuntimeError("模拟 commit 失败")

        monkeypatch.setattr(session, "commit", fail_commit)
        with pytest.raises(RuntimeError, match="模拟 commit 失败"):
            service.publish_v2(
                project_id="project-1",
                run_id=run.id,
                candidates=[_ranked_event()],
            )

        session.expire_all()
        remaining = list(session.scalars(select(EventNode)))
        assert [event.id for event in remaining] == [old.id]
        assert remaining[0].status == "active"
        assert session.get(AnalysisRun, run.id).status == "running"  # type: ignore[union-attr]
    finally:
        session.close()
        database.close()
