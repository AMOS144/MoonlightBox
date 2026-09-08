import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Thread
from threading import Event as ThreadEvent

import pytest
from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource
from moonlightbox.projects.models import Project
from sqlalchemy import create_engine, delete, event, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


def _session_with_import(tmp_path: Path) -> tuple[Database, Session, ImportSource]:
    from moonlightbox.events.models import AnalysisRun

    database = Database(f"sqlite:///{tmp_path / 'analysis-runs.db'}")
    AnalysisRun.metadata.create_all(database.engine)
    session = Session(database.engine)
    project = Project(id="project-1", name="分析运行测试")
    session.add(project)
    session.flush()
    imported = ImportSource(
        id="import-1",
        project_id=project.id,
        preview_id="preview-1",
        source_path="/tmp/chat.json",
        message_count=20,
        confirmed_at=datetime.now(UTC),
    )
    session.add(imported)
    session.commit()
    return database, session, imported


def _create_run(service: object, imported: ImportSource, **changes: object) -> object:
    values = {
        "project_id": imported.project_id,
        "import_id": imported.id,
        "analysis_version": "important-event-v2",
        "prompt_version": "prompt-v1",
        "model": "cloud-model-v1",
        "config": {"threshold": 0.72, "maximum_nodes": 25},
        "window_ids": ["window-0", "window-1", "window-2"],
    }
    values.update(changes)
    return service.get_or_create(**values)  # type: ignore[attr-defined, no-any-return]


def test_get_or_create_is_idempotent_for_same_analysis_identity(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        first = _create_run(service, imported)
        second = _create_run(
            service,
            imported,
            config={"maximum_nodes": 25, "threshold": 0.72},
        )
        session.commit()

        assert second.id == first.id
        assert session.scalar(select(func.count()).select_from(type(first))) == 1
    finally:
        session.close()
        database.close()


def test_get_or_create_uses_sqlite_on_conflict_do_nothing(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService

    database, session, imported = _session_with_import(tmp_path)
    statements: list[str] = []

    def capture_insert(
        _: object,
        __: object,
        statement: str,
        ___: object,
        ____: object,
        _____: object,
    ) -> None:
        if "insert into analysis_runs" in statement.lower():
            statements.append(statement)

    try:
        event.listen(database.engine, "before_cursor_execute", capture_insert)
        _create_run(AnalysisRunService(session), imported)

        assert len(statements) == 1
        assert "ON CONFLICT" in statements[0].upper()
        assert "DO NOTHING" in statements[0].upper()
    finally:
        event.remove(database.engine, "before_cursor_execute", capture_insert)
        session.rollback()
        session.close()
        database.close()


def test_concurrent_get_or_create_returns_single_run(tmp_path: Path) -> None:
    from moonlightbox.events.models import AnalysisRun
    from moonlightbox.events.runs import AnalysisRunService

    database, setup_session, imported = _session_with_import(tmp_path)
    project_id = imported.project_id
    import_id = imported.id
    setup_session.close()
    barrier = Barrier(2)

    def create_run() -> str:
        with Session(database.engine) as session:
            barrier.wait()
            run = AnalysisRunService(session).get_or_create(
                project_id=project_id,
                import_id=import_id,
                analysis_version="important-event-v2",
                prompt_version="prompt-v1",
                model="cloud-model-v1",
                config={"threshold": 0.72, "maximum_nodes": 25},
                window_ids=["window-0", "window-1", "window-2"],
            )
            session.commit()
            return run.id

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            ids = list(executor.map(lambda _: create_run(), range(2)))

        with Session(database.engine) as verification:
            assert len(set(ids)) == 1
            assert verification.scalar(select(func.count()).select_from(AnalysisRun)) == 1
    finally:
        database.close()


def test_get_or_create_retries_locked_database_finitely(tmp_path: Path) -> None:
    from moonlightbox.events.models import AnalysisRun
    from moonlightbox.events.runs import (
        AnalysisDatabaseBusyError,
        AnalysisRunService,
    )

    database_url = f"sqlite:///{tmp_path / 'locked.db'}"
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False, "timeout": 0},
    )
    AnalysisRun.metadata.create_all(engine)
    with Session(engine) as setup:
        project = Project(id="project-1", name="锁重试测试")
        setup.add(project)
        setup.flush()
        imported = ImportSource(
            id="import-1",
            project_id=project.id,
            preview_id="preview-1",
            source_path="/tmp/chat.json",
            message_count=20,
            confirmed_at=datetime.now(UTC),
        )
        setup.add(imported)
        setup.commit()

    insert_attempts = 0

    def count_insert_attempts(
        _: object,
        __: object,
        statement: str,
        ___: object,
        ____: object,
        _____: object,
    ) -> None:
        nonlocal insert_attempts
        if "insert into analysis_runs" in statement.lower():
            insert_attempts += 1

    lock_connection = engine.connect()
    lock_connection.exec_driver_sql("BEGIN IMMEDIATE")
    event.listen(engine, "before_cursor_execute", count_insert_attempts)
    try:
        with Session(engine) as session:
            service = AnalysisRunService(
                session,
                lock_retry_attempts=2,
                lock_retry_delay=0,
            )
            with pytest.raises(AnalysisDatabaseBusyError):
                service.get_or_create(
                    project_id="project-1",
                    import_id="import-1",
                    analysis_version="important-event-v2",
                    prompt_version="prompt-v1",
                    model="cloud-model-v1",
                    config={"threshold": 0.72, "maximum_nodes": 25},
                    window_ids=["window-0", "window-1", "window-2"],
                )

            assert insert_attempts == 2
            assert session.scalar(select(func.count()).select_from(AnalysisRun)) == 0
    finally:
        event.remove(engine, "before_cursor_execute", count_insert_attempts)
        lock_connection.rollback()
        lock_connection.close()
        engine.dispose()


def test_locked_retry_preserves_callers_pending_transaction(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService

    database, session, imported = _session_with_import(tmp_path)
    unrelated = Project(id="project-unrelated", name="修改前")
    session.add(unrelated)
    session.commit()
    unrelated.name = "修改后"
    lock_raised = False

    def raise_lock_once(
        _: object,
        statement: str,
        __: object,
        ___: object,
    ) -> None:
        nonlocal lock_raised
        if not lock_raised and "insert into analysis_runs" in statement.lower():
            lock_raised = True
            raise sqlite3.OperationalError("database is locked")

    event.listen(database.engine, "do_execute", raise_lock_once)
    try:
        run = _create_run(
            AnalysisRunService(
                session,
                lock_retry_attempts=2,
                lock_retry_delay=0,
            ),
            imported,
        )
        session.commit()

        with Session(database.engine) as verification:
            assert run.id
            assert verification.get(Project, unrelated.id).name == "修改后"
    finally:
        event.remove(database.engine, "do_execute", raise_lock_once)
        session.close()
        database.close()


def test_real_sqlite_lock_retry_preserves_pending_project_change(tmp_path: Path) -> None:
    from moonlightbox.events.models import AnalysisRun
    from moonlightbox.events.runs import AnalysisRunService

    database_url = f"sqlite:///{tmp_path / 'real-locked.db'}"
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False, "timeout": 0},
    )
    AnalysisRun.metadata.create_all(engine)
    with Session(engine) as setup:
        project = Project(id="project-1", name="修改前")
        setup.add(project)
        setup.flush()
        setup.add(
            ImportSource(
                id="import-1",
                project_id=project.id,
                preview_id="preview-1",
                source_path="/tmp/chat.json",
                message_count=20,
                confirmed_at=datetime.now(UTC),
            )
        )
        setup.commit()

    lock_ready = ThreadEvent()
    release_lock = ThreadEvent()
    lock_released = ThreadEvent()

    def hold_write_lock() -> None:
        with engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            lock_ready.set()
            release_lock.wait(timeout=2)
            connection.rollback()
        lock_released.set()

    holder = Thread(target=hold_write_lock)
    holder.start()
    assert lock_ready.wait(timeout=2)
    locked_errors = 0

    def release_after_first_lock(context: object) -> None:
        nonlocal locked_errors
        original = getattr(context, "original_exception", None)
        if "locked" not in str(original).lower():
            return
        locked_errors += 1
        if locked_errors == 1:
            release_lock.set()
            assert lock_released.wait(timeout=2)

    event.listen(engine, "handle_error", release_after_first_lock)
    try:
        with Session(engine) as session:
            project = session.get(Project, "project-1")
            assert project is not None
            project.name = "修改后"
            run = AnalysisRunService(
                session,
                lock_retry_attempts=2,
                lock_retry_delay=0,
            ).get_or_create(
                project_id="project-1",
                import_id="import-1",
                analysis_version="important-event-v2",
                prompt_version="prompt-v1",
                model="cloud-model-v1",
                config={"threshold": 0.72},
                window_ids=["window-0"],
            )
            session.commit()
            assert run.id

        with Session(engine) as verification:
            project = verification.get(Project, "project-1")
            assert project is not None
            assert project.name == "修改后"
            assert verification.scalar(select(func.count()).select_from(AnalysisRun)) == 1
        assert locked_errors == 1
    finally:
        event.remove(engine, "handle_error", release_after_first_lock)
        release_lock.set()
        holder.join(timeout=2)
        engine.dispose()


def test_get_or_create_saves_normalized_config_snapshot(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService, config_fingerprint
    from moonlightbox.events.schemas import AnalysisRunRead

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        evidence_weights = [0.2, 0.3]
        config: dict[str, object] = {
            "threshold": 0.72,
            "weights": {"evidence": evidence_weights, "change": 0.5},
        }
        run = _create_run(service, imported, config=config)
        expected = {
            "threshold": 0.72,
            "weights": {"change": 0.5, "evidence": [0.2, 0.3]},
        }

        config["threshold"] = 0.9
        evidence_weights.append(0.5)

        assert run.config == expected
        assert run.config_fingerprint == config_fingerprint(expected)
        assert AnalysisRunRead.model_validate(run).config == expected
    finally:
        session.close()
        database.close()


def test_get_or_create_saves_immutable_window_manifest(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService
    from moonlightbox.events.schemas import AnalysisRunRead

    database, session, imported = _session_with_import(tmp_path)
    try:
        window_ids = ["window-0", "window-1"]
        run = AnalysisRunService(session).get_or_create(
            project_id=imported.project_id,
            import_id=imported.id,
            analysis_version="important-event-v2",
            prompt_version="prompt-v1",
            model="cloud-model-v1",
            config={"threshold": 0.72},
            window_ids=window_ids,
        )
        window_ids.append("window-2")

        assert run.window_ids == ["window-0", "window-1"]
        assert run.total_windows == 2
        assert AnalysisRunRead.model_validate(run).window_ids == [
            "window-0",
            "window-1",
        ]
    finally:
        session.close()
        database.close()


def test_window_result_id_must_match_manifest_for_new_and_replayed_window(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.runs import AnalysisRunService, InvalidAnalysisCheckpointError

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = service.get_or_create(
            project_id=imported.project_id,
            import_id=imported.id,
            analysis_version="important-event-v2",
            prompt_version="prompt-v1",
            model="cloud-model-v1",
            config={"threshold": 0.72},
            window_ids=["window-0", "window-1"],
        )
        service.start(run.id)

        with pytest.raises(InvalidAnalysisCheckpointError):
            service.record_window_results(
                run.id,
                window_index=0,
                window_id="wrong-window",
                candidates=[],
            )
        service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=[],
        )
        with pytest.raises(InvalidAnalysisCheckpointError):
            service.record_window_results(
                run.id,
                window_index=0,
                window_id="wrong-window",
                candidates=[],
            )

        assert service.get(run.id).checkpoint == 1
    finally:
        session.close()
        database.close()


@pytest.mark.parametrize("replacement", [["window-x"], ["window-0", "window-1"]])
def test_window_manifest_mutation_is_rejected(
    tmp_path: Path,
    replacement: list[str],
) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunService,
        AnalysisWindowManifestIntegrityError,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        run = AnalysisRunService(session).get_or_create(
            project_id=imported.project_id,
            import_id=imported.id,
            analysis_version="important-event-v2",
            prompt_version="prompt-v1",
            model="cloud-model-v1",
            config={"threshold": 0.72},
            window_ids=["window-0"],
        )
        session.commit()
        if len(replacement) == 1:
            run.window_ids = replacement
        else:
            run.window_ids.append("window-1")

        with pytest.raises(AnalysisWindowManifestIntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()
        database.close()


def test_empty_window_manifest_run_can_succeed(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = service.get_or_create(
            project_id=imported.project_id,
            import_id=imported.id,
            analysis_version="important-event-v2",
            prompt_version="prompt-v1",
            model="cloud-model-v1",
            config={"threshold": 0.72},
            window_ids=[],
        )
        service.start(run.id)
        succeeded = service.succeed(run.id)

        assert succeeded.status == "succeeded"
        assert succeeded.total_windows == 0
    finally:
        session.close()
        database.close()


@pytest.mark.parametrize("window_ids", [["window-0", "window-0"], [""]])
def test_window_manifest_requires_unique_nonempty_ids(
    tmp_path: Path,
    window_ids: list[str],
) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunService,
        InvalidWindowManifestError,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        with pytest.raises(InvalidWindowManifestError):
            AnalysisRunService(session).get_or_create(
                project_id=imported.project_id,
                import_id=imported.id,
                analysis_version="important-event-v2",
                prompt_version="prompt-v1",
                model="cloud-model-v1",
                config={"threshold": 0.72},
                window_ids=window_ids,
            )
    finally:
        session.close()
        database.close()


def test_replacing_config_with_mismatched_fingerprint_is_rejected(tmp_path: Path) -> None:
    from moonlightbox.events.runs import (
        AnalysisConfigIntegrityError,
        AnalysisRunService,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        run = _create_run(AnalysisRunService(session), imported)
        session.commit()
        run.config = {"threshold": 0.9}

        with pytest.raises(AnalysisConfigIntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()
        database.close()


def test_nested_config_mutation_is_rejected_before_commit(tmp_path: Path) -> None:
    from moonlightbox.events.runs import (
        AnalysisConfigIntegrityError,
        AnalysisRunService,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        run = _create_run(
            AnalysisRunService(session),
            imported,
            config={"weights": {"change": 0.5}},
        )
        session.commit()
        weights = run.config["weights"]
        assert isinstance(weights, dict)
        weights["change"] = 0.8

        with pytest.raises(AnalysisConfigIntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()
        database.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("analysis_version", "hybrid-v3"),
        ("prompt_version", "prompt-v2"),
        ("model", "cloud-model-v2"),
        ("config", {"threshold": 0.8, "maximum_nodes": 25}),
    ],
)
def test_get_or_create_creates_new_run_when_identity_changes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    from moonlightbox.events.runs import AnalysisRunService, build_lane_slot_ids

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        first = _create_run(service, imported)
        changes = {field: value}
        if field == "analysis_version":
            changes["window_ids"] = build_lane_slot_ids(["window-0", "window-1", "window-2"])
        second = _create_run(service, imported, **changes)

        assert second.id != first.id
    finally:
        session.close()
        database.close()


def test_record_window_results_upserts_candidate_without_committing(tmp_path: Path) -> None:
    from moonlightbox.events.models import EventCandidate
    from moonlightbox.events.runs import AnalysisRunService
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(service, imported)
        service.start(run.id)
        candidate = EventCandidateUpsert(
            candidate_key="candidate-0",
            raw_payload={"type": "conflict", "reason": "初次结果"},
            review_payload=None,
            status="pending_review",
            rejection_reason=None,
            scores={"change_magnitude": 0.8},
        )
        first = service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=[candidate],
        )[0]
        updated = candidate.model_copy(
            update={
                "review_payload": {"persistent": True},
                "status": "accepted",
                "scores": {"change_magnitude": 0.8, "persistence": 0.9},
            }
        )
        second = service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=[updated],
        )[0]

        assert second.id == first.id
        assert second.status == "accepted"
        assert second.review_payload == {"persistent": True}
        assert session.scalar(select(func.count()).select_from(EventCandidate)) == 1
        assert session.in_transaction()
    finally:
        session.rollback()
        session.close()
        database.close()


def test_record_window_results_bulk_upserts_without_candidate_n_plus_one(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.runs import AnalysisRunService
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    candidate_selects = 0

    def count_candidate_selects(
        _: object,
        __: object,
        statement: str,
        ___: object,
        ____: object,
        _____: object,
    ) -> None:
        nonlocal candidate_selects
        normalized = statement.lower()
        if normalized.lstrip().startswith("select") and "event_candidates" in normalized:
            candidate_selects += 1

    try:
        service = AnalysisRunService(session)
        run = _create_run(service, imported)
        service.start(run.id)
        candidates = [
            EventCandidateUpsert(
                candidate_key=f"candidate-{index}",
                raw_payload={"index": index},
                status="pending_review",
                scores={"change_magnitude": 0.8},
            )
            for index in range(20)
        ]
        event.listen(database.engine, "before_cursor_execute", count_candidate_selects)

        persisted = service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=candidates,
        )

        assert len(persisted) == 20
        assert candidate_selects <= 1
    finally:
        event.remove(database.engine, "before_cursor_execute", count_candidate_selects)
        session.rollback()
        session.close()
        database.close()


def test_stale_session_replay_cannot_regress_checkpoint(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService

    database, setup_session, imported = _session_with_import(tmp_path)
    try:
        setup_service = AnalysisRunService(setup_session)
        run = _create_run(setup_service, imported)
        setup_service.start(run.id)
        setup_session.commit()

        with Session(database.engine) as first, Session(database.engine) as stale:
            first_service = AnalysisRunService(first)
            stale_service = AnalysisRunService(stale)
            stale_run = stale_service.get(run.id)
            assert stale_run.checkpoint == 0

            first_service.record_window_results(
                run.id,
                window_index=0,
                window_id="window-0",
                candidates=[],
            )
            first_service.record_window_results(
                run.id,
                window_index=1,
                window_id="window-1",
                candidates=[],
            )
            first.commit()

            stale_service.record_window_results(
                run.id,
                window_index=0,
                window_id="window-0",
                candidates=[],
            )
            assert stale_service.get(run.id).checkpoint == 2
    finally:
        setup_session.close()
        database.close()


def test_window_results_reject_skipping_unfinished_window(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService, InvalidAnalysisCheckpointError

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(service, imported)
        service.start(run.id)

        with pytest.raises(InvalidAnalysisCheckpointError):
            service.record_window_results(
                run.id,
                window_index=1,
                window_id="window-1",
                candidates=[],
            )
    finally:
        session.close()
        database.close()


def test_concurrent_same_window_replay_advances_checkpoint_once(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService

    database, setup_session, imported = _session_with_import(tmp_path)
    try:
        setup_service = AnalysisRunService(setup_session)
        run = _create_run(setup_service, imported)
        setup_service.start(run.id)
        setup_session.commit()

        with Session(database.engine) as first, Session(database.engine) as second:
            first_service = AnalysisRunService(first)
            second_service = AnalysisRunService(second)
            first_run = first_service.get(run.id)
            second_run = second_service.get(run.id)
            assert first_run.checkpoint == second_run.checkpoint == 0

            first_service.record_window_results(
                run.id,
                window_index=0,
                window_id="window-0",
                candidates=[],
            )
            first.commit()
            second_service.record_window_results(
                run.id,
                window_index=0,
                window_id="window-0",
                candidates=[],
            )
            second.commit()

        with Session(database.engine) as verification:
            persisted = verification.get(type(run), run.id)
            assert persisted is not None
            assert persisted.completed_windows == persisted.checkpoint == 1
    finally:
        setup_session.close()
        database.close()


def test_checkpoint_is_monotonic_and_resume_lists_unfinished_windows(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.runs import AnalysisRunService

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(service, imported)
        service.start(run.id)

        service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=[],
        )
        service.interrupt(run.id, "worker_interrupted", "执行器中断")

        assert service.unfinished_window_indexes(run.id) == [1, 2]

        resumed = service.resume(run.id)
        assert resumed.status == "running"
        assert resumed.checkpoint == 1

        service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=[],
        )
        assert service.get(run.id).checkpoint == 1
    finally:
        session.close()
        database.close()


def test_rejected_candidate_requires_rejection_reason() -> None:
    from moonlightbox.events.schemas import EventCandidateUpsert
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EventCandidateUpsert(
            candidate_key="candidate-0",
            raw_payload={"type": "conflict"},
            status="rejected",
            scores={},
        )


def test_non_rejected_candidate_cannot_have_rejection_reason() -> None:
    from moonlightbox.events.schemas import EventCandidateUpsert
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EventCandidateUpsert(
            candidate_key="candidate-0",
            raw_payload={"type": "conflict"},
            status="accepted",
            rejection_reason="不应存在",
            scores={},
        )


def test_invalid_status_transitions_are_rejected(tmp_path: Path) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunService,
        InvalidAnalysisRunTransitionError,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(service, imported, window_ids=[])

        with pytest.raises(InvalidAnalysisRunTransitionError):
            service.resume(run.id)

        with pytest.raises(InvalidAnalysisRunTransitionError):
            service.succeed(run.id)

        service.start(run.id)
        service.succeed(run.id)

        with pytest.raises(InvalidAnalysisRunTransitionError):
            service.resume(run.id)
    finally:
        session.close()
        database.close()


def test_stale_session_status_transition_uses_compare_and_swap(tmp_path: Path) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunConcurrencyError,
        AnalysisRunService,
    )

    database, setup_session, imported = _session_with_import(tmp_path)
    try:
        run = _create_run(AnalysisRunService(setup_session), imported)
        setup_session.commit()

        with Session(database.engine) as winner, Session(database.engine) as stale:
            winner_service = AnalysisRunService(winner)
            stale_service = AnalysisRunService(stale)
            stale_run = stale_service.get(run.id)
            assert stale_run.status == "queued"

            winner_service.start(run.id)
            winner.commit()

            with pytest.raises(AnalysisRunConcurrencyError):
                stale_service.start(run.id)
    finally:
        setup_session.close()
        database.close()


def test_model_constraints_enforce_unique_identity_and_cascade_candidates(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.models import AnalysisRun, EventCandidate
    from moonlightbox.events.runs import (
        config_fingerprint,
        window_manifest_fingerprint,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        config = {"threshold": 0.72}
        common = {
            "project_id": imported.project_id,
            "import_id": imported.id,
            "analysis_version": "important-event-v2",
            "prompt_version": "prompt-v1",
            "model": "cloud-model-v1",
            "config": config,
            "config_fingerprint": config_fingerprint(config),
            "window_ids": ["window-0"],
            "window_manifest_fingerprint": window_manifest_fingerprint(["window-0"]),
            "total_windows": 1,
        }
        first = AnalysisRun(**common)
        session.add(first)
        session.flush()
        session.add(
            EventCandidate(
                run_id=first.id,
                window_id="window-0",
                candidate_key="candidate-0",
                raw_payload={"type": "conflict"},
                review_payload=None,
                status="pending_review",
                rejection_reason=None,
                scores={},
            )
        )
        session.commit()

        duplicate = AnalysisRun(**common)
        session.add(duplicate)
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()

        session.execute(delete(AnalysisRun).where(AnalysisRun.id == first.id))
        session.commit()
        assert session.scalar(select(func.count()).select_from(EventCandidate)) == 0
    finally:
        session.close()
        database.close()


def _v3_candidate_payload(
    lane: str,
    **changes: object,
) -> dict[str, object]:
    from moonlightbox.events.v3_reviewer import V3EventCandidate

    payload: dict[str, object] = {
        "candidate_key": "由本地重算",
        "lane": lane,
        "type": "commitment" if lane == "relationship" else "outing",
        "title": "确认长期关系" if lane == "relationship" else "共同看展",
        "event_status": "occurred",
        "start_message_id": "m1",
        "end_message_id": "m2",
        "summary": "双方明确作出长期承诺" if lane == "relationship" else "双方共同看展",
        "before_state": "尚未承诺" if lane == "relationship" else None,
        "after_state": "形成长期承诺" if lane == "relationship" else None,
        "emotion_labels": ["期待"],
        "topic": "关系承诺" if lane == "relationship" else "看展",
        "conflict_level": 0,
        "event_significance": 0.9,
        "relationship_impact": 0.8,
        "model_confidence": 0.85,
        "reason": "原始消息支持候选事实",
        "evidence_ids": ["m1", "m2"],
    }
    payload.update(changes)
    return V3EventCandidate.model_validate(payload).model_dump(mode="json")


def _v3_review_payload(**changes: object) -> dict[str, object]:
    from moonlightbox.events.v3_reviewer import V3CandidateReview

    payload: dict[str, object] = {
        "facts_supported": True,
        "occurrence_supported": True,
        "bilateral_confirmation": True,
        "evidence_alignment": 0.9,
        "persistence": 0.8,
        "type_support": 0.85,
        "relationship_impact": 0.8,
        "event_significance": 0.9,
        "model_confidence": 0.85,
        "evidence_ids": ["m1", "m2"],
        "reason": "原始消息支持候选事实",
    }
    payload.update(changes)
    return V3CandidateReview.model_validate(payload).model_dump(mode="json")


def test_build_lane_slot_ids_uses_stable_lane_order_and_round_trips() -> None:
    from moonlightbox.events.runs import (
        build_lane_slot_ids,
        parse_lane_slot_id,
        resolve_lane_slot_id,
    )

    slots = build_lane_slot_ids(["window-0", "window-1"])

    assert slots == [
        "window-0::relationship",
        "window-0::shared_experience",
        "window-1::relationship",
        "window-1::shared_experience",
    ]
    assert [parse_lane_slot_id(slot) for slot in slots] == [
        ("window-0", "relationship"),
        ("window-0", "shared_experience"),
        ("window-1", "relationship"),
        ("window-1", "shared_experience"),
    ]
    assert resolve_lane_slot_id(slots[1]) == (
        "window-0",
        "shared_experience",
    )


@pytest.mark.parametrize(
    "window_ids",
    [
        [""],
        ["   "],
        ["window-0", "window-0"],
        ["window-0::relationship"],
    ],
)
def test_build_lane_slot_ids_rejects_ambiguous_physical_window_ids(
    window_ids: list[str],
) -> None:
    from moonlightbox.events.runs import InvalidWindowManifestError, build_lane_slot_ids

    with pytest.raises(InvalidWindowManifestError):
        build_lane_slot_ids(window_ids)


@pytest.mark.parametrize(
    "slot_id",
    [
        "",
        "window-0",
        "window-0::forged",
        "window-0::relationship::shared_experience",
        "::relationship",
    ],
)
def test_parse_lane_slot_id_rejects_forged_or_ambiguous_slots(slot_id: str) -> None:
    from moonlightbox.events.runs import InvalidWindowManifestError, parse_lane_slot_id

    with pytest.raises(InvalidWindowManifestError):
        parse_lane_slot_id(slot_id)


@pytest.mark.parametrize("analysis_version", ["hybrid-v3"])
def test_v3_run_persists_lane_slots_as_immutable_manifest(
    tmp_path: Path,
    analysis_version: str,
) -> None:
    from moonlightbox.events.runs import AnalysisRunService, build_lane_slot_ids
    from moonlightbox.events.schemas import AnalysisRunRead

    database, session, imported = _session_with_import(tmp_path)
    try:
        physical_window_ids = ["window-0", "window-1"]
        slot_ids = build_lane_slot_ids(physical_window_ids)
        run = _create_run(
            AnalysisRunService(session),
            imported,
            analysis_version=analysis_version,
            window_ids=slot_ids,
        )
        physical_window_ids.append("window-2")

        expected = build_lane_slot_ids(["window-0", "window-1"])
        assert run.window_ids == expected
        assert run.total_windows == 4
        assert AnalysisRunRead.model_validate(run).window_ids == expected
    finally:
        session.close()
        database.close()


def test_v3_run_rejects_physical_window_ids_without_lane_slots(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunService,
        InvalidWindowManifestError,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        with pytest.raises(InvalidWindowManifestError):
            _create_run(
                AnalysisRunService(session),
                imported,
                analysis_version="hybrid-v3",
                window_ids=["window-0"],
            )
    finally:
        session.close()
        database.close()


def test_v3_failed_second_slot_resumes_without_overwriting_first_slot(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.models import EventCandidate
    from moonlightbox.events.runs import (
        AnalysisRunService,
        InvalidAnalysisCheckpointError,
        build_lane_slot_ids,
    )
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(
            service,
            imported,
            analysis_version="hybrid-v3",
            window_ids=build_lane_slot_ids(["window-0"]),
        )
        service.start(run.id)
        relationship_slot, shared_slot = run.window_ids
        relationship_payload = _v3_candidate_payload(
            "relationship",
            title="第一个槽位候选",
        )
        first = service.record_window_results(
            run.id,
            window_index=0,
            window_id=relationship_slot,
            candidates=[
                EventCandidateUpsert(
                    candidate_key=str(relationship_payload["candidate_key"]),
                    raw_payload=relationship_payload,
                )
            ],
        )[0]

        with pytest.raises(InvalidAnalysisCheckpointError):
            service.record_window_results(
                run.id,
                window_index=1,
                window_id=shared_slot,
                candidates=[
                    EventCandidateUpsert(
                        candidate_key=str(relationship_payload["candidate_key"]),
                        raw_payload=relationship_payload,
                    )
                ],
            )

        assert service.get(run.id).completed_windows == 1
        assert service.unfinished_window_indexes(run.id) == [1]

        service.record_window_results(
            run.id,
            window_index=1,
            window_id=shared_slot,
            candidates=[
                EventCandidateUpsert(
                    candidate_key=str(_v3_candidate_payload("shared_experience")["candidate_key"]),
                    raw_payload=_v3_candidate_payload("shared_experience"),
                )
            ],
        )
        persisted_first = session.get(EventCandidate, first.id)
        assert persisted_first is not None
        assert persisted_first.raw_payload == relationship_payload
    finally:
        session.close()
        database.close()


def test_v3_record_is_idempotent_and_lane_slots_do_not_collide(tmp_path: Path) -> None:
    from moonlightbox.events.models import EventCandidate
    from moonlightbox.events.runs import AnalysisRunService, build_lane_slot_ids
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(
            service,
            imported,
            analysis_version="hybrid-v3",
            window_ids=build_lane_slot_ids(["window-0"]),
        )
        service.start(run.id)
        relationship_slot, shared_slot = run.window_ids
        relationship_payload = _v3_candidate_payload("relationship")
        relationship = EventCandidateUpsert(
            candidate_key=str(relationship_payload["candidate_key"]),
            raw_payload=relationship_payload,
        )
        first = service.record_window_results(
            run.id,
            window_index=0,
            window_id=relationship_slot,
            candidates=[relationship],
        )[0]
        replayed = service.record_window_results(
            run.id,
            window_index=0,
            window_id=relationship_slot,
            candidates=[relationship],
        )[0]
        shared_payload = _v3_candidate_payload("shared_experience")
        shared = service.record_window_results(
            run.id,
            window_index=1,
            window_id=shared_slot,
            candidates=[
                EventCandidateUpsert(
                    candidate_key=str(shared_payload["candidate_key"]),
                    raw_payload=shared_payload,
                )
            ],
        )[0]

        assert replayed.id == first.id
        assert shared.id != first.id
        assert {first.window_id, shared.window_id} == {
            relationship_slot,
            shared_slot,
        }
        assert session.scalar(select(func.count()).select_from(EventCandidate)) == 2
        assert service.get(run.id).completed_windows == 2
    finally:
        session.close()
        database.close()


def test_v3_record_accepts_real_candidate_and_review_model_dumps(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.runs import AnalysisRunService, build_lane_slot_ids
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(
            service,
            imported,
            analysis_version="hybrid-v3",
            window_ids=build_lane_slot_ids(["window-0"]),
        )
        service.start(run.id)
        raw_payload = _v3_candidate_payload("relationship")
        review_payload = _v3_review_payload()

        stored = service.record_window_results(
            run.id,
            window_index=0,
            window_id=run.window_ids[0],
            candidates=[
                EventCandidateUpsert(
                    candidate_key=str(raw_payload["candidate_key"]),
                    raw_payload=raw_payload,
                    review_payload=review_payload,
                )
            ],
        )[0]

        assert stored.raw_payload == raw_payload
        assert stored.review_payload == review_payload
        assert service.get(run.id).completed_windows == 1
    finally:
        session.close()
        database.close()


@pytest.mark.parametrize(
    "review_payload",
    [
        pytest.param(
            {"facts_supported": True},
            id="字段缺失",
        ),
        pytest.param(
            {
                **_v3_review_payload(),
                "invented_detail": "不允许的额外字段",
            },
            id="额外字段",
        ),
    ],
)
def test_v3_record_rejects_invalid_review_without_progress(
    tmp_path: Path,
    review_payload: dict[str, object],
) -> None:
    from moonlightbox.events.models import EventCandidate
    from moonlightbox.events.runs import (
        AnalysisRunService,
        InvalidAnalysisCheckpointError,
        build_lane_slot_ids,
    )
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(
            service,
            imported,
            analysis_version="hybrid-v3",
            window_ids=build_lane_slot_ids(["window-0"]),
        )
        service.start(run.id)
        raw_payload = _v3_candidate_payload("relationship")

        with pytest.raises(InvalidAnalysisCheckpointError):
            service.record_window_results(
                run.id,
                window_index=0,
                window_id=run.window_ids[0],
                candidates=[
                    EventCandidateUpsert(
                        candidate_key=str(raw_payload["candidate_key"]),
                        raw_payload=raw_payload,
                        review_payload=review_payload,
                    )
                ],
            )

        assert service.get(run.id).completed_windows == 0
        assert session.scalar(select(func.count()).select_from(EventCandidate)) == 0
    finally:
        session.close()
        database.close()


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(
            {
                "candidate_key": "candidate-relationship-invalid",
                "raw_payload": {},
            },
            id="raw-payload缺少lane",
        ),
        pytest.param(
            {
                "candidate_key": str(_v3_candidate_payload("shared_experience")["candidate_key"]),
                "raw_payload": _v3_candidate_payload("shared_experience"),
            },
            id="raw-payload的lane不一致",
        ),
        pytest.param(
            {
                "candidate_key": str(_v3_candidate_payload("relationship")["candidate_key"]),
                "raw_payload": _v3_candidate_payload("relationship"),
                "review_payload": {"lane": "shared_experience"},
            },
            id="review-payload的lane不一致",
        ),
    ],
)
def test_v3_lane_payload_mismatch_is_rejected_without_progress(
    tmp_path: Path,
    candidate: dict[str, object],
) -> None:
    from moonlightbox.events.models import EventCandidate
    from moonlightbox.events.runs import (
        AnalysisRunService,
        InvalidAnalysisCheckpointError,
        build_lane_slot_ids,
    )
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(
            service,
            imported,
            analysis_version="hybrid-v3",
            window_ids=build_lane_slot_ids(["window-0"]),
        )
        service.start(run.id)

        with pytest.raises(InvalidAnalysisCheckpointError):
            service.record_window_results(
                run.id,
                window_index=0,
                window_id=run.window_ids[0],
                candidates=[
                    EventCandidateUpsert(
                        **candidate,
                    )
                ],
            )

        assert service.get(run.id).completed_windows == 0
        assert session.scalar(select(func.count()).select_from(EventCandidate)) == 0
    finally:
        session.close()
        database.close()


@pytest.mark.parametrize("tampering", ["lane", "hash"])
def test_v3_candidate_key_tampering_is_rejected_without_progress(
    tmp_path: Path,
    tampering: str,
) -> None:
    from moonlightbox.events.models import EventCandidate
    from moonlightbox.events.runs import (
        AnalysisRunService,
        InvalidAnalysisCheckpointError,
        build_lane_slot_ids,
    )
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(
            service,
            imported,
            analysis_version="hybrid-v3",
            window_ids=build_lane_slot_ids(["window-0"]),
        )
        service.start(run.id)
        raw_payload = _v3_candidate_payload("relationship")
        stable_key = str(raw_payload["candidate_key"])
        if tampering == "lane":
            forged_key = stable_key.replace(
                "candidate-relationship-",
                "candidate-shared_experience-",
                1,
            )
        else:
            forged_key = f"{stable_key[:-1]}{'0' if stable_key[-1] != '0' else '1'}"

        with pytest.raises(InvalidAnalysisCheckpointError):
            service.record_window_results(
                run.id,
                window_index=0,
                window_id=run.window_ids[0],
                candidates=[
                    EventCandidateUpsert(
                        candidate_key=forged_key,
                        raw_payload=raw_payload,
                    )
                ],
            )

        assert service.get(run.id).completed_windows == 0
        assert session.scalar(select(func.count()).select_from(EventCandidate)) == 0
    finally:
        session.close()
        database.close()


def test_v2_candidate_without_lane_payload_remains_compatible(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(service, imported, window_ids=["window-0"])
        service.start(run.id)

        stored = service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=[
                EventCandidateUpsert(
                    candidate_key="candidate-0",
                    raw_payload={"type": "conflict"},
                )
            ],
        )[0]

        assert stored.window_id == "window-0"
        assert service.get(run.id).completed_windows == 1
    finally:
        session.close()
        database.close()


@pytest.mark.parametrize("analysis_version", ["not-v3", "important-event-v3"])
def test_only_explicit_hybrid_v3_enables_lane_slot_validation(
    tmp_path: Path,
    analysis_version: str,
) -> None:
    from moonlightbox.events.runs import AnalysisRunService
    from moonlightbox.events.schemas import EventCandidateUpsert

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(
            service,
            imported,
            analysis_version=analysis_version,
            window_ids=["window-0"],
        )
        service.start(run.id)

        stored = service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=[
                EventCandidateUpsert(
                    candidate_key="candidate-0",
                    raw_payload={"type": "conflict"},
                )
            ],
        )[0]

        assert stored.window_id == "window-0"
        assert service.get(run.id).checkpoint == 1
    finally:
        session.close()
        database.close()


def test_corrupted_completed_manifest_cannot_succeed(tmp_path: Path) -> None:
    from moonlightbox.events.models import AnalysisRun
    from moonlightbox.events.runs import (
        AnalysisRunManifestError,
        AnalysisRunService,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(service, imported, window_ids=["window-0"])
        service.start(run.id)
        service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=[],
        )
        session.execute(text("PRAGMA ignore_check_constraints = ON"))
        session.execute(
            update(AnalysisRun)
            .where(AnalysisRun.id == run.id)
            .values(window_manifest_fingerprint="tampered")
            .execution_options(synchronize_session=False)
        )
        session.commit()

        with pytest.raises(AnalysisRunManifestError):
            service.succeed(run.id)

        persisted = service.get(run.id)
        session.refresh(persisted)
        assert persisted.status == "running"
    finally:
        session.close()
        database.close()


def test_unfinished_indexes_reject_inconsistent_checkpoint(tmp_path: Path) -> None:
    from moonlightbox.events.models import AnalysisRun
    from moonlightbox.events.runs import (
        AnalysisRunManifestError,
        AnalysisRunService,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(service, imported, window_ids=["window-0"])
        service.start(run.id)
        service.record_window_results(
            run.id,
            window_index=0,
            window_id="window-0",
            candidates=[],
        )
        session.execute(text("PRAGMA ignore_check_constraints = ON"))
        session.execute(
            update(AnalysisRun)
            .where(AnalysisRun.id == run.id)
            .values(checkpoint=0)
            .execution_options(synchronize_session=False)
        )
        session.commit()

        with pytest.raises(AnalysisRunManifestError):
            service.unfinished_window_indexes(run.id)
    finally:
        session.close()
        database.close()


def test_resume_rejects_corrupted_manifest_without_state_change(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.models import AnalysisRun
    from moonlightbox.events.runs import (
        AnalysisRunManifestError,
        AnalysisRunService,
    )

    database, session, imported = _session_with_import(tmp_path)
    try:
        service = AnalysisRunService(session)
        run = _create_run(service, imported, window_ids=["window-0"])
        service.start(run.id)
        service.interrupt(run.id, "worker_interrupted", "执行器中断")
        session.execute(text("PRAGMA ignore_check_constraints = ON"))
        session.execute(
            update(AnalysisRun)
            .where(AnalysisRun.id == run.id)
            .values(total_windows=2)
            .execution_options(synchronize_session=False)
        )
        session.commit()

        with pytest.raises(AnalysisRunManifestError):
            service.resume(run.id)

        persisted = service.get(run.id)
        session.refresh(persisted)
        assert persisted.status == "interrupted"
    finally:
        session.close()
        database.close()
