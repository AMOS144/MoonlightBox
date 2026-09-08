from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
from moonlightbox.db import Database
from moonlightbox.events.models import AnalysisRun
from moonlightbox.events.schemas import EventCandidateUpsert
from moonlightbox.imports.models import ImportSource
from moonlightbox.projects.models import Project
from sqlalchemy import update
from sqlalchemy.orm import Session


def _database_with_run(
    tmp_path: Path,
    *,
    window_ids: list[str] | None = None,
) -> tuple[Database, str]:
    from moonlightbox.events.runs import AnalysisRunService

    database = Database(f"sqlite:///{tmp_path / 'lease.db'}")
    AnalysisRun.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="Lease 测试"))
        session.flush()
        session.add(
            ImportSource(
                id="import-1",
                project_id="project-1",
                preview_id="preview-1",
                source_path="/tmp/chat.json",
                message_count=1,
                confirmed_at=datetime.now(UTC),
            )
        )
        session.commit()
        run = AnalysisRunService(session).get_or_create(
            project_id="project-1",
            import_id="import-1",
            analysis_version="hybrid-v2",
            prompt_version="prompt-v2",
            model="cloud-v2",
            config={"threshold": 0.72},
            window_ids=window_ids or [],
        )
        session.commit()
        return database, run.id


def test_only_one_worker_acquires_same_run_with_two_sessions(tmp_path: Path) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunAlreadyRunningError,
        AnalysisRunService,
    )

    database, run_id = _database_with_run(tmp_path)
    barrier = Barrier(2)

    def acquire(owner: str) -> str:
        with Session(database.engine) as session:
            barrier.wait()
            try:
                lease = AnalysisRunService(session).acquire_lease(
                    run_id,
                    owner=owner,
                    duration=timedelta(minutes=1),
                )
                session.commit()
                return lease.token
            except AnalysisRunAlreadyRunningError:
                session.rollback()
                return "already_running"

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(acquire, ["worker-a", "worker-b"]))

        assert results.count("already_running") == 1
        assert len({result for result in results if result != "already_running"}) == 1
    finally:
        database.close()


def test_expired_lease_can_be_reclaimed_from_checkpoint(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunService

    database, run_id = _database_with_run(tmp_path, window_ids=["window-0"])
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        with Session(database.engine) as first:
            service = AnalysisRunService(first)
            old = service.acquire_lease(
                run_id,
                owner="worker-old",
                duration=timedelta(seconds=10),
                now=started_at,
            )
            service.start(run_id)
            service.record_window_results(
                run_id,
                window_index=0,
                window_id="window-0",
                candidates=[],
                lease_token=old.token,
                now=started_at,
            )
            first.commit()

        with Session(database.engine) as second:
            service = AnalysisRunService(second)
            reclaimed = service.acquire_lease(
                run_id,
                owner="worker-new",
                duration=timedelta(minutes=1),
                now=started_at + timedelta(minutes=1),
            )
            second.commit()
            run = service.get(run_id)

            assert reclaimed.token != old.token
            assert run.checkpoint == 1
            assert run.lease_owner == "worker-new"
    finally:
        database.close()


def test_heartbeat_extends_lease_and_release_allows_next_worker(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.runs import AnalysisRunService

    database, run_id = _database_with_run(tmp_path)
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        with Session(database.engine) as session:
            service = AnalysisRunService(session)
            acquired = service.acquire_lease(
                run_id,
                owner="worker-first",
                duration=timedelta(seconds=10),
                now=started_at,
            )
            heartbeat = service.heartbeat_lease(
                run_id,
                token=acquired.token,
                duration=timedelta(seconds=20),
                now=started_at + timedelta(seconds=5),
            )
            service.release_lease(
                run_id,
                token=acquired.token,
                now=started_at + timedelta(seconds=5),
            )
            session.commit()

            assert heartbeat.expires_at == started_at + timedelta(seconds=25)

        with Session(database.engine) as next_session:
            next_lease = AnalysisRunService(next_session).acquire_lease(
                run_id,
                owner="worker-next",
                duration=timedelta(minutes=1),
                now=started_at + timedelta(seconds=6),
            )
            next_session.commit()
            assert next_lease.owner == "worker-next"
    finally:
        database.close()


def test_matching_token_with_wrong_owner_cannot_renew(tmp_path: Path) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunLeaseLostError,
        AnalysisRunService,
    )

    database, run_id = _database_with_run(tmp_path)
    try:
        with Session(database.engine) as session:
            service = AnalysisRunService(session)
            acquired = service.acquire_lease(
                run_id,
                owner="worker-correct",
                duration=timedelta(minutes=1),
            )
            session.commit()

            with pytest.raises(AnalysisRunLeaseLostError):
                service.heartbeat_lease(
                    run_id,
                    token=acquired.token,
                    owner="worker-wrong",
                    duration=timedelta(minutes=1),
                )
    finally:
        database.close()


def test_expired_old_token_cannot_write_candidates(tmp_path: Path) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunLeaseLostError,
        AnalysisRunService,
    )

    database, run_id = _database_with_run(tmp_path, window_ids=["window-0"])
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        with Session(database.engine) as first:
            service = AnalysisRunService(first)
            old = service.acquire_lease(
                run_id,
                owner="worker-old",
                duration=timedelta(seconds=10),
                now=started_at,
            )
            service.start(run_id)
            first.commit()

        with Session(database.engine) as second:
            current = AnalysisRunService(second).acquire_lease(
                run_id,
                owner="worker-new",
                duration=timedelta(minutes=1),
                now=started_at + timedelta(minutes=1),
            )
            second.commit()

        with Session(database.engine) as stale:
            with pytest.raises(AnalysisRunLeaseLostError):
                AnalysisRunService(stale).record_window_results(
                    run_id,
                    window_index=0,
                    window_id="window-0",
                    candidates=[
                        EventCandidateUpsert(
                            candidate_key="candidate-1",
                            raw_payload={},
                            status="accepted",
                        )
                    ],
                    lease_token=old.token,
                    now=started_at + timedelta(minutes=1),
                )
            stale.rollback()

        with Session(database.engine) as verification:
            run = AnalysisRunService(verification).get(run_id)
            assert run.checkpoint == 0
            assert run.lease_token == current.token
    finally:
        database.close()


def test_expired_old_token_cannot_publish(tmp_path: Path) -> None:
    from moonlightbox.events.runs import AnalysisRunLeaseLostError, AnalysisRunService
    from moonlightbox.events.service import EventService

    database, run_id = _database_with_run(tmp_path)
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        with Session(database.engine) as first:
            service = AnalysisRunService(first)
            old = service.acquire_lease(
                run_id,
                owner="worker-old",
                duration=timedelta(seconds=10),
                now=started_at,
            )
            service.start(run_id)
            first.commit()

        with Session(database.engine) as second:
            AnalysisRunService(second).acquire_lease(
                run_id,
                owner="worker-new",
                duration=timedelta(minutes=1),
                now=started_at + timedelta(minutes=1),
            )
            second.commit()

        with Session(database.engine) as stale:
            with pytest.raises(AnalysisRunLeaseLostError):
                EventService(stale).publish_v2(
                    project_id="project-1",
                    run_id=run_id,
                    candidates=[],
                    lease_token=old.token,
                    now=started_at + timedelta(minutes=1),
                )
            stale.rollback()
            assert AnalysisRunService(stale).get(run_id).status == "running"
    finally:
        database.close()


@pytest.mark.parametrize("elapsed_seconds", [10, 11])
def test_assert_and_heartbeat_reject_lease_at_or_after_expiry(
    tmp_path: Path,
    elapsed_seconds: int,
) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunLeaseLostError,
        AnalysisRunService,
    )

    database, run_id = _database_with_run(tmp_path)
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    checked_at = started_at + timedelta(seconds=elapsed_seconds)
    try:
        with Session(database.engine) as session:
            service = AnalysisRunService(session)
            lease = service.acquire_lease(
                run_id,
                owner="worker-old",
                duration=timedelta(seconds=10),
                now=started_at,
            )

            with pytest.raises(AnalysisRunLeaseLostError):
                service.assert_lease(
                    run_id,
                    token=lease.token,
                    owner=lease.owner,
                    now=checked_at,
                )
            with pytest.raises(AnalysisRunLeaseLostError):
                service.heartbeat_lease(
                    run_id,
                    token=lease.token,
                    owner=lease.owner,
                    duration=timedelta(minutes=1),
                    now=checked_at,
                )
    finally:
        database.close()


@pytest.mark.parametrize("elapsed_seconds", [10, 11])
def test_expired_lease_cannot_record_or_succeed(
    tmp_path: Path,
    elapsed_seconds: int,
) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunLeaseLostError,
        AnalysisRunService,
    )

    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    checked_at = started_at + timedelta(seconds=elapsed_seconds)
    record_directory = tmp_path / "record"
    succeed_directory = tmp_path / "succeed"
    record_directory.mkdir()
    succeed_directory.mkdir()
    database, record_run_id = _database_with_run(
        record_directory,
        window_ids=["window-0"],
    )
    succeed_database, succeed_run_id = _database_with_run(succeed_directory)
    try:
        with Session(database.engine) as session:
            service = AnalysisRunService(session)
            lease = service.acquire_lease(
                record_run_id,
                owner="worker-old",
                duration=timedelta(seconds=10),
                now=started_at,
            )
            service.start(record_run_id)

            with pytest.raises(AnalysisRunLeaseLostError):
                service.record_window_results(
                    record_run_id,
                    window_index=0,
                    window_id="window-0",
                    candidates=[],
                    lease_token=lease.token,
                    lease_owner=lease.owner,
                    now=checked_at,
                )
            assert service.get(record_run_id).checkpoint == 0

        with Session(succeed_database.engine) as session:
            service = AnalysisRunService(session)
            lease = service.acquire_lease(
                succeed_run_id,
                owner="worker-old",
                duration=timedelta(seconds=10),
                now=started_at,
            )
            service.start(succeed_run_id)

            with pytest.raises(AnalysisRunLeaseLostError):
                service.succeed(
                    succeed_run_id,
                    lease_token=lease.token,
                    lease_owner=lease.owner,
                    now=checked_at,
                )
            assert service.get(succeed_run_id).status == "running"
    finally:
        database.close()
        succeed_database.close()


def test_record_final_cas_rechecks_lease_expiry(tmp_path: Path) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunLeaseLostError,
        AnalysisRunService,
    )

    class ExpiringDuringWriteService(AnalysisRunService):
        def assert_lease_if_required(self, *args: object, **kwargs: object) -> None:
            super().assert_lease_if_required(*args, **kwargs)  # type: ignore[arg-type]
            run = args[0]
            checked_at = kwargs["now"]
            assert isinstance(run, AnalysisRun)
            assert isinstance(checked_at, datetime)
            self._session.execute(  # type: ignore[attr-defined]
                update(AnalysisRun)
                .where(AnalysisRun.id == run.id)
                .values(lease_expires_at=checked_at)
                .execution_options(synchronize_session=False)
            )

    database, run_id = _database_with_run(tmp_path, window_ids=["window-0"])
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        with Session(database.engine) as session:
            service = ExpiringDuringWriteService(session)
            lease = service.acquire_lease(
                run_id,
                owner="worker-old",
                duration=timedelta(minutes=1),
                now=started_at,
            )
            service.start(run_id)

            with pytest.raises(AnalysisRunLeaseLostError):
                service.record_window_results(
                    run_id,
                    window_index=0,
                    window_id="window-0",
                    candidates=[],
                    lease_token=lease.token,
                    lease_owner=lease.owner,
                    now=started_at + timedelta(seconds=1),
                )
            assert service.get(run_id).checkpoint == 0
    finally:
        database.close()


def test_new_worker_reclaims_exactly_at_expiry_before_old_heartbeat(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunLeaseLostError,
        AnalysisRunService,
    )

    database, run_id = _database_with_run(tmp_path)
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    boundary = started_at + timedelta(seconds=10)
    try:
        with Session(database.engine) as first:
            old = AnalysisRunService(first).acquire_lease(
                run_id,
                owner="worker-old",
                duration=timedelta(seconds=10),
                now=started_at,
            )
            first.commit()

        with Session(database.engine) as second:
            new = AnalysisRunService(second).acquire_lease(
                run_id,
                owner="worker-new",
                duration=timedelta(minutes=1),
                now=boundary,
            )
            second.commit()

        with Session(database.engine) as stale:
            with pytest.raises(AnalysisRunLeaseLostError):
                AnalysisRunService(stale).heartbeat_lease(
                    run_id,
                    token=old.token,
                    owner=old.owner,
                    duration=timedelta(minutes=1),
                    now=boundary,
                )
        assert new.token != old.token
    finally:
        database.close()


def test_expired_lease_cannot_be_released_by_stale_token(tmp_path: Path) -> None:
    from moonlightbox.events.runs import (
        AnalysisRunLeaseLostError,
        AnalysisRunService,
    )

    database, run_id = _database_with_run(tmp_path)
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    try:
        with Session(database.engine) as session:
            service = AnalysisRunService(session)
            lease = service.acquire_lease(
                run_id,
                owner="worker-old",
                duration=timedelta(seconds=10),
                now=started_at,
            )

            with pytest.raises(AnalysisRunLeaseLostError):
                service.release_lease(
                    run_id,
                    token=lease.token,
                    owner=lease.owner,
                    now=started_at + timedelta(seconds=10),
                )
    finally:
        database.close()
