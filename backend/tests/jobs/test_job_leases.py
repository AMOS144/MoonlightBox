from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Thread
from time import sleep

import pytest
from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobLeaseLostError, JobService
from sqlalchemy import select, text
from sqlalchemy.orm import Session


def _database(tmp_path: Path, name: str = "job-leases.db") -> Database:
    database = Database(f"sqlite:///{tmp_path / name}")
    Job.metadata.create_all(database.engine)
    return database


def test_claim_writes_token_and_heartbeat_uses_token_cas(tmp_path: Path) -> None:
    database = _database(tmp_path)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        job_id = JobService(session).enqueue("sample", {}).id
        claimed = JobService(session).claim_next_queued(
            worker_token="worker-token-a",
            lease_duration=timedelta(seconds=30),
            now=now,
        )

        assert claimed is not None
        assert claimed.id == job_id
        assert claimed.worker_token == "worker-token-a"
        assert claimed.lease_expires_at is not None
        assert claimed.lease_expires_at.replace(tzinfo=UTC) == now + timedelta(seconds=30)

        heartbeat = JobService(session).heartbeat(
            job_id,
            token="worker-token-a",
            lease_duration=timedelta(seconds=60),
            now=now,
        )
        assert heartbeat is not None
        assert heartbeat.lease_expires_at is not None
        assert heartbeat.lease_expires_at.replace(tzinfo=UTC) == now + timedelta(seconds=60)
        assert (
            JobService(session).heartbeat(
                job_id,
                token="stale-token",
                lease_duration=timedelta(seconds=60),
                now=now,
            )
            is None
        )


def test_stale_token_cannot_checkpoint_or_finish_after_takeover(tmp_path: Path) -> None:
    database = _database(tmp_path)
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    with Session(database.engine) as session:
        service = JobService(session)
        job_id = service.enqueue("sample", {}).id
        service.claim_next_queued(
            worker_token="old-token",
            lease_duration=timedelta(seconds=1),
            now=expired_at - timedelta(seconds=1),
        )
        assert service.recover_expired_running(now=expired_at) == 1
        replacement = service.claim_next_queued(
            worker_token="new-token",
            lease_duration=timedelta(minutes=1),
            now=expired_at,
        )
        assert replacement is not None

        with pytest.raises(JobLeaseLostError):
            service.checkpoint(
                job_id,
                {"stage": "stale"},
                token="old-token",
            )
        assert service.succeed(job_id, token="old-token") is None
        assert (
            service.fail(
                job_id,
                "stale_failure",
                "旧 Worker 不得写入",
                token="old-token",
            )
            is None
        )
        assert service.get(job_id).status == "running"
        assert service.get(job_id).worker_token == "new-token"


def test_startup_recovery_only_requeues_expired_running_jobs(tmp_path: Path) -> None:
    database = _database(tmp_path)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        service = JobService(session)
        active_id = service.enqueue("sample", {"name": "active"}).id
        service.claim_next_queued(
            worker_token="active-token",
            lease_duration=timedelta(minutes=1),
            now=now,
        )
        expired_id = service.enqueue("sample", {"name": "expired"}).id
        service.claim_next_queued(
            worker_token="expired-token",
            lease_duration=timedelta(seconds=1),
            now=now - timedelta(minutes=1),
        )

        assert service.recover_expired_running(now=now) == 1
        assert service.get(active_id).status == "running"
        assert service.get(active_id).worker_token == "active-token"
        assert service.get(expired_id).status == "queued"
        assert service.get(expired_id).worker_token is None


def test_enqueue_unique_is_atomic_across_sessions(tmp_path: Path) -> None:
    database = _database(tmp_path, "dedupe.db")
    barrier = Barrier(2)
    ids: list[str] = []

    def enqueue() -> None:
        with Session(database.engine) as session:
            barrier.wait()
            job = JobService(session).enqueue_unique(
                "event_analysis_v2",
                {"import_id": "import-1"},
                dedupe_key="event-analysis-v2:import-1:prompt-v2:config-v1",
            )
            ids.append(job.id)

    threads = [Thread(target=enqueue), Thread(target=enqueue)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ids) == 2
    assert len(set(ids)) == 1
    with Session(database.engine) as session:
        assert session.query(Job).count() == 1


def test_enqueue_unique_retries_when_sqlite_is_briefly_write_locked(tmp_path: Path) -> None:
    database = Database(
        f"sqlite:///{tmp_path / 'enqueue-locked.db'}",
        busy_timeout_ms=10,
        lock_retries=3,
    )
    Job.metadata.create_all(database.engine)
    locked = database.engine.connect()
    locked.exec_driver_sql("BEGIN IMMEDIATE")

    def release_lock() -> None:
        # 第一轮写入会遇到锁；随后放开锁，验证入队会重试而非报“找不到幂等记录”。
        sleep(0.02)
        locked.rollback()
        locked.close()

    releaser = Thread(target=release_lock)
    releaser.start()
    try:
        with Session(database.engine) as session:
            job = JobService(session, lock_retries=3).enqueue_unique(
                "sample",
                {"project_id": "project-1"},
                dedupe_key="sample:project-1",
            )
    finally:
        releaser.join()

    assert job.status == "queued"
    with Session(database.engine) as session:
        assert session.scalar(select(Job.id).where(Job.dedupe_key == "sample:project-1")) == job.id


def test_database_enables_wal_and_busy_timeout_for_file_sqlite(tmp_path: Path) -> None:
    database = _database(tmp_path, "pragmas.db")
    with database.engine.connect() as connection:
        journal_mode = connection.execute(text("PRAGMA journal_mode")).scalar_one()
        busy_timeout = connection.execute(text("PRAGMA busy_timeout")).scalar_one()

    assert str(journal_mode).lower() == "wal"
    assert int(busy_timeout) > 0


def test_claim_returns_safely_while_sqlite_is_write_locked(tmp_path: Path) -> None:
    database = Database(
        f"sqlite:///{tmp_path / 'locked.db'}",
        busy_timeout_ms=10,
        lock_retries=1,
    )
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        JobService(session).enqueue("sample", {})

    locked = database.engine.connect()
    locked.exec_driver_sql("BEGIN IMMEDIATE")
    try:
        with Session(database.engine) as session:
            assert (
                JobService(session, lock_retries=1).claim_next_queued(
                    worker_token="worker-token",
                    lease_duration=timedelta(seconds=30),
                )
                is None
            )
    finally:
        locked.rollback()
        locked.close()
