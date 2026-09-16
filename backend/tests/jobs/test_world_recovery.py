from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.runtime_recovery import WORLD_KIND, recovery, scan_recovery
from moonlightbox.jobs.service import InvalidJobTransitionError, JobService


@pytest.fixture
def service(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'recovery.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        yield JobService(session)
    database.close()


def fail(service, job, code="lightrag_timeout"):
    running = service.start(job.id)
    service.checkpoint(job.id, {"progress": .1}, token=running.worker_token)
    return service.fail(job.id, code, "test failure", token=running.worker_token)


def test_world_backoff_keeps_progress_and_exhausts_total_budget(service):
    job = service.enqueue(WORLD_KIND, {})
    running = service.start(job.id)
    service.checkpoint(job.id, {"indexed_bundles": 20, "bundle_count": 50, "progress": .4}, token=running.worker_token)
    job = service.fail(job.id, "lightrag_timeout", "timeout", token=running.worker_token)
    assert recovery(job)["status"] == "waiting"
    assert scan_recovery(service.session, datetime.now(UTC)) == 0
    assert scan_recovery(service.session, datetime.now(UTC) + timedelta(hours=1)) == 1
    service.session.commit()
    for count in range(1, 6):
        job = fail(service, job)
        assert job.checkpoint["job_retry_attempt"] == count
        assert job.checkpoint["indexed_bundles"] == 20
        assert job.progress == .4
        if count < 5:
            job = service.resume(job.id)
    assert recovery(job)["reason"] == "retry_exhausted"
    assert scan_recovery(service.session, datetime.now(UTC) + timedelta(days=1)) == 0
    with pytest.raises(InvalidJobTransitionError):
        service.resume(job.id)


def test_content_rejection_is_terminal_without_consuming_retry_loop(service):
    job = service.enqueue(WORLD_KIND, {})
    running = service.start(job.id)
    service.checkpoint(job.id, {"indexed_bundles": 5, "bundle_count": 30}, token=running.worker_token)
    job = service.fail(job.id, "lightrag_content_rejected", "content rejected", token=running.worker_token)
    assert recovery(job)["status"] == "terminal"
    assert recovery(job)["reason"] == "lightrag_content_rejected"
    assert scan_recovery(service.session, datetime.now(UTC) + timedelta(days=1)) == 0
    assert job.checkpoint["indexed_bundles"] == 5


def test_busy_checks_do_not_spend_retry_budget_and_can_be_cancelled(service):
    job = service.enqueue(WORLD_KIND, {})
    for _ in range(8):
        job = fail(service, job, "lightrag_index_running")
        assert recovery(job)["status"] == "waiting"
        assert recovery(job)["retries"] == 0
        job = service.resume(job.id)
    job = fail(service, job, "lightrag_index_running")
    assert service.cancel(job.id).status == "cancelled"
    assert scan_recovery(service.session, datetime.now(UTC) + timedelta(days=1)) == 0


def test_configuration_failure_requires_manual_recovery(service):
    job = fail(service, service.enqueue(WORLD_KIND, {}), "lightrag_request_failed")
    assert recovery(job)["status"] == "terminal"
    assert scan_recovery(service.session, datetime.now(UTC) + timedelta(days=1)) == 0
    assert service.resume(job.id).status == "queued"
