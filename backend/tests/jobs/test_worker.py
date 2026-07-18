from pathlib import Path

from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobRegistry
from moonlightbox.jobs.service import JobService
from sqlalchemy.orm import Session


def test_unknown_job_kind_is_marked_failed(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'worker.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job = JobService(session).enqueue("unknown", {})
        job_id = job.id

    from moonlightbox.worker import Worker

    Worker(database, JobRegistry()).run_once()

    with Session(database.engine) as session:
        failed = JobService(session).get(job_id)
        assert failed.status == "failed"
        assert failed.error_code == "unknown_job_kind"


def test_registered_job_handler_succeeds(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'registered.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job = JobService(session).enqueue("sample", {"value": 1})
        job_id = job.id

    handled: list[str] = []
    registry = JobRegistry()
    registry.register("sample", lambda _service, current: handled.append(current.id))

    from moonlightbox.worker import Worker

    Worker(database, registry).run_once()

    with Session(database.engine) as session:
        succeeded = JobService(session).get(job_id)
        assert succeeded.status == "succeeded"
    assert handled == [job_id]
