from pathlib import Path

from sqlalchemy.orm import Session


def test_interrupted_job_can_resume(tmp_path: Path) -> None:
    from moonlightbox.db import Database
    from moonlightbox.jobs.models import Job
    from moonlightbox.jobs.service import JobService

    database = Database(f"sqlite:///{tmp_path / 'jobs.db'}")
    Job.metadata.create_all(database.engine)

    with Session(database.engine) as session:
        service = JobService(session)
        job = service.enqueue("sample", {"value": 1})
        running = service.start(job.id)
        token = running.worker_token
        assert token is not None
        service.checkpoint(running.id, {"offset": 12}, token=token)
        interrupted = service.interrupt(
            running.id,
            "worker stopped",
            token=token,
        )
        assert interrupted is not None

        resumed = service.resume(interrupted.id)

    assert resumed.status == "queued"
    assert resumed.checkpoint == {"offset": 12}
