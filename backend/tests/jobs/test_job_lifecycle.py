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


def test_running_cancel_waits_for_worker_exit(tmp_path):
    from threading import Event, Thread

    from moonlightbox.agent_runtime.cancellation import cancellation_requested
    from moonlightbox.db import Database
    from moonlightbox.jobs.models import Job
    from moonlightbox.jobs.registry import JobRegistry
    from moonlightbox.jobs.service import JobService
    from moonlightbox.worker import Worker

    database = Database(f"sqlite:///{tmp_path / 'cancel.db'}")
    Job.metadata.create_all(database.engine)
    entered, release = Event(), Event()
    seen = []

    def handler(service, job):
        entered.set()
        assert release.wait(10)
        seen.append(cancellation_requested())

    registry = JobRegistry()
    registry.register("blocking", handler)
    with Session(database.engine) as session:
        job_id = JobService(session).enqueue("blocking", {}).id
    worker = Worker(database, registry)
    thread = Thread(target=worker.run_once)
    thread.start()
    try:
        assert entered.wait(10)
        with Session(database.engine) as session:
            service = JobService(session)
            job = service.cancel(job_id)
            assert job.status == "cancelling"
            token = job.worker_token
            assert token
            service.acknowledge_cancellation(job_id, token="wrong-worker")
            assert service.get(job_id).status == "cancelling"
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert seen == [True]
    with Session(database.engine) as session:
        job = JobService(session).get(job_id)
        assert job.status == "cancelled"
        assert job.worker_token is None
        assert job.error_code is None
