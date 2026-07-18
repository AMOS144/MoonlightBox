from fastapi.testclient import TestClient
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService
from sqlalchemy.orm import Session


def test_get_and_cancel_job(client: TestClient, settings: Settings) -> None:
    database = Database(settings.database_url)
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job = JobService(session).enqueue("sample", {})
        job_id = job.id

    loaded = client.get(f"/api/jobs/{job_id}")
    cancelled = client.post(f"/api/jobs/{job_id}/cancel")

    assert loaded.status_code == 200
    assert loaded.json()["status"] == "queued"
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"


def test_resume_interrupted_job(client: TestClient, settings: Settings) -> None:
    database = Database(settings.database_url)
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        service = JobService(session)
        job = service.enqueue("sample", {})
        service.start(job.id)
        interrupted = service.interrupt(job.id, "worker stopped")
        job_id = interrupted.id

    resumed = client.post(f"/api/jobs/{job_id}/resume")

    assert resumed.status_code == 200
    assert resumed.json()["status"] == "queued"
