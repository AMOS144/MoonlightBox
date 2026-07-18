from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.jobs.models import Job

TRANSITIONS: dict[str, set[str]] = {
    "queued": {"running", "cancelled"},
    "running": {"succeeded", "failed", "cancelled", "interrupted"},
    "interrupted": {"queued", "cancelled"},
    "failed": {"queued"},
}


class InvalidJobTransitionError(ValueError):
    pass


class JobNotFoundError(LookupError):
    pass


class JobService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def enqueue(self, kind: str, payload: dict[str, Any]) -> Job:
        job = Job(kind=kind, payload=payload)
        self._session.add(job)
        return self._persist(job)

    def next_queued(self) -> Job | None:
        statement = (
            select(Job)
            .where(Job.status == "queued")
            .order_by(Job.created_at.asc())
            .limit(1)
        )
        return self._session.scalar(statement)

    def get(self, job_id: str) -> Job:
        job = self._session.get(Job, job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def start(self, job_id: str) -> Job:
        return self._transition(job_id, "running")

    def cancel(self, job_id: str) -> Job:
        return self._transition(job_id, "cancelled")

    def checkpoint(self, job_id: str, checkpoint: dict[str, Any]) -> Job:
        job = self.get(job_id)
        if job.status != "running":
            raise InvalidJobTransitionError(f"{job.status} cannot checkpoint")
        job.checkpoint = checkpoint
        return self._persist(job)

    def interrupt(self, job_id: str, message: str) -> Job:
        job = self._transition(job_id, "interrupted")
        job.error_code = "worker_interrupted"
        job.error_message = message
        return self._persist(job)

    def resume(self, job_id: str) -> Job:
        job = self._transition(job_id, "queued")
        job.error_code = None
        job.error_message = None
        return self._persist(job)

    def fail(self, job_id: str, code: str, message: str) -> Job:
        job = self._transition(job_id, "failed")
        job.error_code = code
        job.error_message = message
        return self._persist(job)

    def succeed(self, job_id: str) -> Job:
        job = self._transition(job_id, "succeeded")
        job.progress = 1.0
        return self._persist(job)

    def _transition(self, job_id: str, target: str) -> Job:
        job = self.get(job_id)
        if target not in TRANSITIONS.get(job.status, set()):
            raise InvalidJobTransitionError(f"{job.status} cannot transition to {target}")
        job.status = target
        return self._persist(job)

    def _persist(self, job: Job) -> Job:
        self._session.commit()
        self._session.refresh(job)
        return job
