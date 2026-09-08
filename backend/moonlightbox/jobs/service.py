from datetime import UTC, datetime, timedelta
from time import sleep
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Update

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


class JobLeaseLostError(RuntimeError):
    pass


class JobService:
    def __init__(self, session: Session, *, lock_retries: int = 2) -> None:
        self._session = session
        self._lock_retries = max(0, lock_retries)

    @property
    def session(self) -> Session:
        return self._session

    def enqueue(self, kind: str, payload: dict[str, Any]) -> Job:
        job = Job(kind=kind, payload=payload)
        self._session.add(job)
        return self._persist(job)

    def enqueue_unique(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        dedupe_key: str,
        commit: bool = True,
    ) -> Job:
        now = datetime.now(UTC)
        statement = (
            sqlite_insert(Job)
            .values(
                id=str(uuid4()),
                kind=kind,
                payload=payload,
                status="queued",
                progress=0.0,
                checkpoint=None,
                error_code=None,
                error_message=None,
                dedupe_key=dedupe_key,
                worker_token=None,
                lease_expires_at=None,
                created_at=now,
                updated_at=now,
            )
            .on_conflict_do_nothing(index_elements=[Job.dedupe_key])
        )
        try:
            self._session.execute(statement)
            if commit:
                self._session.commit()
            else:
                self._session.flush()
        except OperationalError as error:
            self._session.rollback()
            if not _is_lock_error(error):
                raise
        job = self._session.scalar(
            select(Job)
            .where(Job.dedupe_key == dedupe_key)
            .execution_options(populate_existing=True)
        )
        if job is None:
            raise RuntimeError("原子创建任务后未找到幂等记录")
        return job

    def next_queued(self) -> Job | None:
        statement = (
            select(Job).where(Job.status == "queued").order_by(Job.created_at.asc()).limit(1)
        )
        return self._session.scalar(statement)

    def claim_next_queued(
        self,
        *,
        worker_token: str,
        lease_duration: timedelta,
        allowed_kinds: frozenset[str] | None = None,
        now: datetime | None = None,
    ) -> Job | None:
        """用状态 CAS 原子抢占最早任务，竞争失败时继续寻找可执行任务。"""

        if not worker_token.strip():
            raise ValueError("worker token 不能为空")
        if lease_duration <= timedelta(0):
            raise ValueError("lease duration 必须大于零")
        claimed_at = now or datetime.now(UTC)
        if allowed_kinds is not None and not allowed_kinds:
            return None
        while True:
            query = select(Job.id).where(Job.status == "queued")
            if allowed_kinds is not None:
                query = query.where(Job.kind.in_(allowed_kinds))
            queued_id = self._session.scalar(
                query.order_by(Job.created_at.asc(), Job.id.asc()).limit(1)
            )
            if queued_id is None:
                return None
            result = self._execute_update(
                update(Job)
                .where(Job.id == queued_id, Job.status == "queued")
                .values(
                    status="running",
                    worker_token=worker_token,
                    lease_expires_at=claimed_at + lease_duration,
                    updated_at=claimed_at,
                )
            )
            if result is None:
                return None
            if result.rowcount == 1:
                self._session.commit()
                return self.get(queued_id)
            self._session.rollback()

    def find_latest_by_payload(
        self,
        kind: str,
        key: str,
        value: str,
    ) -> Job | None:
        jobs = self._session.scalars(
            select(Job).where(Job.kind == kind).order_by(Job.created_at.desc())
        )
        return next((job for job in jobs if job.payload.get(key) == value), None)

    def list_for_project(self, project_id: str, *, limit: int = 100) -> list[Job]:
        jobs = self._session.scalars(
            select(Job).order_by(Job.created_at.desc())
        )
        return [
            job
            for job in jobs
            if job.payload.get("project_id") == project_id
        ][:limit]

    def get(self, job_id: str) -> Job:
        job = self._session.scalar(
            select(Job).where(Job.id == job_id).execution_options(populate_existing=True)
        )
        if job is None:
            raise JobNotFoundError(job_id)
        return job

    def start(
        self,
        job_id: str,
        *,
        worker_token: str = "manual-worker",
        lease_duration: timedelta = timedelta(minutes=2),
        now: datetime | None = None,
    ) -> Job:
        started_at = now or datetime.now(UTC)
        result = self._execute_update(
            update(Job)
            .where(Job.id == job_id, Job.status == "queued")
            .values(
                status="running",
                worker_token=worker_token,
                lease_expires_at=started_at + lease_duration,
                updated_at=started_at,
            )
        )
        if result is None or result.rowcount != 1:
            raise InvalidJobTransitionError("任务不能转换为 running")
        self._session.commit()
        return self.get(job_id)

    def cancel(self, job_id: str) -> Job:
        result = self._execute_update(
            update(Job)
            .where(Job.id == job_id, Job.status.in_({"queued", "running"}))
            .values(
                status="cancelled",
                worker_token=None,
                lease_expires_at=None,
                updated_at=datetime.now(UTC),
            )
        )
        if result is None or result.rowcount != 1:
            job = self.get(job_id)
            raise InvalidJobTransitionError(f"{job.status} cannot transition to cancelled")
        self._session.commit()
        return self.get(job_id)

    def checkpoint(
        self,
        job_id: str,
        checkpoint: dict[str, Any],
        *,
        token: str,
    ) -> Job:
        progress = checkpoint.get("progress")
        values: dict[str, Any] = {
            "checkpoint": checkpoint,
            "updated_at": datetime.now(UTC),
        }
        if isinstance(progress, int | float):
            values["progress"] = max(0.0, min(1.0, float(progress)))
        result = self._execute_update(
            update(Job)
            .where(
                Job.id == job_id,
                Job.status == "running",
                Job.worker_token == token,
            )
            .values(**values)
        )
        if result is None or result.rowcount != 1:
            raise JobLeaseLostError("job_lease_lost")
        self._session.commit()
        return self.get(job_id)

    def heartbeat(
        self,
        job_id: str,
        *,
        token: str,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> Job | None:
        if lease_duration <= timedelta(0):
            raise ValueError("lease duration 必须大于零")
        heartbeat_at = now or datetime.now(UTC)
        result = self._execute_update(
            update(Job)
            .where(
                Job.id == job_id,
                Job.status == "running",
                Job.worker_token == token,
            )
            .values(
                lease_expires_at=heartbeat_at + lease_duration,
                updated_at=heartbeat_at,
            )
        )
        if result is None or result.rowcount != 1:
            self._session.rollback()
            return None
        self._session.commit()
        return self.get(job_id)

    def is_running_with_token(self, job_id: str, token: str) -> bool:
        return (
            self._session.scalar(
                select(Job.id).where(
                    Job.id == job_id,
                    Job.status == "running",
                    Job.worker_token == token,
                )
            )
            is not None
        )

    def interrupt(
        self,
        job_id: str,
        message: str,
        *,
        token: str,
    ) -> Job | None:
        return self._finish(
            job_id,
            "interrupted",
            token=token,
            error_code="worker_interrupted",
            error_message=message,
        )

    def resume(self, job_id: str) -> Job:
        result = self._execute_update(
            update(Job)
            .where(Job.id == job_id, Job.status.in_({"failed", "interrupted"}))
            .values(
                status="queued",
                error_code=None,
                error_message=None,
                worker_token=None,
                lease_expires_at=None,
                updated_at=datetime.now(UTC),
            )
        )
        if result is None or result.rowcount != 1:
            job = self.get(job_id)
            raise InvalidJobTransitionError(f"{job.status} cannot transition to queued")
        self._session.commit()
        return self.get(job_id)

    def recover_expired_running(self, *, now: datetime | None = None) -> int:
        recovered_at = now or datetime.now(UTC)
        result = self._execute_update(
            update(Job)
            .where(
                Job.status == "running",
                Job.lease_expires_at <= recovered_at,
            )
            .values(
                status="queued",
                error_code=None,
                error_message=None,
                worker_token=None,
                lease_expires_at=None,
                updated_at=recovered_at,
            )
        )
        if result is None:
            return 0
        self._session.commit()
        return result.rowcount or 0

    def recover_worker_interrupted(self) -> int:
        """重新排队由 Worker 中断且不再持有租约的任务。"""

        result = self._execute_update(
            update(Job)
            .where(
                Job.status == "interrupted",
                Job.error_code == "worker_interrupted",
                Job.worker_token.is_(None),
                Job.lease_expires_at.is_(None),
            )
            .values(
                status="queued",
                error_code=None,
                error_message=None,
                updated_at=datetime.now(UTC),
            )
        )
        if result is None:
            return 0
        self._session.commit()
        return result.rowcount or 0

    def recover_expired_job(
        self,
        job_id: str,
        *,
        token: str,
        now: datetime | None = None,
        commit: bool = True,
    ) -> bool:
        recovered_at = now or datetime.now(UTC)
        result = self._execute_update(
            update(Job)
            .where(
                Job.id == job_id,
                Job.status == "running",
                Job.worker_token == token,
                Job.lease_expires_at <= recovered_at,
            )
            .values(
                status="queued",
                error_code=None,
                error_message=None,
                worker_token=None,
                lease_expires_at=None,
                updated_at=recovered_at,
            )
        )
        if result is None or result.rowcount != 1:
            self._session.rollback()
            return False
        if commit:
            self._session.commit()
        return True

    def fail(
        self,
        job_id: str,
        code: str,
        message: str,
        *,
        token: str,
    ) -> Job | None:
        return self._finish(
            job_id,
            "failed",
            token=token,
            error_code=code,
            error_message=message,
        )

    def succeed(self, job_id: str, *, token: str) -> Job | None:
        return self._finish(job_id, "succeeded", token=token, progress=1.0)

    def succeed_in_transaction(
        self,
        job_id: str,
        *,
        token: str,
        checkpoint: dict[str, Any],
    ) -> bool:
        """在调用方事务内 CAS 完成任务，不单独提交或回滚。"""

        result = cast(
            CursorResult[Any],
            self._session.execute(
                update(Job)
                .where(
                    Job.id == job_id,
                    Job.status == "running",
                    Job.worker_token == token,
                )
                .values(
                    status="succeeded",
                    progress=1.0,
                    checkpoint=checkpoint,
                    error_code=None,
                    error_message=None,
                    worker_token=None,
                    lease_expires_at=None,
                    updated_at=datetime.now(UTC),
                )
                .execution_options(synchronize_session=False)
            ),
        )
        return result.rowcount == 1

    def _finish(
        self,
        job_id: str,
        status: str,
        *,
        token: str,
        error_code: str | None = None,
        error_message: str | None = None,
        progress: float | None = None,
    ) -> Job | None:
        values: dict[str, Any] = {
            "status": status,
            "error_code": error_code,
            "error_message": error_message,
            "worker_token": None,
            "lease_expires_at": None,
            "updated_at": datetime.now(UTC),
        }
        if progress is not None:
            values["progress"] = progress
        result = self._execute_update(
            update(Job)
            .where(
                Job.id == job_id,
                Job.status == "running",
                Job.worker_token == token,
            )
            .values(**values)
        )
        if result is None or result.rowcount != 1:
            self._session.rollback()
            return None
        self._session.commit()
        return self.get(job_id)

    def _execute_update(self, statement: Update) -> CursorResult[Any] | None:
        statement = statement.execution_options(synchronize_session=False)
        for attempt in range(self._lock_retries + 1):
            try:
                return cast(CursorResult[Any], self._session.execute(statement))
            except OperationalError as error:
                self._session.rollback()
                if not _is_lock_error(error):
                    raise
                if attempt >= self._lock_retries:
                    return None
                sleep(0.01 * (attempt + 1))
        return None

    def _persist(self, job: Job) -> Job:
        self._session.commit()
        self._session.refresh(job)
        return job


def _is_lock_error(error: OperationalError) -> bool:
    return "locked" in str(error).lower() or "busy" in str(error).lower()
