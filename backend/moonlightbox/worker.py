from datetime import UTC, datetime, timedelta
from threading import Event, Thread
from time import monotonic
from uuid import uuid4

from sqlalchemy import inspect, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from moonlightbox.db import Database
from moonlightbox.events.models import AnalysisRun
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import (
    JobHandlerError,
    JobRegistry,
    UnknownJobKindError,
)
from moonlightbox.jobs.service import JobService


class Worker:
    def __init__(
        self,
        database: Database,
        registry: JobRegistry,
        *,
        stop_event: Event | None = None,
        allowed_kinds: set[str] | frozenset[str] | None = None,
        lease_duration: timedelta = timedelta(minutes=2),
        heartbeat_interval: timedelta | None = None,
    ) -> None:
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration 必须大于零")
        resolved_interval = heartbeat_interval or timedelta(
            seconds=min(1.0, max(0.05, lease_duration.total_seconds() / 3))
        )
        if resolved_interval <= timedelta(0):
            raise ValueError("heartbeat_interval 必须大于零")
        self._database = database
        self._registry = registry
        self._stop_event = stop_event or Event()
        self._allowed_kinds = frozenset(allowed_kinds) if allowed_kinds is not None else None
        self._lease_duration = lease_duration
        self._heartbeat_interval = resolved_interval

    def run_once(self) -> bool:
        if self._stop_event.is_set():
            return False
        with Session(self._database.engine) as session:
            service = JobService(session, lock_retries=self._database.lock_retries)
            token = str(uuid4())
            running = service.claim_next_queued(
                worker_token=token,
                lease_duration=self._lease_duration,
                allowed_kinds=self._allowed_kinds,
            )
            if running is None:
                return False
            # JobService 的 checkpoint 会提交当前 Session，从而使 ORM 实例过期。
            # 心跳线程只能持有标量 ID，绝不能跨线程延迟读取 ``running.id``。
            job_id = running.id
            job_kind = running.kind
            lease_done = Event()
            lease_thread = Thread(
                target=self._maintain_lease,
                args=(job_id, token, lease_done),
                daemon=True,
            )
            lease_thread.start()

            try:
                handler = self._registry.get(job_kind)
            except UnknownJobKindError:
                lease_done.set()
                lease_thread.join()
                service.fail(
                    job_id,
                    "unknown_job_kind",
                    f"未注册的任务类型：{job_kind}",
                    token=token,
                )
                return True

            try:
                handler(service, running)
            except JobHandlerError as error:
                lease_done.set()
                lease_thread.join()
                session.rollback()
                service.fail(
                    job_id,
                    error.code,
                    error.safe_message,
                    token=token,
                )
                return True
            except Exception:
                lease_done.set()
                lease_thread.join()
                session.rollback()
                service.fail(
                    job_id,
                    "job_handler_failed",
                    "任务处理失败",
                    token=token,
                )
                return True

            lease_done.set()
            lease_thread.join()
            if self._stop_event.is_set():
                service.interrupt(
                    job_id,
                    "Worker 正在退出",
                    token=token,
                )
            service.succeed(job_id, token=token)
            return True

    def _maintain_lease(self, job_id: str, token: str, done: Event) -> None:
        heartbeat_seconds = self._heartbeat_interval.total_seconds()
        next_heartbeat = monotonic() + heartbeat_seconds
        while not done.is_set():
            if self._stop_event.is_set():
                self._interrupt_owned_execution(job_id, token)
                return
            remaining = max(0.0, next_heartbeat - monotonic())
            if done.wait(min(0.1, remaining)):
                return
            if monotonic() < next_heartbeat:
                continue
            with Session(self._database.engine) as session:
                service = JobService(
                    session,
                    lock_retries=self._database.lock_retries,
                )
                if (
                    service.heartbeat(
                        job_id,
                        token=token,
                        lease_duration=self._lease_duration,
                    )
                    is None
                ):
                    self._interrupt_owned_analysis(job_id)
                    return
            next_heartbeat = monotonic() + heartbeat_seconds

    def _interrupt_owned_execution(self, job_id: str, token: str) -> None:
        with Session(self._database.engine) as session:
            JobService(
                session,
                lock_retries=self._database.lock_retries,
            ).interrupt(
                job_id,
                "Worker 正在退出",
                token=token,
            )
        self._interrupt_owned_analysis(job_id)

    def _interrupt_owned_analysis(self, job_id: str) -> None:
        if not inspect(self._database.engine).has_table(AnalysisRun.__tablename__):
            return
        with Session(self._database.engine) as session:
            try:
                job = session.get(Job, job_id)
                if job is None:
                    return
                project_id = job.payload.get("project_id")
                import_id = job.payload.get("import_id")
                if not isinstance(project_id, str) or not isinstance(import_id, str):
                    return
                session.execute(
                    update(AnalysisRun)
                    .where(
                        AnalysisRun.project_id == project_id,
                        AnalysisRun.import_id == import_id,
                        AnalysisRun.status == "running",
                        AnalysisRun.lease_owner == f"job-{job.id}",
                    )
                    .values(
                        status="interrupted",
                        error_category="worker_interrupted",
                        error_message="Job 已取消或 Worker 正在退出",
                        lease_owner=None,
                        lease_token=None,
                        lease_expires_at=None,
                    )
                    .execution_options(synchronize_session=False)
                )
                session.commit()
            except OperationalError:
                session.rollback()


def recover_interrupted_jobs(database: Database) -> int:
    """只恢复租约已过期的任务，活跃 Worker 的任务保持不变。"""

    now = datetime.now(UTC)
    with Session(database.engine) as session:
        service = JobService(session, lock_retries=database.lock_retries)
        recovered = service.recover_worker_interrupted()
        running_jobs = list(
            session.scalars(
                select(Job).where(
                    Job.status == "running",
                    Job.lease_expires_at <= now,
                )
            )
        )
        has_analysis_runs = inspect(database.engine).has_table(AnalysisRun.__tablename__)
        for job in running_jobs:
            token = job.worker_token
            if token is None or not service.recover_expired_job(
                job.id,
                token=token,
                now=now,
                commit=False,
            ):
                continue
            if has_analysis_runs:
                project_id = job.payload.get("project_id")
                import_id = job.payload.get("import_id")
                if isinstance(project_id, str) and isinstance(import_id, str):
                    session.execute(
                        update(AnalysisRun)
                        .where(
                            AnalysisRun.project_id == project_id,
                            AnalysisRun.import_id == import_id,
                            AnalysisRun.status == "running",
                            AnalysisRun.lease_owner == f"job-{job.id}",
                        )
                        .values(
                            status="interrupted",
                            error_category="worker_interrupted",
                            error_message="Worker 上次运行意外中断",
                            lease_owner=None,
                            lease_token=None,
                            lease_expires_at=None,
                        )
                    )
            session.commit()
            recovered += 1
        return recovered
