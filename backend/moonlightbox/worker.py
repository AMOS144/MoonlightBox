from sqlalchemy.orm import Session

from moonlightbox.db import Database
from moonlightbox.jobs.registry import JobRegistry, UnknownJobKindError
from moonlightbox.jobs.service import JobService


class Worker:
    def __init__(self, database: Database, registry: JobRegistry) -> None:
        self._database = database
        self._registry = registry

    def run_once(self) -> bool:
        with Session(self._database.engine) as session:
            service = JobService(session)
            queued = service.next_queued()
            if queued is None:
                return False

            running = service.start(queued.id)
            try:
                handler = self._registry.get(running.kind)
            except UnknownJobKindError:
                service.fail(
                    running.id,
                    "unknown_job_kind",
                    f"未注册的任务类型：{running.kind}",
                )
                return True

            try:
                handler(service, running)
            except Exception as error:
                service.fail(running.id, "job_handler_failed", str(error))
                return True

            service.succeed(running.id)
            return True
