from collections.abc import Callable

from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService

JobHandler = Callable[[JobService, Job], None]


class UnknownJobKindError(LookupError):
    pass


class JobHandlerError(RuntimeError):
    """允许 handler 向 Worker 传递可安全持久化的错误。"""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.safe_message = message
        super().__init__(message)


class JobRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, JobHandler] = {}

    def register(self, kind: str, handler: JobHandler) -> None:
        self._handlers[kind] = handler

    def get(self, kind: str) -> JobHandler:
        try:
            return self._handlers[kind]
        except KeyError as error:
            raise UnknownJobKindError(kind) from error
