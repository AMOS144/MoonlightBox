from collections.abc import Callable

from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService

JobHandler = Callable[[JobService, Job], None]


class UnknownJobKindError(LookupError):
    pass


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
