from collections.abc import Callable
from concurrent.futures import CancelledError
from datetime import UTC, datetime, timedelta
from threading import Event, Lock
from time import monotonic, sleep

import pytest
from moonlightbox.agent.inference import (
    DuplicateInferenceRequestError,
    InferencePreemptedError,
    InferencePriority,
    PersonaInferenceRequest,
    PersonaInferenceResult,
    PersonaInferenceScheduler,
)


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.release = Event()
        self.started = Event()
        self._active = 0
        self.max_active = 0
        self._lock = Lock()

    def infer(self, request: PersonaInferenceRequest) -> PersonaInferenceResult:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        self.calls.append(request.request_id)
        if request.request_id == "blocker":
            self.started.set()
            assert self.release.wait(timeout=2)
        with self._lock:
            self._active -= 1
        return PersonaInferenceResult(
            request_id=request.request_id,
            request_type=request.request_type,
            output={"ok": True},
        )


class CooperativeBackend:
    def __init__(self) -> None:
        self.started: dict[str, Event] = {}
        self.release: dict[str, Event] = {}
        self.stopped: dict[str, Event] = {}

    def prepare(self, request_id: str) -> None:
        self.started[request_id] = Event()
        self.release[request_id] = Event()
        self.stopped[request_id] = Event()

    def infer(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool],
    ) -> PersonaInferenceResult:
        started = self.started.setdefault(request.request_id, Event())
        release = self.release.setdefault(request.request_id, Event())
        stopped = self.stopped.setdefault(request.request_id, Event())
        started.set()
        try:
            while not release.is_set():
                if should_cancel():
                    raise InferencePreemptedError()
                sleep(0.005)
        finally:
            stopped.set()
        return PersonaInferenceResult(
            request_id=request.request_id,
            request_type=request.request_type,
            output={"ok": True},
        )


def _request(
    request_id: str,
    priority: InferencePriority,
    *,
    deadline: datetime | None = None,
) -> PersonaInferenceRequest:
    return PersonaInferenceRequest(
        request_id=request_id,
        request_type="reply",
        priority=priority,
        deadline=deadline or datetime.now(UTC) + timedelta(seconds=5),
        model_version_id="model-1",
        payload={},
    )


def test_scheduler_orders_waiting_requests_by_priority_then_creation() -> None:
    backend = RecordingBackend()
    scheduler = PersonaInferenceScheduler(backend)
    blocker = scheduler.submit(_request("blocker", InferencePriority.REALTIME))
    assert backend.started.wait(timeout=1)

    offline = scheduler.submit(_request("offline", InferencePriority.OFFLINE))
    first_realtime = scheduler.submit(_request("realtime-1", InferencePriority.REALTIME))
    second_realtime = scheduler.submit(_request("realtime-2", InferencePriority.REALTIME))
    backend.release.set()

    for future in (blocker, offline, first_realtime, second_realtime):
        future.result(timeout=2)
    scheduler.shutdown()

    assert backend.calls == ["blocker", "realtime-1", "realtime-2", "offline"]
    assert backend.max_active == 1


def test_scheduler_expires_and_cancels_requests_before_execution() -> None:
    backend = RecordingBackend()
    scheduler = PersonaInferenceScheduler(backend)
    blocker = scheduler.submit(_request("blocker", InferencePriority.REALTIME))
    assert backend.started.wait(timeout=1)

    expired = scheduler.submit(
        _request(
            "expired",
            InferencePriority.OFFLINE,
            deadline=datetime.now(UTC) - timedelta(milliseconds=1),
        )
    )
    cancelled = scheduler.submit(_request("cancelled", InferencePriority.OFFLINE))
    assert scheduler.cancel("cancelled") is True
    backend.release.set()
    blocker.result(timeout=2)

    with pytest.raises(CancelledError):
        expired.result(timeout=2)
    with pytest.raises(CancelledError):
        cancelled.result(timeout=2)
    scheduler.shutdown()
    assert backend.calls == ["blocker"]


def test_scheduler_rejects_duplicate_ids_and_can_restart_after_shutdown() -> None:
    backend = RecordingBackend()
    scheduler = PersonaInferenceScheduler(backend)
    scheduler.submit(_request("once", InferencePriority.REALTIME)).result(timeout=2)
    with pytest.raises(DuplicateInferenceRequestError):
        scheduler.submit(_request("once", InferencePriority.REALTIME))

    scheduler.shutdown()
    scheduler.start()
    scheduler.submit(_request("after-restart", InferencePriority.REALTIME)).result(timeout=2)
    scheduler.shutdown()

    assert backend.calls == ["once", "after-restart"]


def test_realtime_request_preempts_running_offline_request() -> None:
    backend = CooperativeBackend()
    backend.prepare("offline")
    backend.prepare("realtime")
    scheduler = PersonaInferenceScheduler(backend)
    offline = scheduler.submit(_request("offline", InferencePriority.OFFLINE))
    assert backend.started["offline"].wait(timeout=1)

    started_at = monotonic()
    realtime = scheduler.submit(_request("realtime", InferencePriority.REALTIME))
    assert backend.stopped["offline"].wait(timeout=1)
    backend.release.setdefault("realtime", Event()).set()

    with pytest.raises(CancelledError):
        offline.result(timeout=1)
    assert realtime.result(timeout=1).request_id == "realtime"
    assert monotonic() - started_at < 1
    scheduler.shutdown()


def test_offline_request_does_not_preempt_running_realtime_request() -> None:
    backend = CooperativeBackend()
    backend.prepare("realtime")
    backend.prepare("offline")
    scheduler = PersonaInferenceScheduler(backend)
    realtime = scheduler.submit(_request("realtime", InferencePriority.REALTIME))
    assert backend.started["realtime"].wait(timeout=1)

    offline = scheduler.submit(_request("offline", InferencePriority.OFFLINE))
    sleep(0.05)
    assert not backend.stopped["realtime"].is_set()
    assert not backend.started["offline"].is_set()

    backend.release["realtime"].set()
    assert realtime.result(timeout=1).request_id == "realtime"
    assert backend.started.setdefault("offline", Event()).wait(timeout=1)
    backend.release.setdefault("offline", Event()).set()
    assert offline.result(timeout=1).request_id == "offline"
    scheduler.shutdown()


def test_cancel_signals_running_request_and_returns_cancelled_error() -> None:
    backend = CooperativeBackend()
    backend.prepare("active")
    scheduler = PersonaInferenceScheduler(backend)
    active = scheduler.submit(_request("active", InferencePriority.OFFLINE))
    assert backend.started["active"].wait(timeout=1)

    assert scheduler.cancel("active") is True
    assert backend.stopped["active"].wait(timeout=1)
    with pytest.raises(CancelledError):
        active.result(timeout=1)
    scheduler.shutdown()


def test_shutdown_signals_running_request() -> None:
    backend = CooperativeBackend()
    backend.prepare("active")
    scheduler = PersonaInferenceScheduler(backend)
    active = scheduler.submit(_request("active", InferencePriority.PROACTIVE))
    assert backend.started["active"].wait(timeout=1)

    scheduler.shutdown(wait=True)

    assert backend.stopped["active"].is_set()
    with pytest.raises(CancelledError):
        active.result(timeout=1)


def test_active_cancel_discards_result_from_legacy_synchronous_backend() -> None:
    backend = RecordingBackend()
    scheduler = PersonaInferenceScheduler(backend)
    active = scheduler.submit(_request("blocker", InferencePriority.OFFLINE))
    assert backend.started.wait(timeout=1)

    assert scheduler.cancel("blocker") is True
    backend.release.set()

    with pytest.raises(CancelledError):
        active.result(timeout=1)
    scheduler.shutdown()
