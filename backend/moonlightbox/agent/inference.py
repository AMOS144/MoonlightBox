"""单一常驻人格推理服务的领域协议与串行调度器。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from inspect import Parameter, signature
from itertools import count
from queue import Empty, PriorityQueue
from threading import Event, Lock, Thread
from typing import Literal, Protocol


class InferencePriority(IntEnum):
    """人格推理请求的固定优先级。"""

    REALTIME = 0
    PRE_SEND = 1
    PROACTIVE = 2
    OFFLINE = 3


@dataclass(frozen=True)
class PersonaInferenceRequest:
    """不携带 Web 或数据库对象的只读推理请求。"""

    request_id: str
    request_type: Literal[
        "reply",
        "cognition",
        "fused_reply",
        "runtime_director",
        "runtime_actor",
        "runtime_token_count",
    ]
    priority: InferencePriority
    deadline: datetime
    model_version_id: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id 不能为空")
        if not self.model_version_id.strip():
            raise ValueError("model_version_id 不能为空")
        object.__setattr__(self, "priority", InferencePriority(self.priority))
        if self.deadline.tzinfo is None:
            raise ValueError("deadline 必须包含时区")


@dataclass(frozen=True)
class PersonaInferenceResult:
    """人格推理服务返回的结构化结果。"""

    request_id: str
    request_type: Literal[
        "reply",
        "cognition",
        "fused_reply",
        "runtime_director",
        "runtime_actor",
        "runtime_token_count",
    ]
    output: Mapping[str, object]


class PersonaInferenceBackend(Protocol):
    """工作线程中调用的纯领域后端。"""

    def infer(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool] | None = None,
    ) -> PersonaInferenceResult: ...


class DuplicateInferenceRequestError(ValueError):
    """请求 ID 已被当前调度器接收。"""


class InferenceSchedulerStoppedError(RuntimeError):
    """调度器尚未启动或已经停止。"""


class InferencePreemptedError(RuntimeError):
    """推理在安全检查点被抢占。"""


@dataclass(order=True)
class _QueuedRequest:
    priority: int
    sequence: int
    request: PersonaInferenceRequest = field(compare=False)
    future: Future[PersonaInferenceResult] = field(compare=False)


@dataclass(frozen=True)
class _ActiveRequest:
    request_id: str
    priority: InferencePriority
    cancel_event: Event


class PersonaInferenceScheduler:
    """使用单工作线程严格串行调用模型后端。"""

    def __init__(self, backend: PersonaInferenceBackend) -> None:
        self._backend = backend
        self._lock = Lock()
        self._seen_ids: set[str] = set()
        self._futures: dict[str, Future[PersonaInferenceResult]] = {}
        self._sequence = count()
        self._queue: PriorityQueue[_QueuedRequest] = PriorityQueue()
        self._thread: Thread | None = None
        self._active: _ActiveRequest | None = None
        self._running = False
        self.start()

    def start(self) -> None:
        """启动或在正常关闭后重新启动工作线程。"""

        with self._lock:
            if self._running:
                return
            self._queue = PriorityQueue()
            self._running = True
            self._thread = Thread(
                target=self._run,
                name="persona-inference",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        request: PersonaInferenceRequest,
    ) -> Future[PersonaInferenceResult]:
        """提交请求并立即返回可等待结果。"""

        with self._lock:
            if not self._running:
                raise InferenceSchedulerStoppedError("人格推理调度器未运行")
            if request.request_id in self._seen_ids:
                raise DuplicateInferenceRequestError("重复的人格推理 request_id")
            self._seen_ids.add(request.request_id)
            future: Future[PersonaInferenceResult] = Future()
            self._futures[request.request_id] = future
            item = _QueuedRequest(
                int(request.priority),
                next(self._sequence),
                request,
                future,
            )
            self._queue.put(item)
            if (
                request.priority is InferencePriority.REALTIME
                and self._active is not None
                and self._active.priority
                in {InferencePriority.PROACTIVE, InferencePriority.OFFLINE}
            ):
                self._active.cancel_event.set()
            return future

    def cancel(self, request_id: str) -> bool:
        """取消等待中或正在运行的请求。"""

        with self._lock:
            future = self._futures.get(request_id)
            if future is None or future.done():
                return False
            if self._active is not None and self._active.request_id == request_id:
                self._active.cancel_event.set()
                return True
            return future.cancel()

    def shutdown(self, *, wait: bool = True) -> None:
        """停止领取新任务，并取消所有尚未执行的任务。"""

        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._active is not None:
                self._active.cancel_event.set()
            sentinel_future: Future[PersonaInferenceResult] = Future()
            self._queue.put(
                _QueuedRequest(
                    -1,
                    next(self._sequence),
                    PersonaInferenceRequest(
                        request_id="__shutdown__",
                        request_type="reply",
                        priority=InferencePriority.REALTIME,
                        deadline=datetime.max.replace(tzinfo=UTC),
                        model_version_id="__shutdown__",
                        payload={},
                    ),
                    sentinel_future,
                )
            )
            thread = self._thread
        if wait and thread is not None:
            thread.join()
            self._cancel_queued()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item.request.request_id == "__shutdown__":
                    return
                if item.future.cancelled():
                    continue
                if datetime.now(UTC) >= item.request.deadline:
                    item.future.cancel()
                    continue
                cancel_event = Event()
                active = _ActiveRequest(
                    request_id=item.request.request_id,
                    priority=item.request.priority,
                    cancel_event=cancel_event,
                )
                with self._lock:
                    if not self._running:
                        item.future.cancel()
                        continue
                    if not item.future.set_running_or_notify_cancel():
                        continue
                    self._active = active
                try:
                    result = self._infer(item.request, cancel_event.is_set)
                    if cancel_event.is_set():
                        raise InferencePreemptedError("推理结果已因取消而作废")
                except InferencePreemptedError:
                    item.future.set_exception(CancelledError())
                except BaseException as error:
                    item.future.set_exception(error)
                else:
                    item.future.set_result(result)
                finally:
                    with self._lock:
                        if self._active is active:
                            self._active = None
            finally:
                self._queue.task_done()

    def _infer(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool],
    ) -> PersonaInferenceResult:
        infer = self._backend.infer
        parameters = signature(infer).parameters.values()
        supports_cancel = any(
            parameter.name == "should_cancel"
            or parameter.kind is Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if supports_cancel:
            return infer(request, should_cancel=should_cancel)
        return infer(request)

    def _cancel_queued(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                return
            try:
                item.future.cancel()
            finally:
                self._queue.task_done()
