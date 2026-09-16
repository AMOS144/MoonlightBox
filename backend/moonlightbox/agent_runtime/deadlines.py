"""同步工具的协作式截止时间，向网络请求传递剩余时间。"""

from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

_deadline = ContextVar("agent_tool_deadline", default=None)


@contextmanager
def tool_deadline(seconds):
    token = _deadline.set(monotonic() + max(0, seconds))
    try:
        yield
    finally:
        _deadline.reset(token)


def bounded_timeout(seconds):
    from .resilience import check_interruption

    # 合作工具在每次后续网络请求前检查取消，而不只检查外层整个工具的起止。
    check_interruption()
    deadline = _deadline.get()
    if deadline is None:
        return seconds
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("工具截止时间已到，不再启动新网络请求")
    return min(seconds, remaining)
