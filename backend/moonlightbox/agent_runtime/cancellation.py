"""业务取消信号的统一入口；不会把 ORM Session 带进模型或异步请求。"""

from contextlib import contextmanager
from contextvars import ContextVar

_signal = ContextVar("agent_owner_cancellation", default=None)


@contextmanager
def cancellation_scope(callback):
    token = _signal.set(callback)
    try:
        yield
    finally:
        _signal.reset(token)


def cancellation_requested():
    callback = _signal.get()
    return bool(callback and callback())


def current_cancellation_signal():
    """捕获父取消回调，便于异步适配器叠加局部取消，不递归进入新作用域。"""
    return _signal.get()


def job_signal(engine, job_id, worker_token):
    """每次短查询使用独立 Session，取消请求不受执行线程的旧事务快照影响。"""
    from sqlalchemy.orm import Session

    from moonlightbox.jobs.models import Job

    def check():
        with Session(engine) as session:
            job = session.get(Job, job_id)
            return job is None or job.status != "running" or job.worker_token != worker_token

    return check
