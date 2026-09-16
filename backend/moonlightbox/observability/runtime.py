"""Runtime 业务边界观测：Cycle 是流程根，Agent 可独立成树但保留关联。"""

from contextlib import contextmanager
from contextvars import ContextVar

from .phoenix import chain_span, record_span_output

_CURRENT_CYCLE = ContextVar("phoenix_runtime_cycle", default=None)


def current_cycle_attributes():
    """通过任务上下文传播关联键，不将 Trace 内容写入领域状态。"""
    return _CURRENT_CYCLE.get() or {}


@contextmanager
def runtime_cycle_span(*, cycle_id, cycle_key, project_id, branch_id, input_value):
    attributes = {
        "moonlightbox.cycle.id": cycle_id,
        "moonlightbox.cycle.key": cycle_key,
        "moonlightbox.project.id": project_id,
        "moonlightbox.branch.id": branch_id,
    }
    token = _CURRENT_CYCLE.set(attributes)
    try:
        with chain_span(
            "moonlightbox.runtime.cycle", input_value=input_value, attributes=attributes
        ) as span:
            yield span
    finally:
        _CURRENT_CYCLE.reset(token)


def observed_stage(name):
    """为无副作用校验等同步阶段记录耗时、输入与结果；异常交给统一 Span 处理。"""
    from functools import wraps
    from inspect import signature

    def decorate(callback):
        @wraps(callback)
        def call(*args, **kwargs):
            arguments = dict(signature(callback).bind(*args, **kwargs).arguments)
            arguments.pop("self", None)
            with chain_span(name, input_value=arguments) as span:
                result = callback(*args, **kwargs)
                record_span_output(span, result)
                return result

        return call

    return decorate
