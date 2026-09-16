"""统一故障与请求生命周期。仅作用于 Agent 作用域，非 Agent 客户端保持原行为。

自有模型 HTTP 通过异步传输取消并关闭请求；同步工具在边界和退避期间检查取消。
不可中断的同步操作由 timeout 收口，退出前保持 cancelling，迟到结果不交付。
"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
from random import uniform
from time import monotonic, sleep

import httpx
from langchain_core.tools import ToolException
from pydantic import ValidationError


@dataclass(frozen=True)
class ResiliencePolicy:
    request_timeout_seconds: float = 300.0
    max_retries: int = 2
    backoff_seconds: float = 1.0
    max_backoff_seconds: float = 10.0

    def __post_init__(self):
        if self.request_timeout_seconds <= 0 or self.max_retries < 0:
            raise ValueError("请求超时必须大于零，重试次数不能小于零")
        if self.backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("退避时间不能为负数")


@dataclass(frozen=True)
class Failure:
    code: str
    category: str
    retryable: bool
    message: str
    attempts: int = 1


class ExecutionInterrupted(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def classify_failure(error):
    code = str(getattr(error, "code", ""))
    if isinstance(error, (ToolException, ValidationError)):
        code = code or "invalid_tool_arguments"
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        code = "timeout"
    elif isinstance(error, httpx.RequestError):
        code = "network"
    status = (
        error.response.status_code
        if isinstance(error, httpx.HTTPStatusError)
        else getattr(error, "status_code", None)
    )
    permanent_index_codes = {
        "lightrag_content_rejected", "lightrag_provider_authentication",
        "lightrag_provider_request_rejected", "lightrag_index_error",
        "lightrag_configuration_missing",
    }
    if isinstance(status, int) and code not in permanent_index_codes:
        code = (
            "rate_limited"
            if status == 429
            else "server"
            if status >= 500
            else "authentication"
            if status in {401, 403}
            else "invalid_request"
        )
    code = {
        "rate_limit": "rate_limited",
        "missing_api_key": "authentication",
        "lightrag_timeout": "timeout",
        "lightrag_unavailable": "network",
        "lightrag_rate_limited": "rate_limited",
        "lightrag_incomplete_index": "server",
        "lightrag_index_failed": "server",
        "lightrag_authentication": "authentication",
    }.get(code, code)
    categories = {
        "lightrag_content_rejected": ("provider_rejection", False, "模型服务拒绝处理内容，需人工处理"),
        "lightrag_provider_authentication": ("configuration", False, "图谱模型凭据无效"),
        "lightrag_provider_request_rejected": ("protocol", False, "图谱模型拒绝请求"),
        "lightrag_index_error": ("runtime", False, "图谱内部错误，需检查后恢复"),
        "lightrag_configuration_missing": ("configuration", False, "LightRAG Sidecar 缺少模型服务配置"),
        "tool_unavailable": (
            "configuration",
            False,
            "工具绑定资料或依赖未就绪，请报告不可用，不要改写参数或当作无结果",
        ),
        "invalid_tool_arguments": ("tool_input", False, "工具参数无效，请按工具说明修正"),
        "timeout": ("transport", True, "模型或工具请求超时"),
        "network": ("transport", True, "网络连接失败"),
        "branch_writer_busy": ("runtime", True, "分支正在执行其他工作，请稍后恢复"),
        "branch_busy": ("runtime", True, "分支正在执行其他工作，请稍后恢复"),
        "rate_limited": ("transport", True, "服务限流"),
        "server": ("transport", True, "服务暂时不可用"),
        "empty_response": ("transport", True, "服务返回空响应"),
        "authentication": ("configuration", False, "凭据无效或缺失，请检查配置"),
        "disabled": ("configuration", False, "模型服务未启用"),
        "invalid_request": ("protocol", False, "请求参数被服务拒绝，请检查工具输入"),
        "invalid_response": ("protocol", False, "服务响应不符合协议"),
        "cancelled": ("cancellation", False, "执行已取消"),
        "stale_input_revision": ("cancellation", False, "新输入已使本次执行失效"),
        "input_revision_unavailable": ("runtime", False, "无法确认当前输入版本"),
        "wall_deadline": ("budget", False, "本次执行时间预算已耗尽"),
    }
    for budget_code in (
        "emergency_model_step_limit",
        "tool_call_safety_limit",
        "tool_result_context_limit",
        "input_context_limit",
        "execution_attempt_limit",
        "collaboration_attempt_limit",
        "collaboration_budget_exhausted",
    ):
        categories[budget_code] = ("budget", False, "执行预算或尝试次数已达上限")
    category, retryable, message = categories.get(
        code, ("runtime", False, "执行失败，请查看 Trace")
    )
    return Failure(
        code or "model_error", category, retryable, message, getattr(error, "_agent_attempts", 1)
    )


_scope = ContextVar("agent_resilience_scope", default=None)
_operation_deadline = ContextVar("agent_operation_deadline", default=None)


def managed():
    return _scope.get() is not None


@contextmanager
def execution_scope(policy, remaining, interrupt_reason, on_status=None):
    token = _scope.set((policy, remaining, interrupt_reason, on_status))
    try:
        yield
    finally:
        _scope.reset(token)


def check_interruption():
    scope = _scope.get()
    if scope is None:
        return
    _, remaining, interrupt, _ = scope
    reason = interrupt()
    if reason:
        raise ExecutionInterrupted(reason)
    if remaining() <= 0:
        raise ExecutionInterrupted("wall_deadline")


def request_timeout(seconds, *, model_request=True):
    """请求内部和嵌套网络操作使用同一个剩余截止时间。"""
    check_interruption()
    scope = _scope.get()
    if scope:
        seconds = min(seconds, scope[1]())
        if model_request:
            seconds = min(seconds, scope[0].request_timeout_seconds)
    deadline = _operation_deadline.get()
    if deadline is not None:
        seconds = min(seconds, deadline - monotonic())
    if seconds <= 0:
        raise TimeoutError("request deadline exceeded")
    return seconds


def emit_status(event):
    # 复用当前 Phoenix Span，不增加独立执行记录数据库。
    try:
        from opentelemetry import trace

        trace.get_current_span().add_event(
            "agent.request_status",
            {
                "status": str(event["status"]),
                "details": json.dumps(event, ensure_ascii=False),
            },
        )
    except Exception:
        pass
    scope = _scope.get()
    if scope and scope[3]:
        try:
            scope[3](event)
        except Exception:
            # 状态展示失败不能使真实调用重试或重复产生业务副作用。
            pass


def run_operation(callback, *, kind="model", replay_safe=True, timeout_seconds=None):
    scope = _scope.get()
    if scope is None:
        return callback()
    if _operation_deadline.get() is not None:
        # 工具内部再请求模型时不嵌套重试；最外层可重放操作持有唯一重试循环。
        check_interruption()
        return callback()
    policy = scope[0]
    attempts = policy.max_retries + 1 if replay_safe else 1
    for attempt in range(1, attempts + 1):
        check_interruption()
        timeout = request_timeout(
            timeout_seconds or policy.request_timeout_seconds,
            model_request=kind == "model",
        )
        token = _operation_deadline.set(monotonic() + timeout)
        emit_status({"status": "requesting", "operation": kind, "attempt": attempt})
        try:
            value = callback()
            check_interruption()
            if monotonic() > _operation_deadline.get():
                raise TimeoutError("request deadline exceeded")
        except Exception as error:
            from .capacity import RequestCapacityExceeded

            if isinstance(error, RequestCapacityExceeded):
                # 本地准入拒绝尚未发请求，交回循环整理上下文，不报告网络失败或重试。
                emit_status({**error.report, "status": "context_compaction_required"})
                raise
            failure = classify_failure(error)
            try:
                error._agent_attempts = attempt
            except Exception:
                pass
            if not failure.retryable or attempt == attempts:
                emit_status(
                    {
                        "status": "request_failed",
                        "operation": kind,
                        "error": asdict(classify_failure(error)),
                    }
                )
                raise
            delay = min(policy.max_backoff_seconds, policy.backoff_seconds * 2 ** (attempt - 1))
            delay = uniform(delay / 2, delay) if delay else 0
            emit_status(
                {
                    "status": "retrying",
                    "operation": kind,
                    "attempt": attempt,
                    "retry_in_seconds": delay,
                    "error": asdict(failure),
                }
            )
        else:
            emit_status({"status": "request_succeeded", "operation": kind, "attempt": attempt})
            return value
        finally:
            _operation_deadline.reset(token)
        until = monotonic() + delay
        while monotonic() < until:
            check_interruption()
            sleep(min(0.1, max(0, until - monotonic())))


def model_request(function):
    """装饰真实模型请求边界，避免重试整个 Agent 或重复它此前已完成的请求。"""

    @wraps(function)
    def wrapped(*args, **kwargs):
        return run_operation(lambda: function(*args, **kwargs), kind="model")

    return wrapped
