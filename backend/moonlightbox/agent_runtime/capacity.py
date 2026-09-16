"""按实际请求 JSON 估算容量；usage 校正只在同一执行、同一端点和模型内共享。"""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from math import ceil

from .resilience import ExecutionInterrupted, emit_status

_scope = ContextVar("agent_request_capacity", default=None)


class RequestCapacityExceeded(ExecutionInterrupted):
    """实际请求尚未发送；携带同一次准入估算，供 Controller 压缩后重新装配。"""

    def __init__(self, report):
        super().__init__("input_context_limit")
        self.report = report


def estimate_tokens(body):
    # 英文约四字符一 token；非 ASCII 保守按一字符一 token，再保留协议包装余量。
    # 这是启发式而非供应商 tokenizer，后续用真实 input usage 向上修正。
    text = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    ascii_count = sum(ord(char) < 128 for char in text)
    return ceil(ascii_count / 4 + len(text) - ascii_count) + 32


@contextmanager
def capacity_scope(policy, calibration):
    token = _scope.set((policy, calibration))
    try:
        yield
    finally:
        _scope.reset(token)


def check_request(body, endpoint):
    scope = _scope.get()
    if scope is None:
        return None
    policy, calibration = scope
    key = endpoint + "|" + str(body.get("model", ""))
    raw = estimate_tokens(body)
    ratio = calibration.get(key, 1.0)
    estimated = ceil(raw * ratio)
    reserved = int(body.get("max_completion_tokens") or body.get("max_tokens") or 0)
    window = policy.model_context_windows.get(
        str(body.get("model", "")), policy.context_window_tokens
    )
    report = {
        "status": "context_capacity",
        "estimated_input_tokens": estimated,
        "raw_estimated_input_tokens": raw,
        "reserved_output_tokens": reserved,
        "context_window_tokens": window,
        "safety_margin_tokens": policy.context_safety_margin_tokens,
        "usage_correction_ratio": ratio,
    }
    emit_status(report)
    if estimated + reserved + policy.context_safety_margin_tokens > window:
        raise RequestCapacityExceeded(report)
    return key, raw


def observe_usage(sample, payload):
    scope = _scope.get()
    if scope is None or sample is None or not isinstance(payload, dict):
        return
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return
    actual = usage.get("prompt_tokens", usage.get("input_tokens"))
    if not isinstance(actual, int) or isinstance(actual, bool) or actual <= 0:
        return
    key, raw = sample
    calibration = scope[1]
    # 不用一次低 usage 放大可用窗口；不把 cached_tokens 与 prompt_tokens 重复相加。
    calibration[key] = max(calibration.get(key, 1.0), actual / max(1, raw))
    emit_status(
        {
            "status": "context_usage",
            "actual_input_tokens": actual,
            "raw_estimated_input_tokens": raw,
            "usage_correction_ratio": calibration[key],
        }
    )
