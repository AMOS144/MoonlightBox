"""真实 HTTP 尝试诊断，独立于模型回合与计费统计，不改变重试策略。"""

from time import monotonic
from urllib.parse import urlsplit

from .execution_context import current_agent_execution_trace_context
from .phoenix import add_context_snapshot, chain_span, record_span_attributes, record_span_error
from .trace_summary import canonical_hash, canonical_json


def request_diagnostics(body):
    """比较实际请求而非 Prompt 文件；只提供差异线索，不猜供应商缓存算法。"""
    context = current_agent_execution_trace_context()
    messages = body.get("messages", [])
    prefix = {
        "system": [m for m in messages if m.get("role") in {"system", "developer"}],
        "tools": body.get("tools", []),
    }
    parameters = {k: v for k, v in body.items() if k not in {"messages", "tools"}}
    serialized = canonical_json(body).encode()
    model = str(body.get("model", ""))
    previous = context.state.previous_provider_requests.get(model) if context else None
    diagnostic = {
        "request_bytes": len(serialized),
        "request_hash": canonical_hash(body),
        "prefix_hash": canonical_hash(prefix),
        "parameters_hash": canonical_hash(parameters),
        "messages_count": len(messages),
        "tools_count": len(body.get("tools", [])),
        "comparison_available": previous is not None,
    }
    if previous:
        old = previous["bytes"]
        first_diff = next(
            (i for i, pair in enumerate(zip(old, serialized, strict=False)) if pair[0] != pair[1]),
            min(len(old), len(serialized)),
        )
        diagnostic.update(
            {
                "prefix_changed": previous["prefix_hash"] != diagnostic["prefix_hash"],
                "parameters_changed": previous["parameters_hash"] != diagnostic["parameters_hash"],
                "identical_request": old == serialized,
                "first_changed_byte": first_diff,
            }
        )
    if context:
        context.state.previous_provider_requests[model] = {**diagnostic, "bytes": serialized}
        context.state.request_diagnostics.append(diagnostic)
    return diagnostic


def observed_post(client, endpoint, *, attempt, cancellable=False, **kwargs):
    """不记录 Authorization；每次尝试保留请求标识、状态、耗时与原始响应。"""
    url = urlsplit(endpoint)
    body = kwargs.get("json") or {}
    started = monotonic()
    # 观测计算异常不阻止真实 HTTP 请求。
    try:
        diagnostic = request_diagnostics(body)
    except Exception:
        diagnostic = {"diagnostic_unavailable": True}
    with chain_span(
        "moonlightbox.provider.http_attempt",
        attributes={
            "moonlightbox.provider.attempt": attempt,
            "moonlightbox.provider.endpoint": f"{url.scheme}://{url.hostname}{url.path}",
            "moonlightbox.provider.request_hash": canonical_hash(body),
            "moonlightbox.provider.streaming": bool(body.get("stream", False)),
            "moonlightbox.provider.max_tokens": body.get("max_tokens", 0),
            "moonlightbox.provider.model": body.get("model", ""),
        },
    ) as span:
        try:
            record_span_attributes(span, {"moonlightbox.provider.request_diagnostics": diagnostic})
            from moonlightbox.agent_runtime.http_transport import post

            response = post(client, endpoint, cancellable=cancellable, **kwargs)
            record_span_attributes(
                span,
                {
                    "http.response.status_code": response.status_code,
                    "moonlightbox.provider.request_id": response.headers.get("x-request-id")
                    or response.headers.get("request-id")
                    or "",
                },
            )
            try:
                payload = response.json()
                add_context_snapshot(span, event_name="http_response", value=payload)
                record_span_attributes(
                    span,
                    {
                        "moonlightbox.provider.completion_id": payload.get("id", "")
                        if isinstance(payload, dict)
                        else "",
                        "moonlightbox.provider.finish_reasons": [
                            str(c.get("finish_reason"))
                            for c in payload.get("choices", [])
                            if isinstance(c, dict)
                        ]
                        if isinstance(payload, dict)
                        else [],
                    },
                )
            except ValueError:
                record_span_attributes(span, {"moonlightbox.provider.response_json_valid": False})
            if response.status_code >= 400:
                record_span_error(span, RuntimeError(f"http_status_{response.status_code}"))
            return response
        finally:
            record_span_attributes(
                span,
                {
                    "moonlightbox.provider.request_elapsed_ms": int((monotonic() - started) * 1000),
                },
            )
