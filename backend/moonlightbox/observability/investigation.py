"""将 Phoenix 原始树与分块快照组织为可下载的调查材料，不另建 Trace 存储。"""

import hashlib
import json
from collections import defaultdict


def unpack_snapshots(span):
    """必须全部分块到齐且校验摘要一致；缺块时明确标记，绝不拼成伪完整上下文。"""
    groups = defaultdict(list)
    for event in span.get("events", []):
        if str(event.get("name", "")).startswith("moonlightbox.snapshot."):
            attrs = event.get("attributes", {})
            groups[(event["name"], attrs.get("moonlightbox.snapshot.id"))].append(attrs)
    result = []
    for (name, snapshot_id), chunks in groups.items():
        expected = chunks[0].get("moonlightbox.snapshot.chunk_count", 0)
        indexed = {c.get("moonlightbox.snapshot.chunk_index"): c for c in chunks}
        complete = (
            isinstance(expected, int) and expected > 0 and set(indexed) == set(range(expected))
        )
        content = "".join(
            c.get("moonlightbox.snapshot.content", "") for _, c in sorted(indexed.items())
        )
        complete = complete and hashlib.sha256(content.encode()).hexdigest()[:16] == snapshot_id
        value = None
        if complete:
            try:
                value = json.loads(content)
            except ValueError:
                value = content
        result.append(
            {
                "name": name,
                "snapshot_id": snapshot_id,
                "complete": complete,
                "expected_chunks": expected,
                "received_chunks": len(indexed),
                "value": value,
                "partial_content": None if complete else content,
            }
        )
    return result


def build_investigation(spans):
    """按原始 Span ID 去重；调用计数只数真实调用边界，不累计所有红色父 Span。"""
    unique = {
        (s.get("context", {}).get("trace_id"), s.get("context", {}).get("span_id")): s
        for s in spans
    }
    ordered = sorted(unique.values(), key=lambda s: str(s.get("start_time", "")))
    roots = [s for s in ordered if not s.get("parent_id")]
    attempts = [s for s in ordered if s.get("name") == "moonlightbox.provider.http_attempt"]
    agents = [s for s in ordered if s.get("name") == "moonlightbox.agent.run"]
    errors = [
        {
            "trace_id": s.get("context", {}).get("trace_id"),
            "span_id": s.get("context", {}).get("span_id"),
            "name": s.get("name"),
            "status_message": s.get("status_message"),
            "events": [e for e in s.get("events", []) if e.get("name") == "exception"],
        }
        for s in ordered
        if s.get("status_code") == "ERROR"
    ]
    return {
        "source": "phoenix",
        "roots": roots,
        "spans": ordered,
        "snapshots": [
            {"span_id": s.get("context", {}).get("span_id"), "items": unpack_snapshots(s)}
            for s in ordered
            if any(
                str(e.get("name", "")).startswith("moonlightbox.snapshot.")
                for e in s.get("events", [])
            )
        ],
        "summary": {
            "agent_executions": len(agents),
            "http_attempts": len(attempts),
            "http_failed_attempts": sum(s.get("status_code") == "ERROR" for s in attempts),
            "http_attempt_instrumentation_present": bool(attempts),
            "note": "旧记录没有逐次 HTTP 埋点时，0 不代表从未请求；异常父子传播不等于多次请求失败",
        },
        "errors": errors,
    }
