"""Phoenix 根 Span 所需的确定性 Agent 运行摘要。

这里不从聊天文本猜测“模型是否读懂了”。所有进展和消费关系只来自工具返回或最终产物
中的结构化引用（``source_ids``、``evidence_ids`` 等）。这样诊断不会重新引入正则或
字符串命中式的伪证据判断。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

from langchain_core.messages import BaseMessage

_REFERENCE_FIELDS = frozenset(
    {
        "source_ids",
        "source_id",
        "message_ids",
        "message_id",
        "document_ids",
        "document_id",
        "entity_ids",
        "entity_id",
        "relation_ids",
        "relation_id",
        "claim_ids",
        "claim_id",
        "evidence_ids",
        "evidence_id",
        "event_ids",
        "event_id",
        "plan_block_ids",
        "plan_block_id",
    }
)


@dataclass(frozen=True, slots=True)
class ContextSegment:
    """一次模型请求中一个可解释的上下文来源。"""

    kind: str
    source: str
    estimated_tokens: int
    message_index: int | None = None
    reference_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoundObservation:
    """发起模型请求之前的上下文盘点。"""

    phase: str
    round_index: int
    messages_snapshot: list[dict[str, object]]
    tool_schemas: list[dict[str, object]]
    segments: list[ContextSegment]
    payload_estimated_tokens: int
    system_prompt_hash: str | None
    tool_schema_hash: str | None


@dataclass(slots=True)
class AgentTraceState:
    # 仅用于当前执行的缓存差异分析，不参与正常恢复或业务判断。
    previous_provider_requests: dict[str, dict[str, object]] = field(default_factory=dict)
    request_diagnostics: list[dict[str, object]] = field(default_factory=list)
    """单次执行的内存诊断缓冲区；结束时一次性汇总到 Phoenix 根 Span。"""

    rounds: list[dict[str, object]] = field(default_factory=list)
    provider_calls: list[dict[str, object]] = field(default_factory=list)
    tools: list[dict[str, object]] = field(default_factory=list)
    compactions: list[dict[str, object]] = field(default_factory=list)
    pending_tool_results: list[dict[str, object]] = field(default_factory=list)
    exact_tool_call_counts: dict[str, int] = field(default_factory=dict)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    provider_usage_reported: bool = False
    cache_usage_reported: bool = False
    tool_success_calls: int = 0
    tool_error_calls: int = 0
    tool_empty_calls: int = 0
    tool_cancelled_calls: int = 0
    terminal_status: str | None = None
    terminal_reason: str | None = None


def serialize_message(message: object) -> dict[str, object]:
    """把 LangChain 消息投影为可回放 JSON，不丢弃模型工具调用或思考字段。"""

    if not isinstance(message, BaseMessage):
        return {"type": type(message).__name__, "content": _jsonable(message)}
    payload: dict[str, object] = {
        "type": message.type,
        "content": _jsonable(message.content),
    }
    name = getattr(message, "name", None)
    if isinstance(name, str) and name:
        payload["name"] = name
    tool_call_id = getattr(message, "tool_call_id", None)
    if isinstance(tool_call_id, str) and tool_call_id:
        payload["tool_call_id"] = tool_call_id
    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        payload["tool_calls"] = _jsonable(tool_calls)
    additional = getattr(message, "additional_kwargs", None)
    if isinstance(additional, Mapping) and additional:
        payload["additional_kwargs"] = _jsonable(dict(additional))
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, Mapping) and response_metadata:
        payload["response_metadata"] = _jsonable(dict(response_metadata))
    return payload


def serialize_messages(messages: Sequence[object]) -> list[dict[str, object]]:
    return [serialize_message(message) for message in messages]


def make_round_observation(
    *,
    phase: str,
    round_index: int,
    messages: Sequence[object],
    tool_schemas: Sequence[Mapping[str, object]],
) -> RoundObservation:
    snapshot = serialize_messages(messages)
    schemas = [_mapping_jsonable(schema) for schema in tool_schemas]
    segments: list[ContextSegment] = []
    for index, message in enumerate(snapshot):
        message_type = str(message.get("type") or "unknown")
        kind = {
            "system": "system_prompt",
            "human": "user_message",
            "ai": "assistant_message",
            "tool": "tool_result",
        }.get(message_type, "message")
        source = str(message.get("name") or message_type)
        segments.append(
            ContextSegment(
                kind=kind,
                source=source,
                estimated_tokens=estimate_tokens(message),
                message_index=index,
                reference_ids=tuple(sorted(extract_structured_references(message))),
            )
        )
        if message_type == "ai":
            for call in _tool_calls_from_snapshot(message):
                segments.append(
                    ContextSegment(
                        kind="tool_arguments",
                        source=str(call.get("name") or "unknown"),
                        estimated_tokens=estimate_tokens(call.get("args") or {}),
                        message_index=index,
                        reference_ids=tuple(
                            sorted(extract_structured_references(call.get("args") or {}))
                        ),
                    )
                )
    if schemas:
        segments.append(
            ContextSegment(
                kind="tool_schema",
                source="registered_tools",
                estimated_tokens=estimate_tokens(schemas),
            )
        )
    system_payload = [item for item in snapshot if item.get("type") == "system"]
    return RoundObservation(
        phase=phase,
        round_index=round_index,
        messages_snapshot=snapshot,
        tool_schemas=schemas,
        segments=segments,
        payload_estimated_tokens=sum(segment.estimated_tokens for segment in segments),
        system_prompt_hash=canonical_hash(system_payload) if system_payload else None,
        tool_schema_hash=canonical_hash(schemas) if schemas else None,
    )


def record_model_result(
    state: AgentTraceState,
    *,
    observation: RoundObservation,
    output: object,
    model_name: str | None = None,
) -> dict[str, object]:
    """记录一次模型调用的事实使用量、输出引用与上下文变化。"""

    usage, reported, cache_reported = provider_usage(output)
    current_refs = extract_structured_references(output)
    _mark_pending_tool_consumption(state, current_refs)
    entry = {
        "phase": observation.phase,
        "round": observation.round_index,
        "payload_estimated_tokens": observation.payload_estimated_tokens,
        "segments": [asdict(segment) for segment in observation.segments],
        "system_prompt_hash": observation.system_prompt_hash,
        "tool_schema_hash": observation.tool_schema_hash,
        "usage": usage,
        "output_reference_ids": sorted(current_refs),
    }
    state.rounds.append(entry)
    state.provider_usage_reported = state.provider_usage_reported or reported
    state.cache_usage_reported = state.cache_usage_reported or cache_reported
    metadata = getattr(output, "response_metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("provider_usage_recorded"):
        # 原生客户端逐 HTTP 尝试已记录 usage；这里仅保留回合/上下文盘点，避免重复累加。
        return entry
    state.total_prompt_tokens += int(usage["prompt_tokens"])
    state.total_completion_tokens += int(usage["completion_tokens"])
    state.total_tokens += int(usage["total_tokens"])
    state.total_reasoning_tokens += int(usage["reasoning_tokens"])
    state.total_cache_read_tokens += int(usage["cache_read_tokens"])
    state.total_cache_write_tokens += int(usage["cache_write_tokens"])
    if reported:
        # RuntimeCloudChatModel 在一个 Agent round 中恰好发起一笔真实请求；它已经把
        # usage 放到 AIMessage response_metadata。只补充调用目录，不能再次累加 usage。
        state.provider_calls.append(
            {
                "provider": "model_adapter",
                "model_name": model_name or "unknown",
                "operation_id": observation.phase,
                "attempts": 1,
                "usage": usage,
            }
        )
    return entry


def record_external_provider_call(
    state: AgentTraceState,
    *,
    provider: str,
    model_name: str,
    operation_id: str | None,
    usage: Mapping[object, object] | None,
    attempts: int,
) -> dict[str, object]:
    """汇总一次客户端级真实模型请求。

    Controller 的 ``rounds`` 描述 Agent 如何思考；一个 round 可能让结构化编译器发出
    多次 HTTP 请求（含一次 Schema 修复）。这里单独记录供应商调用与实际 usage，既不
    将其误当成新的 Agent round，也不会把它丢在 Controller 的黑盒里。
    """

    normalized_usage, reported, cache_reported = _usage_from_mapping(usage)
    entry = {
        "provider": provider,
        "model_name": model_name,
        "operation_id": operation_id or "",
        "attempts": attempts,
        "usage": normalized_usage,
    }
    state.provider_calls.append(entry)
    state.provider_usage_reported = state.provider_usage_reported or reported
    state.cache_usage_reported = state.cache_usage_reported or cache_reported
    state.total_prompt_tokens += int(normalized_usage["prompt_tokens"])
    state.total_completion_tokens += int(normalized_usage["completion_tokens"])
    state.total_tokens += int(normalized_usage["total_tokens"])
    state.total_reasoning_tokens += int(normalized_usage["reasoning_tokens"])
    state.total_cache_read_tokens += int(normalized_usage["cache_read_tokens"])
    state.total_cache_write_tokens += int(normalized_usage["cache_write_tokens"])
    return entry


def record_tool_result(
    state: AgentTraceState,
    *,
    tool_name: str,
    arguments: object,
    result: object,
    status: str,
    duration_ms: int,
    retry_count: int,
) -> dict[str, object]:
    """记录一次工具事实；重复调用只看规范化参数哈希，不读取聊天文本。"""

    references = extract_structured_references(result)
    call_key = canonical_hash({"tool_name": tool_name, "arguments": arguments})
    occurrence = state.exact_tool_call_counts.get(call_key, 0) + 1
    state.exact_tool_call_counts[call_key] = occurrence
    entry: dict[str, object] = {
        "tool_name": tool_name,
        "arguments_estimated_tokens": estimate_tokens(arguments),
        "result_estimated_tokens": estimate_tokens(result),
        "result_characters": len(canonical_json(result)),
        "duration_ms": duration_ms,
        "retry_count": retry_count,
        "status": status,
        "reference_ids": sorted(references),
        "exact_call_occurrence": occurrence,
        "consumed_in_next_model_turn": "unknown",
    }
    state.tools.append(entry)
    if status == "succeeded":
        state.tool_success_calls += 1
    elif status == "empty":
        state.tool_empty_calls += 1
    elif status == "cancelled":
        state.tool_cancelled_calls += 1
    else:
        state.tool_error_calls += 1
    state.pending_tool_results.append({"entry": entry, "reference_ids": references})
    return entry


def record_compaction(
    state: AgentTraceState,
    *,
    phase: str,
    before_messages: Sequence[object],
    after_messages: Sequence[object],
    duration_ms: int,
    summary: str,
) -> dict[str, object]:
    before = serialize_messages(before_messages)
    after = serialize_messages(after_messages)
    before_tokens = estimate_tokens(before)
    after_tokens = estimate_tokens(after)
    removed = _removed_message_summary(before, after)
    entry: dict[str, object] = {
        "phase": phase,
        "duration_ms": duration_ms,
        "tokens_before": before_tokens,
        "tokens_after": after_tokens,
        "tokens_saved": max(0, before_tokens - after_tokens),
        "summary": summary,
        "removed": removed,
        "before_reference_ids": sorted(extract_structured_references(before)),
        "after_reference_ids": sorted(extract_structured_references(after)),
    }
    state.compactions.append(entry)
    return entry


def build_root_cause_snapshot(state: AgentTraceState) -> dict[str, object]:
    """只根据结构化运行事实生成可筛选的诊断结论，不调用第二个模型。"""

    contexts = [int(item.get("payload_estimated_tokens") or 0) for item in state.rounds]
    total_tools = len(state.tools)
    unconsumed = [
        item
        for item in state.tools
        if item.get("consumed_in_next_model_turn") is False
        and int(item.get("result_estimated_tokens") or 0) > 0
    ]
    repeated = [item for item in state.tools if int(item.get("exact_call_occurrence") or 0) >= 3]
    compression_saved = sum(int(item.get("tokens_saved") or 0) for item in state.compactions)
    primary = _primary_root_cause(
        state=state,
        contexts=contexts,
        repeated=repeated,
        unconsumed=unconsumed,
        compression_saved=compression_saved,
    )
    prefix_hashes = [
        item.get("system_prompt_hash") for item in state.rounds if item.get("system_prompt_hash")
    ]
    schema_hashes = [
        item.get("tool_schema_hash") for item in state.rounds if item.get("tool_schema_hash")
    ]
    return {
        "primary_root_cause": primary,
        "execution": {
            "terminal_status": state.terminal_status,
            "terminal_reason": state.terminal_reason,
            "model_calls": len(state.rounds),
            "provider_model_calls": len(state.provider_calls),
            "tool_calls": total_tools,
            "tool_success_calls": state.tool_success_calls,
            "tool_error_calls": state.tool_error_calls,
        },
        "usage": {
            "provider_usage_reported": state.provider_usage_reported,
            "prompt_tokens": state.total_prompt_tokens,
            "completion_tokens": state.total_completion_tokens,
            "total_tokens": state.total_tokens,
            "reasoning_tokens": state.total_reasoning_tokens,
            "cache_support": (
                "reported" if state.cache_usage_reported else "not_reported_by_provider"
            ),
            "cache_read_tokens": state.total_cache_read_tokens,
            "cache_write_tokens": state.total_cache_write_tokens,
        },
        "context": {
            "round_estimated_tokens": contexts,
            "system_prompt_stable": len(set(prefix_hashes)) <= 1,
            "tool_schema_stable": len(set(schema_hashes)) <= 1,
            "compression_events": len(state.compactions),
            "compression_saved_tokens": compression_saved,
            "request_diagnostics": state.request_diagnostics,
        },
        "tool_diagnostics": {
            "exact_repeat_count": len(repeated),
            "structurally_unconsumed_count": len(unconsumed),
            "unknown_consumption_count": sum(
                item.get("consumed_in_next_model_turn") == "unknown" for item in state.tools
            ),
            "largest_results": sorted(
                state.tools,
                key=lambda item: int(item.get("result_estimated_tokens") or 0),
                reverse=True,
            )[:3],
        },
    }


def canonical_json(value: object) -> str:
    try:
        return json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return repr(value)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:16]


def estimate_tokens(value: object) -> int:
    """本地字节估算，仅用于分解上下文来源，绝不替代供应商 usage。"""

    encoded = canonical_json(value).encode("utf-8")
    return max(1, (len(encoded) + 3) // 4) if encoded else 0


def extract_structured_references(value: object) -> frozenset[str]:
    """从 JSON/模型结构中提取 provenance，不对普通自然语言作任何模式匹配。"""

    references: set[str] = set()

    def walk(item: object, *, depth: int = 0) -> None:
        if depth > 20:
            return
        if isinstance(item, BaseMessage):
            walk(serialize_message(item), depth=depth + 1)
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = str(key)
                if normalized in _REFERENCE_FIELDS:
                    _collect_reference_values(child, references)
                walk(child, depth=depth + 1)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                walk(child, depth=depth + 1)
            return
        if isinstance(item, str):
            # 仅接受完整 JSON 值；自然语言字符串绝不做 substring/正则推测。
            try:
                decoded = json.loads(item)
            except (TypeError, ValueError):
                return
            walk(decoded, depth=depth + 1)

    walk(value)
    return frozenset(references)


def provider_usage(output: object) -> tuple[dict[str, int], bool, bool]:
    metadata = getattr(output, "response_metadata", None)
    raw = metadata.get("usage") if isinstance(metadata, Mapping) else None
    return _usage_from_mapping(raw if isinstance(raw, Mapping) else None)


def _usage_from_mapping(
    raw: Mapping[object, object] | None,
) -> tuple[dict[str, int], bool, bool]:
    if raw is None:
        return _empty_usage(), False, False
    usage = {
        "prompt_tokens": _usage_int(raw, "prompt_tokens", "input_tokens"),
        "completion_tokens": _usage_int(raw, "completion_tokens", "output_tokens"),
        "total_tokens": _usage_int(raw, "total_tokens"),
        "reasoning_tokens": _usage_int(
            raw,
            "reasoning_tokens",
            "reasoning_content_tokens",
        ),
        "cache_read_tokens": _usage_int(
            raw,
            "cached_tokens",
            "cache_read_tokens",
            "cache_read_input_tokens",
        ),
        "cache_write_tokens": _usage_int(
            raw,
            "cache_write_tokens",
            "cache_creation_input_tokens",
        ),
    }
    if usage["total_tokens"] == 0:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    cache_keys = (
        "cached_tokens",
        "cache_read_tokens",
        "cache_read_input_tokens",
        "cache_write_tokens",
        "cache_creation_input_tokens",
    )
    prompt_details = raw.get("prompt_tokens_details")
    completion_details = raw.get("completion_tokens_details")
    cache_reported = any(key in raw for key in cache_keys) or any(
        isinstance(details, Mapping) and any(key in details for key in cache_keys)
        for details in (prompt_details, completion_details)
    )
    return usage, True, cache_reported


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }


def _usage_int(raw: Mapping[object, object], *keys: str) -> int:
    containers: tuple[Mapping[object, object], ...] = (
        raw,
        *tuple(
            value
            for value in (
                raw.get("prompt_tokens_details"),
                raw.get("completion_tokens_details"),
            )
            if isinstance(value, Mapping)
        ),
    )
    for container in containers:
        for key in keys:
            value = container.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return 0


def _mark_pending_tool_consumption(state: AgentTraceState, current_refs: frozenset[str]) -> None:
    pending = list(state.pending_tool_results)
    state.pending_tool_results.clear()
    for item in pending:
        entry = item["entry"]
        result_refs = item["reference_ids"]
        if not result_refs or not current_refs:
            entry["consumed_in_next_model_turn"] = "unknown"
        else:
            entry["consumed_in_next_model_turn"] = bool(result_refs & current_refs)


def _primary_root_cause(
    *,
    state: AgentTraceState,
    contexts: list[int],
    repeated: list[dict[str, object]],
    unconsumed: list[dict[str, object]],
    compression_saved: int,
) -> dict[str, object]:
    if state.terminal_reason == "invalid_final_output":
        return {
            "kind": "invalid_final_output",
            "severity": "high",
            "summary": "最终输出未通过验收，不代表供应商请求失败",
        }
    if state.terminal_status == "failed":
        return {
            "kind": "system_or_provider_failure",
            "severity": "high",
            "summary": state.terminal_reason or "Agent 执行失败",
        }
    if state.terminal_reason == "repeated_tool_cycle":
        return {
            "kind": "repeated_tool_cycle",
            "severity": "high",
            "summary": "已提示调整，但仍连续重复相同工具参数和结果；不是缺少新来源",
        }
    if state.terminal_reason in {"execution_attempt_limit", "collaboration_attempt_limit"}:
        return {
            "kind": "execution_attempt_limit",
            "severity": "medium",
            "summary": "同一任务已达到执行尝试上限，保留工作状态等待处理",
        }
    if repeated:
        return {
            "kind": "exact_tool_call_loop",
            "severity": "high",
            "summary": "同一工具以规范化后的相同参数至少执行了三次",
            "evidence": {"tools": [item.get("tool_name") for item in repeated]},
        }
    if state.terminal_reason == "no_new_semantic_progress":
        return {
            "kind": "no_new_structured_evidence",
            "severity": "medium",
            "summary": "连续工具批次没有产生新的结构化 provenance",
        }
    if state.tools and state.tool_error_calls == len(state.tools):
        return {
            "kind": "all_tools_failed",
            "severity": "high",
            "summary": "本次执行的工具调用全部失败",
        }
    if len(contexts) >= 3 and contexts[0] > 0 and contexts[-1] >= contexts[0] * 3:
        if compression_saved == 0:
            return {
                "kind": "context_bloat_without_effective_compaction",
                "severity": "high",
                "summary": "模型上下文在执行过程中增长至少三倍，且没有有效压缩",
                "evidence": {"first": contexts[0], "last": contexts[-1]},
            }
    if unconsumed:
        largest = max(unconsumed, key=lambda item: int(item.get("result_estimated_tokens") or 0))
        return {
            "kind": "structured_tool_result_not_consumed",
            "severity": "medium",
            "summary": "工具返回了可引用材料，但下一模型回合没有引用这些材料",
            "evidence": {
                "tool_name": largest.get("tool_name"),
                "result_estimated_tokens": largest.get("result_estimated_tokens"),
            },
        }
    return {"kind": "none", "severity": "low", "summary": "没有命中确定性运行故障规则"}


def _mapping_jsonable(value: Mapping[str, object]) -> dict[str, object]:
    rendered = _jsonable(dict(value))
    return rendered if isinstance(rendered, dict) else {"value": rendered}


def _jsonable(value: object) -> object:
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _collect_reference_values(value: object, output: set[str]) -> None:
    if isinstance(value, str) and value:
        output.add(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output.add(str(value))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _collect_reference_values(item, output)


def _tool_calls_from_snapshot(message: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _removed_message_summary(
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    after_hashes = {canonical_hash(item) for item in after}
    removed = [
        {
            "type": item.get("type"),
            "name": item.get("name"),
            "estimated_tokens": estimate_tokens(item),
            "reference_ids": sorted(extract_structured_references(item)),
        }
        for item in before
        if canonical_hash(item) not in after_hashes
    ]
    return sorted(removed, key=lambda item: int(item["estimated_tokens"]), reverse=True)[:20]
