"""统一 Agent Loop 的 Phoenix 观察器。

Controller 只报告模型、工具与压缩事实；本类负责把事实写成 Span、快照和根摘要，避免各
领域 Agent 手写不同的 Trace 语义。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from time import monotonic
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from .execution_context import AgentExecutionTraceContext
from .phoenix import (
    add_context_snapshot,
    annotate_trace_async,
    chain_span,
    compression_span,
    record_span_attributes,
    record_span_error,
    record_span_output,
    tool_span,
)
from .trace_summary import (
    RoundObservation,
    build_root_cause_snapshot,
    make_round_observation,
    record_compaction,
    record_model_result,
    record_tool_result,
    serialize_messages,
)

if TYPE_CHECKING:
    from moonlightbox.agent_runtime.contracts import AgentSpec, ToolContract, ToolExecution


class AgentRunObserver:
    """一次 ``AgentLoopController.run`` 专属的无持久化观察器。"""

    def __init__(
        self,
        *,
        trace_context: AgentExecutionTraceContext,
        spec: AgentSpec[Any],
    ) -> None:
        self._context = trace_context
        self._spec = spec
        self._last_model_end = None
        self._model_call_index = 0

    @property
    def trace_context(self) -> AgentExecutionTraceContext:
        return self._context

    def record_run_input(self, *, root_span: Any | None, messages: Sequence[object]) -> None:
        """根 Span 保留原始启动上下文，便于从 Phoenix 直接回放一次执行。"""

        add_context_snapshot(
            root_span,
            event_name="run_input",
            value=serialize_messages(messages),
        )

    @contextmanager
    def model_call(
        self,
        *,
        phase: str,
        model_name: str,
        messages: Sequence[object],
        visible_tools: Sequence[BaseTool],
    ) -> Iterator[tuple[Any | None, RoundObservation]]:
        observation = make_round_observation(
            phase=phase,
            round_index=len(self._context.state.rounds),
            messages=messages,
            tool_schemas=_tool_schemas(visible_tools),
        )
        attrs: dict[str, object] = {
            "moonlightbox.llm.call_index": self._model_call_index,
            "moonlightbox.llm.model_name": model_name,
            "moonlightbox.llm.gap_since_previous_ms": (
                int((monotonic() - self._last_model_end) * 1000)
                if self._last_model_end is not None
                else 0
            ),
            "moonlightbox.agent.phase": phase,
            "moonlightbox.agent.round": observation.round_index,
            "moonlightbox.context.estimated_tokens": observation.payload_estimated_tokens,
            "moonlightbox.context.segment_count": len(observation.segments),
            "moonlightbox.llm.system_prompt_hash": observation.system_prompt_hash or "",
            "moonlightbox.llm.tool_schema_hash": observation.tool_schema_hash or "",
            "moonlightbox.llm.tool_schema_count": len(observation.tool_schemas),
        }
        self._model_call_index += 1
        with chain_span(
            f"moonlightbox.agent.decision.{phase}",
            input_value=observation.messages_snapshot,
            attributes=attrs,
        ) as span:
            add_context_snapshot(
                span,
                event_name="model_input",
                value={
                    "messages": observation.messages_snapshot,
                    "tool_schemas": observation.tool_schemas,
                    "context_segments": [asdict(segment) for segment in observation.segments],
                },
            )
            try:
                yield span, observation
            except Exception as error:
                record_span_output(span, {"status": "error", "error_type": type(error).__name__})
                raise
            finally:
                self._last_model_end = monotonic()

    def record_model_result(
        self,
        *,
        span: Any | None,
        observation: RoundObservation,
        output: object,
        model_name: str,
    ) -> None:
        record_span_output(span, output)
        entry = record_model_result(
            self._context.state,
            observation=observation,
            output=output,
            model_name=model_name,
        )
        record_span_attributes(
            span,
            {
                "moonlightbox.llm.provider_usage_reported": (
                    self._context.state.provider_usage_reported
                ),
                "moonlightbox.llm.cache_usage_reported": self._context.state.cache_usage_reported,
                "moonlightbox.llm.payload_estimated_tokens": observation.payload_estimated_tokens,
                "moonlightbox.llm.context_inventory": entry["segments"],
                "moonlightbox.llm.cache_support": (
                    "reported"
                    if self._context.state.cache_usage_reported
                    else "not_reported_by_provider"
                ),
            },
        )
        add_context_snapshot(span, event_name="model_output", value=output)

    def record_model_failure(self, *, span: Any | None, error: Exception) -> None:
        record_span_output(
            span,
            {"status": "error", "error_code": getattr(error, "code", type(error).__name__)},
        )
        record_span_attributes(
            span,
            {
                "moonlightbox.llm.error_type": type(error).__name__,
                "moonlightbox.llm.error_code": getattr(error, "code", ""),
                "moonlightbox.llm.error_diagnostic": getattr(error, "diagnostic", {}),
            },
        )
        record_span_error(span, error)

    @contextmanager
    def tool_call(
        self,
        *,
        tool_name: str,
        arguments: object,
        contract: ToolContract | None,
    ) -> Iterator[Any | None]:
        with tool_span(
            f"moonlightbox.agent.tool.{tool_name}",
            tool_name=tool_name,
            arguments=arguments,
            contract=contract,
        ) as span:
            add_context_snapshot(span, event_name="tool_arguments", value=arguments)
            yield span

    def record_tool_result(self, *, span: Any | None, execution: ToolExecution) -> None:
        record_span_output(
            span,
            {
                "status": execution.status,
                "retry_count": execution.retry_count,
                "duration_ms": execution.duration_ms,
                "result": execution.result,
                "progress": {
                    "new_keys": sorted(execution.progress.new_keys),
                    "consumed_keys": sorted(execution.progress.consumed_keys),
                    "is_negative_evidence": execution.progress.is_negative_evidence,
                },
            },
        )
        entry = record_tool_result(
            self._context.state,
            tool_name=execution.tool_name,
            arguments=execution.args,
            result=execution.result,
            status=execution.status,
            duration_ms=execution.duration_ms,
            retry_count=execution.retry_count,
        )
        record_span_attributes(
            span,
            {
                "moonlightbox.tool.status": execution.status,
                "moonlightbox.tool.duration_ms": execution.duration_ms,
                "moonlightbox.tool.retry_count": execution.retry_count,
                "moonlightbox.tool.result_estimated_tokens": entry["result_estimated_tokens"],
                "moonlightbox.tool.result_reference_ids": entry["reference_ids"],
                "moonlightbox.tool.exact_call_occurrence": entry["exact_call_occurrence"],
            },
        )
        if execution.status == "error":
            record_span_error(
                span,
                RuntimeError(f"tool_execution_failed:{execution.tool_name}"),
            )
        add_context_snapshot(span, event_name="tool_result", value=execution.result)

    def observe_compaction(
        self,
        *,
        before_messages: Sequence[object],
        after_messages: Sequence[object],
        summary: str,
        duration_ms: int,
    ) -> None:
        with compression_span("moonlightbox.agent.context_compaction") as span:
            entry = record_compaction(
                self._context.state,
                phase="context_compaction",
                before_messages=before_messages,
                after_messages=after_messages,
                duration_ms=duration_ms,
                summary=summary,
            )
            record_span_attributes(
                span,
                {
                    "moonlightbox.compression.tokens_before": entry["tokens_before"],
                    "moonlightbox.compression.tokens_after": entry["tokens_after"],
                    "moonlightbox.compression.tokens_saved": entry["tokens_saved"],
                    "moonlightbox.compression.removed": entry["removed"],
                },
            )
            add_context_snapshot(
                span,
                event_name="compression_before",
                value=serialize_messages(before_messages),
            )
            add_context_snapshot(
                span,
                event_name="compression_after",
                value=serialize_messages(after_messages),
            )

    def finalize(
        self,
        *,
        root_span: Any | None,
        status: str,
        reason: str,
        result: object | None,
    ) -> dict[str, object]:
        self._context.state.terminal_status = status
        self._context.state.terminal_reason = reason
        snapshot = build_root_cause_snapshot(self._context.state)
        duration_ms = int((monotonic() - self._context.started_monotonic) * 1000)
        record_span_attributes(
            root_span,
            {
                "moonlightbox.agent.duration_ms": duration_ms,
                "moonlightbox.agent.model_calls": len(self._context.state.rounds),
                "moonlightbox.agent.provider_model_calls": len(self._context.state.provider_calls),
                "moonlightbox.agent.tool_calls": len(self._context.state.tools),
                "moonlightbox.agent.tool_success_calls": self._context.state.tool_success_calls,
                "moonlightbox.agent.tool_error_calls": self._context.state.tool_error_calls,
                "moonlightbox.agent.total_prompt_tokens": self._context.state.total_prompt_tokens,
                "moonlightbox.agent.total_completion_tokens": (
                    self._context.state.total_completion_tokens
                ),
                "moonlightbox.agent.total_tokens": self._context.state.total_tokens,
                "moonlightbox.agent.total_reasoning_tokens": (
                    self._context.state.total_reasoning_tokens
                ),
                "moonlightbox.agent.total_cache_read_tokens": (
                    self._context.state.total_cache_read_tokens
                ),
                "moonlightbox.agent.total_cache_write_tokens": (
                    self._context.state.total_cache_write_tokens
                ),
                "moonlightbox.agent.root_cause": snapshot,
                "moonlightbox.agent.root_cause_kind": (
                    snapshot["primary_root_cause"].get("kind")
                    if isinstance(snapshot.get("primary_root_cause"), Mapping)
                    else "unknown"
                ),
            },
        )
        add_context_snapshot(root_span, event_name="final_result", value=result)
        annotate_trace_async(root_span, snapshot)
        return snapshot


def _tool_schemas(tools: Sequence[BaseTool]) -> list[dict[str, object]]:
    schemas: list[dict[str, object]] = []
    for tool in tools:
        try:
            converted = convert_to_openai_tool(tool, strict=True)
        except (TypeError, ValueError):
            converted = convert_to_openai_tool(tool)
        if isinstance(converted, Mapping):
            schemas.append({str(key): value for key, value in converted.items()})
    return schemas
