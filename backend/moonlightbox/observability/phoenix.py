"""MoonlightBox 的 Phoenix / OpenInference 统一埋点。

这里刻意只处理观测：它不会决定 Agent 是否继续，也不会把业务状态交给 Phoenix。这样即
使 Collector 不可用，Agent 仍可按领域业务协议运行。

自动埋点覆盖 LangChain/LangGraph 的 Runnable 细节；统一 Harness 额外创建 Agent、模型
与工具边界 Span，保证自定义模型适配器同样能在 Phoenix 中形成一棵可筛选的调用树。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanLimits
from opentelemetry.trace import Link, Status, StatusCode
from phoenix.otel import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    register,
    using_metadata,
    using_session,
    using_tags,
)

from .execution_context import current_agent_execution_trace_context

if TYPE_CHECKING:
    from moonlightbox.agent_runtime.contracts import (
        AgentExecutionRequest,
        AgentSpec,
        ToolContract,
    )
    from moonlightbox.config import Settings


LOGGER = logging.getLogger(__name__)
_LOCK = Lock()
_API_KEY_PATTERN = re.compile(r"(?i)(?:bearer\s+|sk-)[A-Za-z0-9_\-]{12,}")
_AttributeValue = (
    str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
)


@dataclass(frozen=True, slots=True)
class _PhoenixState:
    enabled: bool
    provider: Any | None
    capture_content: bool
    max_characters: int
    project_name: str
    collector_endpoint: str
    annotations_enabled: bool


_STATE = _PhoenixState(
    enabled=False,
    provider=None,
    capture_content=False,
    max_characters=0,
    project_name="moonlightbox",
    collector_endpoint="",
    annotations_enabled=False,
)

_ANNOTATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="moonlightbox-phoenix-annotations",
)


def initialize_phoenix(settings: Settings, *, service_name: str) -> None:
    """在 API / Worker 进程启动时恰好配置一次 Phoenix。

    使用 OTLP/HTTP，便于本机开发和 Compose 服务使用同一个 ``/v1/traces`` Collector。
    Collector 连接失败只会影响 Trace 导出，绝不能阻断聊天、任务恢复或图谱审核。
    """

    global _STATE
    if not settings.phoenix_enabled:
        return
    with _LOCK:
        if _STATE.enabled:
            if _STATE.project_name != settings.phoenix_project_name:
                LOGGER.warning(
                    "Phoenix 已在本进程初始化，忽略不同 project_name=%s",
                    settings.phoenix_project_name,
                )
            return
        try:
            provider = register(
                project_name=settings.phoenix_project_name,
                endpoint=settings.phoenix_collector_endpoint,
                protocol="http/protobuf",
                batch=True,
                auto_instrument=False,
                span_limits=SpanLimits(
                    max_attributes=512, max_events=2048, max_attribute_length=65536
                ),
            )
            # 自动 Instrumentor 负责 LangChain / LangGraph 图节点；手工 Span 仍是
            # 自定义模型和统一 Harness 的稳定根。TraceConfig 同时约束自动 Span，避免
            # capture_content=false 时自动路径意外导出聊天正文。
            LangChainInstrumentor().instrument(
                tracer_provider=provider,
                config=TraceConfig(
                    hide_inputs=not settings.phoenix_capture_content,
                    hide_outputs=not settings.phoenix_capture_content,
                    hide_input_messages=not settings.phoenix_capture_content,
                    hide_output_messages=not settings.phoenix_capture_content,
                    hide_input_text=not settings.phoenix_capture_content,
                    hide_output_text=not settings.phoenix_capture_content,
                    hide_prompts=not settings.phoenix_capture_content,
                    hide_llm_tools=not settings.phoenix_capture_content,
                ),
            )
        except Exception:
            # 不把可选观测依赖升级为生产路径的单点故障。日志不包含 endpoint 中的
            # 认证信息（Settings 已拒绝 URL credentials）。
            LOGGER.exception("Phoenix 初始化失败；本进程继续以无外部 Trace 模式运行")
            return
        _STATE = _PhoenixState(
            enabled=True,
            provider=provider,
            capture_content=settings.phoenix_capture_content,
            max_characters=settings.phoenix_trace_max_characters,
            project_name=settings.phoenix_project_name,
            collector_endpoint=settings.phoenix_collector_endpoint,
            annotations_enabled=settings.phoenix_annotations_enabled,
        )
        LOGGER.info(
            "Phoenix 已启用 service=%s project=%s content=%s",
            service_name,
            settings.phoenix_project_name,
            settings.phoenix_capture_content,
        )


def shutdown_phoenix() -> None:
    """在进程退出前尽力 flush；导出失败不影响应用正常关闭。"""

    provider = _STATE.provider
    if provider is None:
        return
    try:
        provider.force_flush(timeout_millis=3_000)
    except Exception:
        LOGGER.warning("Phoenix Trace flush 失败", exc_info=True)


@contextmanager
def agent_execution_span(
    spec: AgentSpec[Any], request: AgentExecutionRequest
) -> Iterator[Any | None]:
    """为一次瞬时 ``AgentLoopController.run`` 创建 Phoenix 根 Agent Span。"""

    trace_context = current_agent_execution_trace_context()
    metadata: dict[str, object] = {
        "moonlightbox.agent.name": spec.name,
        "moonlightbox.agent.prompt_version": spec.prompt_version,
        "moonlightbox.owner.type": request.owner_type,
        "moonlightbox.owner.id": request.owner_id,
        "moonlightbox.project.id": request.project_id or "",
        "moonlightbox.branch.id": request.scope.branch_id or "",
        "moonlightbox.target_person.id": request.scope.target_person_id or "",
        "moonlightbox.input_revision": request.input_revision,
    }
    if trace_context is not None:
        metadata["moonlightbox.agent.execution_id"] = trace_context.execution_id
    with _span(
        "moonlightbox.agent.run",
        kind=OpenInferenceSpanKindValues.AGENT,
        input_value={"messages": list(request.messages)},
        attributes=metadata,
        session_id=(
            trace_context.session_id
            if trace_context is not None
            else f"{request.owner_type}:{request.owner_id}"
        ),
        tags=("moonlightbox", request.owner_type, spec.name),
        detach_ambient_parent=True,
    ) as span:
        yield span


def _best_effort(callback):
    """观测写入失败不能改变模型结果、工具结果或领域事务。"""

    @wraps(callback)
    def wrapped(*args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception:
            LOGGER.warning("Phoenix %s 写入失败", callback.__name__, exc_info=True)
            return None

    return wrapped


@_best_effort
def record_agent_outcome(
    span: Any | None,
    *,
    status: str,
    reason: str,
    execution_id: str,
    result: object | None,
) -> None:
    """把本次执行的终态回写到根 Span，便于在 Phoenix 直接筛选失败原因。"""

    if span is None:
        return
    record_span_attributes(
        span,
        {
            "moonlightbox.agent.execution_id": execution_id,
            "moonlightbox.agent.status": status,
            "moonlightbox.agent.terminal_reason": reason,
            "moonlightbox.agent.output_valid": result is not None
            and reason in {"success", "decision"},
            "moonlightbox.agent.incomplete": status != "succeeded"
            or agent_outcome_is_error(status, reason),
        },
    )
    if result is not None:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _trace_value(result))
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)
    # 证据不足、等待用户、输入过期和用户取消均是预期领域终态。它们必须能在
    # Phoenix 筛选，但不能被误报成系统故障；只有未恢复的执行失败进入 ERROR。
    try:
        if agent_outcome_is_error(status, reason):
            span.set_status(Status(StatusCode.ERROR, f"{status}:{reason}"))
        else:
            span.set_status(Status(StatusCode.OK))
    except Exception:
        LOGGER.warning("Phoenix 终态写入失败", exc_info=True)


@contextmanager
def llm_span(
    name: str,
    *,
    model_name: str,
    input_value: object,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[Any | None]:
    """记录项目自定义 HTTP/结构化模型调用，避免被 LangChain 自动埋点遗漏。"""

    with _span(
        name,
        kind=OpenInferenceSpanKindValues.LLM,
        input_value=input_value,
        attributes={"llm.model_name": model_name, **dict(attributes or {})},
    ) as span:
        yield span


@contextmanager
def chain_span(
    name: str,
    *,
    input_value: object | None = None,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[Any | None]:
    """记录不等同于一次供应商请求的 Agent 决策阶段。

    例如 PersonWorld 的一轮决策可能在内部发起多次结构化模型请求。这里保留该轮的
    上下文盘点，而每一次真实 HTTP 模型请求由调用客户端另建 LLM Span，避免把一轮
    调度错误地当成一笔模型账单。
    """

    with _span(
        name,
        kind=OpenInferenceSpanKindValues.CHAIN,
        input_value=input_value,
        attributes=attributes,
    ) as span:
        yield span


@contextmanager
def retriever_span(
    name: str,
    *,
    input_value: object,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[Any | None]:
    """记录 LightRAG 等检索后端的一次真实读取请求。"""

    with _span(
        name,
        kind=OpenInferenceSpanKindValues.RETRIEVER,
        input_value=input_value,
        attributes=attributes,
    ) as span:
        yield span


@contextmanager
def tool_span(
    name: str,
    *,
    tool_name: str,
    arguments: object,
    contract: ToolContract | None = None,
) -> Iterator[Any | None]:
    """记录每一个实际执行的 LangChain Tool，而不是仅记录“计划过工具”。"""

    attributes: dict[str, object] = {"tool.name": tool_name}
    if contract is not None:
        attributes.update(
            {
                "moonlightbox.tool.side_effect": contract.side_effect,
                "moonlightbox.tool.timeout_seconds": contract.timeout_seconds,
                "moonlightbox.tool.cancellation": contract.execution.cancellation,
                "moonlightbox.tool.timeout_enforcement": contract.execution.timeout_enforcement,
                "moonlightbox.tool.parallelism": contract.execution.parallelism,
                "moonlightbox.tool.exclusive_resources": list(
                    contract.execution.exclusive_resources
                ),
                "moonlightbox.tool.execution_policy_reason": contract.execution.reason,
                # 能力声明不等于已经开启并行调度。
                "moonlightbox.tool.dispatch": "serial",
            }
        )
    with _span(
        name,
        kind=OpenInferenceSpanKindValues.TOOL,
        input_value=arguments,
        attributes=attributes,
        classification="project_private",
    ) as span:
        yield span


@contextmanager
def compression_span(name: str) -> Iterator[Any | None]:
    """上下文压缩是独立的 CHAIN Span，便于与模型调用耗时及收益对照。"""

    with _span(name, kind=OpenInferenceSpanKindValues.CHAIN) as span:
        yield span


def record_span_attributes(span: Any | None, attributes: Mapping[str, object]) -> None:
    """以 fail-open 方式写结构化诊断属性，不让观测故障影响 Agent。"""

    if span is None:
        return
    try:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, _attribute_value(value))
    except Exception:
        LOGGER.warning("Phoenix Span 属性写入失败", exc_info=True)


def record_span_error(span: Any | None, error: Exception) -> None:
    """在调用方已接住异常时，仍把子 Span 明确标为 ERROR。"""

    if span is None:
        return
    try:
        record_span_attributes(
            span,
            {
                "moonlightbox.error.code": str(getattr(error, "code", "") or type(error).__name__),
            },
        )
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, type(error).__name__))
    except Exception:
        LOGGER.warning("Phoenix Span 错误状态写入失败", exc_info=True)


def add_context_snapshot(span: Any | None, *, event_name: str, value: object) -> None:
    """将完整输入、输出或压缩快照分块附着为 Phoenix Span event。

    ``input.value`` / ``output.value`` 仍保留用于 UI 快速查看，但其长度受通用属性
    限制；快照事件以 60K 字符分块，因此本机 Phoenix 可以复盘完整内容，而非只看到
    hash 或截断前缀。
    """

    if span is None or not _STATE.enabled or not _STATE.capture_content:
        return
    try:
        text = _API_KEY_PATTERN.sub("[REDACTED_SECRET]", _serialize(value))
        chunk_size = 60_000
        chunks = [
            text[index : index + chunk_size] for index in range(0, len(text), chunk_size)
        ] or [""]
        snapshot_id = sha256(text.encode("utf-8")).hexdigest()[:16]
        for index, chunk in enumerate(chunks):
            span.add_event(
                f"moonlightbox.snapshot.{event_name}",
                attributes={
                    "moonlightbox.snapshot.id": snapshot_id,
                    "moonlightbox.snapshot.chunk_index": index,
                    "moonlightbox.snapshot.chunk_count": len(chunks),
                    "moonlightbox.snapshot.content": chunk,
                },
            )
    except Exception:
        LOGGER.warning("Phoenix 上下文快照写入失败", exc_info=True)


def annotate_trace_async(span: Any | None, snapshot: Mapping[str, object]) -> None:
    """把确定性根因写为 Phoenix CODE annotation，不阻塞 Agent 主流程。"""

    if span is None or not _STATE.enabled or not _STATE.annotations_enabled:
        return
    try:
        context = span.get_span_context()
        trace_id = f"{int(context.trace_id):032x}"
        if not trace_id.strip("0"):
            return
        primary = snapshot.get("primary_root_cause")
        if not isinstance(primary, Mapping):
            return
        annotation = {
            "trace_id": trace_id,
            "name": "moonlightbox_agent_root_cause",
            "annotator_kind": "CODE",
            "result": {
                "label": str(primary.get("kind") or "unknown"),
                "score": {"high": 1.0, "medium": 0.5, "low": 0.0}.get(
                    str(primary.get("severity") or "low"), 0.0
                ),
                "explanation": str(primary.get("summary") or ""),
            },
            "metadata": dict(snapshot),
            "identifier": "moonlightbox-agent-root-cause",
        }
        _ANNOTATION_EXECUTOR.submit(_post_annotation, annotation)
    except Exception:
        LOGGER.warning("Phoenix Annotation 排程失败", exc_info=True)


def _post_annotation(annotation: Mapping[str, object]) -> None:
    try:
        base_url = _STATE.collector_endpoint.removesuffix("/v1/traces")
        if not base_url:
            return
        with httpx.Client(timeout=1.0) as client:
            response = client.post(
                f"{base_url}/v1/trace_annotations",
                params={"sync": "false"},
                json={"data": [dict(annotation)]},
            )
            response.raise_for_status()
    except Exception:
        # Annotation 是增强信息；Collector 尚未启动或版本不支持都不能影响主执行。
        LOGGER.warning("Phoenix Annotation 写入失败", exc_info=True)


@_best_effort
def record_span_output(span: Any | None, value: object) -> None:
    """写入受隐私开关与字符上限约束的输出，而不是直接序列化任意对象。"""

    if span is None:
        return
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, _trace_value(value))
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)
    # RuntimeCloudChatModel 把供应商 usage 放在 LangChain response_metadata；映射为
    # OpenInference 标准字段后，Phoenix 可直接按 token 与延迟比较不同 Agent/模型。
    response_metadata = getattr(value, "response_metadata", None)
    usage = response_metadata.get("usage") if isinstance(response_metadata, Mapping) else None
    if not isinstance(usage, Mapping):
        return
    _set_token_attribute(span, "llm.token_count.prompt", usage, "prompt_tokens", "input_tokens")
    _set_token_attribute(
        span,
        "llm.token_count.completion",
        usage,
        "completion_tokens",
        "output_tokens",
    )
    _set_token_attribute(span, "llm.token_count.total", usage, "total_tokens")
    _set_token_attribute(
        span,
        "llm.token_count.reasoning",
        usage,
        "reasoning_tokens",
        "reasoning_content_tokens",
    )
    _set_token_attribute(
        span,
        "llm.token_count.cache_read",
        usage,
        "cached_tokens",
        "cache_read_tokens",
        "cache_read_input_tokens",
    )
    _set_token_attribute(
        span,
        "llm.token_count.cache_write",
        usage,
        "cache_write_tokens",
        "cache_creation_input_tokens",
    )


@contextmanager
def _span(
    name: str,
    *,
    kind: OpenInferenceSpanKindValues,
    input_value: object | None = None,
    attributes: Mapping[str, object] | None = None,
    classification: str = "project_private",
    session_id: str | None = None,
    tags: tuple[str, ...] = (),
    detach_ambient_parent: bool = False,
) -> Iterator[Any | None]:
    """统一类型与关联；观测初始化失败放行，业务异常记录后原样抛出。"""
    if not _STATE.enabled:
        yield None
        return
    stack = ExitStack()
    span = None
    try:
        from .runtime import current_cycle_attributes

        cycle = current_cycle_attributes()
        span_attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: kind.value,
            **{key: _attribute_value(value) for key, value in cycle.items()},
            **{key: _attribute_value(value) for key, value in (attributes or {}).items()},
        }
        if input_value is not None:
            span_attributes[SpanAttributes.INPUT_VALUE] = _trace_value(
                input_value, classification=classification
            )
            span_attributes[SpanAttributes.INPUT_MIME_TYPE] = "application/json"
        trace_context = current_agent_execution_trace_context()
        if trace_context is not None:
            span_attributes.update(
                {
                    "moonlightbox.agent.execution_id": trace_context.execution_id,
                    "moonlightbox.agent.name": trace_context.agent_name,
                    "moonlightbox.agent.prompt_version": trace_context.prompt_version,
                    "moonlightbox.owner.type": trace_context.owner_type,
                    "moonlightbox.owner.id": trace_context.owner_id,
                    "moonlightbox.project.id": trace_context.project_id or "",
                    "moonlightbox.branch.id": trace_context.branch_id or "",
                    "moonlightbox.input_revision": trace_context.input_revision,
                }
            )
            session_id = session_id or trace_context.session_id
        if cycle:
            session_id = f"branch:{cycle['moonlightbox.branch.id']}"
        links = ()
        if detach_ambient_parent:
            parent = trace.get_current_span().get_span_context()
            if parent.is_valid:
                links = (Link(parent),)
                span_attributes["moonlightbox.agent.parent_trace_id"] = f"{parent.trace_id:032x}"
                span_attributes["moonlightbox.agent.parent_span_id"] = f"{parent.span_id:016x}"
            token = otel_context.attach(otel_context.Context())
            stack.callback(otel_context.detach, token)
        if session_id:
            stack.enter_context(using_session(session_id=session_id))
        if tags:
            stack.enter_context(using_tags(tags=list(tags)))
        stack.enter_context(using_metadata({"moonlightbox.project": _STATE.project_name}))
        span = stack.enter_context(
            trace.get_tracer("moonlightbox.agent_runtime").start_as_current_span(
                name,
                attributes=span_attributes,
                links=links,
                record_exception=False,
                set_status_on_exception=False,
            )
        )
    except Exception:
        LOGGER.warning("Phoenix Span 初始化失败，继续业务执行", exc_info=True)
        try:
            stack.close()
        except Exception:
            LOGGER.warning("Phoenix 清理失败", exc_info=True)
        span = None
    try:
        yield span
    except BaseException as error:
        code = str(getattr(error, "code", "") or error)
        if (
            code in {"stale_input_revision", "cancelled", "user_cancelled"}
            or type(error).__name__ == "CancelledError"
        ):
            record_span_attributes(
                span,
                {
                    "moonlightbox.outcome": "superseded_or_cancelled",
                    "moonlightbox.exit_reason": code,
                },
            )
            if span is not None:
                try:
                    span.set_status(Status(StatusCode.OK))
                except Exception:
                    pass
        else:
            record_span_error(span, error)
        raise
    else:
        try:
            if (
                span is not None
                and getattr(getattr(span, "status", None), "status_code", None) == StatusCode.UNSET
            ):
                span.set_status(Status(StatusCode.OK))
        except Exception:
            LOGGER.warning("Phoenix 成功状态写入失败", exc_info=True)
    finally:
        try:
            stack.close()
        except Exception:
            LOGGER.warning("Phoenix Span 结束失败，不影响业务结果", exc_info=True)


def agent_outcome_is_error(status: str, reason: str) -> bool:
    """业务等待不是故障；即使旧检查点误记成功，无效最终输出也不能显示成功。"""
    return status == "failed" or reason in {
        "invalid_final_output",
        "model_error",
        "tool_result_context_limit",
        "context_limit",
        "deadline_exceeded",
    }


@contextmanager
def _null_context() -> Iterator[None]:
    yield None


def _trace_value(value: object, *, classification: str = "project_private") -> str:
    serialized = _serialize(value)
    if not _STATE.capture_content or classification == "sensitive":
        return json.dumps(
            {
                "captured": False,
                "classification": classification,
                "sha256": sha256(serialized.encode("utf-8")).hexdigest(),
                "character_count": len(serialized),
            },
            ensure_ascii=False,
        )
    redacted = _API_KEY_PATTERN.sub("[REDACTED_SECRET]", serialized)
    if len(redacted) <= _STATE.max_characters:
        return redacted
    return json.dumps(
        {
            "truncated": True,
            "character_count": len(redacted),
            "sha256": sha256(redacted.encode("utf-8")).hexdigest(),
            "preview": redacted[: _STATE.max_characters],
        },
        ensure_ascii=False,
    )


def _serialize(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=_json_default, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _json_default(value: object) -> object:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except (TypeError, ValueError):
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "content"):
        return {
            "type": getattr(value, "type", type(value).__name__),
            "content": getattr(value, "content", ""),
            "tool_calls": getattr(value, "tool_calls", []),
        }
    return str(value)


def _attribute_value(value: object) -> _AttributeValue:
    if isinstance(value, (str, bool, int, float)):
        return value
    return _serialize(value)


def _set_token_attribute(
    span: Any,
    attribute: str,
    usage: Mapping[object, object],
    *keys: str,
) -> None:
    containers: tuple[Mapping[object, object], ...] = (
        usage,
        *tuple(
            value
            for value in (
                usage.get("prompt_tokens_details"),
                usage.get("completion_tokens_details"),
            )
            if isinstance(value, Mapping)
        ),
    )
    for container in containers:
        for key in keys:
            value = container.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                span.set_attribute(attribute, value)
                return
