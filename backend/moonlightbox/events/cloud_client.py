import json
import logging
import math
import random
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC
from email.utils import parsedate_to_datetime
from enum import StrEnum
from types import TracebackType
from typing import Any, Literal, Self, TypeVar

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from moonlightbox.agent_runtime.resilience import managed, model_request, request_timeout
from moonlightbox.observability import (
    add_context_snapshot,
    current_agent_execution_trace_context,
    llm_span,
    record_span_attributes,
    record_span_output,
)
from moonlightbox.observability.trace_summary import record_external_provider_call

logger = logging.getLogger(__name__)


class NodeAnalysisCloudErrorCode(StrEnum):
    DISABLED = "disabled"
    MISSING_API_KEY = "missing_api_key"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    TIMEOUT = "timeout"
    NETWORK = "network"
    INVALID_RESPONSE = "invalid_response"


_ERROR_MESSAGES = {
    NodeAnalysisCloudErrorCode.DISABLED: "节点分析云端客户端未启用",
    NodeAnalysisCloudErrorCode.MISSING_API_KEY: "节点分析云端客户端缺少 API key",
    NodeAnalysisCloudErrorCode.AUTHENTICATION: "节点分析云端服务认证失败",
    NodeAnalysisCloudErrorCode.RATE_LIMIT: "节点分析云端服务请求受限",
    NodeAnalysisCloudErrorCode.SERVER: "节点分析云端服务暂时不可用",
    NodeAnalysisCloudErrorCode.TIMEOUT: "节点分析云端服务请求超时",
    NodeAnalysisCloudErrorCode.NETWORK: "节点分析云端服务网络错误",
    NodeAnalysisCloudErrorCode.INVALID_RESPONSE: "节点分析云端服务响应无效",
}

_RETRYABLE_CODES = {
    NodeAnalysisCloudErrorCode.RATE_LIMIT,
    NodeAnalysisCloudErrorCode.SERVER,
    NodeAnalysisCloudErrorCode.TIMEOUT,
    NodeAnalysisCloudErrorCode.NETWORK,
}


class NodeAnalysisCloudError(RuntimeError):
    def __init__(
        self,
        code: NodeAnalysisCloudErrorCode,
        *,
        attempts: int,
        context: Mapping[str, str] | None = None,
        diagnostic: Mapping[str, str | int] | None = None,
    ) -> None:
        self.code = code
        self.attempts = attempts
        self.retryable = code in _RETRYABLE_CODES
        self.context = dict(context or {})
        self.diagnostic = dict(diagnostic or {})
        super().__init__(_ERROR_MESSAGES[code])


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
_STABLE_STRICT_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "description",
    "enum",
    "items",
    "properties",
    "required",
    "title",
    "type",
}
_UNREPRESENTABLE_SCHEMA_KEYS = {
    "allOf",
    "contains",
    "dependentSchemas",
    "else",
    "if",
    "not",
    "oneOf",
    "patternProperties",
    "prefixItems",
    "propertyNames",
    "then",
    "unevaluatedProperties",
}
_SCHEMA_MAP_KEYWORDS = {"$defs", "properties"}
_ALL_SCHEMA_MAP_KEYWORDS = {
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
}
_ALL_SCHEMA_NODE_KEYWORDS = {
    "contains",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
}
_ALL_SCHEMA_LIST_KEYWORDS = {"allOf", "anyOf", "oneOf", "prefixItems"}


def _copy_schema_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_copy_schema_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _copy_schema_value(item) for key, item in value.items()}
    return value


def _to_openai_strict_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        key: (
            {name: _to_openai_strict_schema(child) for name, child in item.items()}
            if key in _SCHEMA_MAP_KEYWORDS and isinstance(item, Mapping)
            else _to_openai_strict_schema(item)
            if key == "items" and isinstance(item, Mapping)
            else [_to_openai_strict_schema(child) for child in item if isinstance(child, Mapping)]
            if key == "anyOf" and isinstance(item, list)
            else _copy_schema_value(item)
        )
        for key, item in value.items()
        if key in _STABLE_STRICT_SCHEMA_KEYS
    }
    properties = normalized.get("properties")
    if "const" in value:
        normalized["enum"] = [_copy_schema_value(value["const"])]
    if normalized.get("type") == "object" or isinstance(properties, Mapping):
        object_properties = properties if isinstance(properties, Mapping) else {}
        normalized["additionalProperties"] = False
        normalized["required"] = list(object_properties)
    return normalized


def _contains_open_map(schema: Mapping[str, Any]) -> bool:
    additional_properties = schema.get("additionalProperties")
    if additional_properties is True or isinstance(additional_properties, Mapping):
        return True

    for keyword in _SCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, Mapping) and any(
            isinstance(child, Mapping) and _contains_open_map(child) for child in children.values()
        ):
            return True
    child = schema.get("items")
    if isinstance(child, Mapping) and _contains_open_map(child):
        return True
    children = schema.get("anyOf")
    if isinstance(children, list) and any(
        isinstance(item, Mapping) and _contains_open_map(item) for item in children
    ):
        return True
    return False


def _normalize_tagged_unions(source: Mapping[str, Any]) -> dict[str, Any]:
    """将互斥标签联合转换为供应商支持的 anyOf；普通 oneOf 不降级。

    Pydantic 用 discriminator + oneOf 表达模块 kind/心理维度。
    当各分支标签值不相交且标签必填时，anyOf 与 oneOf 等价。
    """
    from copy import deepcopy

    root = deepcopy(dict(source))

    def visit(node):
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            discriminator = node.get("discriminator", {})
            tag = discriminator.get("propertyName") if isinstance(discriminator, dict) else None
            alternatives = node.get("oneOf")
            if tag and isinstance(alternatives, list):
                variants, seen = [], set()
                for alternative in alternatives:
                    variant = _resolve_local_schema_ref(alternative, root)
                    field = variant.get("properties", {}).get(tag, {})
                    values = [field["const"]] if "const" in field else field.get("enum", [])
                    if not values or seen.intersection(values):
                        break
                    seen.update(values)
                    variants.append(variant)
                else:
                    for variant in variants:
                        variant["required"] = list(
                            dict.fromkeys([*variant.get("required", []), tag])
                        )
                    node["anyOf"] = node.pop("oneOf")
                    node.pop("discriminator", None)
            for child in list(node.values()):
                visit(child)

    visit(root)
    return root


def _contains_unrepresentable_structure(schema: Mapping[str, Any]) -> bool:
    if any(keyword in schema for keyword in _UNREPRESENTABLE_SCHEMA_KEYS):
        return True
    for keyword in _SCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, Mapping) and any(
            isinstance(child, Mapping) and _contains_unrepresentable_structure(child)
            for child in children.values()
        ):
            return True
    child = schema.get("items")
    if isinstance(child, Mapping) and _contains_unrepresentable_structure(child):
        return True
    children = schema.get("anyOf")
    return isinstance(children, list) and any(
        isinstance(item, Mapping) and _contains_unrepresentable_structure(item) for item in children
    )


def _contains_boolean_schema(schema: Any) -> bool:
    if isinstance(schema, bool):
        return True
    if not isinstance(schema, Mapping):
        return False
    for keyword in _ALL_SCHEMA_MAP_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, Mapping) and any(
            _contains_boolean_schema(child) for child in children.values()
        ):
            return True
    for keyword in _ALL_SCHEMA_NODE_KEYWORDS:
        if keyword in schema and _contains_boolean_schema(schema[keyword]):
            return True
    for keyword in _ALL_SCHEMA_LIST_KEYWORDS:
        children = schema.get(keyword)
        if isinstance(children, list) and any(
            _contains_boolean_schema(child) for child in children
        ):
            return True
    return False


def _resolve_local_schema_ref(
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
) -> Mapping[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return schema
    current: Any = root_schema
    for part in reference[2:].split("/"):
        if not isinstance(current, Mapping) or part not in current:
            return schema
        current = current[part]
    return current if isinstance(current, Mapping) else schema


def _minimal_json_example(
    schema: Mapping[str, Any],
    *,
    root_schema: Mapping[str, Any],
    visited_refs: frozenset[str] = frozenset(),
) -> Any:
    """从 JSON schema 确定性生成不含业务数据的最小示例。"""

    maximum_example_array_items = 16
    reference = schema.get("$ref")
    if isinstance(reference, str):
        if reference in visited_refs:
            return None
        resolved = _resolve_local_schema_ref(schema, root_schema)
        return _minimal_json_example(
            resolved,
            root_schema=root_schema,
            visited_refs=visited_refs | {reference},
        )

    if "const" in schema:
        return _copy_schema_value(schema["const"])
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return _copy_schema_value(enum[0])

    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        for alternative in alternatives:
            if isinstance(alternative, Mapping):
                return _minimal_json_example(
                    alternative,
                    root_schema=root_schema,
                    visited_refs=visited_refs,
                )

    schema_type = schema.get("type")
    if schema_type == "object" or isinstance(schema.get("properties"), Mapping):
        properties = schema.get("properties")
        if not isinstance(properties, Mapping):
            return {}
        return {
            name: _minimal_json_example(
                child,
                root_schema=root_schema,
                visited_refs=visited_refs,
            )
            for name, child in properties.items()
            if isinstance(name, str) and isinstance(child, Mapping)
        }
    if schema_type == "array":
        item_schema = schema.get("items")
        if not isinstance(item_schema, Mapping):
            return []
        maximum_items = schema.get("maxItems")
        if maximum_items == 0:
            return []
        minimum_items = schema.get("minItems", 1)
        count = minimum_items if isinstance(minimum_items, int) else 1
        count = min(maximum_example_array_items, max(1, count))
        return [
            _minimal_json_example(
                item_schema,
                root_schema=root_schema,
                visited_refs=visited_refs,
            )
            for _ in range(count)
        ]
    if schema_type == "boolean":
        return False
    if schema_type == "integer":
        minimum = schema.get("minimum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(minimum, int):
            return minimum
        if isinstance(exclusive_minimum, int):
            return exclusive_minimum + 1
        maximum = schema.get("maximum")
        return maximum if isinstance(maximum, int) and maximum < 0 else 0
    if schema_type == "number":
        minimum = schema.get("minimum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(minimum, int | float):
            return minimum
        if isinstance(exclusive_minimum, int | float):
            return exclusive_minimum + 1
        maximum = schema.get("maximum")
        return maximum if isinstance(maximum, int | float) and maximum < 0 else 0
    if schema_type == "null":
        return None
    if schema_type == "string":
        string_format = schema.get("format")
        if string_format == "date-time":
            return "1970-01-01T00:00:00Z"
        if string_format == "date":
            return "1970-01-01"
        minimum_length = schema.get("minLength", 0)
        return "x" * (minimum_length if isinstance(minimum_length, int) else 0)
    return None


class NodeAnalysisCloudClient:
    def create_agent_chat_model(self, *, sensitive_handler=None):
        """将同一认知模型配置提供给原生工具 Agent，保留思考字段与统一 Phoenix 链路。

        不从全局环境另选模型，不复制 key；HTTP 生命周期仍由本客户端管理。
        工具 Schema 由 LangChain bind_tools 注册，循环由 AgentLoopController 管理。
        sensitive_handler 可选：云端风控拒绝时由它做最小遮罩降级。
        """
        from moonlightbox.runtime_v1.cloud_models import RuntimeCloudChatModel, RuntimeCloudClient

        client = RuntimeCloudClient(
            endpoint=self._endpoint,
            model=self._model,
            api_key=self._api_key.get_secret_value() if self._api_key else None,
            timeout_seconds=self._timeout_seconds,
            thinking_mode=self._thinking_mode,
            max_output_tokens=self._max_output_tokens,
            max_retries=self._max_retries,
            client=self._client,
            cancellable_http=self._owns_client,
        )
        return RuntimeCloudChatModel(client, temperature=0.3, sensitive_handler=sensitive_handler)

    def __init__(
        self,
        *,
        enabled: bool,
        endpoint: str,
        model: str,
        api_key: str | SecretStr | None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 60.0,
        max_retry_after_seconds: float = 3600.0,
        response_format: Literal["json_schema", "json_object"] = "json_schema",
        # 这是所有结构化调查 Agent 的底层默认值；调用方没有显式覆盖时也不能静默
        # 关闭推理。批处理若真需要低延迟，必须显式传入 disabled。
        thinking_mode: Literal["default", "disabled"] = "default",
        max_output_tokens: int = 8192,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if not 0 < timeout_seconds <= 7200:
            raise ValueError("timeout_seconds 必须在 0 到 7200 秒之间")
        if not 0 <= max_retries <= 10:
            raise ValueError("max_retries 必须在 0 到 10 之间")
        if not 0 < backoff_seconds <= 60:
            raise ValueError("backoff_seconds 必须在 0 到 60 秒之间")
        if not 0 < max_backoff_seconds <= 300:
            raise ValueError("max_backoff_seconds 必须在 0 到 300 秒之间")
        if not 0 < max_retry_after_seconds <= 86400:
            raise ValueError("max_retry_after_seconds 必须在 0 到 86400 秒之间")
        if response_format not in {"json_schema", "json_object"}:
            raise ValueError("response_format 必须是 json_schema 或 json_object")
        if thinking_mode not in {"default", "disabled"}:
            raise ValueError("thinking_mode 必须是 default 或 disabled")
        if not 0 < max_output_tokens <= 65536:
            raise ValueError("max_output_tokens 必须在 1 到 65536 之间")
        if not endpoint.strip() or not model.strip():
            raise ValueError("endpoint 和 model 不能为空")

        self._enabled = enabled
        self._endpoint = endpoint
        self._model = model
        self._api_key = (
            api_key
            if isinstance(api_key, SecretStr)
            else SecretStr(api_key)
            if api_key is not None
            else None
        )
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._max_backoff_seconds = max_backoff_seconds
        self._max_retry_after_seconds = max_retry_after_seconds
        self._response_format = response_format
        self._thinking_mode = thinking_mode
        self._max_output_tokens = max_output_tokens
        self._is_deepseek = self._detect_deepseek_endpoint(endpoint)
        self._is_minimax = self._detect_minimax_endpoint(endpoint)
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds)

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    def estimate_structured_prompt_overhead(
        self,
        *,
        response_model: type[ResponseModel],
        json_schema: Mapping[str, Any] | None = None,
    ) -> int:
        """返回结构化输出追加内容的确定性字符数。"""

        if self._response_format != "json_object":
            return 0
        source_schema = response_model.model_json_schema() if json_schema is None else json_schema
        if isinstance(source_schema, Mapping):
            source_schema = _normalize_tagged_unions(source_schema)
        return len("\n\n" + self._json_object_instruction(source_schema))

    @model_request
    def create_structured_completion(
        self,
        *,
        system_content: str,
        user_content: str,
        response_model: type[ResponseModel],
        json_schema: Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        window_id: str | None = None,
        run_id: str | None = None,
        max_prompt_chars: int | None = None,
        thinking_mode_override: Literal["default", "disabled"] | None = None,
        max_output_tokens: int | None = None,
        request_timeout_seconds: float | None = None,
        request_max_retries: int | None = None,
        include_minimal_example: bool = True,
    ) -> ResponseModel:
        context = self._safe_context(
            operation_id=operation_id,
            window_id=window_id,
            run_id=run_id,
        )
        self._ensure_configured(context)
        source_schema = response_model.model_json_schema() if json_schema is None else json_schema
        if isinstance(source_schema, Mapping):
            source_schema = _normalize_tagged_unions(source_schema)
        if _contains_boolean_schema(source_schema):
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
                attempts=0,
                context=context,
                diagnostic={"configuration": "unsupported_boolean_schema"},
            )
        if not isinstance(source_schema, Mapping) or source_schema.get("type") != "object":
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
                attempts=0,
                context=context,
                diagnostic={"configuration": "root_schema_must_be_object"},
            )
        if _contains_unrepresentable_structure(source_schema):
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
                attempts=0,
                context=context,
                diagnostic={"configuration": "unsupported_schema_structure"},
            )
        if _contains_open_map(source_schema):
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
                attempts=0,
                context=context,
                diagnostic={"configuration": "unsupported_open_map"},
            )
        schema = _to_openai_strict_schema(source_schema)
        if self._response_format == "json_object":
            system_content = self._with_json_schema_instruction(
                system_content,
                source_schema,
                include_example=include_minimal_example,
            )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        if (
            max_prompt_chars is not None
            and sum(len(message["content"]) for message in messages) > max_prompt_chars
        ):
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
                attempts=0,
                context=context,
                diagnostic={"configuration": "prompt_budget_exceeded"},
            )
        effective_max_output_tokens = (
            self._max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        effective_timeout_seconds = (
            self._timeout_seconds if request_timeout_seconds is None else request_timeout_seconds
        )
        effective_max_retries = (
            self._max_retries if request_max_retries is None else request_max_retries
        )
        if managed():
            effective_timeout_seconds = request_timeout(effective_timeout_seconds)
            effective_max_retries = 0
        if not 0 < effective_max_output_tokens <= 65536:
            raise ValueError("max_output_tokens 必须在 1 到 65536 之间")
        if not 0 < effective_timeout_seconds <= 7200:
            raise ValueError("request_timeout_seconds 必须在 0 到 7200 秒之间")
        if not 0 <= effective_max_retries <= 10:
            raise ValueError("request_max_retries 必须在 0 到 10 之间")

        request_body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": effective_max_output_tokens,
            "response_format": (
                {"type": "json_object"}
                if self._response_format == "json_object"
                else {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                }
            ),
        }
        effective_thinking_mode = thinking_mode_override or self._thinking_mode
        # MiniMax M3 也支持 ``thinking: {type: disabled}``。此前这里只处理了
        # DeepSeek，导致 M3 即使环境明确设为 disabled，仍会把整个 completion
        # 预算消耗在 reasoning 中，最终拿不到可解析的结构化 JSON。
        if (self._is_deepseek or self._is_minimax_m3) and effective_thinking_mode == "disabled":
            request_body["thinking"] = {"type": "disabled"}
        if self._is_minimax:
            # MiniMax M3 的 OpenAI 兼容接口以 max_completion_tokens 表示完整
            # completion（思考 + 正文）预算。max_tokens 在部分网关虽可兼容，
            # 却会产生版本相关的截断语义；统一改用 M3 的原生字段。
            request_body.pop("max_tokens")
            request_body["max_completion_tokens"] = effective_max_output_tokens
            request_body["reasoning_split"] = True

        # 这是结构化编译器真正向供应商发出的请求。外层 Agent decision Span 用于说明
        # 为什么走到这里；本 Span 则保存精确 prompt/schema、重试数与供应商 usage，
        # 让一轮中可能发生的 Schema 修复不再变成 Phoenix 黑盒。
        with llm_span(
            "moonlightbox.provider.structured_completion",
            model_name=self._model,
            input_value={"messages": messages, "response_schema": source_schema},
            attributes={
                "moonlightbox.provider": "cognition",
                "moonlightbox.provider.operation_id": context.get("operation_id", ""),
                "moonlightbox.provider.response_model": response_model.__name__,
                "moonlightbox.provider.thinking_mode": effective_thinking_mode,
                "moonlightbox.provider.max_output_tokens": effective_max_output_tokens,
                "moonlightbox.provider.request_timeout_seconds": effective_timeout_seconds,
            },
        ) as span:
            add_context_snapshot(span, event_name="provider_request", value=request_body)
            response, attempts = self._post_with_retries(
                request_body,
                context,
                timeout_seconds=effective_timeout_seconds,
                max_retries=effective_max_retries,
            )
            try:
                response_body: object = response.json()
            except (json.JSONDecodeError, UnicodeDecodeError):
                response_body = {}
            add_context_snapshot(
                span,
                event_name="provider_response",
                value=response_body,
            )
            _record_structured_completion_usage(
                span=span,
                response_body=response_body,
                model_name=self._model,
                operation_id=context.get("operation_id"),
                attempts=attempts,
            )
            payload = self._extract_content(response, context, attempts)
            try:
                if isinstance(payload, str):
                    # MiniMax 在 reasoning_split 模式下偶尔会把合法 JSON 包在
                    # ```json``` 或 <think>...</think> 外壳里。先去掉这些传输层
                    # 包装，再交给 Pydantic 做唯一的结构校验；不能靠正则修补
                    # JSON 字段内容，否则会把模型的语义错误掩盖掉。
                    parsed = response_model.model_validate_json(
                        _unwrap_json_content(payload), strict=True
                    )
                else:
                    parsed = response_model.model_validate(payload, strict=True)
            except ValidationError as error:
                # 保留最小诊断，便于定位第三方模型返回的字段偏差；不记录完整
                # 聊天上下文或 API 密钥。调用方仍收到统一的 INVALID_RESPONSE。
                logger.warning(
                    (
                        "structured completion validation failed operation=%s model=%s "
                        "errors=%s payload_type=%s"
                    ),
                    context.get("operation_id", ""),
                    self._model,
                    error.errors(include_url=False)[:8],
                    type(payload).__name__,
                )
                # 调用方只需要知道“哪个结构约束没有满足”。这个有界诊断用于同一模型
                # 的一次 Schema 修复提示；完整原始响应已仅记录在本机 Phoenix 快照。
                compact_errors = error.errors(include_url=False)[:5]
                paths = [
                    ".".join(str(part) for part in item.get("loc", ()))[:180]
                    for item in compact_errors
                    if isinstance(item, Mapping)
                ]
                types = [
                    str(item.get("type", "invalid"))[:80]
                    for item in compact_errors
                    if isinstance(item, Mapping)
                ]
                validation_diagnostic: dict[str, str | int | bool] = {
                    **self._safe_response_metadata(response_body, response.status_code),
                    "validation_error_count": len(error.errors(include_url=False)),
                    "validation_error_paths": ",".join(paths),
                    "validation_error_types": ",".join(types),
                    "validation_error_messages": "; ".join(
                        str(item.get("msg", ""))[:1200] for item in compact_errors
                    ),
                }
            else:
                record_span_output(span, parsed)
                add_context_snapshot(span, event_name="provider_parsed_output", value=parsed)
                return parsed
        raise NodeAnalysisCloudError(
            NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
            attempts=attempts,
            context=context,
            diagnostic=validation_diagnostic,
        )

    def _post_with_retries(
        self,
        request_body: Mapping[str, Any],
        context: Mapping[str, str],
        *,
        timeout_seconds: float,
        max_retries: int,
    ) -> tuple[httpx.Response, int]:
        from moonlightbox.agent_runtime.capacity import check_request, observe_usage

        for attempt in range(max_retries + 1):
            terminal_error: NodeAnalysisCloudErrorCode | None = None
            terminal_diagnostic: dict[str, str | int] = {}
            try:
                from moonlightbox.observability.provider import observed_post

                sample = check_request(request_body, self._endpoint)
                response = observed_post(
                    self._client,
                    self._endpoint,
                    cancellable=self._owns_client,
                    attempt=attempt + 1,
                    headers={"Authorization": f"Bearer {self._api_key_value()}"},
                    json=request_body,
                    timeout=request_timeout(timeout_seconds),
                )
                if response.is_success:
                    try:
                        observe_usage(sample, response.json())
                    except ValueError:
                        pass
            except httpx.InvalidURL as error:
                terminal_diagnostic = {"exception_type": type(error).__name__}
                terminal_error = NodeAnalysisCloudErrorCode.INVALID_RESPONSE
            except httpx.TimeoutException as error:
                terminal_diagnostic = {"exception_type": type(error).__name__}
                if attempt < max_retries:
                    self._backoff(attempt, None)
                    continue
                terminal_error = NodeAnalysisCloudErrorCode.TIMEOUT
            except httpx.RequestError as error:
                terminal_diagnostic = {"exception_type": type(error).__name__}
                if attempt < max_retries:
                    self._backoff(attempt, None)
                    continue
                terminal_error = NodeAnalysisCloudErrorCode.NETWORK

            if terminal_error is not None:
                raise NodeAnalysisCloudError(
                    terminal_error,
                    attempts=attempt + 1,
                    context=context,
                    diagnostic=terminal_diagnostic,
                )

            status_error = self._classify_status(response.status_code)
            if status_error is None:
                if (
                    self._response_format == "json_object"
                    and self._has_blank_message_content(response)
                    and attempt < max_retries
                ):
                    self._backoff(attempt, None)
                    continue
                return response, attempt + 1
            if (
                status_error
                in {
                    NodeAnalysisCloudErrorCode.RATE_LIMIT,
                    NodeAnalysisCloudErrorCode.SERVER,
                }
                and attempt < max_retries
            ):
                self._backoff(attempt, response.headers.get("Retry-After"))
                continue
            raise NodeAnalysisCloudError(
                status_error,
                attempts=attempt + 1,
                context=context,
                diagnostic={"http_status": response.status_code},
            )

        raise AssertionError("重试循环不应在此处结束")

    @staticmethod
    def _has_blank_message_content(response: httpx.Response) -> bool:
        try:
            body = response.json()
            message = body["choices"][0]["message"]
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            KeyError,
            IndexError,
            TypeError,
        ):
            return False
        return (
            isinstance(message, Mapping)
            and not message.get("refusal")
            and not isinstance(message.get("parsed"), Mapping)
            and isinstance(message.get("content"), str)
            and not message["content"].strip()
        )

    def _extract_content(
        self,
        response: httpx.Response,
        context: Mapping[str, str],
        attempts: int,
    ) -> Any:
        try:
            body = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = None
        if body is None:
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
                attempts=attempts,
                context=context,
                diagnostic={"http_status": response.status_code},
            )
        # 有些供应商会以 200 返回空 content（例如安全过滤或耗尽生成预算）。不能记录
        # 正文、reasoning 或供应商错误文本，但需要把这几个枚举/布尔元数据留下，才能将
        # “模型未按 Schema 输出”与“供应商没有输出可解析内容”区分开。
        response_diagnostic = self._safe_response_metadata(body, response.status_code)
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            message = None
        if not isinstance(message, Mapping):
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
                attempts=attempts,
                context=context,
                diagnostic=response_diagnostic,
            )
        if message.get("refusal"):
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
                attempts=attempts,
                context=context,
                diagnostic={**response_diagnostic, "refusal": True},
            )
        parsed = message.get("parsed")
        if isinstance(parsed, Mapping):
            return parsed
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                block["text"]
                for block in content
                if isinstance(block, Mapping)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            if text_parts:
                return "".join(text_parts)
        raise NodeAnalysisCloudError(
            NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
            attempts=attempts,
            context=context,
            diagnostic=response_diagnostic,
        )

    @staticmethod
    def _safe_response_metadata(body: object, http_status: int) -> dict[str, str | int | bool]:
        """提取无正文的供应商响应状态，供审计和重试决策使用。"""

        diagnostic: dict[str, str | int | bool] = {"http_status": http_status}
        if not isinstance(body, Mapping):
            return diagnostic
        for key in ("input_sensitive", "output_sensitive"):
            value = body.get(key)
            if isinstance(value, bool):
                diagnostic[key] = value
        choices = body.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            finish_reason = choices[0].get("finish_reason")
            if isinstance(finish_reason, str):
                diagnostic["finish_reason"] = finish_reason[:80]
        base_response = body.get("base_resp")
        if isinstance(base_response, Mapping):
            status_code = base_response.get("status_code")
            if isinstance(status_code, int) and not isinstance(status_code, bool):
                diagnostic["provider_status_code"] = status_code
        return diagnostic

    @staticmethod
    def _parse_json(
        content: str,
        context: Mapping[str, str],
        attempts: int,
        status_code: int,
    ) -> Any:
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass
        raise NodeAnalysisCloudError(
            NodeAnalysisCloudErrorCode.INVALID_RESPONSE,
            attempts=attempts,
            context=context,
            diagnostic={"http_status": status_code},
        )

    def _ensure_configured(self, context: Mapping[str, str]) -> None:
        if not self._enabled:
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.DISABLED,
                attempts=0,
                context=context,
            )
        if not self._api_key or not self._api_key.get_secret_value().strip():
            raise NodeAnalysisCloudError(
                NodeAnalysisCloudErrorCode.MISSING_API_KEY,
                attempts=0,
                context=context,
            )

    def _api_key_value(self) -> str:
        if self._api_key is None:
            return ""
        return self._api_key.get_secret_value()

    def _backoff(self, attempt: int, retry_after_header: str | None) -> None:
        exponential = self._backoff_seconds * (2**attempt)
        jitter_value = min(max(self._jitter(), 0.0), 1.0)
        jittered = exponential * (1 + 0.25 * jitter_value)
        retry_after = self._parse_retry_after(retry_after_header)
        local_backoff = min(jittered, self._max_backoff_seconds)
        self._sleep(max(local_backoff, retry_after or 0.0))

    def _parse_retry_after(self, header: str | None) -> float | None:
        if header is None:
            return None
        try:
            seconds = float(header)
        except ValueError:
            pass
        else:
            if not math.isfinite(seconds) or seconds < 0:
                return None
            return min(seconds, self._max_retry_after_seconds)
        try:
            retry_at = parsedate_to_datetime(header)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        try:
            seconds = retry_at.timestamp() - self._clock()
        except (OverflowError, OSError, ValueError):
            return None
        if not math.isfinite(seconds) or seconds < 0:
            return None
        return min(seconds, self._max_retry_after_seconds)

    @staticmethod
    def _detect_deepseek_endpoint(endpoint: str) -> bool:
        try:
            return httpx.URL(endpoint).host == "api.deepseek.com"
        except httpx.InvalidURL:
            return False

    @staticmethod
    def _detect_minimax_endpoint(endpoint: str) -> bool:
        try:
            return httpx.URL(endpoint).host in {"api.minimaxi.com", "api.minimax.io"}
        except httpx.InvalidURL:
            return False

    @property
    def _is_minimax_m3(self) -> bool:
        return self._is_minimax and self._model.strip().casefold() in {
            "minimax-m3",
            "minimax/minimax-m3",
        }

    @staticmethod
    def _with_json_schema_instruction(
        content: str,
        schema: Mapping[str, Any],
        *,
        include_example: bool = True,
    ) -> str:
        instruction = NodeAnalysisCloudClient._json_object_instruction(
            schema, include_example=include_example
        )
        return f"{content}\n\n{instruction}"

    @staticmethod
    def _json_object_instruction(schema: Mapping[str, Any], *, include_example: bool = True) -> str:
        serialized_schema = json.dumps(
            schema,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if not include_example:
            # 复杂画像有完整维度和情境模块，空列表式“最小示例”会误导模型省略交付。
            return (
                "仅返回符合下列结构的 json 对象，不输出解释或额外字段。\njson schema："
                + serialized_schema
            )
        serialized_example = json.dumps(
            _minimal_json_example(schema, root_schema=schema),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            "仅返回一个符合下列结构的 json 对象，不要输出解释或额外字段。\n"
            f"json schema：{serialized_schema}\n"
            f"最小合法 JSON 示例：{serialized_example}"
        )

    @staticmethod
    def _classify_status(status_code: int) -> NodeAnalysisCloudErrorCode | None:
        if 200 <= status_code < 300:
            return None
        if status_code in {401, 403}:
            return NodeAnalysisCloudErrorCode.AUTHENTICATION
        if status_code == 429:
            return NodeAnalysisCloudErrorCode.RATE_LIMIT
        if 500 <= status_code < 600:
            return NodeAnalysisCloudErrorCode.SERVER
        return NodeAnalysisCloudErrorCode.INVALID_RESPONSE

    @staticmethod
    def _safe_context(
        *,
        operation_id: str | None,
        window_id: str | None,
        run_id: str | None,
    ) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "operation_id": operation_id,
                "window_id": window_id,
                "run_id": run_id,
            }.items()
            if value is not None
        }

    def close(self) -> None:
        if self._owns_client and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _record_structured_completion_usage(
    *,
    span: object | None,
    response_body: object,
    model_name: str,
    operation_id: str | None,
    attempts: int,
) -> None:
    """把供应商原始 usage 原样计入 Phoenix 和当前 Agent 根摘要。

    没有 ``usage`` 时保持 ``not_reported``：本地字符估算只用于上下文盘点，绝不能伪造
    成模型或缓存 token。该函数也不从响应正文推断缓存命中。
    """

    usage = response_body.get("usage") if isinstance(response_body, Mapping) else None
    normalized_usage = usage if isinstance(usage, Mapping) else None
    attributes: dict[str, object] = {"moonlightbox.provider.attempts": attempts}
    for attribute, keys in (
        ("llm.token_count.prompt", ("prompt_tokens", "input_tokens")),
        ("llm.token_count.completion", ("completion_tokens", "output_tokens")),
        ("llm.token_count.total", ("total_tokens",)),
        ("llm.token_count.reasoning", ("reasoning_tokens", "reasoning_content_tokens")),
        (
            "llm.token_count.cache_read",
            ("cached_tokens", "cache_read_tokens", "cache_read_input_tokens"),
        ),
        (
            "llm.token_count.cache_write",
            ("cache_write_tokens", "cache_creation_input_tokens"),
        ),
    ):
        value = _usage_value(normalized_usage, *keys)
        if value is not None:
            attributes[attribute] = value
    record_span_attributes(span, attributes)
    trace_context = current_agent_execution_trace_context()
    if trace_context is not None:
        record_external_provider_call(
            trace_context.state,
            provider="cognition",
            model_name=model_name,
            operation_id=operation_id,
            usage=normalized_usage,
            attempts=attempts,
        )


def _usage_value(usage: Mapping[object, object] | None, *keys: str) -> int | None:
    if usage is None:
        return None
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
                return value
    return None


def _unwrap_json_content(content: str) -> str:
    """提取模型返回的 JSON 对象，兼容 MiniMax 的思考/Markdown 外壳。"""

    text = content.strip()
    # reasoning_split=false 或网关降级时，思考文本可能直接混入 content。
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    # 只剥离首尾代码围栏；JSON 内部的反引号保持不动。
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    # 个别兼容网关会在 JSON 后附带第二段 JSON 或说明文字。不能用
    # rfind("}")，否则会把尾部内容一起切进来；按字符串状态寻找第一个
    # 完整的根对象，保留 JSON 内部的括号和转义字符。
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return text
