"""Runtime Agent 共用的 OpenAI-compatible 云端模型适配器。

LangChain ``bind_tools`` 生成供应商原生 tools；统一 ``AgentLoopController`` 执行本地
只读工具并维护账本。本模块只负责协议转换，不维护工具名到 schema 的第二份注册表。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import monotonic
from typing import Any, Literal

import httpx
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from moonlightbox.agent_runtime.resilience import managed, model_request, request_timeout
from moonlightbox.config import Settings


class RuntimeCloudInferenceError(RuntimeError):
    """云端大模型不可用或返回了无效的 Runtime 协议。"""

    def __init__(
        self, code: str, message: str, *, diagnostic: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.diagnostic = diagnostic or {}


@dataclass(frozen=True, slots=True)
class RuntimeCloudCompletion:
    """供应商完整 assistant 消息；工具回合必须原样带回下一次请求。"""

    message: dict[str, Any]
    usage: dict[str, Any]

    @property
    def content(self) -> str:
        return _message_text(self.message.get("content"))


class RuntimeCloudClient:
    """最小 OpenAI-compatible Chat Completions 客户端。"""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None,
        timeout_seconds: float,
        thinking_mode: Literal["default", "disabled"],
        max_output_tokens: int,
        max_retries: int = 0,
        client: httpx.Client | None = None,
        cancellable_http: bool | None = None,
    ) -> None:
        self._endpoint = endpoint.strip()
        self._model = model.strip()
        self._api_key = (api_key or "").strip()
        self._timeout_seconds = timeout_seconds
        self._thinking_mode = thinking_mode
        self._max_output_tokens = max_output_tokens
        self._max_retries = max_retries
        self._is_deepseek = _host(endpoint) == "api.deepseek.com"
        self._is_minimax = _host(endpoint) in {"api.minimaxi.com", "api.minimax.io"}
        self._owns_client = client is None
        self._cancellable_http = client is None if cancellable_http is None else cancellable_http
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._runtime_settings: Settings | None = None
        if not self._endpoint or not self._model:
            raise ValueError("Runtime 云端模型 endpoint 和 model 不能为空")

    @classmethod
    def from_settings(cls, settings: Settings) -> RuntimeCloudClient:
        """复用项目认知模型配置，不增加另一套密钥。"""

        key = settings.resolved_cognition_api_key()
        client = cls(
            endpoint=settings.cognition_endpoint,
            model=settings.cognition_model,
            api_key=key.get_secret_value() if key is not None else None,
            timeout_seconds=settings.cognition_timeout_seconds,
            thinking_mode=settings.cognition_thinking_mode,
            max_output_tokens=settings.cognition_max_output_tokens,
            max_retries=settings.cognition_max_retries,
        )
        # Worker 会原地刷新这个 Settings 对象。保留引用，让后续模型调用
        # 无需重启进程即可使用设置页刚保存的 endpoint/model/key。
        client._runtime_settings = settings
        return client

    def _refresh_runtime_settings(self) -> None:
        settings = self._runtime_settings
        if settings is None:
            return
        key = settings.resolved_cognition_api_key()
        self._endpoint = settings.cognition_endpoint.strip()
        self._model = settings.cognition_model.strip()
        self._api_key = key.get_secret_value().strip() if key is not None else ""
        self._is_deepseek = _host(self._endpoint) == "api.deepseek.com"
        self._is_minimax = _host(self._endpoint) in {"api.minimaxi.com", "api.minimax.io"}

    @model_request
    def complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        temperature: float,
        tools: list[dict[str, Any]] | None = None,
    ) -> RuntimeCloudCompletion:
        self._refresh_runtime_settings()
        if not self._api_key:
            raise RuntimeCloudInferenceError("missing_api_key", "Runtime 云端大模型缺少 API key")
        request_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
        body: dict[str, object] = {
            "model": self._model,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": self._max_output_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        else:
            # 工具回合不能同时要求普通 JSON response_format；最终无工具回合才可用。
            body["response_format"] = {"type": "json_object"}
        if self._is_deepseek and self._thinking_mode == "disabled":
            body["thinking"] = {"type": "disabled"}
        if self._is_minimax:
            # MiniMax OpenAI-compatible 接口用 reasoning_details 分离思考；官方要求
            # 多轮工具调用时把该字段连同 assistant.tool_calls 完整放回历史。
            body.pop("response_format", None)
            body.pop("max_tokens", None)
            body["max_completion_tokens"] = self._max_output_tokens
            body["reasoning_split"] = True
            if self._thinking_mode == "disabled":
                body["thinking"] = {"type": "disabled"}
        retry_error = RuntimeCloudInferenceError("server", "Runtime 云端大模型暂时不可用")
        deadline = monotonic() + request_timeout(self._timeout_seconds)
        max_retries = 0 if managed() else self._max_retries
        for attempt in range(max_retries + 1):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise RuntimeCloudInferenceError("timeout", "Runtime 云端模型总请求时间已耗尽")
            try:
                message, usage = self._send_attempt(
                    body, attempt=attempt, timeout_seconds=remaining
                )
            except httpx.TimeoutException as raw_error:
                retry_error = RuntimeCloudInferenceError("timeout", "Runtime 云端大模型请求超时")
                retry_error.__cause__ = raw_error
            except httpx.RequestError as raw_error:
                retry_error = RuntimeCloudInferenceError("network", "Runtime 云端大模型网络不可用")
                retry_error.__cause__ = raw_error
            except RuntimeCloudInferenceError as error:
                retry_error = error
            else:
                return RuntimeCloudCompletion(message=message, usage=usage)
            if retry_error.code not in {
                "timeout",
                "network",
                "rate_limited",
                "server",
                "empty_response",
            }:
                raise retry_error
            if attempt >= max_retries:
                raise retry_error
        raise retry_error

    def _send_attempt(self, body, *, attempt, timeout_seconds=None):
        """每笔原生 HTTP 请求各有 LLM Span，不能把两次请求算成一个 Agent 回合。"""
        from moonlightbox.agent_runtime.capacity import check_request, observe_usage

        sample = check_request(body, self._endpoint)
        from moonlightbox.observability import (
            add_context_snapshot,
            current_agent_execution_trace_context,
            llm_span,
            record_span_attributes,
            record_span_output,
        )
        from moonlightbox.observability.trace_summary import record_external_provider_call

        usage = None
        with llm_span(
            "moonlightbox.provider.chat_completion",
            model_name=self._model,
            input_value=body,
            attributes={"moonlightbox.provider.attempt": attempt + 1},
        ) as span:
            add_context_snapshot(span, event_name="provider_request", value=body)
            try:
                from moonlightbox.observability.provider import observed_post

                response = observed_post(
                    self._client,
                    self._endpoint,
                    cancellable=self._cancellable_http,
                    attempt=attempt + 1,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=body,
                    timeout=timeout_seconds
                    if timeout_seconds is not None
                    else self._timeout_seconds,
                )
                try:
                    response_payload = response.json()
                    if response.is_success:
                        observe_usage(sample, response_payload)
                    add_context_snapshot(
                        span, event_name="provider_response", value=response_payload
                    )
                except ValueError:
                    record_span_attributes(
                        span, {"http.response.status_code": response.status_code}
                    )
                message, usage = _response_message(response, api_key=self._api_key, is_minimax=self._is_minimax)
                record_span_output(span, message)
                for attribute, key in (
                    ("prompt", "prompt_tokens"),
                    ("completion", "completion_tokens"),
                    ("total", "total_tokens"),
                ):
                    if isinstance(usage.get(key), int):
                        record_span_attributes(span, {f"llm.token_count.{attribute}": usage[key]})
                return message, usage
            except RuntimeCloudInferenceError as error:
                record_span_attributes(
                    span, {"moonlightbox.provider.error_diagnostic": error.diagnostic}
                )
                raise
            finally:
                context = current_agent_execution_trace_context()
                if context is not None:
                    record_external_provider_call(
                        context.state,
                        provider="cognition",
                        model_name=self._model,
                        operation_id="native_chat",
                        usage=usage,
                        attempts=1,
                    )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def model_name(self) -> str:
        """仅供安全 Trace 标识实际供应商模型，不暴露 API key 或 endpoint。"""

        return self._model


class RuntimeCloudChatModel:
    """满足当前 LangGraph 所需最小接口的云端 ChatModel。"""

    has_managed_request_boundaries = True
    # 原生请求在发送前用完整消息、工具 Schema 和输出预留进行容量准入。
    has_managed_request_capacity = True

    def __init__(
        self,
        client: RuntimeCloudClient,
        *,
        temperature: float,
        tool_schemas: tuple[dict[str, Any], ...] = (),
        sensitive_handler=None,
    ) -> None:
        self._client = client
        self._temperature = temperature
        self._tool_schemas = tool_schemas
        self._sensitive_handler = sensitive_handler

    @property
    def agent_resilience(self):
        from moonlightbox.agent_runtime.resilience import ResiliencePolicy

        return ResiliencePolicy(
            request_timeout_seconds=self._client._timeout_seconds,
            max_retries=self._client._max_retries,
        )

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> RuntimeCloudChatModel:
        """让 LangChain 从 StructuredTool 生成唯一一份 OpenAI tool schema。"""

        strict = kwargs.get("strict")
        schemas = tuple(convert_to_openai_tool(tool, strict=strict) for tool in tools)
        return RuntimeCloudChatModel(
            self._client,
            temperature=self._temperature,
            tool_schemas=schemas,
            sensitive_handler=self._sensitive_handler,
        )

    def with_timeout_seconds(self, seconds: float) -> RuntimeCloudChatModel:
        """每次请求采用 Harness 剩余时间，不更改共享 HTTP 客户端或其他 Agent。"""
        from copy import copy

        client = copy(self._client)
        # Controller 已注入实际网络策略；这里仅传剩余任务时间，避免旧客户端默认值
        # 再次压低显式的 Agent 请求超时。
        client._timeout_seconds = seconds
        client._owns_client = False
        return RuntimeCloudChatModel(
            client,
            temperature=self._temperature,
            tool_schemas=self._tool_schemas,
            sensitive_handler=self._sensitive_handler,
        )

    def invoke(self, messages: list[Any]) -> AIMessage:
        handler = self._sensitive_handler
        if handler is not None:
            messages = handler.masked_view(messages)
        try:
            result = self._complete_turn(messages)
        except RuntimeCloudInferenceError as error:
            if handler is None or error.code != "provider_input_rejected":
                raise
            result = self._recover_sensitive(messages, handler, error)
        if handler is not None:
            handler.note_accepted(messages)
        return result

    def _recover_sensitive(self, messages, handler, original_error) -> AIMessage:
        from moonlightbox.agent_runtime.sensitive_mask import StillRejected

        def try_call(view):
            try:
                return self._complete_turn(view)
            except RuntimeCloudInferenceError as error:
                if error.code == "provider_input_rejected":
                    raise StillRejected() from error
                raise

        result = handler.recover(messages, try_call)
        if result is None:
            raise original_error
        return result

    def _complete_turn(self, messages: list[Any]) -> AIMessage:
        system_prompt, payload = _split_system_message(messages)
        completion = self._client.complete(
            system_prompt=system_prompt,
            messages=payload,
            temperature=self._temperature,
            tools=list(self._tool_schemas),
        )
        allowed_names = frozenset(
            str(schema.get("function", {}).get("name", "")) for schema in self._tool_schemas
        )
        tool_calls = _native_tool_calls(completion.message.get("tool_calls"), allowed_names)
        invalid_calls = [call for call in tool_calls if call["type"] == "invalid_tool_call"]
        valid_calls = [call for call in tool_calls if call["type"] == "tool_call"]
        # 最终 JSON 可以清除供应商偶发的 fence；工具回合的 content 与推理字段必须
        # 保持完整，否则 MiniMax 下一轮无法延续 interleaved thinking。
        content = completion.content
        if not tool_calls:
            content = _unwrap_json(content)
        provider_fields = {
            key: completion.message[key]
            for key in ("reasoning_details", "reasoning_content", "reasoning")
            if key in completion.message
        }
        if invalid_calls:
            # 保存原始调用及顺序，下一轮必须用同一 call_id 回传错误，不能伪造修好的参数。
            provider_fields["runtime_raw_tool_calls"] = completion.message["tool_calls"]
        return AIMessage(
            content=content,
            tool_calls=valid_calls,
            invalid_tool_calls=invalid_calls,
            additional_kwargs=provider_fields,
            response_metadata={
                "usage": completion.usage,
                "model": self._client.model_name,
                "provider_usage_recorded": True,
            },
        )


def create_cloud_models(
    client: RuntimeCloudClient,
) -> tuple[RuntimeCloudChatModel, RuntimeCloudChatModel]:
    """Director、DayPlanAgent 与 PersonaActor 共用同一供应商模型配置。"""

    return (
        RuntimeCloudChatModel(client, temperature=0.15),
        RuntimeCloudChatModel(client, temperature=0.65),
    )


def _split_system_message(messages: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """把 LangChain 消息无损投影到 OpenAI 工具调用协议。"""

    system_parts: list[str] = []
    payload: list[dict[str, Any]] = []
    for message in messages:
        role = str(getattr(message, "type", None) or getattr(message, "role", "user"))
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        content = _message_text(getattr(message, "content", message))
        if role == "system":
            system_parts.append(content)
            continue
        if isinstance(message, AIMessage):
            item: dict[str, Any] = {"role": "assistant", "content": content}
            if "runtime_raw_tool_calls" in message.additional_kwargs:
                item["tool_calls"] = message.additional_kwargs["runtime_raw_tool_calls"]
            elif message.tool_calls:
                item["tool_calls"] = [_openai_tool_call(call) for call in message.tool_calls]
            for key in ("reasoning_details", "reasoning_content", "reasoning"):
                if key in message.additional_kwargs:
                    item[key] = message.additional_kwargs[key]
            payload.append(item)
            continue
        if isinstance(message, ToolMessage):
            item = {
                "role": "tool",
                "content": content,
                "tool_call_id": message.tool_call_id,
            }
            if message.name:
                item["name"] = message.name
            payload.append(item)
            continue
        payload.append({"role": "user", "content": content})
    if not system_parts:
        raise RuntimeCloudInferenceError(
            "missing_system_prompt", "Runtime 云端模型调用缺少 system prompt"
        )
    return "\n\n".join(system_parts), payload


def _native_tool_calls(raw_calls: object, allowed_names: frozenset[str]) -> list[dict[str, Any]]:
    """保留可回执的错误调用；只有缺失身份等协议损坏才中断整次响应。"""

    if raw_calls in (None, []):
        return []
    if not isinstance(raw_calls, list):
        raise RuntimeCloudInferenceError("invalid_response", "Runtime 工具调用格式无效")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_calls:
        function = raw.get("function") if isinstance(raw, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        raw_arguments = function.get("arguments") if isinstance(function, dict) else None
        call_id = raw.get("id") if isinstance(raw, dict) else None
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(call_id, str)
            or not call_id
            or call_id in seen_ids
        ):
            raise RuntimeCloudInferenceError("invalid_response", "Runtime 工具名称或调用 ID 无效")
        seen_ids.add(call_id)
        argument_error = None
        try:
            arguments = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            )
        except json.JSONDecodeError:
            arguments = None
            argument_error = "工具参数不是合法 JSON；请修正后重新调用"
        if argument_error is None and not isinstance(arguments, dict):
            argument_error = "工具参数必须是 JSON 对象，不能是数组、字符串或 null"
        if argument_error:
            normalized.append(
                {
                    "name": name,
                    "id": call_id,
                    "type": "invalid_tool_call",
                    "args": raw_arguments
                    if isinstance(raw_arguments, str)
                    else json.dumps(raw_arguments),
                    "error": argument_error,
                }
            )
            continue
        # 未知工具名交给 Harness 分发层拒绝，不能在模型适配层冒充云端故障。
        normalized.append({"name": name, "args": arguments, "id": call_id, "type": "tool_call"})
    return normalized


def _openai_tool_call(call: ToolCall) -> dict[str, Any]:
    return {
        "id": str(call.get("id", "")),
        "type": "function",
        "function": {
            "name": str(call.get("name", "")),
            "arguments": json.dumps(
                call.get("args", {}),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


# MiniMax 官方错误码（platform.minimax.cn/docs/api-reference/errorcode）。
# 限流类（1002/1041/2045）稍后重试即可；配额类（1008/2056）须充值、升级套餐或等
# 下一计费窗口，自动重试无意义；鉴权类（1004/2049）与涉敏类（1026/1027）须人工处理。
_MINIMAX_ERROR_CODES: dict[int, tuple[str, str]] = {
    1000: ("server", "Runtime 云端大模型暂时不可用"),
    1001: ("timeout", "Runtime 云端大模型请求超时"),
    1002: ("rate_limited", "Runtime 云端大模型请求受限"),
    1004: ("authentication", "Runtime 云端大模型认证失败"),
    1008: ("billing_unavailable", "Runtime 云端大模型账户余额不足"),
    1024: ("server", "Runtime 云端大模型暂时不可用"),
    1026: ("provider_input_rejected", "Runtime 云端大模型判定输入涉敏"),
    1027: ("provider_output_rejected", "Runtime 云端大模型判定输出涉敏"),
    1033: ("server", "Runtime 云端大模型暂时不可用"),
    1039: ("invalid_request", "Runtime 云端大模型 max_tokens 超出限制"),
    1041: ("rate_limited", "Runtime 云端大模型连接数受限"),
    1042: ("invalid_request", "Runtime 云端大模型请求含过多非法字符"),
    2013: ("invalid_request", "Runtime 云端大模型请求参数无效"),
    2045: ("rate_limited", "Runtime 云端大模型请求频率增长超限"),
    2049: ("authentication", "Runtime 云端大模型认证失败"),
    2056: ("quota_exhausted", "Runtime 云端大模型套餐用量已达上限"),
}


def _minimax_status(payload: Any) -> int | None:
    """MiniMax 两种错误外壳：base_resp.status_code（可伴随 HTTP 200）与
    OpenAI 兼容 error.message 尾部的 "(2056)"。"""
    if not isinstance(payload, dict):
        return None
    base = payload.get("base_resp")
    if isinstance(base, dict) and isinstance(base.get("status_code"), int) and base["status_code"]:
        return int(base["status_code"])
    error = payload.get("error")
    message = error.get("message") if isinstance(error, dict) else None
    if isinstance(message, str):
        match = re.search(r"\((\d{4,5})\)\s*$", message)
        if match:
            return int(match.group(1))
    return None


def _response_message(
    response: httpx.Response, *, api_key: str = "", is_minimax: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """提取完整 assistant 消息，错误详情只映射成本地安全错误码。"""

    try:
        payload: Any = response.json()
    except ValueError:
        payload = None
    # MiniMax 官方错误码优先于 HTTP 状态：同属 429 的“频率超限”（1002，可稍后重试）
    # 与“Token Plan 用量上限”（2056，须等下一计费窗口或升级套餐）不能混为一谈。
    minimax_status = _minimax_status(payload) if is_minimax else None
    if minimax_status in _MINIMAX_ERROR_CODES:
        code, message = _MINIMAX_ERROR_CODES[minimax_status]
        raise RuntimeCloudInferenceError(
            code,
            message,
            diagnostic={
                "http_status": response.status_code,
                "minimax_status": minimax_status,
            },
        )
    if response.status_code in {401, 403}:
        raise RuntimeCloudInferenceError("authentication", "Runtime 云端大模型认证失败")
    if response.status_code == 402:
        raise RuntimeCloudInferenceError(
            "billing_unavailable", "Runtime 云端大模型账户不可用或额度不足"
        )
    if response.status_code == 429:
        raise RuntimeCloudInferenceError("rate_limited", "Runtime 云端大模型请求受限")
    if response.status_code >= 500:
        raise RuntimeCloudInferenceError("server", "Runtime 云端大模型暂时不可用")
    if response.status_code >= 400:
        detail = payload.get("error", payload) if isinstance(payload, dict) else {}
        description = (
            str(detail.get("message", "")) if isinstance(detail, dict) else str(detail)
        ) if payload is not None else "非 JSON 错误响应"
        if api_key:
            description = description.replace(api_key, "[REDACTED_SECRET]")
        # 供应商明确拒绝输入不能伪装成格式错误，也不靠同内容重试掩盖。
        rejection = "input" in description.lower() and "sensitive" in description.lower()
        raise RuntimeCloudInferenceError(
            "provider_input_rejected" if rejection else "invalid_request",
            "Runtime 云端大模型请求无效",
            diagnostic={
                "http_status": response.status_code,
                "provider_message": description[:2000],
            },
        )
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeCloudInferenceError(
            "invalid_response", "Runtime 云端大模型响应无效"
        ) from error
    if not isinstance(message, dict):
        raise RuntimeCloudInferenceError("invalid_response", "Runtime 云端大模型响应无效")
    has_text = bool(_message_text(message.get("content")).strip())
    has_calls = isinstance(message.get("tool_calls"), list) and bool(message["tool_calls"])
    if not has_text and not has_calls:
        raise RuntimeCloudInferenceError("empty_response", "Runtime 云端大模型返回空内容")
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return dict(message), dict(usage) if isinstance(usage, dict) else {}


def _message_text(content: object) -> str:
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    if content is None:
        return ""
    return str(content)


def _unwrap_json(content: str) -> str:
    """只清除最终文本的传输外壳，工具回合不会调用此函数。"""

    stripped = re.sub(r"<think>[\s\S]*?</think>\s*", "", content, flags=re.IGNORECASE).strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped.removeprefix("```json").removesuffix("```").strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.removeprefix("```").removesuffix("```").strip()
    return _first_json_object(stripped) or stripped


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
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
    return None


def _host(endpoint: str) -> str:
    try:
        return httpx.URL(endpoint).host or ""
    except httpx.InvalidURL:
        return ""
