"""Runtime v1 到独立人格推理服务的 LangChain 适配器。

本模块刻意不导入 PyTorch、Transformers 或 PEFT。它允许 API 与 Worker 运行在
Linux/WSL，而把模型权重、显存与 LoRA 仅留在 ``moonlightbox.persona_runtime`` 服务中。
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage
from sqlalchemy.orm import Session

from .branch_models import Branch
from .inference_client import RuntimeInferenceClient
from .schemas import GetStyleExamplesArgs, SearchMemoryArgs


def create_remote_models(
    session: Session,
    branch_id: str,
    *,
    client: RuntimeInferenceClient,
) -> tuple[RemoteDirectorChatModel, RemotePersonaActorChatModel]:
    """按分支解析模型版本，创建只走 HTTP 的 Director 与 PersonaActor。"""

    branch = session.get(Branch, branch_id)
    if branch is None:
        raise LookupError("时间分支不存在")
    model_version_id = branch.model_version_id
    return (
        RemoteDirectorChatModel(client, model_version_id),
        RemotePersonaActorChatModel(client, model_version_id),
    )


class RemoteDirectorChatModel:
    """把 LangGraph Director 的消息投影到远端基础模型。

    远端模型使用统一的受限 JSON 协议表示内部工具调用，再由本适配器转换为
    LangChain ``tool_calls``。工具本身仍在 Worker 的只读闭包中执行，远端模型
    无法借此获得数据库连接、写能力或用户身份参数。
    """

    def __init__(
        self,
        client: RuntimeInferenceClient,
        model_version_id: str,
        *,
        tool_names: frozenset[str] = frozenset(),
    ) -> None:
        self._client = client
        self._model_version_id = model_version_id
        self._tool_names = tool_names

    def bind_tools(self, tools: list[Any], **_kwargs: Any) -> RemoteDirectorChatModel:
        """把 LangChain tool 名称传给远端适配器；身份参数仍由本地 Tool 闭包绑定。"""

        return RemoteDirectorChatModel(
            self._client,
            self._model_version_id,
            tool_names=frozenset(
                str(getattr(tool, "name", "")) for tool in tools if getattr(tool, "name", None)
            ),
        )

    def invoke(self, messages: list[Any]) -> AIMessage:
        system_prompt, payload = _split_system_message(messages, keep_tool_role=True)
        content = self._client.generate_runtime_director(
            self._model_version_id,
            system_prompt,
            payload,
        )
        tool_message = _tool_call_message(content, self._tool_names)
        return tool_message or AIMessage(content=content)

    def count_text_tokens(self, text: str) -> int:
        """供 ContextAssembler 使用当前 Linux Director tokenizer 的精确计数。"""

        return self._client.count_runtime_director_tokens(self._model_version_id, text)


class RemotePersonaActorChatModel:
    """用现有远端 LoRA 将批准的表达意图改写为人物口吻。

    不复用普通回复端点：普通端点会先将模型输出解析成气泡协议，导致 Actor
    无法进行严格 JSON 校验或调用只读风格示例工具。专用端点保留原始输出，
    仍固定挂载当前版本的 LoRA。
    """

    def __init__(
        self,
        client: RuntimeInferenceClient,
        model_version_id: str,
        *,
        tool_names: frozenset[str] = frozenset(),
    ) -> None:
        self._client = client
        self._model_version_id = model_version_id
        self._tool_names = tool_names

    def bind_tools(self, tools: list[Any], **_kwargs: Any) -> RemotePersonaActorChatModel:
        return RemotePersonaActorChatModel(
            self._client,
            self._model_version_id,
            tool_names=frozenset(
                str(getattr(tool, "name", "")) for tool in tools if getattr(tool, "name", None)
            ),
        )

    def invoke(self, messages: list[Any]) -> AIMessage:
        # 通用聊天模板不保证接受 tool role；示例结果已由 LangGraph 包进
        # <tool_results>，降为 user 上下文不会改变它的只读语义。
        system_prompt, payload = _split_system_message(messages, keep_tool_role=False)
        content = self._client.generate_runtime_actor(
            self._model_version_id,
            system_prompt,
            payload,
        )
        tool_message = _tool_call_message(content, self._tool_names)
        if tool_message is not None:
            return tool_message
        return AIMessage(content=content)


def _split_system_message(
    messages: list[Any], *, keep_tool_role: bool
) -> tuple[str, list[dict[str, str]]]:
    """将 LangChain 消息稳定投影为人格服务的无状态 HTTP 协议。"""

    system_parts: list[str] = []
    payload: list[dict[str, str]] = []
    for message in messages:
        role = str(getattr(message, "type", None) or getattr(message, "role", "user"))
        role = {"human": "user", "ai": "assistant"}.get(role, role)
        content = getattr(message, "content", message)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        normalized = str(content)
        if role == "system":
            system_parts.append(normalized)
        elif role in {"user", "assistant"}:
            payload.append({"role": role, "content": normalized})
        elif role == "tool":
            payload.append({"role": "tool" if keep_tool_role else "user", "content": normalized})
        else:
            payload.append({"role": "user", "content": normalized})
    if not system_parts:
        raise ValueError("Runtime 模型调用缺少系统提示词")
    return "\n\n".join(system_parts), payload


def _tool_call_message(content: str, allowed_names: frozenset[str]) -> AIMessage | None:
    """将远端内部 JSON 工具协议转换为 LangChain 标准 tool_calls。"""

    if not allowed_names:
        return None
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        return None
    raw_calls = decoded.get("tool_calls") if isinstance(decoded, dict) else None
    if not isinstance(raw_calls, list) or not raw_calls:
        return None
    normalized: list[dict[str, object]] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            return None
        name = raw.get("name")
        args = raw.get("args")
        if not isinstance(name, str) or name not in allowed_names or not isinstance(args, dict):
            return None
        try:
            if name == "search_memory":
                SearchMemoryArgs.model_validate(args)
            elif name == "get_style_examples":
                GetStyleExamplesArgs.model_validate(args)
            else:
                return None
        except ValueError:
            return None
        normalized.append({"name": name, "args": args, "id": f"runtime-{uuid4()}"})
    return AIMessage(content="", tool_calls=normalized)
