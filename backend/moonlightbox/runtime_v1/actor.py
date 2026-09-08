"""LangGraph PersonaActor：把已批准意图表达成待提交消息。"""

from __future__ import annotations

import json
from typing import Any, Protocol, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from .config import PERSONA_ACTOR_SYSTEM_PROMPT
from .schemas import ActorMessage, ContextPacket


class ActorModel(Protocol):
    def invoke(self, messages: list[Any]) -> Any: ...


class ActorState(TypedDict, total=False):
    messages: list[Any]
    model: ActorModel | None
    style_tool: BaseTool | None
    output: Any
    message: ActorMessage | None
    style_calls: int


class PersonaActor:
    """Actor 不改变决定，只负责生成消息文本。"""

    def __init__(self, model: ActorModel | None = None) -> None:
        self.model = model

    def run(
        self,
        *,
        packet: ContextPacket,
        intent: str,
        content_points: list[str],
        speech_mode: str,
        style_tool: BaseTool | None = None,
    ) -> ActorMessage:
        if self.model is None:
            return ActorMessage(
                text="\n".join(content_points) or intent, bubbles=content_points or [intent]
            )
        graph = self._build_graph()
        result = graph.invoke(
            {
                "model": self.model,
                "style_tool": style_tool,
                "style_calls": 0,
                "messages": [
                    SystemMessage(content=PERSONA_ACTOR_SYSTEM_PROMPT),
                    HumanMessage(
                        content="<actor_context>" + json.dumps(
                            {
                                "communication_intent": intent,
                                "content_points": content_points,
                                "speech_mode": speech_mode,
                                "runtime_context": packet.model_dump(mode="json"),
                            },
                            ensure_ascii=False,
                        ) + "</actor_context>"
                    ),
                ],
            }
        )
        message = result.get("message")
        if isinstance(message, ActorMessage):
            return message
        return ActorMessage(
            text="\n".join(content_points) or intent, bubbles=content_points or [intent]
        )

    def _build_graph(self) -> Any:
        graph = StateGraph(ActorState)
        graph.add_node("actor_model", self._actor_node)
        graph.add_node("style_tool", self._style_tool_node)
        graph.add_node("validate_output", self._validate_node)
        graph.add_edge(START, "actor_model")
        graph.add_conditional_edges(
            "actor_model",
            self._should_style,
            {"style": "style_tool", "validate": "validate_output"},
        )
        graph.add_edge("style_tool", "actor_model")
        graph.add_edge("validate_output", END)
        return graph.compile()

    def _actor_node(self, state: ActorState) -> dict[str, Any]:
        model = state.get("model")
        if model is None:
            return {}
        bound = model
        tool = state.get("style_tool")
        bind_tools = getattr(model, "bind_tools", None)
        if tool is not None and callable(bind_tools):
            bound = bind_tools([tool], strict=True)
        try:
            output = bound.invoke(state["messages"])
            return {"output": output, "messages": [*state["messages"], output]}
        except Exception:
            return {"output": None}

    def _should_style(self, state: ActorState) -> str:
        output = state.get("output")
        if (
            isinstance(output, AIMessage)
            and output.tool_calls
            and state.get("style_tool") is not None
            and state.get("style_calls", 0) < 1
        ):
            return "style"
        return "validate"

    def _style_tool_node(self, state: ActorState) -> dict[str, Any]:
        tool = state.get("style_tool")
        output = state.get("output")
        if tool is None or not isinstance(output, AIMessage):
            return {"style_calls": 1}
        messages = list(state["messages"])
        for call in output.tool_calls[:1]:
            try:
                result = tool.invoke(call.get("args", {}))
            except Exception:
                result = {"source_ids": [], "data": []}
            messages.append(
                ToolMessage(
                    content=(
                        "<tool_results>"
                        + json.dumps(result, ensure_ascii=False)
                        + "</tool_results>"
                    ),
                    tool_call_id=call.get("id", "style"),
                )
            )
        return {"messages": messages, "style_calls": state.get("style_calls", 0) + 1}

    def _validate_node(self, state: ActorState) -> dict[str, Any]:
        output = state.get("output")
        content = output.content if isinstance(output, AIMessage) else output
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        try:
            return {
                "message": (
                    ActorMessage.model_validate_json(content)
                    if isinstance(content, str)
                    else ActorMessage.model_validate(content)
                )
            }
        except Exception:
            return {"message": None}
