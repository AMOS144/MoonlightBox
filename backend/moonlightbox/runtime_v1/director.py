"""LangGraph Director：模型只产出结构化 LifeDecision，写入仍交给 Executor。"""

from __future__ import annotations

import json
from time import monotonic
from typing import Any, Protocol, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from .config import (
    DIRECTOR_SYSTEM_PROMPT,
    MAX_DIRECTOR_STEPS,
    MAX_TOOL_CALLS,
    MAX_TOOL_DEADLINE_SECONDS,
    MAX_TOOL_TOKENS,
)
from .schemas import ContextPacket, LifeDecision


class DirectorModel(Protocol):
    """兼容 LangChain ChatModel 的最小接口，方便注入现有 LoRA/云模型。"""

    def invoke(self, messages: list[Any]) -> Any: ...


class DirectorLoopState(TypedDict, total=False):
    messages: list[Any]
    packet: dict[str, Any]
    tool: BaseTool | None
    model: DirectorModel | None
    output: Any
    decision: LifeDecision | None
    model_steps: int
    tool_calls: int
    tool_tokens: int
    repeated_call_count: int
    last_fingerprint: str | None
    no_progress_count: int
    tool_error_count: int
    started_at: float
    terminal_reason: str


class DirectorAgent:
    """受硬上限保护的 Director 图；模型不可直接访问数据库。"""

    def __init__(self, model: DirectorModel | None = None) -> None:
        self.model = model

    def run(self, packet: ContextPacket, *, search_tool: BaseTool | None = None) -> LifeDecision:
        if packet.budgets.get("budget_exceeded") or self.model is None:
            return _safe_wait(packet, "模型不可用或上下文超过硬上限")
        graph = self._build_graph(search_tool)
        result = graph.invoke(
            {
                "packet": packet.model_dump(mode="json"),
                "messages": [
                    SystemMessage(content=DIRECTOR_SYSTEM_PROMPT),
                    HumanMessage(
                        content=_runtime_context_message(packet)
                    ),
                ],
                "tool": search_tool,
                "model": self.model,
                "model_steps": 0,
                "tool_calls": 0,
                "tool_tokens": 0,
                "repeated_call_count": 0,
                "last_fingerprint": None,
                "no_progress_count": 0,
                "tool_error_count": 0,
                "started_at": monotonic(),
            }
        )
        decision = result.get("decision")
        return (
            decision
            if isinstance(decision, LifeDecision)
            else _safe_wait(packet, "Director 输出无法解析")
        )

    def _build_graph(self, search_tool: BaseTool | None) -> Any:
        graph = StateGraph(DirectorLoopState)
        graph.add_node("director_model", self._director_node)
        graph.add_node("tool_node", self._tool_node)
        graph.add_node("force_final", self._force_final_node)
        graph.add_edge(START, "director_model")
        graph.add_conditional_edges(
            "director_model",
            self._should_continue,
            {"tool": "tool_node", "final": END, "force_final": "force_final"},
        )
        graph.add_edge("tool_node", "director_model")
        graph.add_edge("force_final", END)
        return graph.compile()

    def _director_node(self, state: DirectorLoopState) -> dict[str, Any]:
        model = state.get("model")
        if model is None:
            return {
                "decision": _safe_wait_from_payload(state["packet"], "模型不可用"),
                "terminal_reason": "invalid_output",
            }
        bound = model
        tool = state.get("tool")
        bind_tools = getattr(model, "bind_tools", None)
        if tool is not None and callable(bind_tools):
            bound = bind_tools([tool], strict=True)
        try:
            output = bound.invoke(state["messages"])
        except Exception:
            return {
                "output": None,
                "model_steps": state.get("model_steps", 0) + 1,
                "terminal_reason": "model_error",
            }
        messages = list(state["messages"])
        messages.append(output)
        decision = _parse_decision(output)
        return {
            "messages": messages,
            "output": output,
            "decision": decision,
            "model_steps": state.get("model_steps", 0) + 1,
        }

    def _tool_node(self, state: DirectorLoopState) -> dict[str, Any]:
        tool = state.get("tool")
        output = state.get("output")
        if tool is None or not isinstance(output, AIMessage) or not output.tool_calls:
            return {"terminal_reason": "invalid_output"}
        messages = list(state["messages"])
        calls = state.get("tool_calls", 0)
        tokens = state.get("tool_tokens", 0)
        repeated = state.get("repeated_call_count", 0)
        no_progress = state.get("no_progress_count", 0)
        errors = state.get("tool_error_count", 0)
        last = state.get("last_fingerprint")
        for call in output.tool_calls:
            if calls >= MAX_TOOL_CALLS:
                break
            args = call.get("args", {})
            fingerprint = json.dumps(args, sort_keys=True, ensure_ascii=False)
            if fingerprint == last:
                repeated += 1
            else:
                repeated = 0
            if repeated >= 2:
                break
            try:
                result = tool.invoke(args)
            except Exception as error:
                result = {"tool_name": "search_memory", "error": str(error), "source_ids": []}
                errors += 1
            serialized = json.dumps(result, ensure_ascii=False)
            tokens += max(1, len(serialized) // 4)
            source_ids = result.get("source_ids", []) if isinstance(result, dict) else []
            if not source_ids:
                no_progress += 1
            else:
                no_progress = 0
            messages.append(
                ToolMessage(
                    content=f"<tool_results>{serialized[:6000]}</tool_results>",
                    tool_call_id=call.get("id", "tool"),
                )
            )
            calls += 1
            last = fingerprint
        return {
            "messages": messages,
            "tool_calls": calls,
            "tool_tokens": tokens,
            "repeated_call_count": repeated,
            "last_fingerprint": last,
            "no_progress_count": no_progress,
            "tool_error_count": errors,
        }

    def _should_continue(self, state: DirectorLoopState) -> str:
        output = state.get("output")
        if isinstance(state.get("decision"), LifeDecision):
            return "final"
        if state.get("model_steps", 0) >= MAX_DIRECTOR_STEPS:
            return "force_final"
        if state.get("tool_calls", 0) >= MAX_TOOL_CALLS:
            return "force_final"
        if state.get("tool_tokens", 0) >= MAX_TOOL_TOKENS:
            return "force_final"
        if monotonic() - state.get("started_at", monotonic()) >= MAX_TOOL_DEADLINE_SECONDS:
            return "force_final"
        if state.get("tool_error_count", 0) > 2:
            return "force_final"
        if state.get("no_progress_count", 0) >= 2:
            return "force_final"
        if state.get("terminal_reason") in {"model_error", "invalid_output"}:
            return "force_final"
        if isinstance(output, AIMessage) and output.tool_calls and state.get("tool") is not None:
            if state.get("repeated_call_count", 0) >= 2:
                return "force_final"
            return "tool"
        return "force_final"

    def _force_final_node(self, state: DirectorLoopState) -> dict[str, Any]:
        # 再给模型一次无工具机会；失败时由代码安全等待。
        model = state.get("model")
        if model is not None:
            try:
                output = model.invoke(
                    [
                        SystemMessage(content=DIRECTOR_SYSTEM_PROMPT),
                        *state["messages"],
                        HumanMessage(
                            content="TOOL_BUDGET_EXHAUSTED：仅依据已有证据输出 LifeDecision JSON。"
                        ),
                    ]
                )
                decision = _parse_decision(output)
                if decision is not None:
                    return {"decision": decision, "terminal_reason": "budget"}
            except Exception:
                pass
        return {
            "decision": _safe_wait_from_payload(state["packet"], "Director 循环达到安全上限"),
            "terminal_reason": "budget",
        }


def _parse_decision(output: Any) -> LifeDecision | None:
    content = output.content if isinstance(output, AIMessage) else output
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    if isinstance(content, dict):
        try:
            return LifeDecision.model_validate(content)
        except Exception:
            return None
    if not isinstance(content, str):
        return None
    try:
        # JSON 输入用 model_validate_json，确保 ISO 时间字符串按协议解析；
        # dict 输入则保留给真实 LangChain structured output 的 Python 值。
        return LifeDecision.model_validate_json(content)
    except (ValueError, TypeError):
        return None


def _safe_wait(packet: ContextPacket, reason: str) -> LifeDecision:
    return _safe_wait_from_payload(packet.model_dump(mode="json"), reason)


def _safe_wait_from_payload(packet: dict[str, Any], reason: str) -> LifeDecision:
    return LifeDecision(action="wait", private_reason=reason[:500])


def _runtime_context_message(packet: ContextPacket) -> str:
    """固定标记 ContextPacket，避免运行数据和 system 规则混在同一段文本。"""

    return "<runtime_context>" + json.dumps(
        packet.model_dump(mode="json"), ensure_ascii=False
    ) + "</runtime_context>"
