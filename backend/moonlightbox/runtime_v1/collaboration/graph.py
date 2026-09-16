"""平级 LangGraph 节点；内部推理仍由各自 AgentLoopController 负责。"""

from time import time

from langgraph.graph import END, START, StateGraph

from moonlightbox.observability.phoenix import add_context_snapshot, chain_span, record_span_output

from ..config import (
    COLLABORATION_MAX_HANDOFFS,
    COLLABORATION_MAX_MODEL_STEPS,
    COLLABORATION_MAX_PROVIDER_TOKENS,
    COLLABORATION_MAX_TOOL_CALLS,
    COLLABORATION_TIMEOUT_SECONDS,
)
from .routing import project_turn
from .state import RuntimeState


def resume_window():
    """Job 已批准恢复；这里只重开时间窗口，不重置累计用量或再决定重试次数。"""
    return {"started_at": time()}


def build_collaboration_graph(
    *,
    director,
    day_planner,
    executor,
    checkpointer=None,
    input_is_current=lambda: True,
    one_turn=False,
    usage_observer=None,
    yield_director=False,
):
    builder = StateGraph(RuntimeState)

    def route(state):
        if state.get("route") == "end":
            return {}  # 已提交完成的回执不会被随后到达的新消息改写成失败。
        if not input_is_current():
            raise RuntimeError("stale_input_revision")
        if (
            state.get("handoffs", 0) >= COLLABORATION_MAX_HANDOFFS
            or time() - state["started_at"] >= COLLABORATION_TIMEOUT_SECONDS
        ):
            from moonlightbox.agent_runtime.resilience import ExecutionInterrupted

            raise ExecutionInterrupted("collaboration_budget_exhausted")
        usage = state.get("usage", {})
        if (
            usage.get("model_steps", 0) >= COLLABORATION_MAX_MODEL_STEPS
            or usage.get("tool_calls", 0) >= COLLABORATION_MAX_TOOL_CALLS
            or usage.get("provider_tokens", 0) >= COLLABORATION_MAX_PROVIDER_TOKENS
        ):
            from moonlightbox.agent_runtime.resilience import ExecutionInterrupted

            raise ExecutionInterrupted("collaboration_budget_exhausted")
        return {}

    def guarded(name, callback):
        def run(state):
            with chain_span(
                f"runtime.collaboration.{name}",
                input_value=state,
                attributes={
                    "moonlightbox.branch.id": state["branch_id"],
                    "moonlightbox.cycle.key": state["cycle_id"],
                },
            ) as span:
                add_context_snapshot(span, event_name="collaboration_input", value=state)
                from moonlightbox.agent_runtime.persistence import checkpoint_scope

                if checkpointer and name != "executor":
                    thread = (
                        f"runtime-inner:{state['branch_id']}:{state['cycle_id']}:"
                        f"{state['input_revision']}:{state.get('handoffs', 0)}:{name}:v3"
                    )
                    if one_turn:
                        thread += f":context:{state.get('life', {}).get('context_epoch', 0)}"
                    with checkpoint_scope(checkpointer, thread) as usage:
                        try:
                            result = callback(state)
                        finally:
                            # 后台任务连失败调用也计入预算；恢复不依赖 Phoenix 文本回读。
                            if usage_observer is not None:
                                usage_observer(thread, usage)
                    result["usage"] = {
                        key: state.get("usage", {}).get(key, 0) + value
                        for key, value in usage.items()
                    }
                else:
                    result = callback(state)
                if one_turn and result.get("route") != "end":
                    result["status"] = "waiting"
                record_span_output(span, result)
                add_context_snapshot(span, event_name="collaboration_output", value=result)
                return {
                    **result,
                    **project_turn(state, result, actor=name),
                    "handoffs": state.get("handoffs", 0) + 1,
                }

        return run

    builder.add_node("route", route)
    for name, callback in (
        ("director", director),
        ("day_planner", day_planner),
        ("executor", executor),
    ):
        if callback is None:
            continue
        builder.add_node(name, guarded(name, callback))
        builder.add_edge(name, "route")
    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {
            "director": END if yield_director else "director",
            "day_planner": "day_planner",
            "executor": "executor",
            "end": END,
        },
    )
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["director", "day_planner", "executor"] if one_turn else None,
    )
