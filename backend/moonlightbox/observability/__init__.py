"""应用级可观测性入口。

Phoenix 是 Agent 调用链的唯一观测来源；领域数据库只保留各自的业务状态、审核和执行
结果，不维护通用 Agent 调用账本。
"""

from .execution_context import (
    AgentExecutionTraceContext,
    agent_execution_trace_scope,
    current_agent_execution_trace_context,
)
from .phoenix import (
    add_context_snapshot,
    agent_execution_span,
    annotate_trace_async,
    chain_span,
    compression_span,
    initialize_phoenix,
    llm_span,
    record_agent_outcome,
    record_span_attributes,
    record_span_error,
    record_span_output,
    retriever_span,
    shutdown_phoenix,
    tool_span,
)

__all__ = [
    "add_context_snapshot",
    "agent_execution_span",
    "agent_execution_trace_scope",
    "AgentExecutionTraceContext",
    "annotate_trace_async",
    "chain_span",
    "compression_span",
    "current_agent_execution_trace_context",
    "initialize_phoenix",
    "llm_span",
    "record_agent_outcome",
    "record_span_attributes",
    "record_span_error",
    "record_span_output",
    "retriever_span",
    "shutdown_phoenix",
    "tool_span",
]
