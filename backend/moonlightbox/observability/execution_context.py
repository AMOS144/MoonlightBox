"""一次 Agent 执行共享的 Phoenix 观测上下文。

这个上下文只存在于当前进程和当前逻辑任务中。它不是 ORM 实体，也绝不能被当成业务
状态保存到数据库：Phoenix 才是 Agent 调用过程的观测后端，领域表只保存业务结果。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING

from .trace_summary import AgentTraceState

if TYPE_CHECKING:
    from moonlightbox.agent_runtime.contracts import AgentExecutionRequest, AgentSpec


@dataclass(slots=True)
class AgentExecutionTraceContext:
    """贯穿 Controller、LangGraph 节点和工具调用的非模型运行上下文。"""

    execution_id: str
    agent_name: str
    prompt_version: str
    owner_type: str
    owner_id: str
    project_id: str | None
    branch_id: str | None
    target_person_id: str | None
    input_revision: int
    started_monotonic: float = field(default_factory=monotonic)
    state: AgentTraceState = field(default_factory=AgentTraceState)

    @property
    def session_id(self) -> str:
        """Phoenix Session 用领域 owner 聚合同一个业务对象的多次执行。"""

        return f"{self.owner_type}:{self.owner_id}"

    @classmethod
    def from_request(
        cls,
        *,
        execution_id: str,
        spec: AgentSpec[object],
        request: AgentExecutionRequest,
    ) -> AgentExecutionTraceContext:
        return cls(
            execution_id=execution_id,
            agent_name=spec.name,
            prompt_version=spec.prompt_version,
            owner_type=request.owner_type,
            owner_id=request.owner_id,
            project_id=request.project_id or request.scope.project_id,
            branch_id=request.scope.branch_id,
            target_person_id=request.scope.target_person_id,
            input_revision=request.input_revision,
        )


_CURRENT_TRACE_CONTEXT: ContextVar[AgentExecutionTraceContext | None] = ContextVar(
    "moonlightbox_agent_execution_trace_context",
    default=None,
)


def current_agent_execution_trace_context() -> AgentExecutionTraceContext | None:
    """返回当前 asyncio 任务绑定的执行上下文；没有 Agent 时返回 ``None``。"""

    return _CURRENT_TRACE_CONTEXT.get()


@contextmanager
def agent_execution_trace_scope(
    context: AgentExecutionTraceContext,
) -> Iterator[AgentExecutionTraceContext]:
    """把一次 Agent 调用绑定到当前逻辑执行流，并在退出时严格恢复父上下文。"""

    token = _CURRENT_TRACE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_TRACE_CONTEXT.reset(token)
