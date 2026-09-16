"""MoonlightBox 统一 Agent 运行时。

此包只负责一次进程内的 Agent 循环与工具调度；它不拥有聊天、DayPlan 或 PersonWorld
的领域写入权限，也不维护通用执行账本。领域 Agent 通过 :class:`AgentSpec` 声明目标和
工具，最终业务写入始终留在各自 Executor 中。
"""

from .contracts import (
    AgentBudgetPolicy,
    AgentExecutionResult,
    AgentSpec,
    RunScope,
    ToolContract,
)
from .controller import AgentLoopController
from .resilience import ResiliencePolicy

__all__ = [
    "AgentBudgetPolicy",
    "AgentLoopController",
    "AgentExecutionResult",
    "AgentSpec",
    "RunScope",
    "ToolContract",
    "ResiliencePolicy",
]
