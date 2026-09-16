"""工具执行能力声明，不是给模型的参数，也不冒充线程锁或强制取消机制。"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    # boundaries：只能在调用前后停止；cooperative：工具内部也会检查取消。
    cancellation: Literal["boundaries", "cooperative"] = "boundaries"
    # 当前 Controller 始终串行；parallel_safe 仅表示已审查的工具体具备并行资格。
    parallelism: Literal["serial", "parallel_safe"] = "serial"
    timeout_enforcement: Literal["boundary_only", "cooperative_io"] = "boundary_only"
    exclusive_resources: tuple[str, ...] = ("execution_state",)
    reason: str = "未审查资源隔离，保守串行；不能中断正在执行的同步代码"

    def __post_init__(self):
        if self.cancellation not in {"boundaries", "cooperative"}:
            raise ValueError("未知取消能力")
        if self.parallelism not in {"serial", "parallel_safe"}:
            raise ValueError("未知并行策略")
        if self.timeout_enforcement not in {"boundary_only", "cooperative_io"}:
            raise ValueError("未知超时保障")
        if self.parallelism == "parallel_safe" and self.exclusive_resources:
            raise ValueError("占用独占资源的工具不能直接声明 parallel_safe")
        if not self.reason.strip():
            raise ValueError("执行策略必须说明依据")


def serial(*resources, reason, cooperative_io=False):
    return ToolExecutionPolicy(
        cancellation="cooperative" if cooperative_io else "boundaries",
        timeout_enforcement="cooperative_io" if cooperative_io else "boundary_only",
        exclusive_resources=tuple(resources),
        reason=reason,
    )


def submission_policy(name=""):
    # 校验回调可能读取 Session、更新素材授权等状态；未知回调不能标记纯函数。
    domain_resources = {
        "submit_expression": ("asset_authorizations", "runtime_result_cache"),
        "submit_section": ("investigation_artifacts", "module_read_tracker"),
        "submit_day_plan": ("planner_work",),
    }.get(name, ())
    return serial(
        "submission_receipt",
        "execution_state",
        "sqlalchemy_session",
        *domain_resources,
        reason="提交必须单独执行；校验回调可能访问会话及授权状态，不能并发或强制中断",
    )
