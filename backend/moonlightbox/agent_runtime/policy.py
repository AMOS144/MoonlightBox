"""单一策略注入点：领域保留任务预算，模型连接提供网络策略。"""

from dataclasses import dataclass, replace

from .contracts import AgentBudgetPolicy
from .resilience import ResiliencePolicy


def inject_policy(spec, model):
    """显式 Agent 覆盖优先；否则继承实际模型客户端的请求配置。"""
    policy = getattr(model, "agent_resilience", None)
    if policy is not None and spec.resilience is None:
        spec = replace(spec, resilience=policy)
    if spec.resilience is None:
        spec = replace(spec, resilience=ResiliencePolicy())
    return spec


@dataclass(frozen=True, slots=True)
class AgentRuntimePolicy:
    """统一描述一个有状态 Agent 的系统安全边界。

    ``emergency_max_model_steps`` 与 ``max_tool_calls`` 是最后一道熔断器，不是业务完成条件。
    无新增来源不代表无进展；只有持续重复相同调用及结果才触发循环保护。
    """

    # 这是仅在语义停滞、墙钟和工具预算均未触发时的最后一道防线；不是流程轮数。
    emergency_max_model_steps: int
    max_tool_calls: int
    # 累计传输量，不等于当前上下文窗口；单次结果和窗口容量分别配置。
    max_tool_result_chars: int
    deadline_seconds: float
    max_stalled_tool_cycles: int = 2
    tool_timeout_seconds: float = 120.0
    max_single_tool_result_chars: int = 32_000
    compaction_threshold_chars: int = 100_000
    max_context_chars: int = 300_000


# Director 是每个 Runtime Cycle 的主决策 Agent：给它足够的检索/反思空间，但不能让一
# 条聊天事件永久占住 Worker。64 次工具调用是供应商或图谱异常时的保险丝。
DIRECTOR_RUNTIME_POLICY = AgentRuntimePolicy(
    emergency_max_model_steps=96,
    max_tool_calls=64,
    max_tool_result_chars=64 * 32_000,
    # 单个 Director 的调查预算；跨角色协作另有累计时间熔断器。
    deadline_seconds=1200.0,
)

# 独立起点理解比普通聊天允许更长调查，但仍使用同一预算契约。
DIRECTOR_INITIALIZATION_POLICY = replace(DIRECTOR_RUNTIME_POLICY, deadline_seconds=1800.0)
# 全历史调查使用更长的单次预算；等待用户正常结束本回合，故障恢复保留累计消耗。
NODE_INVESTIGATION_POLICY = AgentRuntimePolicy(
    emergency_max_model_steps=256,
    max_tool_calls=256,
    max_tool_result_chars=256 * 48_000,
    deadline_seconds=1800.0,
)
# 预读只占可用窗口的一部分，保留继续调查空间；最终仍由统一完整请求准入检查。
INITIALIZATION_HISTORY_SHARE = 0.45
INITIALIZATION_OUTPUT_RESERVE = 24_576

# DayPlan 是跨证据、约束和日历的调查式规划，允许比聊天决策更长的探索；计划合法与否由
# Pydantic/Executor 校验，而不是用“第 N 轮”猜测是否该停。
DAY_PLAN_RUNTIME_POLICY = AgentRuntimePolicy(
    emergency_max_model_steps=128,
    max_tool_calls=96,
    max_tool_result_chars=96 * 32_000,
    deadline_seconds=900.0,
    max_stalled_tool_cycles=3,
)

# PersonaActor 的目标很窄，但仍允许多次、不同查询的风格核对；没有把“一次取风格”硬编码
# 成它的完成路径。常态由有效 ActorMessage 停止。
PERSONA_ACTOR_RUNTIME_POLICY = AgentRuntimePolicy(
    emergency_max_model_steps=48,
    max_tool_calls=32,
    max_tool_result_chars=32 * 32_000,
    deadline_seconds=180.0,
    max_stalled_tool_cycles=2,
)


@dataclass(frozen=True, slots=True)
class SectionAgentRuntimePolicy:
    """部署层事故熔断器；从不写入 Prompt/frontmatter。"""

    max_tool_calls: int = 48
    tool_timeout_seconds: float = 300.0
    deadline_seconds: float = 900.0
    max_stalled_cycles: int = 2
    model_request_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not 12 <= self.max_tool_calls <= 200:
            raise ValueError("PersonWorld 栏目工具安全上限必须在 12 到 200 之间")
        if not 60 <= self.deadline_seconds <= 7200:
            raise ValueError("PersonWorld 栏目截止时间必须在 60 到 7200 秒之间")
        if not 1 <= self.max_stalled_cycles <= 8:
            raise ValueError("PersonWorld 无进展周期必须在 1 到 8 之间")
        if self.tool_timeout_seconds <= 0 or self.model_request_timeout_seconds <= 0:
            raise ValueError("模型与工具请求超时必须分别大于零")


def section_budget(policy: SectionAgentRuntimePolicy) -> AgentBudgetPolicy:
    return AgentBudgetPolicy(
        max_wall_seconds=policy.deadline_seconds,
        max_tool_result_chars=96_000 * policy.max_tool_calls,
        max_unchanged_state_steps=policy.max_stalled_cycles,
        max_total_tool_calls=policy.max_tool_calls,
        emergency_max_model_steps=96,
        compaction_threshold_chars=120_000,
    )


@dataclass(frozen=True, slots=True)
class RevisionAgentRuntimePolicy:
    """纠正会话单次 Agent turn 的系统熔断器。

    用户确认是跨 turn 的收敛机制；本值只防止一次后台执行因工具异常失控，不能由
    Markdown Prompt 调低或改变。
    """

    max_tool_calls: int = 32
    deadline_seconds: float = 900.0
    model_request_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not 8 <= self.max_tool_calls <= 100:
            raise ValueError("纠正 Agent 工具安全上限必须在 8 到 100 之间")
        if not 10 <= self.model_request_timeout_seconds <= 1800:
            raise ValueError("纠正 Agent 模型超时必须在 10 到 1800 秒之间")


def section_policy(settings):
    return SectionAgentRuntimePolicy(
        max_tool_calls=settings.person_world_section_max_tool_calls,
        deadline_seconds=settings.person_world_section_deadline_seconds,
        max_stalled_cycles=settings.person_world_section_max_stalled_cycles,
        model_request_timeout_seconds=settings.person_world_section_model_timeout_seconds,
        tool_timeout_seconds=settings.person_world_section_tool_timeout_seconds,
    )


def model_resilience(model, *, timeout_seconds):
    return replace(
        getattr(model, "agent_resilience", None) or ResiliencePolicy(),
        request_timeout_seconds=timeout_seconds,
    )


def task_budget():
    """Graph Patch、回归评估和别名调查的共享默认预算。"""
    return AgentBudgetPolicy(
        max_wall_seconds=900,
        max_tool_result_chars=3_000_000,
        max_total_tool_calls=64,
        emergency_max_model_steps=96,
    )


def controller_budget(policy):
    return AgentBudgetPolicy(
        max_wall_seconds=policy.deadline_seconds,
        # 累计传输量与当前窗口分离。每次工具正文最多 32K，累计允许覆盖全部安全调用数。
        max_tool_result_chars=policy.max_tool_result_chars,
        max_unchanged_state_steps=policy.max_stalled_tool_cycles,
        emergency_max_model_steps=policy.emergency_max_model_steps,
        max_total_tool_calls=policy.max_tool_calls,
        compaction_threshold_chars=policy.compaction_threshold_chars,
        max_context_chars=policy.max_context_chars,
    )


def revision_budget(policy: RevisionAgentRuntimePolicy) -> AgentBudgetPolicy:
    return AgentBudgetPolicy(
        max_wall_seconds=policy.deadline_seconds,
        max_tool_result_chars=96_000,
        max_unchanged_state_steps=2,
        max_total_tool_calls=policy.max_tool_calls,
        emergency_max_model_steps=48,
        compaction_threshold_chars=48_000,
    )
