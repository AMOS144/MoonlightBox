"""统一 Agent 运行时的纯领域无关契约。

这里的类型刻意不导入 Runtime v1 或 PersonWorld。这样工具、Prompt 和循环控制各自可
维护，新增 Agent 时不用复制一份 LangGraph 条件边。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .resilience import ResiliencePolicy
from .tool_execution import ToolExecutionPolicy


class ChatModel(Protocol):
    """统一模型调用接口；具体绑定、容量估算和网络实现由客户端负责。"""

    def invoke(self, messages: list[BaseMessage]) -> object: ...


class RunScope(BaseModel):
    """不由模型参数决定的读取范围和权限。

    ``project_id``、``branch_id``、graph 版本和权限由启动 Agent 的应用服务写入。
    工具 schema 不应包含这些字段，避免模型通过伪造 ID 扩大查询边界。
    """

    project_id: str | None = None
    branch_id: str | None = None
    target_person_id: str | None = None
    allowed_source_snapshot_id: str | None = None
    graph_read_version: str | None = None
    actor_user_id: str | None = None
    permissions: frozenset[str] = Field(default_factory=frozenset)
    extra: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProgressDelta:
    """一次工具调用带来的、无需读取消息正文即可验证的状态变化。"""

    new_keys: frozenset[str] = frozenset()
    consumed_keys: frozenset[str] = frozenset()
    summary: str = ""
    is_negative_evidence: bool = False

    @property
    def made_progress(self) -> bool:
        return bool(self.new_keys)


class ProgressEvaluator(Protocol):
    def __call__(
        self,
        result: object,
        *,
        state_revision: int,
        seen_keys: frozenset[str],
    ) -> ProgressDelta: ...


class ResultNormalizer(Protocol):
    """把工具实现返回的对象收束为可审计的稳定结构。"""

    def __call__(self, result: object) -> object: ...


class ModelResultProjector(Protocol):
    """把完整工具结果投影成可安全交给下一次模型调用的摘要。

    工具的完整返回仍会交给进度判断、提交工具和 Phoenix。投影只决定 ToolMessage
    中模型实际可见的内容，因此不能拿它替代业务证据或审计数据。
    """

    def __call__(self, result: object) -> object: ...


def identity_result_normalizer(result: object) -> object:
    """没有领域规范化需求的工具保持原有结构，绝不从消息正文猜测事实。"""

    return result


def identity_model_result_projector(result: object) -> object:
    """默认工具保持原有模型可见结果；专用工具可声明更小的投影。"""

    return result


def source_reference_progress(
    result: object,
    *,
    state_revision: int,
    seen_keys: frozenset[str],
) -> ProgressDelta:
    """从工具的结构化 provenance 字段计算增量，不解释或匹配聊天正文。

    所有项目现有只读工具都会返回 ``source_ids``、``message_ids``、实体 ID、约束
    版本或计划版本中的至少一项。未知结构不被乐观地视为进展。
    """

    if not isinstance(result, Mapping):
        return ProgressDelta(summary="工具返回不是可审计的结构化对象")
    keys: set[str] = set()
    for field_name in (
        "source_ids",
        "message_ids",
        "document_ids",
        "entity_ids",
        "relation_ids",
        "claim_ids",
        "evidence_ids",
    ):
        values = result.get(field_name)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            keys.update(f"{field_name}:{item}" for item in values if isinstance(item, str))
    for field_name in ("version", "constraint_version", "retrieval_version", "graph_version"):
        value = result.get(field_name)
        if isinstance(value, (str, int)):
            keys.add(f"{field_name}:{value}")
    new_keys = frozenset(item for item in keys if item not in seen_keys)
    return ProgressDelta(
        new_keys=new_keys,
        summary=("获得新的可引用材料" if new_keys else "没有新的可引用材料"),
        is_negative_evidence=bool(result.get("negative_evidence")),
    )


@dataclass(frozen=True, slots=True)
class ToolContract:
    """LangChain tool 之外，Harness 需要知道的确定性调度属性。"""

    name: str
    side_effect: Literal["read_only", "proposal", "communication", "executor_only"] = "read_only"
    timeout_seconds: float = 30.0
    result_normalizer: ResultNormalizer = identity_result_normalizer
    model_result_projector: ModelResultProjector = identity_model_result_projector
    # 工具作者声明比较视图；排除执行编号，保留正文/草稿/状态变化，不解释材料语义。
    comparison_projection: ModelResultProjector = identity_model_result_projector
    progress_evaluator: ProgressEvaluator = source_reference_progress
    required_permissions: frozenset[str] = frozenset()
    max_result_chars: int = 6_000
    execution: ToolExecutionPolicy = field(default_factory=ToolExecutionPolicy)
    # 外部规则表／间接依赖改变时显式递增；不能只靠 Prompt hash 判定恢复兼容。
    contract_version: str = "1"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("工具 timeout_seconds 必须大于零")
        if self.max_result_chars < 256:
            raise ValueError("工具 max_result_chars 至少为 256")
        if self.side_effect != "read_only" and self.execution.parallelism != "serial":
            raise ValueError("状态提案和 Executor 操作必须串行")


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """模型可见的 LangChain tool 与其运行时契约的组合。"""

    tool: BaseTool
    contract: ToolContract
    is_submission: bool = False


@dataclass(frozen=True, slots=True)
class AgentBudgetPolicy:
    """统一事故熔断器；其中数字不定义业务完成。"""

    max_wall_seconds: float
    max_tool_result_chars: int
    max_unchanged_state_steps: int = 2
    emergency_max_model_steps: int | None = None
    max_total_tool_calls: int | None = None
    # 仅为未提供完整请求容量准入的模型适配器保留字符兜底。
    # 原生客户端使用下方 token 窗口和实际请求估算，不再并行套用这套阈值。
    compaction_threshold_chars: int = 24_000
    max_context_chars: int | None = None
    # 同一检查点最多执行几次（含初次）；每次重新计墙钟，累计调用预算不清零。
    max_execution_attempts: int = 3
    # 未声明模型窗口时的保守运行上限，不宣称是供应商实际窗口；可按部署覆盖。
    context_window_tokens: int = 128_000
    context_safety_margin_tokens: int = 2048
    # M3 官方保证至少 512K，最高 1M；采用保证下限，不把未知模型的
    # 128K 兜底误用于它。仍保留完整请求估算、输出预留和安全余量。
    # https://www.minimax.io/models/text/m3
    model_context_windows: dict[str, int] = field(default_factory=lambda: {"MiniMax-M3": 512_000})


@dataclass(frozen=True, slots=True)
class SubmissionContext:
    """提交工具可读取的结构化运行时状态，不含未授权原文。"""

    tool_results: tuple[object, ...]
    source_refs: tuple[str, ...]
    state_revision: int
    input_revision: int


class ContextCompactor(Protocol):
    def __call__(
        self,
        messages: Sequence[BaseMessage],
        *,
        source_refs: Sequence[str],
        unresolved: Sequence[str],
    ) -> tuple[list[BaseMessage], str]: ...


@dataclass(frozen=True, slots=True)
class AgentSpec[FinalT]:
    """领域 Agent 对统一运行时的唯一声明入口。"""

    name: str
    prompt_version: str
    budget: AgentBudgetPolicy
    submission_tool_name: str
    tools: tuple[RegisteredTool, ...] = ()
    context_compactor: ContextCompactor | None = None
    # 模型调用前续接外部输入；结果进入同一消息历史与检查点，不另开执行或消耗失败重试。
    refresh_inputs: Callable[[Sequence[BaseMessage]], list[BaseMessage]] | None = None
    resilience: ResiliencePolicy | None = None
    restore_tool_results: Callable[[Sequence[object]], None] | None = None
    # 只保存可序列化工作材料，不保存 Session、客户端或回调；与节点结果一起落盘。
    snapshot_work_state: Callable[[], dict[str, Any]] | None = None
    restore_work_state: Callable[[dict[str, Any]], None] | None = None
    # 恢复时只刷新易变的任务投影，领域回调必须保留已有工具协议和调查结果。
    resume_messages: (
        Callable[[Sequence[BaseMessage], Sequence[BaseMessage]], list[BaseMessage]] | None
    ) = None
    contract_version: str = "1"
    state_version: str = "1"


@dataclass(frozen=True, slots=True)
class AgentExecutionRequest:
    """启动一次瞬时 Agent 执行所需的应用侧输入。

    这是调用参数，不对应数据库中的 Run 行。完整模型、工具和图状态由 Phoenix Trace
    观测；业务持久化只留在各领域的 Job、Revision 和 Executor 中。
    """

    owner_type: Literal[
        "runtime_cycle",
        "runtime_expression",
        "day_plan",
        "person_world",
        "revision",
        "conversation_summary",
        "director_initialization",
        "node_investigation",
    ]
    owner_id: str
    scope: RunScope
    messages: tuple[BaseMessage, ...]
    input_revision: int = 1
    project_id: str | None = None
    # 应用层可提供 owner 当前输入版本的只读查询。它绝不进入 Prompt；Controller 在
    # 模型和工具边界比对它，防止本次内存执行把旧结果交给领域 Executor。
    input_revision_resolver: Callable[[], int | None] | None = None
    # 应用提供取消信号和状态推送入口；均不写入检查点或模型上下文。
    cancellation_requested: Callable[[], bool] | None = None
    on_status: Callable[[dict[str, Any]], None] | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True, slots=True)
class AgentExecutionResult[FinalT]:
    execution_id: str
    status: Literal["succeeded", "blocked", "waiting_for_user", "cancelled", "stale", "failed"]
    terminal_reason: str
    value: FinalT | None
    messages: tuple[BaseMessage, ...]
    error: dict[str, Any] | None = None


@dataclass(slots=True)
class SubmissionReceipt:
    """工具内已校验的 typed 产物；不是模型可伪造的 JSON accepted 标记。"""

    value: object = None
    status: Literal["succeeded", "waiting_for_user"] = "succeeded"


@dataclass(frozen=True, slots=True)
class ToolExecution:
    call_id: str
    tool_name: str
    args: dict[str, Any]
    result: object
    status: Literal["succeeded", "empty", "error", "cancelled"]
    progress: ProgressDelta
    retry_count: int = 0
    duration_ms: int = 0
    submission: SubmissionReceipt | None = None


@dataclass(slots=True)
class BudgetLedger:
    """只存可观测资源，不把本地字符估算伪装为供应商 token 使用量。"""

    request_calibration: dict[str, float] = field(default_factory=dict)
    model_steps: int = 0
    tool_calls_by_name: dict[str, int] = field(default_factory=dict)
    tool_result_chars: int = 0
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    local_context_chars: int = 0
