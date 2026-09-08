"""主体认知领域中的不可变输入与输出类型。"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

from moonlightbox.branches.replies import GeneratedReplyTurn

JsonObject = Mapping[str, object]


@dataclass(frozen=True)
class ExpressionDecision:
    """主体选择表达或保持沉默的决定。"""

    express: bool
    content: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class WakeupSuggestion:
    """认知生成器建议的下一次唤醒。"""

    wake_at: datetime
    reason: str
    idempotency_key: str


@dataclass(frozen=True)
class CognitionRequest:
    """一次认知生成所需的只读上下文。"""

    project_id: str
    branch_id: str
    trigger_event: JsonObject
    deadline: datetime
    current_mental_state: JsonObject
    goals: tuple[JsonObject, ...]
    relevant_context: tuple[JsonObject, ...]
    decision_context: JsonObject = field(default_factory=dict)
    model_version_id: str = ""
    authoritative_expression: bool = False
    expression_required: bool | None = None


@dataclass(frozen=True)
class CognitionDraft:
    """认知生成器给出的影子认知草稿。"""

    private_content: str
    subjective_feelings: JsonObject
    attention_target: JsonObject
    desired_actions: tuple[JsonObject, ...]
    expression_decision: ExpressionDecision
    suggested_next_wakeup: WakeupSuggestion | None
    structured_changes: JsonObject = field(default_factory=dict)
    confidence: float = 1.0


@dataclass(frozen=True)
class FusedAgentTurn:
    """一次实时生成得到的私密认知、表达决定和可选公开回复。"""

    cognition: CognitionDraft
    expression_decision: ExpressionDecision
    reply: GeneratedReplyTurn | None
