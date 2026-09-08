import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from moonlightbox.events.change_points import CandidateBoundary
from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.events.schemas import ReviewedEvent
from moonlightbox.events.windowing import AnalysisWindow, PersistenceContext

EventType = Literal[
    "relationship_started",
    "intimacy_increased",
    "commitment",
    "boundary_change",
    "conflict",
    "distancing",
    "reconciliation",
    "separation",
    "reconnection",
]


class EventCandidate(BaseModel):
    """第一阶段云端提取出的关系节点候选。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_key: str = Field(min_length=1)
    type: EventType
    start_message_id: str = Field(min_length=1)
    end_message_id: str = Field(min_length=1)
    before_state: str = Field(min_length=1)
    after_state: str = Field(min_length=1)
    emotion_labels: list[str]
    topic: str
    conflict_level: int = Field(ge=0, le=5)
    state_change_strength: float = Field(ge=0, le=1)
    model_confidence: float = Field(ge=0, le=1)
    reason: str
    evidence_ids: list[str]


class RawEventCandidate(BaseModel):
    """云端严格 schema 使用的宽候选，具体语义由本地模型二次校验。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_key: str | None = None
    type: str | None = None
    start_message_id: str | None = None
    end_message_id: str | None = None
    before_state: str | None = None
    after_state: str | None = None
    emotion_labels: list[str] | None = None
    topic: str | None = None
    conflict_level: int | float | str | None = None
    state_change_strength: int | float | str | None = None
    model_confidence: int | float | str | None = None
    reason: str | None = None
    evidence_ids: list[str] | None = None


class EventCandidateBatch(BaseModel):
    """第一阶段云端响应 envelope。"""

    model_config = ConfigDict(extra="forbid")

    candidates: list[RawEventCandidate]


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    """不携带原始内容的候选拒绝诊断。"""

    index: int
    error_code: str


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """带可观测拒绝诊断的第一阶段结果。"""

    candidates: tuple[EventCandidate, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_candidates)

    @property
    def diagnostics(self) -> tuple[RejectedCandidate, ...]:
        return self.rejected_candidates

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> EventCandidate:
        return self.candidates[index]


class CandidateBatchStructureError(ValueError):
    """云端返回了候选项，但没有任何一项通过结构校验。"""

    def __init__(self, diagnostics: tuple[RejectedCandidate, ...]) -> None:
        self.diagnostics = diagnostics
        self.rejected_count = len(diagnostics)
        super().__init__("候选批次全部无效")


class PromptBudgetExceededError(ValueError):
    """固定复核内容已经超过本地字符预算。"""

    def __init__(self, *, character_budget: int, required_characters: int) -> None:
        self.character_budget = character_budget
        self.required_characters = required_characters
        super().__init__("复核提示词固定内容超过字符预算")


class CandidateEvidenceUnavailableError(ValueError):
    """候选证据无法映射到当前规范化窗口。"""


class EventCandidateReview(BaseModel):
    """第二阶段对候选持续性的结构化复核。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    accepted: bool
    type_supported: bool
    evidence_alignment: float = Field(ge=0, le=1)
    decisive_event: bool
    persistence: float = Field(ge=0, le=1)
    evidence_ids: list[str]
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("复核理由不能为空")
        return value


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class NodeAnalysisClient(Protocol):
    """任务 4 云端客户端的最小依赖接口，便于测试注入 fake。"""

    def create_structured_completion(
        self,
        *,
        system_content: str,
        user_content: str,
        response_model: type[ResponseModel],
        json_schema: Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        window_id: str | None = None,
        run_id: str | None = None,
        max_prompt_chars: int | None = None,
    ) -> ResponseModel: ...


class TwoStageEventReviewer:
    """使用两次独立云端调用提取候选并核查持续性。"""

    def __init__(
        self,
        client: NodeAnalysisClient,
        *,
        persistence_character_budget: int = 12000,
        persistence_session_limit: int = 3,
    ) -> None:
        if persistence_character_budget <= 0:
            raise ValueError("持续性上下文字符预算必须大于零")
        if not 0 <= persistence_session_limit <= 3:
            raise ValueError("持续性上下文会话数必须在 0 到 3 之间")
        self._client = client
        self._persistence_character_budget = persistence_character_budget
        self._persistence_session_limit = persistence_session_limit

    def extract_candidates(
        self,
        window: AnalysisWindow,
        *,
        run_id: str | None = None,
    ) -> ExtractionResult:
        response = self._client.create_structured_completion(
            system_content=_EXTRACTION_SYSTEM_PROMPT,
            user_content=json.dumps(
                {
                    "元数据": {
                        "run_id": run_id,
                        "window_id": window.window_id,
                    },
                    "消息": [_message_payload(message) for message in window.messages],
                },
                ensure_ascii=False,
            ),
            response_model=EventCandidateBatch,
            operation_id="event_candidate_extraction",
            window_id=window.window_id,
            run_id=run_id,
        )
        candidates: list[EventCandidate] = []
        rejected: list[RejectedCandidate] = []
        for index, raw_candidate in enumerate(response.candidates):
            payload = raw_candidate.model_dump()
            payload["candidate_key"] = "pending"
            try:
                payload = _normalize_candidate_payload(payload)
                candidate = EventCandidate.model_validate(payload)
            except (TypeError, ValueError, ValidationError):
                rejected.append(
                    RejectedCandidate(
                        index=index,
                        error_code="candidate_validation_failed",
                    )
                )
                continue
            candidates.append(
                candidate.model_copy(update={"candidate_key": stable_candidate_key(candidate)})
            )
        diagnostics = tuple(rejected)
        if response.candidates and not candidates:
            raise CandidateBatchStructureError(diagnostics)
        return ExtractionResult(
            candidates=tuple(candidates),
            rejected_candidates=diagnostics,
        )

    def review_candidate(
        self,
        candidate: EventCandidate,
        window: AnalysisWindow,
        persistence_context: PersistenceContext,
        *,
        run_id: str | None = None,
    ) -> EventCandidateReview:
        estimator = getattr(
            self._client,
            "estimate_structured_prompt_overhead",
            None,
        )
        prompt_overhead = (
            estimator(response_model=EventCandidateReview) if callable(estimator) else 0
        )
        user_content = _build_review_user_content(
            candidate,
            window,
            persistence_context,
            run_id=run_id,
            character_budget=self._persistence_character_budget,
            persistence_session_limit=self._persistence_session_limit,
            reserved_prompt_chars=prompt_overhead,
        )
        return self._client.create_structured_completion(
            system_content=_REVIEW_SYSTEM_PROMPT,
            user_content=user_content,
            response_model=EventCandidateReview,
            operation_id="event_candidate_review",
            window_id=window.window_id,
            run_id=run_id,
            max_prompt_chars=self._persistence_character_budget,
        )


_EXTRACTION_SYSTEM_PROMPT = """
你负责从按时间排序的规范化聊天消息中提取关系状态变化候选，只返回指定结构。
必须通读完整窗口后再判断，不能因示例为空、为求谨慎或不确定而无条件返回空列表。
若窗口中存在至少一个满足定义的关系状态变化，必须至少返回一个候选；确实不存在时才返回空列表。
不得凑数，不满足关系状态变化定义的内容不得作为候选。
候选 type 必须精确使用以下 9 个英文 ID，不得翻译、改写或创造新值：
- relationship_started：关系建立
- intimacy_increased：亲密程度提升
- commitment：作出关系承诺
- boundary_change：关系边界变化
- conflict：关系冲突
- distancing：关系疏远
- reconciliation：冲突后和解
- separation：关系分离或结束
- reconnection：关系重新连接
state_change_strength 和 model_confidence 必须是 0..1 小数，例如 0.8，不是 8 或 80。
conflict_level 才是 0..5 的整数。candidate_key 会由本地代码稳定重算。
start/end 与 evidence_ids 只能引用输入中真实存在的消息 ID，不得补写或猜测证据。
普通长间隔本身、一次性争吵和短暂情绪都不是关系节点。
""".strip()

_REVIEW_SYSTEM_PROMPT = """
你负责复核候选关系变化是否被证据语义和后续行为支持，只返回指定结构。
必须逐条判断证据内容是否支持候选类型、before_state 与 after_state，不能仅因证据 ID 合法而接受。
原始证据消息是不可截断的固定内容，必须逐条阅读后再判断。
天气闲聊不能支持 conflict。
与候选类型或状态变化无关的内容必须令 type_supported=false 或降低 evidence_alignment。
普通长间隔、一次性争吵、短暂情绪不算关系节点。
后续没有证据时必须拒绝，不能虚构消息、证据 ID 或未发生的互动。
普通事件 accepted=true 时 persistence 必须大于 0，且 evidence_ids 必须引用事件结束后的消息。
只有明确决定关系状态的终局事件才能标记 decisive_event=true；事件范围内必须至少包含行动与对方回应。
""".strip()


def _message_payload(message: NormalizedMessage) -> dict[str, str]:
    return {
        "id": message.source_id,
        "角色": message.sender,
        "时间": message.timestamp.isoformat(),
        "内容": message.content,
    }


_EVENT_TYPE_ALIASES: dict[str, EventType] = {
    "relationship_establishment": "relationship_started",
    "relationship_upgrade": "intimacy_increased",
    "intimacy_increase": "intimacy_increased",
    "commitment_establishment": "commitment",
    "boundary_setting": "boundary_change",
    "conflict_escalation": "conflict",
    "alienation": "distancing",
    "emotional_distancing": "distancing",
    "conflict_resolution": "reconciliation",
    "breakup": "separation",
    "relationship_reconnection": "reconnection",
}


def _normalize_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """仅按明确白名单和确定数值规则规范化候选。"""

    normalized = dict(payload)
    event_type = normalized.get("type")
    if isinstance(event_type, str):
        normalized["type"] = _EVENT_TYPE_ALIASES.get(event_type, event_type)
    for field_name in ("state_change_strength", "model_confidence"):
        normalized[field_name] = _normalize_unit_score(normalized.get(field_name))
    normalized["conflict_level"] = _normalize_conflict_level(normalized.get("conflict_level"))
    return normalized


def _normalize_unit_score(value: object) -> float:
    """接受 0..1 原值，或将明确的 10 分制数值缩放到 0..1。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("分数字段必须是 JSON 数值")
    numeric = float(value)
    if 0 <= numeric <= 1:
        return numeric
    if 1 < numeric <= 10:
        return numeric / 10
    raise ValueError("分数字段超出可规范化范围")


def _normalize_conflict_level(value: object) -> int:
    """接受 0..5 的整数；JSON 中 5.0 视为整数 5。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("冲突等级必须是 JSON 数值")
    numeric = float(value)
    if not numeric.is_integer() or not 0 <= numeric <= 5:
        raise ValueError("冲突等级必须是 0..5 的整数")
    return int(numeric)


def _evidence_message_payload(message: NormalizedMessage) -> dict[str, str]:
    return {
        "id": message.source_id,
        "sender": message.sender,
        "timestamp": message.timestamp.isoformat(),
        "kind": message.kind.value,
        "content": message.content,
    }


def stable_candidate_key(candidate: EventCandidate) -> str:
    """根据规范化语义字段生成稳定且抗碰撞的候选键。"""

    identity = {
        "type": _canonical_text(candidate.type),
        "start_message_id": _canonical_identifier(candidate.start_message_id),
        "end_message_id": _canonical_identifier(candidate.end_message_id),
        "evidence_ids": [
            _canonical_identifier(message_id) for message_id in candidate.evidence_ids
        ],
        "topic": _canonical_text(candidate.topic),
        "before_state": _canonical_state(candidate.before_state),
        "after_state": _canonical_state(candidate.after_state),
    }
    serialized = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"candidate-{sha256(serialized.encode()).hexdigest()}"


def _canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _canonical_identifier(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _canonical_state(value: str) -> str:
    normalized = _canonical_text(value)
    return re.sub(r"^(?:已经|已)", "", normalized, count=1)


def _messages_after_candidate(
    candidate: EventCandidate,
    window: AnalysisWindow,
) -> tuple[NormalizedMessage, ...]:
    for index, message in enumerate(window.messages):
        if message.source_id == candidate.end_message_id:
            return window.messages[index + 1 :]
    return ()


def _build_review_user_content(
    candidate: EventCandidate,
    window: AnalysisWindow,
    persistence_context: PersistenceContext,
    *,
    run_id: str | None,
    character_budget: int,
    persistence_session_limit: int,
    reserved_prompt_chars: int = 0,
) -> str:
    """逐条序列化消息，确保完整 system+user 提示词不超过预算。"""

    evidence_id_set = set(candidate.evidence_ids)
    evidence_messages = tuple(
        message for message in window.messages if message.source_id in evidence_id_set
    )
    if {message.source_id for message in evidence_messages} != evidence_id_set:
        raise CandidateEvidenceUnavailableError("候选证据不属于当前分析窗口")

    same_window_messages = tuple(
        message
        for message in _messages_after_candidate(candidate, window)
        if message.source_id not in evidence_id_set
    )
    allowed_sessions = persistence_context.following_sessions[:persistence_session_limit]
    raw_persistence_messages = persistence_context.remaining_session_messages + tuple(
        message for session in allowed_sessions for message in session.messages
    )
    seen_ids = evidence_id_set | {message.source_id for message in same_window_messages}
    persistence_messages: tuple[NormalizedMessage, ...] = ()
    unique_persistence_messages: list[NormalizedMessage] = []
    for message in raw_persistence_messages:
        if message.source_id in seen_ids:
            continue
        seen_ids.add(message.source_id)
        unique_persistence_messages.append(message)
    persistence_messages = tuple(unique_persistence_messages)
    selected_window: list[NormalizedMessage] = []
    selected_persistence: list[NormalizedMessage] = []
    omitted_sessions = len(persistence_context.following_sessions) > len(allowed_sessions)
    ordered_messages = [("window", message) for message in same_window_messages] + [
        ("persistence", message) for message in persistence_messages
    ]

    def serialize(*, truncated: bool) -> str:
        return json.dumps(
            {
                "元数据": {
                    "run_id": run_id,
                    "window_id": window.window_id,
                    "character_budget": character_budget,
                    "truncated": truncated,
                },
                "候选": candidate.model_dump(mode="json"),
                "原始证据消息": [
                    _evidence_message_payload(message) for message in evidence_messages
                ],
                "同窗口后续消息": [_message_payload(message) for message in selected_window],
                "持续性上下文": [_message_payload(message) for message in selected_persistence],
            },
            ensure_ascii=False,
        )

    base_content = serialize(
        truncated=bool(ordered_messages) or omitted_sessions,
    )
    base_size = len(_REVIEW_SYSTEM_PROMPT) + len(base_content) + reserved_prompt_chars
    if base_size > character_budget:
        raise PromptBudgetExceededError(
            character_budget=character_budget,
            required_characters=base_size,
        )

    for index, (group_name, message) in enumerate(ordered_messages):
        selected = selected_window if group_name == "window" else selected_persistence
        selected.append(message)
        has_omitted_content = index < len(ordered_messages) - 1 or omitted_sessions
        trial_content = serialize(truncated=has_omitted_content)
        if (
            len(_REVIEW_SYSTEM_PROMPT) + len(trial_content) + reserved_prompt_chars
            <= character_budget
        ):
            base_content = trial_content
            continue
        selected.pop()
        return serialize(truncated=True)
    return base_content


class LlmClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class JsonEventReviewer:
    def __init__(self, client: LlmClient) -> None:
        self._client = client

    def review(
        self,
        candidate: CandidateBoundary,
        context: dict[str, str],
    ) -> ReviewedEvent | None:
        prompt = json.dumps(
            {
                "任务": "判断这是否为关系转折点，并只返回 JSON",
                "候选分数": candidate.score,
                "信号": candidate.signals,
                "对话": context,
            },
            ensure_ascii=False,
        )
        try:
            event = ReviewedEvent.model_validate_json(self._client.complete(prompt))
        except ValidationError:
            return None

        allowed_evidence = set(candidate.evidence_ids) & set(context)
        if not event.evidence_ids or not set(event.evidence_ids).issubset(allowed_evidence):
            return None
        if event.start_message_id not in context or event.end_message_id not in context:
            return None
        return event
