import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.events.v3_types import EventLane, EventStatus, is_valid_event_type
from moonlightbox.events.windowing import AnalysisWindow


class V3EventCandidate(BaseModel):
    """V3 双通道提取出的结构化事件候选。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate_key: str = Field(min_length=1)
    lane: EventLane
    type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    event_status: EventStatus
    start_message_id: str = Field(min_length=1)
    end_message_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    before_state: str | None = None
    after_state: str | None = None
    emotion_labels: list[str]
    topic: str
    conflict_level: int = Field(ge=0, le=5)
    event_significance: float = Field(ge=0, le=1)
    relationship_impact: float = Field(ge=0, le=1)
    model_confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lane_contract(self) -> "V3EventCandidate":
        if not is_valid_event_type(self.lane, self.type):
            raise ValueError("候选类型与通道不匹配")
        if self.lane == "relationship" and not (
            self.before_state
            and self.before_state.strip()
            and self.after_state
            and self.after_state.strip()
        ):
            raise ValueError("关系候选必须同时提供非空前后状态")
        object.__setattr__(self, "candidate_key", stable_v3_candidate_key(self))
        return self


class RawV3EventCandidate(BaseModel):
    """隔离云端单项错误的宽候选，仅在本地收紧语义和数值。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate_key: str | None = Field(default=None, min_length=1)
    lane: str | None = Field(default=None, min_length=1)
    type: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    event_status: str | None = Field(default=None, min_length=1)
    start_message_id: str | None = Field(default=None, min_length=1)
    end_message_id: str | None = Field(default=None, min_length=1)
    summary: str | None = Field(default=None, min_length=1)
    before_state: str | None = None
    after_state: str | None = None
    emotion_labels: list[str] | None = None
    topic: str | None = None
    conflict_level: int | float | str | bool | None = None
    event_significance: int | float | str | bool | None = None
    relationship_impact: int | float | str | bool | None = None
    model_confidence: int | float | str | bool | None = None
    reason: str | None = Field(default=None, min_length=1)
    evidence_ids: list[str] | None = Field(default=None, min_length=1)


class RawV3EventCandidateBatch(BaseModel):
    """只校验根对象和 candidates 数组的宽 envelope。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidates: list[Any]


class _V3ExtractionRequestBatch(BaseModel):
    """仅用于生成云端结构化输出 schema，不参与整批本地校验。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidates: list[RawV3EventCandidate]


_V3_EXTRACTION_REQUEST_SCHEMA = _V3ExtractionRequestBatch.model_json_schema()


class V3CandidateReview(BaseModel):
    """通道复核返回的事实判断与六维评分输入。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    facts_supported: bool
    occurrence_supported: bool
    bilateral_confirmation: bool
    evidence_alignment: float = Field(ge=0, le=1)
    persistence: float = Field(ge=0, le=1)
    type_support: float = Field(ge=0, le=1)
    relationship_impact: float = Field(ge=0, le=1)
    event_significance: float = Field(ge=0, le=1)
    model_confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str]
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("复核理由不能为空")
        return value


class V3GlobalSelectionItem(BaseModel):
    """全局比较后保留的节点及其相对重要性。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    candidate_key: str = Field(min_length=1)
    relative_importance: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)


class V3GlobalSelectionResult(BaseModel):
    """允许模型返回空集合的全局节点选择结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    selected: list[V3GlobalSelectionItem]


class RawV3CandidateReview(BaseModel):
    """允许有限数值规范化的复核响应模型。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    facts_supported: bool | None = None
    occurrence_supported: bool | None = None
    bilateral_confirmation: bool | None = None
    evidence_alignment: int | float | str | bool | None = None
    persistence: int | float | str | bool | None = None
    type_support: int | float | str | bool | None = None
    relationship_impact: int | float | str | bool | None = None
    event_significance: int | float | str | bool | None = None
    model_confidence: int | float | str | bool | None = None
    evidence_ids: list[str] | None = None
    reason: str | None = Field(default=None, min_length=1)


class V3RejectedCandidate(BaseModel):
    """不包含候选正文的安全拒绝诊断。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    index: int
    error_code: str
    field_locations: tuple[tuple[str | int, ...], ...]


@dataclass(frozen=True, slots=True)
class V3ExtractionResult:
    """候选结果及单项安全诊断。"""

    candidates: tuple[V3EventCandidate, ...]
    rejected_candidates: tuple[V3RejectedCandidate, ...]

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_candidates)

    @property
    def diagnostics(self) -> tuple[V3RejectedCandidate, ...]:
        return self.rejected_candidates

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> V3EventCandidate:
        return self.candidates[index]


class V3CandidateBatchStructureError(ValueError):
    """云端返回了候选，但整批候选均无法安全校验。"""

    def __init__(self, diagnostics: tuple[V3RejectedCandidate, ...]) -> None:
        self.diagnostics = diagnostics
        self.rejected_count = len(diagnostics)
        super().__init__("V3 候选批次全部无效")


class V3ReviewStructureError(ValueError):
    """云端复核响应无法通过安全结构校验。"""

    def __init__(self) -> None:
        super().__init__("V3 候选复核结构无效")


class V3CandidateIntegrityError(ValueError):
    """公开复核边界收到无效、伪造或陈旧候选。"""

    def __init__(self) -> None:
        super().__init__("V3 候选完整性校验失败")


class V3PromptBudgetExceededError(ValueError):
    """完整 V3 复核提示词超过本地字符预算。"""

    def __init__(self, *, character_budget: int, required_characters: int) -> None:
        self.character_budget = character_budget
        self.required_characters = required_characters
        super().__init__("V3 复核提示词超过字符预算")


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class V3AnalysisClient(Protocol):
    """V3 审阅器复用的云端结构化调用接口。"""

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


_NO_INVENTION_RULE = "不得补写消息 ID、人物、地点或行为；所有事实与证据 ID 只能来自输入消息。"

_RELATIONSHIP_EXTRACTION_PROMPT = f"""
你负责提取关系状态变化候选，只返回指定结构。
候选 lane 必须是 relationship，type 只能是以下 9 类：
relationship_started、intimacy_increased、commitment、boundary_change、conflict、
distancing、reconciliation、separation、reconnection。
关系候选必须同时给出非空 before_state 与 after_state。
occurred 表示关系变化已有直接证据；confirmed 表示双方已经明确确认但尚待落实。
普通闲聊、短暂情绪或一次性分歧不得凑成候选。
必须相对窗口中已经存在的关系基线判断“变化”：重复示爱、日常晚安、调情或延续既有
亲密状态不是 intimacy_increased；before_state 与 after_state 语义相同不得输出候选。
{_NO_INVENTION_RULE}
candidate_key 由本地稳定重算。所有 0..1 分数请返回 JSON 数值，conflict_level 返回 0..5 整数。
""".strip()

_SHARED_EXTRACTION_PROMPT = f"""
你负责提取有意义的共同经历候选，只返回指定结构。
候选 lane 必须是 shared_experience，type 只能是以下 10 类：
date（约会）、outing（出游）、travel（旅行）、celebration（庆祝）、gift（礼物）、
family_social（见亲友）、support_care（照顾）、shared_project（共同项目）、
important_plan（重要计划）、life_milestone（人生里程碑）。
meaningful 指事件具有明确安排、明显体验、后续回顾、关系影响或非日常记录价值；
普通吃饭、购物、通勤和随口提议默认不属于 meaningful。
一次普通线上通话或游戏不是 date、outing 或 shared_project；shared_project 必须是双方
持续投入并形成共同成果的事项。付费购买、转卖或报销不是 gift，除非证据明确表达赠予
及其情感意义。类型不匹配时不得靠提高分数保留候选。
occurred 指消息证明活动过程、结束、回程或事后回顾；confirmed 指事件尚未发生，
但双方已经明确同意具体计划。单方面提议、模糊回应和“以后有空”不算 confirmed。
共同经历允许 before_state 与 after_state 为 null。
{_NO_INVENTION_RULE}
candidate_key 由本地稳定重算。所有 0..1 分数请返回 JSON 数值，conflict_level 返回 0..5 整数。
""".strip()

_RELATIONSHIP_REVIEW_PROMPT = f"""
你负责独立复核 relationship 候选，只返回指定结构。
逐项核对候选类型、关系前后状态、发生状态、事实、证据对齐和评分。
只按以下 9 类判断：relationship_started、intimacy_increased、commitment、
boundary_change、conflict、distancing、reconciliation、separation、reconnection。
不得仅因证据 ID 存在就认定事实成立。
先比较 before_state 与 after_state 是否构成相对既有基线的真实转变。重复示爱、日常晚安、
普通调情、一次争执等若没有改变关系状态，必须 facts_supported=false；不能用高分补偿。
{_NO_INVENTION_RULE}
""".strip()

_SHARED_REVIEW_PROMPT = f"""
你负责独立复核 shared_experience 候选，只返回指定结构。
逐项核对约会、出游、旅行、庆祝、礼物、见亲友、照顾、共同项目、重要计划或人生里程碑。
meaningful 必须具有明确安排、明显体验、后续回顾、关系影响或非日常记录价值。
occurred 必须有活动发生或事后证据；confirmed 必须有双方明确确认。
单方面提议、模糊回应和普通日常活动不应获得事实支持。
严格检查类型语义：普通视频/语音通话不是 date，一局游戏不是 shared_project，票务买卖
不是 gift。只有非日常、持续投入、形成共同成果或具有明确关系意义的经历才可支持。
{_NO_INVENTION_RULE}
""".strip()

_GLOBAL_SELECTION_PROMPT = f"""
你是整段关系时间线的总编辑。请同时比较所有候选及其原始证据，只保留真正值得成为
平行时间分支起点的节点。核心判断是：删除后是否会改变关系轨迹、双方对彼此
的理解、后续选择，或一段具有持续叙事价值的共同记忆。
不要按事件类型、关键词或候选自评分机械判断；普通活动也可能重要，重大活动也可能
只是流水账。必须做候选之间的相对比较，并参考用户过去标记的误报校准标准。
可以返回空集合；selected 按相对重要性从高到低排列，数量不得超过 maximum_nodes。
relative_importance 只表示本批候选中的相对重要程度，不是固定门槛分。
选择前必须逐项执行反事实检查：如果删除该事件，关系轨迹、重要承诺、共同生活选择或
可长期辨认的共同记忆是否真的会缺失。仅仅证明“发生过”不等于“重要”。以下内容默认
排除，除非原始证据明确证明其产生了超出日常的长期影响：重复示爱、晚安、普通调情、
一次视频/语音通话、临时玩一局游戏、日常吃饭购物、票务买卖、单方面或模糊计划。
必须检查候选类型是否与证据一致；错分为 date、gift、shared_project 等的候选直接排除。
relationship 候选若前后状态相同或只是既有亲密程度的延续，直接排除。
候选自述、reason、模型分数仅供交叉核对，不能替代原始证据，也不能作为入选理由。
入选理由必须具体说明该事件相对整段时间线为何不可替代；无法说明则不要选择。
{_NO_INVENTION_RULE}
""".strip()

_EXTRACTION_PROMPTS: dict[EventLane, str] = {
    "relationship": _RELATIONSHIP_EXTRACTION_PROMPT,
    "shared_experience": _SHARED_EXTRACTION_PROMPT,
}
_REVIEW_PROMPTS: dict[EventLane, str] = {
    "relationship": _RELATIONSHIP_REVIEW_PROMPT,
    "shared_experience": _SHARED_REVIEW_PROMPT,
}


class DualChannelEventReviewer:
    """以完全独立的调用和提示词处理两个 V3 通道。"""

    def __init__(
        self,
        client: V3AnalysisClient,
        *,
        review_character_budget: int = 12000,
        global_selection_character_budget: int = 48000,
    ) -> None:
        if review_character_budget <= 0:
            raise ValueError("复核字符预算必须大于零")
        if global_selection_character_budget <= 0:
            raise ValueError("全局复核字符预算必须大于零")
        self._client = client
        self._review_character_budget = review_character_budget
        self._global_selection_character_budget = global_selection_character_budget

    def extract_candidates(
        self,
        window: AnalysisWindow,
        *,
        lane: EventLane,
        run_id: str | None = None,
    ) -> V3ExtractionResult:
        response = self._client.create_structured_completion(
            system_content=_EXTRACTION_PROMPTS[lane],
            user_content=json.dumps(
                {
                    "元数据": {
                        "run_id": run_id,
                        "window_id": window.window_id,
                        "lane": lane,
                    },
                    "消息": [_message_payload(message) for message in window.messages],
                },
                ensure_ascii=False,
            ),
            response_model=RawV3EventCandidateBatch,
            json_schema=_V3_EXTRACTION_REQUEST_SCHEMA,
            operation_id=f"v3_{lane}_extraction",
            window_id=window.window_id,
            run_id=run_id,
        )
        accepted: list[V3EventCandidate] = []
        rejected: list[V3RejectedCandidate] = []
        for index, raw_item in enumerate(response.candidates):
            try:
                raw_candidate = RawV3EventCandidate.model_validate(raw_item)
            except ValidationError as error:
                rejected.append(
                    _safe_rejected_candidate(
                        index=index,
                        field_locations=_safe_validation_locations(error),
                    )
                )
                continue
            payload = raw_candidate.model_dump()
            payload["candidate_key"] = "pending"
            try:
                normalized = _normalize_candidate_payload(payload)
                candidate = V3EventCandidate.model_validate(normalized)
                if candidate.lane != lane:
                    raise _CandidateFieldError("lane")
            except _CandidateFieldError as error:
                rejected.append(
                    _safe_rejected_candidate(
                        index=index,
                        field_locations=((error.field_name,),),
                    )
                )
                continue
            except ValidationError as error:
                rejected.append(
                    _safe_rejected_candidate(
                        index=index,
                        field_locations=_safe_validation_locations(error),
                    )
                )
                continue
            accepted.append(candidate)
        diagnostics = tuple(rejected)
        if response.candidates and not accepted:
            raise V3CandidateBatchStructureError(diagnostics)
        return V3ExtractionResult(
            candidates=tuple(accepted),
            rejected_candidates=diagnostics,
        )

    def select_globally(
        self,
        *,
        candidates: Sequence[Mapping[str, object]],
        rejected_examples: Sequence[Mapping[str, object]],
        maximum_nodes: int,
        run_id: str | None = None,
    ) -> V3GlobalSelectionResult:
        response = self._client.create_structured_completion(
            system_content=_GLOBAL_SELECTION_PROMPT,
            user_content=json.dumps(
                {
                    "元数据": {
                        "run_id": run_id,
                        "maximum_nodes": maximum_nodes,
                    },
                    "候选": list(candidates),
                    "用户历史误报": list(rejected_examples),
                },
                ensure_ascii=False,
            ),
            response_model=V3GlobalSelectionResult,
            json_schema=V3GlobalSelectionResult.model_json_schema(),
            operation_id="v3_global_selection",
            run_id=run_id,
            max_prompt_chars=self._global_selection_character_budget,
        )
        known_keys = {
            str(candidate["candidate_key"])
            for candidate in candidates
            if "candidate_key" in candidate
        }
        selected_keys = [item.candidate_key for item in response.selected]
        if (
            len(selected_keys) > maximum_nodes
            or len(selected_keys) != len(set(selected_keys))
            or not set(selected_keys).issubset(known_keys)
        ):
            raise V3CandidateIntegrityError()
        return response

    def review_candidate(
        self,
        candidate: V3EventCandidate,
        window: AnalysisWindow,
        *,
        run_id: str | None = None,
    ) -> V3CandidateReview:
        candidate = _validate_candidate_integrity(candidate)
        prompt = _REVIEW_PROMPTS[candidate.lane]
        estimator = getattr(self._client, "estimate_structured_prompt_overhead", None)
        overhead = estimator(response_model=RawV3CandidateReview) if callable(estimator) else 0
        review_messages = _fit_review_messages(
            candidate,
            window,
            run_id=run_id,
            prompt=prompt,
            overhead=overhead,
            character_budget=self._review_character_budget,
        )
        user_content = _review_user_content(
            candidate,
            window,
            review_messages,
            run_id=run_id,
        )
        response = self._client.create_structured_completion(
            system_content=prompt,
            user_content=user_content,
            response_model=RawV3CandidateReview,
            operation_id=f"v3_{candidate.lane}_review",
            window_id=window.window_id,
            run_id=run_id,
            max_prompt_chars=self._review_character_budget,
        )
        try:
            return V3CandidateReview.model_validate(
                _normalize_review_payload(response.model_dump())
            )
        except (TypeError, ValueError, ValidationError):
            raise V3ReviewStructureError from None


def _fit_review_messages(
    candidate: V3EventCandidate,
    window: AnalysisWindow,
    *,
    run_id: str | None,
    prompt: str,
    overhead: int,
    character_budget: int,
) -> tuple[NormalizedMessage, ...]:
    """保留候选证据，并按与事件范围的距离填充可用复核上下文。"""

    all_messages = tuple(window.messages)
    complete_content = _review_user_content(
        candidate,
        window,
        all_messages,
        run_id=run_id,
    )
    if len(prompt) + len(complete_content) + overhead <= character_budget:
        return all_messages

    positions = {message.source_id: index for index, message in enumerate(all_messages)}
    mandatory_ids = {
        candidate.start_message_id,
        candidate.end_message_id,
        *candidate.evidence_ids,
    }
    selected_indexes = {
        index for message_id in mandatory_ids if (index := positions.get(message_id)) is not None
    }
    selected = tuple(
        message for index, message in enumerate(all_messages) if index in selected_indexes
    )
    required_characters = (
        len(prompt)
        + len(
            _review_user_content(
                candidate,
                window,
                selected,
                run_id=run_id,
            )
        )
        + overhead
    )
    if required_characters > character_budget:
        raise V3PromptBudgetExceededError(
            character_budget=character_budget,
            required_characters=required_characters,
        )

    lower_bound = min(selected_indexes, default=0)
    upper_bound = max(selected_indexes, default=0)
    optional_indexes = sorted(
        set(range(len(all_messages))) - selected_indexes,
        key=lambda index: (
            min(abs(index - lower_bound), abs(index - upper_bound)),
            index,
        ),
    )
    for index in optional_indexes:
        proposed_indexes = selected_indexes | {index}
        proposed = tuple(
            message
            for message_index, message in enumerate(all_messages)
            if message_index in proposed_indexes
        )
        proposed_size = (
            len(prompt)
            + len(
                _review_user_content(
                    candidate,
                    window,
                    proposed,
                    run_id=run_id,
                )
            )
            + overhead
        )
        if proposed_size <= character_budget:
            selected_indexes = proposed_indexes
            selected = proposed
    return selected


def _review_user_content(
    candidate: V3EventCandidate,
    window: AnalysisWindow,
    messages: Sequence[NormalizedMessage],
    *,
    run_id: str | None,
) -> str:
    return json.dumps(
        {
            "元数据": {
                "run_id": run_id,
                "window_id": window.window_id,
                "lane": candidate.lane,
            },
            "候选": candidate.model_dump(mode="json"),
            "消息": [_review_message_payload(message) for message in messages],
        },
        ensure_ascii=False,
    )


def _validate_candidate_integrity(candidate: V3EventCandidate) -> V3EventCandidate:
    """重新校验公开边界，并拒绝绕过校验产生的伪造或陈旧候选。"""

    payload = candidate.model_dump(mode="python")
    provided_key = payload.get("candidate_key")
    try:
        validated = V3EventCandidate.model_validate(payload)
    except (TypeError, ValueError, ValidationError):
        raise V3CandidateIntegrityError from None
    if not isinstance(provided_key, str) or provided_key != validated.candidate_key:
        raise V3CandidateIntegrityError
    return validated


def _message_payload(message: NormalizedMessage) -> dict[str, str]:
    return {
        "id": message.source_id,
        "角色": message.sender,
        "时间": message.timestamp.isoformat(),
        "内容": message.content,
    }


def _review_message_payload(message: NormalizedMessage) -> dict[str, str]:
    return {
        "id": message.source_id,
        "角色": message.sender,
        "时间": message.timestamp.isoformat(),
        "类型": message.kind.value,
        "内容": message.content,
    }


class _CandidateFieldError(ValueError):
    """只记录字段名，不保留无效原值。"""

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name
        super().__init__("候选字段无法规范化")


_CANDIDATE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "candidate_key",
        "lane",
        "type",
        "title",
        "event_status",
        "start_message_id",
        "end_message_id",
        "summary",
        "before_state",
        "after_state",
        "emotion_labels",
        "topic",
        "conflict_level",
        "event_significance",
        "relationship_impact",
        "model_confidence",
        "reason",
        "evidence_ids",
    }
)
_EXTRA_FIELD_LOCATION = "<extra_field>"
_MAX_DIAGNOSTIC_LOCATION_DEPTH = 3


def _safe_rejected_candidate(
    *,
    index: int,
    field_locations: tuple[tuple[str | int, ...], ...],
) -> V3RejectedCandidate:
    return V3RejectedCandidate(
        index=index,
        error_code="candidate_validation_failed",
        field_locations=_sanitize_field_locations(field_locations),
    )


def _sanitize_field_locations(
    field_locations: tuple[tuple[object, ...], ...],
) -> tuple[tuple[str | int, ...], ...]:
    """仅保留固定字段名、固定标记与有限深度的整数索引。"""

    sanitized_locations: list[tuple[str | int, ...]] = []
    for raw_location in field_locations:
        safe_parts: list[str | int] = []
        for part in raw_location[:_MAX_DIAGNOSTIC_LOCATION_DEPTH]:
            if type(part) is int:
                safe_parts.append(part)
            elif isinstance(part, str) and part in _CANDIDATE_DIAGNOSTIC_FIELDS:
                safe_parts.append(part)
            else:
                safe_parts.append(_EXTRA_FIELD_LOCATION)
        safe_location = tuple(safe_parts) or (_EXTRA_FIELD_LOCATION,)
        if safe_location not in sanitized_locations:
            sanitized_locations.append(safe_location)
    return tuple(sanitized_locations)


def _safe_validation_locations(
    error: ValidationError,
) -> tuple[tuple[str | int, ...], ...]:
    """仅提取字段路径，主动丢弃 Pydantic 的 input、消息与上下文。"""

    locations: list[tuple[str | int, ...]] = []
    for detail in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = tuple(part for part in detail["loc"] if isinstance(part, str | int))
        locations.append(location)
    return _sanitize_field_locations(tuple(locations))


def _normalize_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for field_name in (
        "event_significance",
        "relationship_impact",
        "model_confidence",
    ):
        try:
            normalized[field_name] = _normalize_unit_score(normalized.get(field_name))
        except (TypeError, ValueError):
            raise _CandidateFieldError(field_name) from None
    try:
        normalized["conflict_level"] = _normalize_conflict_level(normalized.get("conflict_level"))
    except (TypeError, ValueError):
        raise _CandidateFieldError("conflict_level") from None
    return normalized


def _normalize_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for field_name in (
        "evidence_alignment",
        "persistence",
        "type_support",
        "relationship_impact",
        "event_significance",
        "model_confidence",
    ):
        normalized[field_name] = _normalize_unit_score(normalized.get(field_name))
    return normalized


def _normalize_unit_score(value: object) -> float:
    """只接受有限 JSON 数值，并明确兼容 0..1 与 10 分制。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("分数必须是 JSON 数值")
    try:
        numeric = float(value)
    except OverflowError:
        raise ValueError("分数必须是有限数值") from None
    if not math.isfinite(numeric):
        raise ValueError("分数必须是有限数值")
    if 0 <= numeric <= 1:
        return numeric
    if 1 < numeric <= 10:
        return round(numeric / 10, 12)
    raise ValueError("分数超出可规范化范围")


def _normalize_conflict_level(value: object) -> int:
    """只接受 0..5 的整数或数值形式 x.0。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("冲突等级必须是 JSON 数值")
    try:
        numeric = float(value)
    except OverflowError:
        raise ValueError("冲突等级必须是有限数值") from None
    if not math.isfinite(numeric) or not numeric.is_integer() or not 0 <= numeric <= 5:
        raise ValueError("冲突等级必须是 0..5 的整数")
    return int(numeric)


def stable_v3_candidate_key(candidate: V3EventCandidate) -> str:
    """将完整规范化候选载荷纳入稳定键。"""

    identity = candidate.model_dump(mode="json", exclude={"candidate_key"})
    for field_name in ("type", "title", "summary", "topic", "reason"):
        identity[field_name] = _canonical_text(identity[field_name])
    for field_name in ("before_state", "after_state"):
        identity[field_name] = _canonical_optional_text(identity[field_name])
    for field_name in ("start_message_id", "end_message_id"):
        identity[field_name] = _canonical_identifier(identity[field_name])
    identity["emotion_labels"] = [_canonical_text(label) for label in identity["emotion_labels"]]
    identity["evidence_ids"] = [
        _canonical_identifier(message_id) for message_id in identity["evidence_ids"]
    ]
    serialized = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = sha256(serialized.encode()).hexdigest()
    return f"candidate-{candidate.lane}-{digest}"


def _canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())


def _canonical_optional_text(value: str | None) -> str | None:
    return _canonical_text(value) if value is not None else None


def _canonical_identifier(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()
