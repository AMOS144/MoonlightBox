from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from moonlightbox.events.v3_types import (
    RELATIONSHIP_EVENT_TYPES,
    SHARED_EXPERIENCE_EVENT_TYPES,
    EventLane,
    EventStatus,
    is_valid_event_type,
)


def _default_source_lanes() -> list[EventLane]:
    return ["relationship"]


def _validate_lane_contract(
    *,
    lane: EventLane,
    event_type: str,
    source_lanes: list[EventLane],
    allow_legacy_type: bool,
) -> None:
    if lane not in source_lanes:
        raise ValueError("source_lanes 必须包含主 lane")
    if is_valid_event_type(lane, event_type):
        return
    known_v3_types = RELATIONSHIP_EVENT_TYPES | SHARED_EXPERIENCE_EVENT_TYPES
    if allow_legacy_type and lane == "relationship" and event_type not in known_v3_types:
        return
    raise ValueError("事件类型与 lane 不匹配")


class ReviewedEvent(BaseModel):
    type: str
    start_message_id: str
    end_message_id: str
    before_state: str
    after_state: str
    emotion_labels: list[str]
    topic: str
    conflict_level: int = Field(ge=0, le=5)
    importance: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence_ids: list[str]


class V3ReviewedEvent(BaseModel):
    lane: EventLane
    event_status: EventStatus
    type: str
    title: str
    summary: str
    start_message_id: str
    end_message_id: str
    started_at: datetime | None
    ended_at: datetime | None
    source_lanes: list[EventLane] = Field(min_length=1)
    before_state: str | None
    after_state: str | None
    evidence_ids: list[str]

    @model_validator(mode="after")
    def validate_v3_contract(self) -> Self:
        _validate_lane_contract(
            lane=self.lane,
            event_type=self.type,
            source_lanes=self.source_lanes,
            allow_legacy_type=False,
        )
        if self.lane == "relationship" and not (
            self.before_state
            and self.before_state.strip()
            and self.after_state
            and self.after_state.strip()
        ):
            raise ValueError("关系事件必须同时提供非空 before_state 和 after_state")
        return self


class V2EventScoreComponents(BaseModel):
    state_change_strength: float
    persistence: float
    evidence_quality: float
    model_confidence: float


class EventScoreComponents(BaseModel):
    event_significance: float = Field(ge=0.0, le=1.0)
    relationship_impact: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    persistence: float = Field(ge=0.0, le=1.0)
    type_support: float = Field(ge=0.0, le=1.0)
    model_confidence: float = Field(ge=0.0, le=1.0)


class EventEvidenceSummary(BaseModel):
    message_id: str
    sender: str
    timestamp: datetime
    content: str


class EventNodeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    type: str
    start_message_id: str
    end_message_id: str
    before_state: str | None
    after_state: str | None
    emotion_labels: list[str] = Field(default_factory=list)
    topic: str = ""
    conflict_level: int = Field(default=0, ge=0, le=5)
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    lane: EventLane = "relationship"
    event_status: EventStatus = "occurred"
    title: str = ""
    summary: str = ""
    display_summary: str | None = None
    summary_status: str = "pending"
    summary_model: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    source_lanes: list[EventLane] = Field(
        default_factory=_default_source_lanes,
        min_length=1,
    )
    status: str
    created_at: datetime
    score_components: EventScoreComponents | V2EventScoreComponents | None = None
    evidence_summaries: list[EventEvidenceSummary] = Field(default_factory=list)
    analysis_version: str | None = None
    prompt_version: str | None = None
    model: str | None = None
    revision_number: int | None = None
    analysis_run_id: str | None = None

    @model_validator(mode="after")
    def validate_lane_states(self) -> Self:
        _validate_lane_contract(
            lane=self.lane,
            event_type=self.type,
            source_lanes=self.source_lanes,
            allow_legacy_type=True,
        )
        if self.lane == "relationship" and not (
            self.before_state
            and self.before_state.strip()
            and self.after_state
            and self.after_state.strip()
        ):
            raise ValueError("关系事件必须同时提供非空 before_state 和 after_state")
        return self


class EventRevisionRequest(BaseModel):
    changes: dict[str, object]
    reason: str = Field(min_length=1)


AnalysisRunStatus = Literal[
    "queued",
    "running",
    "interrupted",
    "failed",
    "succeeded",
    "cancelled",
]
EventCandidateStatus = Literal[
    "pending_review",
    "accepted",
    "rejected",
]


class EventCandidateUpsert(BaseModel):
    candidate_key: str = Field(min_length=1)
    raw_payload: dict[str, object]
    review_payload: dict[str, object] | None = None
    status: EventCandidateStatus = "pending_review"
    rejection_reason: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_rejection_reason(self) -> Self:
        if self.status == "rejected" and not (
            self.rejection_reason and self.rejection_reason.strip()
        ):
            raise ValueError("被拒绝的候选必须提供 rejection_reason")
        if self.status != "rejected" and self.rejection_reason is not None:
            raise ValueError("未被拒绝的候选不能提供 rejection_reason")
        return self


class EventCandidateRead(EventCandidateUpsert):
    model_config = ConfigDict(from_attributes=True)

    id: str
    run_id: str
    window_id: str
    created_at: datetime
    updated_at: datetime


class AnalysisRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    import_id: str
    analysis_version: str
    prompt_version: str
    model: str
    config: dict[str, object]
    config_fingerprint: str
    window_ids: list[str]
    window_manifest_fingerprint: str
    status: AnalysisRunStatus
    total_windows: int
    completed_windows: int
    checkpoint: int
    error_category: str | None
    error_message: str | None
    lease_owner: str | None
    lease_token: str | None
    lease_expires_at: datetime | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
