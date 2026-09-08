from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MentalStateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    version: int
    state: dict[str, object]
    evidence: dict[str, object]
    model_version_id: str | None
    model_protocol_version: str
    created_at: datetime


class AgentGoalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    goal_type: str
    content: str
    priority: float
    evidence: dict[str, object]
    review_at: datetime | None
    created_at: datetime


class AgentIntentionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    intention_type: str
    content: str
    goal_id: str | None
    trigger_event_id: str
    expression_plan: dict[str, object]
    evidence: dict[str, object]
    created_at: datetime


class AgentWakeupRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    wake_at: datetime
    reason: str
    goal_id: str | None
    event_id: str | None
    evidence: dict[str, object]
    created_at: datetime


class CognitionStateRead(BaseModel):
    mental_state: MentalStateRead | None
    active_goals: list[AgentGoalRead]
    active_intentions: list[AgentIntentionRead]
    next_wakeup: AgentWakeupRead | None


class PrivateCognitionNoteRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    trigger_event_id: str
    content: str
    subjective_feelings: dict[str, object]
    attention_target: dict[str, object]
    desired_actions: list[dict[str, object]]
    memory_ids: list[str]
    evidence: dict[str, object]
    confidence: float
    model_version_id: str
    model_protocol_version: str
    created_at: datetime


class CognitiveCycleRead(BaseModel):
    id: str
    trigger_event_id: str
    input_cutoff_at: datetime
    starting_state_version_id: str | None
    memory_ids: list[str]
    goal_ids: list[str]
    structured_changes: dict[str, object]
    final_decision: dict[str, object]
    evidence: dict[str, object]
    status: str
    model_version_id: str
    model_protocol_version: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    private_note: PrivateCognitionNoteRead | None


class PerceptionEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    event_type: str
    occurred_at: datetime
    source: str
    confidence: float
    idempotency_key: str
    visible_through: datetime
    evidence: dict[str, object]
    model_protocol_version: str
    created_at: datetime


class ReplayObservationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["historical_replay", "labeled_behavior"]
    cutoff: datetime
    expected_express: bool
    predicted_express: bool
    extraction_succeeded: bool
    fact_safe: bool
    latency_ms: float = Field(ge=0)

    @field_validator("cutoff")
    @classmethod
    def require_aware_cutoff(cls, cutoff: datetime) -> datetime:
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("cutoff 必须包含时区")
        return cutoff


class AcceptanceEvaluateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version_id: str = Field(min_length=1)
    direct_lora_p95: float = Field(gt=0)
    observations: list[ReplayObservationInput]


class SubjectAgentActivateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version_id: str = Field(min_length=1)


class SubjectAgentAcceptanceReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    branch_id: str
    model_version_id: str
    sample_count: int
    structure_extraction_success_rate: float
    fact_safety_rate: float
    expression_decision_accuracy: float
    p95_cognition_latency_ms: float
    direct_lora_p95: float
    passed: bool
    failure_reasons: list[str]
    evidence: dict[str, object]
    created_at: datetime


class SubjectAgentModeRead(BaseModel):
    id: str
    project_id: str
    model_version_id: str
    subject_agent_mode: Literal["shadow", "preview", "active"]
