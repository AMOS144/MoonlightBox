from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BranchCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    origin_event_id: str = Field(min_length=1)
    model_version_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    origin_time: datetime


class BranchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    origin_event_id: str
    model_version_id: str
    title: str
    origin_time: datetime
    state_snapshot: dict[str, object]
    lifecycle_status: str
    replacement_branch_id: str | None
    generation_policy_version: str
    origin_import_id: str | None
    origin_boundary_message_id: str | None
    baseline_manifest_id: str | None
    baseline_job_id: str | None
    baseline_status: str
    baseline_error_code: str | None
    baseline_error_message: str | None
    baseline_ready_at: datetime | None
    subject_agent_mode: str
    created_at: datetime


class BranchPreparationRead(BaseModel):
    branch_id: str
    status: str
    progress: float
    stage: str
    message_count: int
    event_count: int
    error_code: str | None
    error_message: str | None


class BranchHistoryMessageRead(BaseModel):
    id: str
    source_id: str
    role: str
    content: str
    type: str
    media_asset_id: str | None
    timestamp: datetime


class BranchHistoryPageRead(BaseModel):
    items: list[BranchHistoryMessageRead]
    next_cursor: str | None
    has_more: bool
    manifest_id: str


class BranchMessageCreate(BaseModel):
    content: str = Field(min_length=1)
    client_message_id: str | None = Field(default=None, min_length=1, max_length=64)


class BranchMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    branch_id: str
    sequence: int
    role: str
    content: str
    type: str
    media_asset_id: str | None
    turn_id: str
    bubble_index: int
    delay_ms: int
    generation_status: str
    generation_metadata: dict[str, object]
    client_message_id: str | None
    observed_at: datetime | None
    expression_plan_id: str | None
    actor_intent: str | None
    is_proactive: bool
    created_at: datetime


class BranchReplyTurnRead(BaseModel):
    user_turn_id: str
    assistant_turn_id: str
    bubbles: list[BranchMessageRead]


class ConversationActorStateRead(BaseModel):
    status: str
    typing: bool
    observed_message_sequence: int
    version: int


class ConversationTypingUpdate(BaseModel):
    typing: bool


SituationalSlot = Literal[
    "activity",
    "location",
    "availability",
    "physical_state",
    "emotion",
    "environment",
    "current_facts",
]


class SituationalStateWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[SituationalSlot, str] = Field(min_length=1)
    valid_until: datetime
    idempotency_key: str = Field(min_length=1, max_length=128)
    confidence: float = Field(default=1.0, ge=0, le=1)


class SituationalStateRead(BaseModel):
    event_id: str
    state: dict[str, object]
