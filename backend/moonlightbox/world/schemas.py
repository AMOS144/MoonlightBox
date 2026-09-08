"""人物世界 API 的请求和响应结构。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceStatus = Literal["direct", "summarized", "inferred", "superseded"]
AssertionKind = Literal[
    "self_fact", "other_person_fact", "general_rule", "plan", "desire",
    "report", "joke", "question", "unknown",
]
Referent = Literal[
    "target_person", "user", "third_person", "organization", "place",
    "general_topic", "unknown",
]
TemporalStatus = Literal["current", "past", "planned", "recurring", "one_off", "timeless", "unknown"]


class SourcedStatement(BaseModel):
    # Profile JSON is persisted through SQLAlchemy's JSON column, so datetime
    # values are serialized to ISO-8601 strings on disk and parsed back here.
    # Keep the payload shape strict while allowing this normal JSON round trip.
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1000)
    source_status: SourceStatus
    source_document_ids: list[str] = Field(default_factory=list, max_length=40)
    assertion_kind: AssertionKind = "unknown"
    referent: Referent = "unknown"
    temporal_status: TemporalStatus = "unknown"
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_message_ids: list[str] = Field(default_factory=list, max_length=80)
    evidence_quotes: list[str] = Field(default_factory=list, max_length=8)
    qualifiers: list[str] = Field(default_factory=list, max_length=20)


class IdentityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    names: list[SourcedStatement] = Field(default_factory=list)
    aliases: list[SourcedStatement] = Field(default_factory=list)
    self_descriptions: list[SourcedStatement] = Field(default_factory=list)
    roles: list[SourcedStatement] = Field(default_factory=list)


class RoutineProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    workdays: list[SourcedStatement] = Field(default_factory=list)
    weekends: list[SourcedStatement] = Field(default_factory=list)
    other_patterns: list[SourcedStatement] = Field(default_factory=list)


class RelationshipProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    overview: list[SourcedStatement] = Field(default_factory=list)
    changes_over_time: list[SourcedStatement] = Field(default_factory=list)


class WorldProfileDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    identity: IdentityProfile
    work_and_education: list[SourcedStatement]
    places: list[SourcedStatement]
    social_relationships: list[SourcedStatement]
    preferences: list[SourcedStatement]
    recurring_activities: list[SourcedStatement]
    routine_summary: RoutineProfile
    life_phases: list[SourcedStatement]
    relationship_with_user: RelationshipProfile
    important_events: list[SourcedStatement]
    unresolved_candidates: list[SourcedStatement]


class WorldGraphVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    message_count: int
    bundle_count: int
    lightrag_version: str | None
    embedding_model: str | None
    embedding_dimension: int | None
    extraction_model: str | None
    chunking_strategy: str | None
    chunk_token_size: int | None
    chunk_overlap_token_size: int | None
    entity_prompt_version: str | None
    compiler_version: str
    error_code: str | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None
    build_progress: "WorldBuildProgressRead | None" = None


class WorldBuildProgressRead(BaseModel):
    """当前人物世界任务的可展示进度。"""

    job_id: str
    status: str
    stage: str | None = None
    progress: float = Field(ge=0, le=1)
    completed_questions: int | None = None
    question_count: int | None = None
    indexed_bundles: int | None = None
    bundle_count: int | None = None


class PersonWorldProfileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    subject_person_id: str
    identity: IdentityProfile
    work_and_education: list[SourcedStatement]
    places: list[SourcedStatement]
    social_relationships: list[SourcedStatement]
    preferences: list[SourcedStatement]
    recurring_activities: list[SourcedStatement]
    routine_summary: RoutineProfile
    life_phases: list[SourcedStatement]
    relationship_with_user: RelationshipProfile
    important_events: list[SourcedStatement]
    unresolved_candidates: list[SourcedStatement]
    source_message_ids: list[str]
    compiler_version: str
    created_at: datetime
    graph: WorldGraphVersionRead


class WorldBuildRead(BaseModel):
    job_id: str
    status: str


class WorldSourceMessageRead(BaseModel):
    id: str
    timestamp: datetime
    participant: str
    role: str
    kind: str
    content: str
    is_carry_in: bool


class WorldSourceRead(BaseModel):
    document_id: str
    started_at: datetime
    ended_at: datetime
    messages: list[WorldSourceMessageRead]


class AliasEvidenceRead(WorldSourceMessageRead):
    """别名 Agent 原文定位结果，补充所属 Bundle 文档。"""

    document_id: str | None = None


class EntityMergeProposalCreate(BaseModel):
    source_entities: list[str] = Field(min_length=1, max_length=20)
    target_entity: str = Field(min_length=1, max_length=500)
    reason: str = ""
    evidence: list[dict[str, object]] = Field(default_factory=list)


class EntityMergeProposalRead(EntityMergeProposalCreate):
    id: str
    project_id: str
    graph_version_id: str
    decision: str
    review_note: str | None
    merge_result: dict[str, object] | None
    created_at: datetime
    reviewed_at: datetime | None
