from datetime import datetime

from pydantic import BaseModel, ConfigDict

from moonlightbox.branches.continuity_migration import ProjectMigrationReport


class IdentityKernelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_version_id: str
    schema_version: str
    content: dict[str, object]
    evidence_message_ids: list[str]
    field_evidence: dict[str, list[str]]
    field_confidence: dict[str, float]
    acceptance_report_id: str | None
    created_at: datetime
    locked_at: datetime


class MemoryEpisodeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_turn_id: str
    user_messages: list[dict[str, object]]
    assistant_turn_id: str
    user_content: str
    assistant_bubbles: list[dict[str, object]]
    importance: float
    processing_status: str
    started_at: datetime
    ended_at: datetime


class MemoryItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    content: str
    subject: str
    predicate: str
    object: str
    confidence: float
    importance: float
    valid_from: datetime
    valid_to: datetime | None
    source_episode_ids: list[str]
    source_item_ids: list[str]
    review_status: str
    verification_status: str
    claim_key: str | None
    stance: str
    state_version_id: str | None
    root_episode_hashes: list[str]


class BranchStateVersionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    version: int
    previous_version_id: str | None
    persona_state: dict[str, object]
    relationship_state: dict[str, object]
    user_model: dict[str, object]
    emotional_tendency: dict[str, object]
    active_belief_ids: list[str]
    contested_belief_ids: list[str]
    current_goals: dict[str, object]
    current_concerns: dict[str, object]
    memory_cutoff_version: int | None
    rollback_of_version_id: str | None
    reason: str
    source_episode_ids: list[str]
    field_evidence: dict[str, list[str]]
    field_confidence: dict[str, float]
    is_current: bool
    created_at: datetime
    rolled_back_at: datetime | None


class ReflectionRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    trigger_episode_id: str
    input_item_ids: list[str]
    input_importance_sum: float
    output_item_ids: list[str]
    status: str
    attempt_count: int
    created_at: datetime
    completed_at: datetime | None


class JobStatusRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kind: str
    status: str
    checkpoint: dict[str, object] | None
    error_code: str | None
    error_message: str | None


class LongitudinalGrowthHealthRead(BaseModel):
    status: str
    processed_episode_count: int
    observation_span_days: float
    state_version_count: int
    approved_memory_count: int
    approved_reflection_count: int
    successful_cognitive_cycle_count: int
    failed_cognitive_cycle_count: int
    evidence_coverage_rate: float
    duplicate_lineage_count: int
    unmet_requirements: list[str]


class BranchMemoryOverviewRead(BaseModel):
    identity_kernel: IdentityKernelRead
    current_state: BranchStateVersionRead
    active_beliefs: list[MemoryItemRead]
    competing_beliefs: list[MemoryItemRead]
    recent_reflections: list[MemoryItemRead]
    pending_jobs: int
    failed_jobs: int
    evolution_frozen: bool
    growth_health: LongitudinalGrowthHealthRead


class MigrationRead(ProjectMigrationReport):
    pass
