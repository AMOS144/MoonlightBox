from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class IdentityKernel(Base):
    __tablename__ = "identity_kernels"
    __table_args__ = (
        UniqueConstraint("model_version_id"),
        Index("ix_identity_kernels_project_id", "project_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE")
    )
    schema_version: Mapped[str] = mapped_column(String(32), default="v1")
    content: Mapped[dict[str, object]] = mapped_column(JSON)
    evidence_message_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    field_evidence: Mapped[dict[str, list[str]]] = mapped_column(JSON, default=dict)
    field_confidence: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    acceptance_report_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    locked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class BranchMemoryEpisode(Base):
    __tablename__ = "branch_memory_episodes"
    __table_args__ = (
        UniqueConstraint(
            "branch_id",
            "user_turn_id",
            "assistant_turn_id",
            name="ux_branch_memory_episode_turns",
        ),
        UniqueConstraint("episode_hash"),
        Index("ix_branch_memory_episodes_branch_status", "branch_id", "processing_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    user_turn_id: Mapped[str] = mapped_column(String(36))
    assistant_turn_id: Mapped[str] = mapped_column(String(36))
    user_content: Mapped[str] = mapped_column(Text)
    user_messages: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    assistant_bubbles: Mapped[list[dict[str, object]]] = mapped_column(JSON)
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT")
    )
    episode_hash: Mapped[str] = mapped_column(String(64))
    importance: Mapped[float] = mapped_column(Float, default=1.0)
    processing_status: Mapped[str] = mapped_column(String(32), default="pending")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class BranchMemoryItem(Base):
    __tablename__ = "branch_memory_items"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('fact', 'experience', 'self_narrative', 'belief', 'reflection')",
            name="ck_branch_memory_items_kind",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_branch_memory_items_confidence",
        ),
        CheckConstraint(
            "importance >= 1 AND importance <= 10",
            name="ck_branch_memory_items_importance",
        ),
        Index(
            "ix_branch_memory_items_branch_valid",
            "branch_id",
            "review_status",
            "valid_to",
        ),
        Index("ix_branch_memory_items_lineage", "branch_id", "lineage_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(String(128))
    predicate: Mapped[str] = mapped_column(String(128))
    object: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    importance: Mapped[float] = mapped_column(Float)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    supersedes_id: Mapped[str | None] = mapped_column(
        ForeignKey("branch_memory_items.id", ondelete="SET NULL"), nullable=True
    )
    source_episode_ids: Mapped[list[str]] = mapped_column(JSON)
    source_item_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    lineage_hash: Mapped[str] = mapped_column(String(64))
    review_status: Mapped[str] = mapped_column(String(32), default="pending")
    verification_status: Mapped[str] = mapped_column(String(32), default="inferred")
    claim_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stance: Mapped[str] = mapped_column(String(16), default="support")
    state_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("branch_state_versions.id", ondelete="SET NULL"), nullable=True
    )
    root_episode_hashes: Mapped[list[str]] = mapped_column(JSON, default=list)


class BranchBeliefEvidence(Base):
    __tablename__ = "branch_belief_evidence"
    __table_args__ = (
        UniqueConstraint(
            "belief_id",
            "episode_id",
            "stance",
            name="ux_branch_belief_episode_stance",
        ),
        CheckConstraint(
            "stance IN ('support', 'oppose')",
            name="ck_branch_belief_evidence_stance",
        ),
        Index("ix_branch_belief_evidence_branch", "branch_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    belief_id: Mapped[str] = mapped_column(ForeignKey("branch_memory_items.id", ondelete="CASCADE"))
    episode_id: Mapped[str] = mapped_column(
        ForeignKey("branch_memory_episodes.id", ondelete="CASCADE")
    )
    stance: Mapped[str] = mapped_column(String(16))
    source_role: Mapped[str] = mapped_column(String(32))
    evidence_type: Mapped[str] = mapped_column(String(64))
    weight: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class BranchStateVersion(Base):
    __tablename__ = "branch_state_versions"
    __table_args__ = (
        UniqueConstraint("branch_id", "version"),
        Index("ix_branch_state_versions_current", "branch_id", "is_current"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    previous_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("branch_state_versions.id", ondelete="SET NULL"), nullable=True
    )
    persona_state: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    relationship_state: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    user_model: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    emotional_tendency: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    active_belief_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    contested_belief_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    current_goals: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    current_concerns: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    memory_cutoff_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rollback_of_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("branch_state_versions.id", ondelete="SET NULL"), nullable=True
    )
    reason: Mapped[str] = mapped_column(Text)
    source_episode_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    field_evidence: Mapped[dict[str, list[str]]] = mapped_column(JSON, default=dict)
    field_confidence: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    rolled_back_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    baseline_manifest_id: Mapped[str | None] = mapped_column(
        ForeignKey("branch_baseline_manifests.id", ondelete="SET NULL"),
        nullable=True,
    )
    baseline_state_id: Mapped[str | None] = mapped_column(
        ForeignKey("branch_baseline_states.id", ondelete="SET NULL"),
        nullable=True,
    )


class BranchReflectionRun(Base):
    __tablename__ = "branch_reflection_runs"
    __table_args__ = (Index("ix_branch_reflection_runs_branch_status", "branch_id", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    trigger_episode_id: Mapped[str] = mapped_column(
        ForeignKey("branch_memory_episodes.id", ondelete="CASCADE")
    )
    input_item_ids: Mapped[list[str]] = mapped_column(JSON)
    input_importance_sum: Mapped[float] = mapped_column(Float)
    local_proposal: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    review_result: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    output_item_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
