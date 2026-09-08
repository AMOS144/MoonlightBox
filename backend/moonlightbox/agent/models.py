from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return str(uuid4())


class PerceptionEvent(Base):
    """分支内不可变的主体感知事件。"""

    __tablename__ = "perception_events"
    __table_args__ = (
        ForeignKeyConstraint(
            ["branch_id", "project_id"],
            ["branches.id", "branches.project_id"],
            ondelete="CASCADE",
            name="fk_perception_events_branch_project",
        ),
        UniqueConstraint(
            "branch_id",
            "idempotency_key",
            name="ux_perception_events_branch_idempotency",
        ),
        UniqueConstraint("id", "branch_id", name="ux_perception_events_id_branch"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_perception_events_confidence",
        ),
        Index("ix_perception_events_branch_occurred", "branch_id", "occurred_at"),
        Index("ix_perception_events_branch_type", "branch_id", "event_type"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String(36))
    branch_id: Mapped[str] = mapped_column(String(36))
    event_type: Mapped[str] = mapped_column(String(64))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(128))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    idempotency_key: Mapped[str] = mapped_column(String(255))
    visible_through: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    evidence: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    model_protocol_version: Mapped[str] = mapped_column(
        String(64), default="subject-cognition-v1"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AgentGoal(Base):
    """主体在单一分支中的长期或临时目标。"""

    __tablename__ = "agent_goals"
    __table_args__ = (
        ForeignKeyConstraint(
            ["branch_id", "project_id"],
            ["branches.id", "branches.project_id"],
            ondelete="CASCADE",
            name="fk_agent_goals_branch_project",
        ),
        UniqueConstraint("id", "branch_id", name="ux_agent_goals_id_branch"),
        CheckConstraint(
            "status IN ('active', 'suspended', 'completed', 'abandoned', 'conflicted')",
            name="ck_agent_goals_status",
        ),
        Index("ix_agent_goals_branch_status", "branch_id", "status"),
        Index("ix_agent_goals_review", "status", "review_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String(36))
    branch_id: Mapped[str] = mapped_column(String(36))
    goal_type: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    supporting_evidence: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    opposing_evidence: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="active")
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=True
    )
    model_protocol_version: Mapped[str] = mapped_column(
        String(64), default="subject-cognition-v1"
    )
    review_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class PrivateCognitionNote(Base):
    """人格模型生成且不向聊天用户展示的简短认知记录。"""

    __tablename__ = "private_cognition_notes"
    __table_args__ = (
        ForeignKeyConstraint(
            ["branch_id", "project_id"],
            ["branches.id", "branches.project_id"],
            ondelete="CASCADE",
            name="fk_private_cognition_notes_branch_project",
        ),
        ForeignKeyConstraint(
            ["trigger_event_id", "branch_id"],
            ["perception_events.id", "perception_events.branch_id"],
            ondelete="RESTRICT",
            name="fk_private_cognition_notes_trigger_branch",
        ),
        UniqueConstraint("id", "branch_id", name="ux_private_cognition_notes_id_branch"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_private_cognition_notes_confidence",
        ),
        Index("ix_private_cognition_notes_branch_created", "branch_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String(36))
    branch_id: Mapped[str] = mapped_column(String(36))
    trigger_event_id: Mapped[str] = mapped_column(String(36))
    content: Mapped[str] = mapped_column(Text)
    subjective_feelings: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    attention_target: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    desired_actions: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    memory_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT")
    )
    model_protocol_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class MentalStateVersion(Base):
    """主体心理状态的不可变版本快照。"""

    __tablename__ = "mental_state_versions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["branch_id", "project_id"],
            ["branches.id", "branches.project_id"],
            ondelete="CASCADE",
            name="fk_mental_state_versions_branch_project",
        ),
        ForeignKeyConstraint(
            ["previous_version_id", "branch_id"],
            ["mental_state_versions.id", "mental_state_versions.branch_id"],
            ondelete="RESTRICT",
            name="fk_mental_state_versions_previous_branch",
        ),
        ForeignKeyConstraint(
            ["source_cycle_id", "branch_id"],
            ["cognitive_cycles.id", "cognitive_cycles.branch_id"],
            ondelete="RESTRICT",
            name="fk_mental_state_versions_cycle_branch",
        ),
        UniqueConstraint(
            "branch_id", "version", name="ux_mental_state_versions_branch_version"
        ),
        UniqueConstraint("id", "branch_id", name="ux_mental_state_versions_id_branch"),
        Index(
            "ux_mental_state_versions_current_branch",
            "branch_id",
            unique=True,
            sqlite_where=text("is_current = 1"),
            postgresql_where=text("is_current"),
        ),
        Index("ix_mental_state_versions_branch_created", "branch_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String(36))
    branch_id: Mapped[str] = mapped_column(String(36))
    version: Mapped[int] = mapped_column(Integer)
    state: Mapped[dict[str, object]] = mapped_column(JSON)
    previous_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source_cycle_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    evidence: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=True
    )
    model_protocol_version: Mapped[str] = mapped_column(
        String(64), default="subject-cognition-v1"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class CognitiveCycle(Base):
    """一次可审计、可作废的主体认知周期。"""

    __tablename__ = "cognitive_cycles"
    __table_args__ = (
        ForeignKeyConstraint(
            ["branch_id", "project_id"],
            ["branches.id", "branches.project_id"],
            ondelete="CASCADE",
            name="fk_cognitive_cycles_branch_project",
        ),
        ForeignKeyConstraint(
            ["trigger_event_id", "branch_id"],
            ["perception_events.id", "perception_events.branch_id"],
            ondelete="RESTRICT",
            name="fk_cognitive_cycles_trigger_branch",
        ),
        ForeignKeyConstraint(
            ["starting_state_version_id", "branch_id"],
            ["mental_state_versions.id", "mental_state_versions.branch_id"],
            ondelete="RESTRICT",
            name="fk_cognitive_cycles_start_state_branch",
        ),
        ForeignKeyConstraint(
            ["private_note_id", "branch_id"],
            ["private_cognition_notes.id", "private_cognition_notes.branch_id"],
            ondelete="RESTRICT",
            name="fk_cognitive_cycles_note_branch",
        ),
        UniqueConstraint(
            "branch_id",
            "trigger_event_id",
            name="ux_cognitive_cycles_branch_trigger",
        ),
        UniqueConstraint("id", "branch_id", name="ux_cognitive_cycles_id_branch"),
        CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'invalidated', 'failed')",
            name="ck_cognitive_cycles_status",
        ),
        Index("ix_cognitive_cycles_branch_status", "branch_id", "status"),
        Index("ix_cognitive_cycles_trigger", "trigger_event_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String(36))
    branch_id: Mapped[str] = mapped_column(String(36))
    trigger_event_id: Mapped[str] = mapped_column(String(36))
    input_cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    starting_state_version_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    memory_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    goal_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    private_note_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    structured_changes: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    final_decision: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    evidence: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT")
    )
    model_protocol_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class AgentIntention(Base):
    """主体近期准备采取或回避的行为意图。"""

    __tablename__ = "agent_intentions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["branch_id", "project_id"],
            ["branches.id", "branches.project_id"],
            ondelete="CASCADE",
            name="fk_agent_intentions_branch_project",
        ),
        ForeignKeyConstraint(
            ["goal_id", "branch_id"],
            ["agent_goals.id", "agent_goals.branch_id"],
            ondelete="RESTRICT",
            name="fk_agent_intentions_goal_branch",
        ),
        ForeignKeyConstraint(
            ["trigger_event_id", "branch_id"],
            ["perception_events.id", "perception_events.branch_id"],
            ondelete="RESTRICT",
            name="fk_agent_intentions_event_branch",
        ),
        CheckConstraint(
            "status IN ('active', 'suspended', 'fulfilled', 'cancelled')",
            name="ck_agent_intentions_status",
        ),
        Index("ix_agent_intentions_branch_status", "branch_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String(36))
    branch_id: Mapped[str] = mapped_column(String(36))
    intention_type: Mapped[str] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text)
    goal_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    trigger_event_id: Mapped[str] = mapped_column(String(36))
    expression_plan: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    evidence: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="active")
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=True
    )
    model_protocol_version: Mapped[str] = mapped_column(
        String(64), default="subject-cognition-v1"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class AgentWakeup(Base):
    """主体按心理因果安排的下一次动态唤醒。"""

    __tablename__ = "agent_wakeups"
    __table_args__ = (
        ForeignKeyConstraint(
            ["branch_id", "project_id"],
            ["branches.id", "branches.project_id"],
            ondelete="CASCADE",
            name="fk_agent_wakeups_branch_project",
        ),
        ForeignKeyConstraint(
            ["goal_id", "branch_id"],
            ["agent_goals.id", "agent_goals.branch_id"],
            ondelete="RESTRICT",
            name="fk_agent_wakeups_goal_branch",
        ),
        ForeignKeyConstraint(
            ["event_id", "branch_id"],
            ["perception_events.id", "perception_events.branch_id"],
            ondelete="RESTRICT",
            name="fk_agent_wakeups_event_branch",
        ),
        UniqueConstraint(
            "branch_id", "idempotency_key", name="ux_agent_wakeups_branch_idempotency"
        ),
        CheckConstraint(
            "status IN ('scheduled', 'claimed', 'completed', 'cancelled', 'expired')",
            name="ck_agent_wakeups_status",
        ),
        Index("ix_agent_wakeups_due", "status", "wake_at"),
        Index("ix_agent_wakeups_branch_status", "branch_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String(36))
    branch_id: Mapped[str] = mapped_column(String(36))
    wake_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(Text)
    goal_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(255))
    evidence: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="scheduled")
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=True
    )
    model_protocol_version: Mapped[str] = mapped_column(
        String(64), default="subject-cognition-v1"
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class SubjectAgentAcceptanceReport(Base):
    """分支与模型范围内的主体认知 Agent 影子验收报告。"""

    __tablename__ = "subject_agent_acceptance_reports"
    __table_args__ = (
        ForeignKeyConstraint(
            ["branch_id", "project_id"],
            ["branches.id", "branches.project_id"],
            ondelete="CASCADE",
            name="fk_subject_agent_acceptance_reports_branch_project",
        ),
        CheckConstraint(
            "sample_count >= 0",
            name="ck_subject_agent_acceptance_reports_sample_count",
        ),
        Index(
            "ix_subject_agent_acceptance_reports_latest",
            "project_id",
            "branch_id",
            "model_version_id",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(String(36))
    branch_id: Mapped[str] = mapped_column(String(36))
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT")
    )
    sample_count: Mapped[int] = mapped_column(Integer)
    structure_extraction_success_rate: Mapped[float] = mapped_column(Float)
    fact_safety_rate: Mapped[float] = mapped_column(Float)
    expression_decision_accuracy: Mapped[float] = mapped_column(Float)
    p95_cognition_latency_ms: Mapped[float] = mapped_column(Float)
    direct_lora_p95: Mapped[float] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean)
    failure_reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
