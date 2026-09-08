from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion

AUTHORITATIVE_SUBJECT_AGENT_MODES = frozenset({"active", "preview"})


def uses_authoritative_cognition(mode: str) -> bool:
    """Return whether public expression must wait for private cognition."""

    return mode in AUTHORITATIVE_SUBJECT_AGENT_MODES


class Branch(Base):
    __tablename__ = "branches"
    __table_args__ = (
        UniqueConstraint("id", "project_id", name="ux_branches_id_project_id"),
        CheckConstraint(
            "subject_agent_mode IN ('shadow', 'preview', 'active')",
            name="ck_branches_subject_agent_mode",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey(f"{Project.__tablename__}.id", ondelete="CASCADE"), index=True
    )
    origin_event_id: Mapped[str] = mapped_column(ForeignKey(f"{EventNode.__tablename__}.id"))
    model_version_id: Mapped[str] = mapped_column(ForeignKey(f"{ModelVersion.__tablename__}.id"))
    title: Mapped[str] = mapped_column(String(255))
    origin_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    state_snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    lifecycle_status: Mapped[str] = mapped_column(String(16), default="active")
    replacement_branch_id: Mapped[str | None] = mapped_column(
        ForeignKey("branches.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    generation_policy_version: Mapped[str] = mapped_column(
        String(32),
        default="hybrid-v1",
    )
    origin_import_id: Mapped[str | None] = mapped_column(
        ForeignKey("import_sources.id", ondelete="RESTRICT"), nullable=True
    )
    origin_boundary_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("messages.id", ondelete="RESTRICT"), nullable=True
    )
    baseline_manifest_id: Mapped[str | None] = mapped_column(
        ForeignKey("branch_baseline_manifests.id", ondelete="SET NULL"),
        nullable=True,
    )
    baseline_job_id: Mapped[str | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    baseline_status: Mapped[str] = mapped_column(String(16), default="ready")
    baseline_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    baseline_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    baseline_ready_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    subject_agent_mode: Mapped[str] = mapped_column(
        String(16),
        default="shadow",
        server_default="shadow",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class BranchMessage(Base):
    __tablename__ = "branch_messages"
    __table_args__ = (
        UniqueConstraint("branch_id", "sequence"),
        UniqueConstraint("branch_id", "turn_id", "bubble_index"),
        UniqueConstraint("branch_id", "client_message_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(16), default="text")
    media_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("media_assets.id", ondelete="SET NULL"), nullable=True
    )
    turn_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()), index=True)
    bubble_index: Mapped[int] = mapped_column(Integer, default=0)
    delay_ms: Mapped[int] = mapped_column(Integer, default=0)
    generation_status: Mapped[str] = mapped_column(String(16), default="completed", index=True)
    generation_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    client_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    observed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expression_plan_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversation_expression_plans.id", ondelete="SET NULL"),
        nullable=True,
    )
    actor_intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_proactive: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
