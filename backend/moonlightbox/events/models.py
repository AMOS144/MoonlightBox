from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
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
    event,
)
from sqlalchemy.orm import Mapped, Session, mapped_column

from moonlightbox.db import Base
from moonlightbox.events.config import (
    ensure_config_integrity,
    ensure_window_manifest_integrity,
)
from moonlightbox.events.v3_types import EventLane, EventStatus
from moonlightbox.projects.models import Project


@dataclass(frozen=True)
class Episode:
    message_ids: list[str]
    started_at: datetime
    ended_at: datetime
    boundary_reasons: list[str]


def _uuid() -> str:
    return str(uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _default_source_lanes() -> list[EventLane]:
    return ["relationship"]


class EventNode(Base):
    __tablename__ = "event_nodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey(f"{Project.__tablename__}.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String(64))
    lane: Mapped[EventLane] = mapped_column(String(32), default="relationship")
    event_status: Mapped[EventStatus] = mapped_column(String(32), default="occurred")
    title: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    display_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_status: Mapped[str] = mapped_column(String(32), default="pending")
    summary_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    start_message_id: Mapped[str] = mapped_column(String(255))
    end_message_id: Mapped[str] = mapped_column(String(255))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_lanes: Mapped[list[EventLane]] = mapped_column(JSON, default=_default_source_lanes)
    before_state: Mapped[str | None] = mapped_column(Text, nullable=True)
    after_state: Mapped[str | None] = mapped_column(Text, nullable=True)
    emotion_labels: Mapped[list[str]] = mapped_column(JSON)
    topic: Mapped[str] = mapped_column(Text)
    conflict_level: Mapped[int] = mapped_column(Integer)
    importance: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    evidence_ids: Mapped[list[str]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AnalysisRevision(Base):
    __tablename__ = "analysis_revisions"
    __table_args__ = (UniqueConstraint("event_id", "revision_number"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    event_id: Mapped[str] = mapped_column(
        ForeignKey("event_nodes.id", ondelete="CASCADE"), index=True
    )
    revision_number: Mapped[int] = mapped_column(Integer)
    snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    action_reason: Mapped[str] = mapped_column(Text)
    analysis_version: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(128))
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("event_candidates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["import_id", "project_id"],
            ["import_sources.id", "import_sources.project_id"],
            ondelete="CASCADE",
            name="fk_analysis_runs_import_project",
        ),
        UniqueConstraint(
            "import_id",
            "analysis_version",
            "prompt_version",
            "model",
            "config_fingerprint",
            name="uq_analysis_runs_identity",
        ),
        CheckConstraint("total_windows >= 0", name="ck_analysis_runs_total_windows"),
        CheckConstraint(
            "json_type(window_ids) = 'array' AND json_array_length(window_ids) = total_windows",
            name="ck_analysis_runs_window_manifest",
        ),
        CheckConstraint(
            "completed_windows >= 0 AND completed_windows <= total_windows",
            name="ck_analysis_runs_completed_windows",
        ),
        CheckConstraint(
            "checkpoint >= 0 AND checkpoint <= total_windows",
            name="ck_analysis_runs_checkpoint",
        ),
        CheckConstraint(
            "completed_windows = checkpoint",
            name="ck_analysis_runs_progress_match",
        ),
        CheckConstraint(
            "status IN ('queued', 'running', 'interrupted', 'failed', 'succeeded', 'cancelled')",
            name="ck_analysis_runs_status",
        ),
        CheckConstraint(
            "(status IN ('succeeded', 'failed', 'cancelled') "
            "AND completed_at IS NOT NULL) OR "
            "(status IN ('queued', 'running', 'interrupted') "
            "AND completed_at IS NULL)",
            name="ck_analysis_runs_completion_time",
        ),
        CheckConstraint(
            "status != 'succeeded' OR completed_windows = total_windows",
            name="ck_analysis_runs_succeeded_progress",
        ),
        CheckConstraint(
            "(status IN ('failed', 'interrupted') "
            "AND error_category IS NOT NULL "
            "AND length(trim(error_category)) > 0 "
            "AND error_message IS NOT NULL "
            "AND length(trim(error_message)) > 0) OR "
            "(status NOT IN ('failed', 'interrupted') "
            "AND error_category IS NULL AND error_message IS NULL)",
            name="ck_analysis_runs_error_fields",
        ),
        CheckConstraint(
            "(lease_owner IS NULL AND lease_token IS NULL "
            "AND lease_expires_at IS NULL) OR "
            "(lease_owner IS NOT NULL AND length(trim(lease_owner)) > 0 "
            "AND lease_token IS NOT NULL AND length(trim(lease_token)) > 0 "
            "AND lease_expires_at IS NOT NULL)",
            name="ck_analysis_runs_lease_fields",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    import_id: Mapped[str] = mapped_column(String(36), index=True)
    analysis_version: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(128))
    model: Mapped[str] = mapped_column(String(255))
    config: Mapped[dict[str, object]] = mapped_column(JSON)
    config_fingerprint: Mapped[str] = mapped_column(String(64))
    window_ids: Mapped[list[str]] = mapped_column(JSON)
    window_manifest_fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    total_windows: Mapped[int] = mapped_column(Integer)
    completed_windows: Mapped[int] = mapped_column(Integer, default=0)
    checkpoint: Mapped[int] = mapped_column(Integer, default=0)
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(255), nullable=True)
    lease_token: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EventCandidate(Base):
    __tablename__ = "event_candidates"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "window_id",
            "candidate_key",
            name="uq_event_candidates_window_key",
        ),
        Index("ix_event_candidates_run_status", "run_id", "status"),
        CheckConstraint(
            "status IN ('pending_review', 'accepted', 'rejected')",
            name="ck_event_candidates_status",
        ),
        CheckConstraint(
            "(status = 'rejected' AND rejection_reason IS NOT NULL "
            "AND length(trim(rejection_reason)) > 0) OR "
            "(status != 'rejected' AND rejection_reason IS NULL)",
            name="ck_event_candidates_rejection_reason",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    window_id: Mapped[str] = mapped_column(String(255))
    candidate_key: Mapped[str] = mapped_column(String(255))
    raw_payload: Mapped[dict[str, object]] = mapped_column(JSON)
    review_payload: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending_review")
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    scores: Mapped[dict[str, float]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )


def _validate_analysis_configs(session: Session) -> None:
    objects = set(session.identity_map.values()) | set(session.new)
    for instance in objects:
        if isinstance(instance, AnalysisRun) and instance not in session.deleted:
            ensure_config_integrity(instance.config, instance.config_fingerprint)
            ensure_window_manifest_integrity(
                instance.window_ids,
                instance.window_manifest_fingerprint,
                instance.total_windows,
            )


def _validate_analysis_configs_before_flush(
    session: Session,
    _: object,
    __: object,
) -> None:
    _validate_analysis_configs(session)


event.listen(Session, "before_commit", _validate_analysis_configs)
event.listen(Session, "before_flush", _validate_analysis_configs_before_flush)
