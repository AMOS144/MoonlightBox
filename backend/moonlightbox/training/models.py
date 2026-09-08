from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base
from moonlightbox.projects.models import Project


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TimelineConfirmation(Base):
    __tablename__ = "timeline_confirmations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey(f"{Project.__tablename__}.id", ondelete="CASCADE"), index=True
    )
    import_id: Mapped[str] = mapped_column(
        ForeignKey("import_sources.id", ondelete="CASCADE"), index=True
    )
    analysis_run_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    confirmation_fingerprint: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    active_event_ids: Mapped[list[str]] = mapped_column(JSON)
    rejected_event_ids: Mapped[list[str]] = mapped_column(JSON)
    event_revision_snapshots: Mapped[list[dict[str, object]]] = mapped_column(JSON)
    config_snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="confirmed")
    training_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey(f"{Project.__tablename__}.id", ondelete="CASCADE"), index=True
    )
    base_model: Mapped[str] = mapped_column(String(255))
    adapter_path: Mapped[str] = mapped_column(String(500))
    dataset_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="ready")
    metrics: Mapped[dict[str, float]] = mapped_column(JSON)
    recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    timeline_confirmation_id: Mapped[str | None] = mapped_column(
        ForeignKey("timeline_confirmations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    training_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    training_config: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
