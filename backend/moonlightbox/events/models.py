from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base
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


class EventNode(Base):
    __tablename__ = "event_nodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey(f"{Project.__tablename__}.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String(64))
    start_message_id: Mapped[str] = mapped_column(String(255))
    end_message_id: Mapped[str] = mapped_column(String(255))
    before_state: Mapped[str] = mapped_column(Text)
    after_state: Mapped[str] = mapped_column(Text)
    emotion_labels: Mapped[list[str]] = mapped_column(JSON)
    topic: Mapped[str] = mapped_column(Text)
    conflict_level: Mapped[int] = mapped_column(Integer)
    importance: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text)
    evidence_ids: Mapped[list[str]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
