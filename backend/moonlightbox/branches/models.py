from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion


class Branch(Base):
    __tablename__ = "branches"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey(f"{Project.__tablename__}.id", ondelete="CASCADE"), index=True
    )
    origin_event_id: Mapped[str] = mapped_column(
        ForeignKey(f"{EventNode.__tablename__}.id")
    )
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey(f"{ModelVersion.__tablename__}.id")
    )
    title: Mapped[str] = mapped_column(String(255))
    origin_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    state_snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class BranchMessage(Base):
    __tablename__ = "branch_messages"
    __table_args__ = (UniqueConstraint("branch_id", "sequence"),)

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    generation_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
