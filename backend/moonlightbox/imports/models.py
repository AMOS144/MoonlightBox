from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


class ImportSource(Base):
    __tablename__ = "import_sources"
    __table_args__ = (UniqueConstraint("project_id", "preview_id"),)

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    preview_id: Mapped[str] = mapped_column(String(36))
    source_path: Mapped[str] = mapped_column(String(500))
    message_count: Mapped[int] = mapped_column(Integer)
    confirmed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Participant(Base):
    __tablename__ = "participants"
    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(20))


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("import_id", "source_id"),)

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    import_id: Mapped[str] = mapped_column(
        ForeignKey("import_sources.id", ondelete="CASCADE")
    )
    participant_id: Mapped[str] = mapped_column(
        ForeignKey("participants.id", ondelete="RESTRICT")
    )
    source_id: Mapped[str] = mapped_column(String(100))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    kind: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(String)
    raw: Mapped[dict[str, Any]] = mapped_column(JSON)
