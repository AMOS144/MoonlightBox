from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class HumanBlindStudy(Base):
    __tablename__ = "human_blind_studies"
    __table_args__ = (
        CheckConstraint(
            "status IN ('open', 'passed', 'failed', 'superseded')",
            name="ck_human_blind_studies_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    model_version_id: Mapped[str] = mapped_column(
        ForeignKey("model_versions.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    minimum_ratings: Mapped[int] = mapped_column(Integer, default=20)
    minimum_preference: Mapped[float] = mapped_column(Float, default=0.45)
    valid_rating_count: Mapped[int] = mapped_column(Integer, default=0)
    candidate_preference_rate: Mapped[float] = mapped_column(Float, default=0.0)
    report: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class HumanBlindCase(Base):
    __tablename__ = "human_blind_cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    study_id: Mapped[str] = mapped_column(
        ForeignKey("human_blind_studies.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    context: Mapped[list[str]] = mapped_column(JSON)
    human_reply: Mapped[str] = mapped_column(Text)
    candidate_reply: Mapped[str] = mapped_column(Text)
    candidate_option: Mapped[str] = mapped_column(String(1))
    order_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class HumanBlindRating(Base):
    __tablename__ = "human_blind_ratings"
    __table_args__ = (
        UniqueConstraint("case_id", "rater_key", name="ux_human_blind_case_rater"),
        CheckConstraint(
            "choice IN ('a', 'b', 'tie')",
            name="ck_human_blind_ratings_choice",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    study_id: Mapped[str] = mapped_column(
        ForeignKey("human_blind_studies.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(
        ForeignKey("human_blind_cases.id", ondelete="CASCADE"), index=True
    )
    rater_key: Mapped[str] = mapped_column(String(128))
    choice: Mapped[str] = mapped_column(String(8))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
