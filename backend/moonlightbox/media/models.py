from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
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


class MediaAsset(Base):
    __tablename__ = "media_assets"
    __table_args__ = (UniqueConstraint("project_id", "sha256", name="uq_media_asset_project_hash"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(20), index=True)
    sha256: Mapped[str] = mapped_column(String(64))
    relative_path: Mapped[str] = mapped_column(String(500))
    mime_type: Mapped[str] = mapped_column(String(100))
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_key: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class MediaSemanticAnnotation(Base):
    __tablename__ = "media_semantic_annotations"
    __table_args__ = (
        UniqueConstraint("asset_id", name="uq_media_semantic_annotation_asset"),
        CheckConstraint(
            "status IN ('pending', 'succeeded', 'failed', 'needs_review')",
            name="ck_media_semantic_annotations_status",
        ),
        CheckConstraint(
            "reuse_decision IN ('pending', 'approved', 'blocked')",
            name="ck_media_semantic_annotations_reuse",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_media_semantic_annotations_confidence",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    asset_id: Mapped[str] = mapped_column(
        ForeignKey("media_assets.id", ondelete="CASCADE"), index=True
    )
    modality: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    transcript: Mapped[str] = mapped_column(Text, default="")
    ocr_text: Mapped[str] = mapped_column(Text, default="")
    safety_tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_model: Mapped[str] = mapped_column(String(255), default="")
    source_version: Mapped[str] = mapped_column(String(64), default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    reusable: Mapped[bool] = mapped_column(Boolean, default=False)
    reuse_decision: Mapped[str] = mapped_column(String(16), default="pending")
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    review_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
