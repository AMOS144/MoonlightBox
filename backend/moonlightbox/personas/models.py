"""LoRA 训练产出的、可审计的人格与风格读模型。"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


class IdentityKernel(Base):
    """训练后锁定的人格与表达风格证据，不承载运行时短期状态。"""

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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    locked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
