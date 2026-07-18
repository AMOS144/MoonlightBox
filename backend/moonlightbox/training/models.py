from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base
from moonlightbox.projects.models import Project


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey(f"{Project.__tablename__}.id", ondelete="CASCADE"), index=True
    )
    base_model: Mapped[str] = mapped_column(String(255))
    adapter_path: Mapped[str] = mapped_column(String(500))
    dataset_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="ready")
    metrics: Mapped[dict[str, float]] = mapped_column(JSON)
    recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
