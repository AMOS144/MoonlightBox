from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BranchBaselineManifest(Base):
    __tablename__ = "branch_baseline_manifests"
    __table_args__ = (UniqueConstraint("branch_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    import_id: Mapped[str] = mapped_column(
        ForeignKey("import_sources.id", ondelete="RESTRICT"), index=True
    )
    origin_event_id: Mapped[str] = mapped_column(ForeignKey("event_nodes.id", ondelete="RESTRICT"))
    boundary_message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="RESTRICT"))
    boundary_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    boundary_source_id: Mapped[str] = mapped_column(String(100))
    boundary_inclusive: Mapped[bool] = mapped_column(Boolean, default=False)
    message_count: Mapped[int] = mapped_column(Integer)
    event_snapshot_count: Mapped[int] = mapped_column(Integer)
    message_digest: Mapped[str] = mapped_column(String(64))
    event_digest: Mapped[str] = mapped_column(String(64))
    index_fingerprint: Mapped[str] = mapped_column(String(64))
    recent_tail_message_ids: Mapped[list[str]] = mapped_column(JSON)
    protocol_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    validated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class BranchBaselineEventSnapshot(Base):
    __tablename__ = "branch_baseline_event_snapshots"
    __table_args__ = (UniqueConstraint("manifest_id", "source_event_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    manifest_id: Mapped[str] = mapped_column(
        ForeignKey("branch_baseline_manifests.id", ondelete="CASCADE"),
        index=True,
    )
    source_event_id: Mapped[str] = mapped_column(String(36))
    source_revision_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    snapshot: Mapped[dict[str, object]] = mapped_column(JSON)
    evidence_message_ids: Mapped[list[str]] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class BranchBaselineState(Base):
    __tablename__ = "branch_baseline_states"
    __table_args__ = (
        UniqueConstraint("manifest_id"),
        UniqueConstraint("branch_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    manifest_id: Mapped[str] = mapped_column(
        ForeignKey("branch_baseline_manifests.id", ondelete="CASCADE")
    )
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), index=True
    )
    persona_state: Mapped[dict[str, object]] = mapped_column(JSON)
    relationship_state: Mapped[dict[str, object]] = mapped_column(JSON)
    emotional_tendency: Mapped[dict[str, object]] = mapped_column(JSON)
    user_model: Mapped[dict[str, object]] = mapped_column(JSON)
    historical_belief_ids: Mapped[list[str]] = mapped_column(JSON)
    evidence_message_ids: Mapped[list[str]] = mapped_column(JSON)
    evidence_event_snapshot_ids: Mapped[list[str]] = mapped_column(JSON)
    local_proposal: Mapped[dict[str, object]] = mapped_column(JSON)
    review_result: Mapped[dict[str, object]] = mapped_column(JSON)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
