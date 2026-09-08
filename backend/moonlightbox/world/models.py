from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base

"""人物世界数据库模型：图版本、Bundle、人物档案和归并审核记录。"""


class WorldGraphVersion(Base):
    __tablename__ = "world_graph_versions"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "source_fingerprint",
            "config_fingerprint",
            name="uq_world_graph_source_config",
        ),
        Index("ix_world_graph_versions_project_created", "project_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    trigger_import_id: Mapped[str] = mapped_column(
        ForeignKey("import_sources.id", ondelete="CASCADE")
    )
    workspace_key: Mapped[str] = mapped_column(String(128), unique=True)
    status: Mapped[str] = mapped_column(String(24), default="building")
    source_fingerprint: Mapped[str] = mapped_column(String(64))
    config_fingerprint: Mapped[str] = mapped_column(String(64))
    source_import_ids: Mapped[list[str]] = mapped_column(JSON)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    bundle_count: Mapped[int] = mapped_column(Integer, default=0)
    lightrag_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extraction_model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    chunking_strategy: Mapped[str | None] = mapped_column(String(80), nullable=True)
    chunk_token_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_overlap_token_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    entity_prompt_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    compiler_version: Mapped[str] = mapped_column(String(100))
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ConversationBundle(Base):
    __tablename__ = "conversation_bundles"
    __table_args__ = (
        UniqueConstraint("graph_version_id", "document_id"),
        Index("ix_conversation_bundles_project_started", "project_id", "started_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="CASCADE")
    )
    document_id: Mapped[str] = mapped_column(String(80))
    source_name: Mapped[str] = mapped_column(String(180))
    ordinal: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    primary_message_count: Mapped[int] = mapped_column(Integer)
    carry_in_message_count: Mapped[int] = mapped_column(Integer, default=0)
    indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class ConversationBundleMessage(Base):
    __tablename__ = "conversation_bundle_messages"
    __table_args__ = (
        UniqueConstraint("bundle_id", "message_id"),
        Index("ix_bundle_messages_message", "message_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    bundle_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_bundles.id", ondelete="CASCADE")
    )
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    is_carry_in: Mapped[bool] = mapped_column(Boolean, default=False)


class PersonWorldProfile(Base):
    __tablename__ = "person_world_profiles"
    __table_args__ = (
        UniqueConstraint("graph_version_id"),
        Index("ix_person_world_profiles_project_created", "project_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    subject_person_id: Mapped[str] = mapped_column(
        ForeignKey("participants.id", ondelete="RESTRICT")
    )
    graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="CASCADE")
    )
    identity: Mapped[dict[str, Any]] = mapped_column(JSON)
    work_and_education: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    places: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    social_relationships: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    preferences: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    recurring_activities: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    routine_summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    life_phases: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    relationship_with_user: Mapped[dict[str, Any]] = mapped_column(JSON)
    important_events: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    unresolved_candidates: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    source_message_ids: Mapped[list[str]] = mapped_column(JSON)
    retrieval_manifest: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    compiler_version: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class EntityMergeProposal(Base):
    __tablename__ = "entity_merge_proposals"
    __table_args__ = (Index("ix_entity_merge_proposals_graph_decision", "graph_version_id", "decision"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    graph_version_id: Mapped[str] = mapped_column(ForeignKey("world_graph_versions.id", ondelete="CASCADE"))
    source_entities: Mapped[list[str]] = mapped_column(JSON)
    target_entity: Mapped[str] = mapped_column(String(500))
    decision: Mapped[str] = mapped_column(String(32), default="pending")
    reason: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    merge_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
