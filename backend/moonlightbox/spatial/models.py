from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


def _uuid() -> str:
    return str(uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SpatialAnalysisRun(Base):
    __tablename__ = "spatial_analysis_runs"
    __table_args__ = (
        UniqueConstraint(
            "import_id",
            "config_hash",
            name="uq_spatial_runs_import_config",
        ),
        Index("ix_spatial_runs_project_status", "project_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    import_id: Mapped[str] = mapped_column(
        ForeignKey("import_sources.id", ondelete="CASCADE"), index=True
    )
    target_person_id: Mapped[str] = mapped_column(
        ForeignKey("participants.id", ondelete="RESTRICT"), index=True
    )
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    config: Mapped[dict[str, object]] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(64))
    statistics: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SpatialConversationEpisode(Base):
    __tablename__ = "spatial_conversation_episodes"
    __table_args__ = (
        Index("ix_spatial_episodes_import_time", "import_id", "started_at"),
        Index("ix_spatial_episodes_project_time", "project_id", "started_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    import_id: Mapped[str] = mapped_column(
        ForeignKey("import_sources.id", ondelete="CASCADE"), index=True
    )
    created_run_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    message_ids: Mapped[list[str]] = mapped_column(JSON)
    participant_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    media_asset_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    index_text: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    segmentation_version: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64))
    time_partition: Mapped[str] = mapped_column(String(16), index=True)
    has_structured_location: Mapped[bool] = mapped_column(Boolean, default=False)
    previous_episode_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    next_episode_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SpatialEpisodeCandidate(Base):
    __tablename__ = "spatial_episode_candidates"
    __table_args__ = (
        UniqueConstraint("run_id", "episode_id", name="uq_spatial_candidate_episode"),
        Index("ix_spatial_candidate_run_score", "run_id", "max_similarity"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    episode_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_conversation_episodes.id", ondelete="CASCADE"), index=True
    )
    query_matches: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    max_similarity: Mapped[float] = mapped_column(Float, default=0.0)
    time_partition: Mapped[str] = mapped_column(String(16))
    forced_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recall_hint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SpatialAnalysisBundle(Base):
    __tablename__ = "spatial_analysis_bundles"
    __table_args__ = (
        UniqueConstraint("run_id", "bundle_hash", name="uq_spatial_bundle_hash"),
        Index("ix_spatial_bundles_run_time", "run_id", "started_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    bundle_hash: Mapped[str] = mapped_column(String(64))
    episode_ids: Mapped[list[str]] = mapped_column(JSON)
    candidate_episode_ids: Mapped[list[str]] = mapped_column(JSON)
    message_ids: Mapped[list[str]] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    analyzer_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    analyzer_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PlaceMention(Base):
    __tablename__ = "place_mentions"
    __table_args__ = (
        Index("ix_place_mentions_run_subject", "run_id", "subject_id"),
        Index("ix_place_mentions_project_message", "project_id", "message_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    import_id: Mapped[str] = mapped_column(
        ForeignKey("import_sources.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    bundle_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_analysis_bundles.id", ondelete="CASCADE"), index=True
    )
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    span_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    span_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)
    mention_type: Mapped[str] = mapped_column(String(32))
    type_distribution: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    candidate_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    speaker_id: Mapped[str] = mapped_column(
        ForeignKey("participants.id", ondelete="RESTRICT"), index=True
    )
    subject_id: Mapped[str | None] = mapped_column(
        ForeignKey("participants.id", ondelete="SET NULL"), nullable=True, index=True
    )
    subject_resolution: Mapped[str] = mapped_column(String(32))
    relation: Mapped[str] = mapped_column(String(32))
    movement_phase: Mapped[str] = mapped_column(String(32))
    assertion_mode: Mapped[str] = mapped_column(String(32))
    time_start_lower: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    time_start_upper: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    time_end_lower: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_end_upper: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    time_precision: Mapped[str] = mapped_column(String(32), default="unknown")
    context_message_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence_message_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    extractor_method: Mapped[str] = mapped_column(String(64))
    raw_score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    retrieved_by_alias_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PlaceCandidate(Base):
    __tablename__ = "place_candidates"
    __table_args__ = (
        UniqueConstraint("mention_id", "candidate_key", name="uq_place_candidate_key"),
        Index("ix_place_candidates_mention_rank", "mention_id", "rank"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    mention_id: Mapped[str] = mapped_column(
        ForeignKey("place_mentions.id", ondelete="CASCADE"), index=True
    )
    candidate_key: Mapped[str] = mapped_column(String(255))
    candidate_source: Mapped[str] = mapped_column(String(32))
    canonical_name: Mapped[str] = mapped_column(Text)
    place_type: Mapped[str] = mapped_column(String(64), default="unknown")
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    coordinate_crs: Mapped[str | None] = mapped_column(String(32), nullable=True)
    uncertainty_radius_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    address_components: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    provider_payload_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    feature_scores: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    contradictions: Mapped[list[str]] = mapped_column(JSON, default=list)
    raw_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    rank: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PlaceEntity(Base):
    __tablename__ = "place_entities"
    __table_args__ = (
        Index("ix_place_entities_project_owner", "project_id", "owner_person_id"),
        Index("ix_place_entities_project_status", "project_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    owner_person_id: Mapped[str | None] = mapped_column(
        ForeignKey("participants.id", ondelete="SET NULL"), nullable=True, index=True
    )
    canonical_name: Mapped[str] = mapped_column(Text)
    place_type: Mapped[str] = mapped_column(String(64), default="unknown")
    role_distribution: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    geo_resolution: Mapped[str] = mapped_column(String(32), default="semantic_only")
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    coordinate_crs: Mapped[str | None] = mapped_column(String(32), nullable=True)
    uncertainty_radius_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    address_components: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    field_provenance: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="provisional", index=True)
    superseded_by_id: Mapped[str | None] = mapped_column(
        ForeignKey("place_entities.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class PlaceAlias(Base):
    __tablename__ = "place_aliases"
    __table_args__ = (
        Index("ix_place_aliases_project_normalized", "project_id", "normalized_alias"),
        Index("ix_place_aliases_place", "place_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    place_id: Mapped[str] = mapped_column(
        ForeignKey("place_entities.id", ondelete="CASCADE"), index=True
    )
    raw_alias: Mapped[str] = mapped_column(Text)
    normalized_alias: Mapped[str] = mapped_column(Text)
    alias_kind: Mapped[str] = mapped_column(String(32))
    speaker_id: Mapped[str | None] = mapped_column(
        ForeignKey("participants.id", ondelete="SET NULL"), nullable=True
    )
    perspective: Mapped[str] = mapped_column(String(32), default="unknown")
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    supporting_mention_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    opposing_mention_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    support_quality: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_eligible: Mapped[bool] = mapped_column(Boolean, default=False)
    confirmation_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    alias_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PlaceResolution(Base):
    __tablename__ = "place_resolutions"
    __table_args__ = (
        UniqueConstraint("run_id", "mention_id", name="uq_place_resolution_mention"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    mention_id: Mapped[str] = mapped_column(
        ForeignKey("place_mentions.id", ondelete="CASCADE"), index=True
    )
    place_id: Mapped[str | None] = mapped_column(
        ForeignKey("place_entities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("place_candidates.id", ondelete="SET NULL"), nullable=True
    )
    decision: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    margin: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(Text)
    resolver_version: Mapped[str] = mapped_column(String(64))
    is_manual: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class VisitObservation(Base):
    __tablename__ = "visit_observations"
    __table_args__ = (
        Index("ix_visit_observations_subject_time", "subject_id", "started_at_lower"),
        Index("ix_visit_observations_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    subject_id: Mapped[str] = mapped_column(
        ForeignKey("participants.id", ondelete="CASCADE"), index=True
    )
    place_id: Mapped[str | None] = mapped_column(
        ForeignKey("place_entities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    mention_ids: Mapped[list[str]] = mapped_column(JSON)
    visit_state: Mapped[str] = mapped_column(String(32))
    assertion_mode: Mapped[str] = mapped_column(String(32))
    started_at_lower: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at_upper: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at_lower: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at_upper: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    purpose_distribution: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    companion_person_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    transport_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_authority: Mapped[str] = mapped_column(String(32))
    verification_status: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    contradiction_group_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    evidence_message_ids: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class VisitEpisode(Base):
    __tablename__ = "visit_episodes"
    __table_args__ = (
        Index("ix_visit_episodes_subject_time", "subject_id", "started_at_lower"),
        Index("ix_visit_episodes_run_place", "run_id", "place_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    subject_id: Mapped[str] = mapped_column(
        ForeignKey("participants.id", ondelete="CASCADE"), index=True
    )
    place_id: Mapped[str] = mapped_column(
        ForeignKey("place_entities.id", ondelete="CASCADE"), index=True
    )
    observation_ids: Mapped[list[str]] = mapped_column(JSON)
    started_at_lower: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at_upper: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ended_at_lower: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at_upper: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dwell_hours_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    dwell_hours_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    left_censored: Mapped[bool] = mapped_column(Boolean, default=False)
    right_censored: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float)
    evidence_root_ids: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PlaceGraphSnapshot(Base):
    __tablename__ = "place_graph_snapshots"
    __table_args__ = (
        UniqueConstraint("run_id", "person_id", name="uq_place_graph_snapshot_run_person"),
        Index("ix_place_graph_snapshots_project_person", "project_id", "person_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    person_id: Mapped[str] = mapped_column(
        ForeignKey("participants.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str] = mapped_column(
        ForeignKey("spatial_analysis_runs.id", ondelete="CASCADE"), index=True
    )
    graph_version: Mapped[str] = mapped_column(String(64))
    cutoff_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Shanghai")
    parameters: Mapped[dict[str, object]] = mapped_column(JSON)
    statistics: Mapped[dict[str, object]] = mapped_column(JSON)
    content_hash: Mapped[str] = mapped_column(String(64))
    reliability: Mapped[float] = mapped_column(Float, default=0.0)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PlaceGraphNode(Base):
    __tablename__ = "place_graph_nodes"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "place_id", name="uq_place_graph_node"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("place_graph_snapshots.id", ondelete="CASCADE"), index=True
    )
    place_id: Mapped[str] = mapped_column(
        ForeignKey("place_entities.id", ondelete="CASCADE"), index=True
    )
    metrics: Mapped[dict[str, object]] = mapped_column(JSON)
    activity_weight: Mapped[float] = mapped_column(Float)
    salience_score: Mapped[float] = mapped_column(Float, default=0.0)
    role_distribution: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)


class PlaceGraphEdge(Base):
    __tablename__ = "place_graph_edges"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id", "from_place_id", "to_place_id", name="uq_place_graph_edge"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("place_graph_snapshots.id", ondelete="CASCADE"), index=True
    )
    from_place_id: Mapped[str] = mapped_column(
        ForeignKey("place_entities.id", ondelete="CASCADE"), index=True
    )
    to_place_id: Mapped[str] = mapped_column(
        ForeignKey("place_entities.id", ondelete="CASCADE"), index=True
    )
    metrics: Mapped[dict[str, object]] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)


class PlaceProviderEnrichment(Base):
    __tablename__ = "place_provider_enrichments"
    __table_args__ = (
        UniqueConstraint(
            "place_id", "provider", "provider_place_id", name="uq_place_provider_ref"
        ),
        Index("ix_place_provider_lookup", "provider", "provider_place_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    place_id: Mapped[str] = mapped_column(
        ForeignKey("place_entities.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    provider_place_id: Mapped[str] = mapped_column(String(255))
    provider_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_category: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    coordinate_crs: Mapped[str | None] = mapped_column(String(32), nullable=True)
    administrative: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    provider_api_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    response_hash: Mapped[str] = mapped_column(String(64))
    queried_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    persistence_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
