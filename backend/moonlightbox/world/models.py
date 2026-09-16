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
    text,
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
    # 人工纠正后的图谱使用父版本和修订号形成不可变版本链。旧数据保持
    # parent_version_id=None、revision=1，运行时仍可读取。
    parent_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="RESTRICT"), nullable=True
    )
    revision: Mapped[int] = mapped_column(Integer, default=1)
    correction_head_hash: Mapped[str] = mapped_column(String(64), default="")
    change_set_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
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
        Index(
            "ux_profile_global_graph",
            "graph_version_id",
            unique=True,
            sqlite_where=text("node_boundary_hash IS NULL"),
            postgresql_where=text("node_boundary_hash IS NULL"),
        ),
        Index("ix_person_world_profiles_project_created", "project_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    subject_person_id: Mapped[str] = mapped_column(
        ForeignKey("participants.id", ondelete="RESTRICT")
    )
    node_boundary_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
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
    agent_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    generation_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # v3 是新人物画像；v1/v2 原样保留供历史页面和已绑定分支读取，不原地迁移解释。
    profile_v2: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    profile_v3: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    profile_schema_version: Mapped[str] = mapped_column(String(24), default="v1")
    investigation_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class EntityMergeProposal(Base):
    __tablename__ = "entity_merge_proposals"
    __table_args__ = (
        Index("ix_entity_merge_proposals_graph_decision", "graph_version_id", "decision"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="CASCADE")
    )
    source_entities: Mapped[list[str]] = mapped_column(JSON)
    target_entity: Mapped[str] = mapped_column(String(500))
    decision: Mapped[str] = mapped_column(String(32), default="pending")
    reason: Mapped[str] = mapped_column(Text, default="")
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    merge_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PersonWorldAgentRun(Base):
    """一次可恢复的人物世界调查；大文本只保存引用和审计摘要。"""

    __tablename__ = "person_world_agent_runs"
    __table_args__ = (
        Index("ix_person_world_agent_runs_project_created", "project_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="CASCADE")
    )
    mode: Mapped[str] = mapped_column(String(32), default="initial_compile")
    status: Mapped[str] = mapped_column(String(32), default="researching")
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    prompt_version: Mapped[str] = mapped_column(String(100))
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PersonWorldSectionTask(Base):
    """一次调查运行中一个顶层栏目的独立、可恢复任务状态。"""

    __tablename__ = "person_world_section_tasks"
    __table_args__ = (
        UniqueConstraint("agent_run_id", "section", name="uq_world_section_task_run_section"),
        Index("ix_world_section_tasks_run_status", "agent_run_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    agent_run_id: Mapped[str] = mapped_column(
        ForeignKey("person_world_agent_runs.id", ondelete="CASCADE")
    )
    section: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(40), default="pending")
    attempted_queries: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    retrievals: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    research_round: Mapped[int] = mapped_column(Integer, default=0)
    tool_call_count: Mapped[int] = mapped_column(Integer, default=0)
    unresolved_questions: Mapped[list[str]] = mapped_column(JSON, default=list)
    result_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error_diagnostic: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SectionEvidenceLedgerEntry(Base):
    """栏目证据的轻量 provenance；正文仍只保存在原始消息表。"""

    __tablename__ = "section_evidence_ledger_entries"
    __table_args__ = (
        UniqueConstraint(
            "section_task_id",
            "message_id",
            "query_id",
            "reference_rank",
            "context_window_id",
            name="uq_section_ledger_entry_provenance",
        ),
        Index("ix_section_ledger_task_message", "section_task_id", "message_id"),
        Index("ix_section_ledger_task_round", "section_task_id", "research_round"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    section_task_id: Mapped[str] = mapped_column(
        ForeignKey("person_world_section_tasks.id", ondelete="CASCADE")
    )
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    retrieval_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    query_id: Mapped[str] = mapped_column(String(100), default="context")
    research_round: Mapped[int] = mapped_column(Integer, default=1)
    reference_rank: Mapped[int] = mapped_column(Integer, default=0)
    document_rank: Mapped[int] = mapped_column(Integer, default=0)
    is_primary_match: Mapped[bool] = mapped_column(Boolean, default=False)
    context_window_id: Mapped[str] = mapped_column(String(120), default="")
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class WorldEvidence(Base):
    """模型结论回到 SQL 原始消息后的不可伪造证据定位。"""

    __tablename__ = "world_evidence"
    __table_args__ = (
        UniqueConstraint("graph_version_id", "message_id", name="uq_world_evidence_graph_message"),
        Index("ix_world_evidence_project_message", "project_id", "message_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="CASCADE")
    )
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    bundle_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_bundles.id", ondelete="CASCADE")
    )
    # 旧版本的证据行没有这四个身份快照；允许为空以便平滑迁移，但所有 v2 新写入都填充。
    timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    participant_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    participant_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    participant_role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    exact_quote: Mapped[str] = mapped_column(Text)
    context_before_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    context_after_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    content_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class AtomicWorldClaim(Base):
    """PersonWorldProfile 的可核验最小事实单元。"""

    __tablename__ = "atomic_world_claims"
    __table_args__ = (
        Index("ix_atomic_world_claims_run_section", "agent_run_id", "profile_section"),
        Index("ix_atomic_world_claims_project_subject", "project_id", "subject_kind"),
        Index(
            "ix_atomic_world_claims_graph_domain_fact",
            "graph_version_id",
            "primary_domain",
            "fact_type",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="CASCADE")
    )
    agent_run_id: Mapped[str] = mapped_column(
        ForeignKey("person_world_agent_runs.id", ondelete="CASCADE")
    )
    # Profile 的顶层七栏目和栏目内事实类型是独立的结构索引。它们让 Revision 的
    # 相关事实查询只按持久化列执行，不需要读取或匹配中文正文。
    primary_domain: Mapped[str] = mapped_column(String(80), default="unknown")
    profile_section: Mapped[str] = mapped_column(String(80))
    fact_type: Mapped[str] = mapped_column(String(80), default="unknown")
    subject_kind: Mapped[str] = mapped_column(String(32))
    subject_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    speaker_role: Mapped[str] = mapped_column(String(32), default="unknown")
    speaker_name: Mapped[str | None] = mapped_column(String(500), nullable=True)
    addressee_kind: Mapped[str] = mapped_column(String(32), default="unknown")
    predicate: Mapped[str] = mapped_column(String(160))
    object_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)
    assertion_kind: Mapped[str] = mapped_column(String(40), default="unknown")
    temporal_status: Mapped[str] = mapped_column(String(40), default="unknown")
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    derivation: Mapped[str] = mapped_column(String(32), default="inferred")
    evidence_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    qualifiers: Mapped[list[str]] = mapped_column(JSON, default=list)
    admission_status: Mapped[str] = mapped_column(String(32), default="accepted")
    supersedes_claim_id: Mapped[str | None] = mapped_column(
        ForeignKey("atomic_world_claims.id", ondelete="SET NULL"), nullable=True
    )
    created_by: Mapped[str] = mapped_column(String(32), default="agent")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class PersonWorldProfileDraft(Base):
    __tablename__ = "person_world_profile_drafts"
    __table_args__ = (UniqueConstraint("agent_run_id", name="uq_person_world_profile_draft_run"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="CASCADE")
    )
    agent_run_id: Mapped[str] = mapped_column(
        ForeignKey("person_world_agent_runs.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(String(32), default="awaiting_review")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    claim_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    generation_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    profile_v2: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    profile_v3: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    profile_schema_version: Mapped[str] = mapped_column(String(24), default="v1")
    investigation_report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class PersonWorldRevisionSession(Base):
    __tablename__ = "person_world_revision_sessions"
    __table_args__ = (
        Index("ix_world_revision_sessions_project_created", "project_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    base_graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="RESTRICT")
    )
    base_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("person_world_profiles.id", ondelete="RESTRICT"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(40), default="exploring")
    # 所有可见会话变化（包含 Worker 阶段）都推进版本，供 API/SSE 做乐观锁和事件游标。
    session_revision: Mapped[int] = mapped_column(Integer, default=0)
    # 仅用户意图、选择范围或明确重试才推进。它与 ``session_revision`` 分离，避免
    # Worker 上报“正在检索/正在组装”等进度时，把仍在执行的 Harness Run 误判为 stale。
    input_revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    scope: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    context_snapshot_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    pending_turn_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    understanding_revision: Mapped[int] = mapped_column(Integer, default=0)
    understanding_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    understanding_payload_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    profile_change_set_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    graph_change_set_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class PersonWorldRevisionMessage(Base):
    __tablename__ = "person_world_revision_messages"
    __table_args__ = (
        Index("ix_world_revision_messages_session_created", "session_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(
        ForeignKey("person_world_revision_sessions.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(24))
    turn_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    kind: Mapped[str] = mapped_column(String(32), default="message")
    in_reply_to_turn_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    context_snapshot_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    session_revision: Mapped[int] = mapped_column(Integer, default=0)
    idempotency_key: Mapped[str | None] = mapped_column(String(120), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class PersonWorldRevisionContextSnapshot(Base):
    """一个 Revision Agent 回合实际可读取的不可变上下文清单。

    业务表记录范围、版本、原始消息引用和 hash；模型已读工作材料另由统一 LangGraph
    检查点保存，以便普通重启后继续调查，不依赖 Phoenix 反向恢复正文。
    """

    __tablename__ = "person_world_revision_context_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "session_id", "session_revision", name="uq_world_revision_snapshot_version"
        ),
        Index("ix_world_revision_snapshots_session_created", "session_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(
        ForeignKey("person_world_revision_sessions.id", ondelete="CASCADE")
    )
    session_revision: Mapped[int] = mapped_column(Integer)
    base_graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="RESTRICT")
    )
    base_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("person_world_profiles.id", ondelete="RESTRICT"), nullable=True
    )
    scope: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    evidence_message_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    related_claim_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    graph_manifest: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    context_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class WorldCorrection(Base):
    __tablename__ = "world_corrections"
    __table_args__ = (Index("ix_world_corrections_project_status", "project_id", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    revision_session_id: Mapped[str] = mapped_column(
        ForeignKey("person_world_revision_sessions.id", ondelete="CASCADE")
    )
    correction_type: Mapped[str] = mapped_column(String(40))
    target_key: Mapped[str] = mapped_column(String(500))
    original_interpretation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    corrected_interpretation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    user_explanation: Mapped[str] = mapped_column(Text, default="")
    source_message_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(24), default="proposed")
    supersedes_id: Mapped[str | None] = mapped_column(
        ForeignKey("world_corrections.id", ondelete="SET NULL"), nullable=True
    )
    approved_change_set_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class WorldGraphChangeSet(Base):
    __tablename__ = "world_graph_change_sets"
    __table_args__ = (
        UniqueConstraint("revision_session_id", "revision", name="uq_world_change_set_revision"),
        Index("ix_world_change_sets_project_status", "project_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    revision_session_id: Mapped[str] = mapped_column(
        ForeignKey("person_world_revision_sessions.id", ondelete="CASCADE")
    )
    base_graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="RESTRICT")
    )
    revision: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="draft")
    profile_patch: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    graph_operations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    affected_entities: Mapped[list[str]] = mapped_column(JSON, default=list)
    affected_relations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    regression_queries: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    canonical_payload_hash: Mapped[str] = mapped_column(String(64))
    execution_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    candidate_graph_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class WorldChangeApproval(Base):
    __tablename__ = "world_change_approvals"
    __table_args__ = (Index("ix_world_change_approvals_change_set", "change_set_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    change_set_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_change_sets.id", ondelete="CASCADE")
    )
    approval_stage: Mapped[str] = mapped_column(String(32))
    revision: Mapped[int] = mapped_column(Integer)
    payload_hash: Mapped[str] = mapped_column(String(64))
    approved_by: Mapped[str] = mapped_column(String(120), default="local_user")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class WorldGraphOperationLog(Base):
    __tablename__ = "world_graph_operation_logs"
    __table_args__ = (
        UniqueConstraint(
            "change_set_id",
            "graph_version_id",
            "operation_id",
            name="uq_world_graph_operation_log",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    change_set_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_change_sets.id", ondelete="CASCADE")
    )
    graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="CASCADE")
    )
    operation_id: Mapped[str] = mapped_column(String(80))
    operation_type: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(24), default="pending")
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    response_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WorldPublication(Base):
    """把 Graph 和 Profile 作为不可拆分的一对发布。"""

    __tablename__ = "world_publications"
    __table_args__ = (
        Index("ix_world_publications_project_status", "project_id", "status"),
        Index(
            "ux_node_publication_active",
            "project_id",
            "node_boundary_hash",
            unique=True,
            sqlite_where=text("status = 'active'"),
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    node_boundary_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    graph_version_id: Mapped[str] = mapped_column(
        ForeignKey("world_graph_versions.id", ondelete="RESTRICT")
    )
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("person_world_profiles.id", ondelete="RESTRICT")
    )
    correction_head_hash: Mapped[str] = mapped_column(String(64), default="")
    previous_publication_id: Mapped[str | None] = mapped_column(
        ForeignKey("world_publications.id", ondelete="RESTRICT"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(24), default="active")
    published_by: Mapped[str] = mapped_column(String(120), default="system")
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
