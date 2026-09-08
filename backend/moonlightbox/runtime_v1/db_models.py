"""Runtime v1 的独立持久化模型。"""

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


class RuntimeEventRow(Base):
    __tablename__ = "runtime_events"
    __table_args__ = (
        Index("ix_runtime_events_branch_status", "branch_id", "status", "occurred_at"),
        UniqueConstraint("branch_id", "idempotency_key"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    priority: Mapped[int] = mapped_column(Integer, default=0)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RuntimeLifeStateRow(Base):
    __tablename__ = "runtime_life_states"
    __table_args__ = (
        UniqueConstraint("branch_id", "version"),
        Index("ix_runtime_life_states_current", "branch_id", "is_current"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer, default=1)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    virtual_now: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    current_plan_block_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    availability: Mapped[str] = mapped_column(String(16), default="unknown")
    energy: Mapped[str] = mapped_column(String(16), default="unknown")
    activity: Mapped[str | None] = mapped_column(String(160), nullable=True)
    location_role: Mapped[str | None] = mapped_column(String(160), nullable=True)
    social_context: Mapped[str | None] = mapped_column(String(160), nullable=True)
    mood: Mapped[str | None] = mapped_column(String(160), nullable=True)
    attention: Mapped[str | None] = mapped_column(String(300), nullable=True)
    current_goal: Mapped[str | None] = mapped_column(String(300), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reason: Mapped[str] = mapped_column(String(200), default="initial")
    source_event_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    # 每个字段的来源单独保存，避免把短期生活状态误当成长期记忆。
    field_sources: Mapped[dict[str, list[str]]] = mapped_column(JSON, default=dict)
    previous_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("runtime_life_states.id", ondelete="SET NULL"), nullable=True
    )
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class RuntimeDayPlanRow(Base):
    __tablename__ = "runtime_day_plans"
    __table_args__ = (UniqueConstraint("branch_id", "plan_date"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    plan_date: Mapped[str] = mapped_column(String(10))
    blocks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class RuntimeMemoryRow(Base):
    __tablename__ = "runtime_memory_records"
    __table_args__ = (Index("ix_runtime_memory_scope_branch", "scope", "branch_id", "status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    scope: Mapped[str] = mapped_column(String(16))
    branch_id: Mapped[str | None] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), nullable=True
    )
    snapshot_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    subject: Mapped[str] = mapped_column(String(200))
    predicate: Mapped[str] = mapped_column(String(120))
    object: Mapped[str] = mapped_column(String(500))
    summary: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="asserted")
    source_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(default=1.0)
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    supersedes_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class RuntimeMemoryIndexRow(Base):
    """分支统一记忆索引的不可变读取边界。

    Runtime v1 退役旧连续记忆表后，只读取 ``RuntimeMemoryRow``。本表记录当前
    可检索的记录 ID，方便查询、回放和回滚在同一个明确边界上工作。
    """

    __tablename__ = "runtime_memory_index_versions"
    __table_args__ = (
        UniqueConstraint("branch_id", "version"),
        Index("ix_runtime_memory_index_current", "branch_id", "is_current"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer, default=1)
    active_record_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    previous_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("runtime_memory_index_versions.id", ondelete="SET NULL"), nullable=True
    )
    reason: Mapped[str] = mapped_column(String(120), default="bootstrap")
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class RuntimeContextSummaryRow(Base):
    """ContextAssembler 压缩过的窗口摘要及其精确来源。

    摘要只保存已有消息的压缩表达与 source IDs，不是新的记忆或事实写入入口。
    """

    __tablename__ = "runtime_context_summaries"
    __table_args__ = (
        Index("ix_runtime_context_summaries_branch_covered", "branch_id", "covered_until"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    covered_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_event_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_message_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    text: Mapped[str] = mapped_column(Text)
    unresolved_items: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_by: Mapped[str] = mapped_column(String(80), default="runtime-context-summarizer-v1")
    compression_round: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class RuntimeSnapshotRow(Base):
    """分支创建时冻结的世界资料；v1 直接使用最新图谱，cutoff 记为导入末时刻。"""

    __tablename__ = "runtime_origin_snapshots"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), unique=True
    )
    graph_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    profile_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    source_node_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    cutoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    snapshot_mode: Mapped[str] = mapped_column(String(32), default="latest_profile")
    # LightRAG 暂不能按时间查询；这里记录当前完整图谱实际引用的消息证据，
    # 以便后续支持 historical_cutoff 时可以无损替换编译器。
    source_message_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    profile: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    routine_profile: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    compiler_version: Mapped[str] = mapped_column(String(64), default="runtime-v1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    @property
    def source_graph_version_id(self) -> str | None:
        return self.graph_version_id

    @property
    def source_profile_id(self) -> str | None:
        return self.profile_id

    @property
    def person_world_profile(self) -> dict[str, Any]:
        return self.profile


class RuntimeClockRow(Base):
    """VirtualClock 的独立持久化锚点，暂停/恢复不会改写历史事件。"""

    __tablename__ = "runtime_clocks"
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), primary_key=True
    )
    virtual_anchor: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    wall_anchor: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    time_scale: Mapped[float] = mapped_column(default=1.0)
    status: Mapped[str] = mapped_column(String(16), default="running")
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")


class RuntimeWakeupRow(Base):
    """统一事件队列中的定时唤醒。"""

    __tablename__ = "runtime_wakeups"
    __table_args__ = (
        UniqueConstraint("branch_id", "idempotency_key"),
        Index("ix_runtime_wakeups_due", "status", "wake_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    wake_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    reason: Mapped[str] = mapped_column(String(200))
    trigger_type: Mapped[str] = mapped_column(String(32))
    idempotency_key: Mapped[str] = mapped_column(String(160))
    status: Mapped[str] = mapped_column(String(20), default="scheduled")
    executing_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RuntimeLifeEventRow(Base):
    """Executor 提交的分支生活事件及审计证据。"""

    __tablename__ = "runtime_life_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(ForeignKey("branches.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(32))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    idempotency_key: Mapped[str] = mapped_column(String(160), unique=True)


# 语义别名：保留 Row 后缀作为数据库层实现名，供按设计文档命名的调用方使用。
LifeStateVersion = RuntimeLifeStateRow
OriginWorldSnapshotRow = RuntimeSnapshotRow
DayPlanRow = RuntimeDayPlanRow
EventQueueRow = RuntimeEventRow
