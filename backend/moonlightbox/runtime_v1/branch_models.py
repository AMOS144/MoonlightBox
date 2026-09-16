"""Runtime v1 的分支与对话持久化模型。

表名沿用既有资料库的 ``branches`` 与 ``branch_messages``，以便已存在的
分支不必重建；这里仅保留 Runtime 真正使用的字段。旧的 Baseline、认知模式和
表达计划字段由退役迁移移除，不能再成为新的 Runtime 依赖。
"""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


class Branch(Base):
    """一条独立运行的虚拟时间线。"""

    __tablename__ = "branches"
    __table_args__ = (UniqueConstraint("id", "project_id", name="ux_branches_id_project_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    # 旧分支引用仅保留可读；新入口不要求事件节点或训练产物。
    origin_event_id: Mapped[str | None] = mapped_column(ForeignKey("event_nodes.id"), nullable=True)
    model_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    origin_boundary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    title: Mapped[str] = mapped_column(String(255))
    origin_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    lifecycle_status: Mapped[str] = mapped_column(String(16), default="active")
    # 保留数据库列名，但仅表示分支执行代际：时钟控制、显式背景切换等使旧任务失效。
    # 普通聊天消息不递增；消息顺序/待处理状态由消息和事件队列维护。
    runtime_input_revision: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class BranchMessage(Base):
    """Runtime 的用户输入与已提交人物表达。

    气泡、输入窗口与幂等客户端 ID 仍是用户界面需要的投影；它们不是旧
    ConversationActor 的表达计划或后台 pending-bubble 队列。
    """

    __tablename__ = "branch_messages"
    __table_args__ = (
        UniqueConstraint("branch_id", "sequence"),
        UniqueConstraint("branch_id", "turn_id", "bubble_index"),
        UniqueConstraint("branch_id", "client_message_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(String(16), default="text")
    media_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("media_assets.id", ondelete="SET NULL"), nullable=True
    )
    turn_id: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()), index=True)
    bubble_index: Mapped[int] = mapped_column(Integer, default=0)
    delay_ms: Mapped[int] = mapped_column(Integer, default=0)
    generation_status: Mapped[str] = mapped_column(String(16), default="completed", index=True)
    generation_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    client_message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_proactive: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
