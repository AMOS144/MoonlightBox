"""可恢复业务状态，不另建 Agent 调用观测账本。"""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


class LifeOpportunityRow(Base):
    __tablename__ = "runtime_life_opportunities"
    __table_args__ = (UniqueConstraint("branch_id", "check_key", name="uq_life_check"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), index=True
    )
    check_key: Mapped[str | None] = mapped_column(String(120), nullable=True)
    kind: Mapped[str] = mapped_column(String(24), default="opportunity", server_default="legacy")
    status: Mapped[str] = mapped_column(String(24), default="scheduled", index=True)
    seed: Mapped[str] = mapped_column(String(36), default=lambda: str(uuid4()))
    policy: Mapped[dict] = mapped_column(JSON, default=dict)
    brief: Mapped[dict] = mapped_column(JSON, default=dict)
    cursor_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # 下面的 hazard / expires_at / attempts 仅保留旧记录可读性，不参与新版调度或重试。
    hazard: Mapped[float] = mapped_column(default=0.0)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    work: Mapped[dict] = mapped_column(JSON, default=dict)
    step: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    event_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(120), nullable=True)


class LifeScheduleCursor(Base):
    """每分支一条调度游标；未命中也推进，不以最后一个机会代替节拍。"""

    __tablename__ = "runtime_life_schedule_cursors"
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), primary_key=True
    )
    sequence: Mapped[int] = mapped_column(Integer, default=0)
    last_check_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    next_check_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_wall: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_virtual: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    interval_minutes: Mapped[int] = mapped_column(Integer)
    clock_status: Mapped[str] = mapped_column(String(20), default="running")
    last_check: Mapped[dict] = mapped_column(JSON, default=dict)


class DayPlanVersionRow(Base):
    __tablename__ = "runtime_day_plan_versions"
    # 计划行 ID + 版本作为不可变归档身份。
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), index=True
    )
    plan_date: Mapped[str] = mapped_column(String(10))
    version: Mapped[int] = mapped_column(Integer)
    blocks: Mapped[list] = mapped_column(JSON)
    generation_metadata: Mapped[dict] = mapped_column(JSON)
