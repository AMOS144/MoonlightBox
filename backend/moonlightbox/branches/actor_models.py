from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ConversationActorState(Base):
    __tablename__ = "conversation_actor_states"
    __table_args__ = (
        Index("ix_conversation_actor_states_next_review", "next_review_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), unique=True
    )
    attention_focus: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    shared_ground: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    open_sequences: Mapped[list[dict[str, object]]] = mapped_column(JSON, default=list)
    expression_intentions: Mapped[list[dict[str, object]]] = mapped_column(
        JSON, default=list
    )
    approach_motivation: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    inhibition: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    observed_message_sequence: Mapped[int] = mapped_column(Integer, default=-1)
    status: Mapped[str] = mapped_column(String(16), default="idle")
    user_typing_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    assistant_typing_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    next_review_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


def assistant_typing_active(
    state: ConversationActorState,
    *,
    now: datetime | None = None,
) -> bool:
    """输入提示只反映限时的数字人表达阶段。"""

    current = now or datetime.now(UTC)
    until = state.assistant_typing_until
    if until is None or state.status not in {"thinking", "expressing"}:
        return False
    aware_until = until if until.tzinfo is not None else until.replace(tzinfo=UTC)
    aware_current = (
        current if current.tzinfo is not None else current.replace(tzinfo=UTC)
    )
    return aware_until > aware_current


class ConversationExpressionPlan(Base):
    __tablename__ = "conversation_expression_plans"
    __table_args__ = (
        Index("ix_conversation_expression_plans_branch_status", "branch_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE")
    )
    trigger_sequence_from: Mapped[int] = mapped_column(Integer)
    trigger_sequence_to: Mapped[int] = mapped_column(Integer)
    intent: Mapped[str] = mapped_column(Text)
    state_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="planned")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    proactive: Mapped[bool] = mapped_column(Boolean, default=False)
    generation_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ConversationPendingBubble(Base):
    __tablename__ = "conversation_pending_bubbles"
    __table_args__ = (
        Index("ix_conversation_pending_bubbles_plan_status", "plan_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    plan_id: Mapped[str] = mapped_column(
        ForeignKey("conversation_expression_plans.id", ondelete="CASCADE")
    )
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE")
    )
    position: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text, default="")
    message_type: Mapped[str] = mapped_column(String(16), default="text")
    media_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("media_assets.id", ondelete="SET NULL"), nullable=True
    )
    delay_ms: Mapped[int] = mapped_column(Integer, default=0)
    send_not_before: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    basis_sequence: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    sent_message_id: Mapped[str | None] = mapped_column(
        ForeignKey("branch_messages.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
