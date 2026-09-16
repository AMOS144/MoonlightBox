"""为人物世界纠正会话增加回合、乐观锁和上下文快照协议。"""

from collections.abc import Sequence

import moonlightbox.world.models  # noqa: F401
import sqlalchemy as sa
from alembic import op
from moonlightbox.db import Base

revision: str = "0048_add_revision_context_protocol"
down_revision: str | None = "0047_add_section_evidence_ledger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    session_columns = {
        item["name"] for item in sa.inspect(bind).get_columns("person_world_revision_sessions")
    }
    session_additions = (
        (
            "session_revision",
            sa.Column("session_revision", sa.Integer(), nullable=False, server_default="0"),
        ),
        (
            "scope",
            sa.Column("scope", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        ),
        ("context_snapshot_id", sa.Column("context_snapshot_id", sa.String(36), nullable=True)),
        ("pending_turn_id", sa.Column("pending_turn_id", sa.String(36), nullable=True)),
        (
            "understanding_payload_hash",
            sa.Column("understanding_payload_hash", sa.String(64), nullable=True),
        ),
    )
    for name, column in session_additions:
        if name not in session_columns:
            op.add_column("person_world_revision_sessions", column)

    message_columns = {
        item["name"] for item in sa.inspect(bind).get_columns("person_world_revision_messages")
    }
    message_additions = (
        ("turn_id", sa.Column("turn_id", sa.String(36), nullable=False, server_default="")),
        ("kind", sa.Column("kind", sa.String(32), nullable=False, server_default="message")),
        ("in_reply_to_turn_id", sa.Column("in_reply_to_turn_id", sa.String(36), nullable=True)),
        ("context_snapshot_id", sa.Column("context_snapshot_id", sa.String(36), nullable=True)),
        (
            "session_revision",
            sa.Column("session_revision", sa.Integer(), nullable=False, server_default="0"),
        ),
        ("idempotency_key", sa.Column("idempotency_key", sa.String(120), nullable=True)),
    )
    for name, column in message_additions:
        if name not in message_columns:
            op.add_column("person_world_revision_messages", column)
    Base.metadata.tables["person_world_revision_context_snapshots"].create(
        bind=bind, checkfirst=True
    )


def downgrade() -> None:
    bind = op.get_bind()
    if "person_world_revision_context_snapshots" in sa.inspect(bind).get_table_names():
        op.drop_table("person_world_revision_context_snapshots")
    message_columns = {
        item["name"] for item in sa.inspect(bind).get_columns("person_world_revision_messages")
    }
    with op.batch_alter_table("person_world_revision_messages") as batch:
        for name in (
            "idempotency_key",
            "session_revision",
            "context_snapshot_id",
            "in_reply_to_turn_id",
            "kind",
            "turn_id",
        ):
            if name in message_columns:
                batch.drop_column(name)
    session_columns = {
        item["name"] for item in sa.inspect(bind).get_columns("person_world_revision_sessions")
    }
    with op.batch_alter_table("person_world_revision_sessions") as batch:
        for name in (
            "understanding_payload_hash",
            "pending_turn_id",
            "context_snapshot_id",
            "scope",
            "session_revision",
        ):
            if name in session_columns:
                batch.drop_column(name)
