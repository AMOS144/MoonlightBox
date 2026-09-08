"""为分支消息增加轮次和多气泡字段。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014_add_branch_message_turn_fields"
down_revision: str | Sequence[str] | None = "0013_add_timeline_confirmations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("branch_messages") as batch:
        batch.add_column(sa.Column("turn_id", sa.String(36), nullable=True))
        batch.add_column(
            sa.Column(
                "bubble_index",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch.add_column(sa.Column("delay_ms", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(
            sa.Column(
                "generation_status",
                sa.String(16),
                nullable=False,
                server_default="completed",
            )
        )
    op.execute("UPDATE branch_messages SET turn_id = id WHERE turn_id IS NULL")
    with op.batch_alter_table("branch_messages") as batch:
        batch.alter_column("turn_id", existing_type=sa.String(36), nullable=False)
        batch.create_unique_constraint(
            "uq_branch_message_turn_bubble",
            ["branch_id", "turn_id", "bubble_index"],
        )
        batch.create_index("ix_branch_messages_turn_id", ["turn_id"])
        batch.create_index(
            "ix_branch_messages_generation_status",
            ["generation_status"],
        )


def downgrade() -> None:
    with op.batch_alter_table("branch_messages") as batch:
        batch.drop_index("ix_branch_messages_generation_status")
        batch.drop_index("ix_branch_messages_turn_id")
        batch.drop_constraint("uq_branch_message_turn_bubble", type_="unique")
        batch.drop_column("generation_status")
        batch.drop_column("delay_ms")
        batch.drop_column("bubble_index")
        batch.drop_column("turn_id")
