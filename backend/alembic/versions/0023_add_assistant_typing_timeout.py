"""为数字人输入提示增加独立超时时间。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023_add_assistant_typing_timeout"
down_revision: str | None = "0022_add_event_display_summaries"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversation_actor_states",
        sa.Column("assistant_typing_until", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversation_actor_states", "assistant_typing_until")
