"""保留触发一次回复的全部用户贡献及其证据标识。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0028_episode_user_messages"
down_revision: str | None = "0027_state_field_provenance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("branch_memory_episodes") as batch:
        batch.add_column(
            sa.Column(
                "user_messages",
                sa.JSON(),
                nullable=False,
                server_default="[]",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("branch_memory_episodes") as batch:
        batch.drop_column("user_messages")
