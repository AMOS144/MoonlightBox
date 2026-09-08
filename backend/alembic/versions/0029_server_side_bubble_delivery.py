"""Store expression metadata for server-side bubble delivery."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0029_server_side_bubble_delivery"
down_revision: str | None = "0028_episode_user_messages"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("conversation_expression_plans") as batch:
        batch.add_column(
            sa.Column(
                "generation_metadata",
                sa.JSON(),
                nullable=False,
                server_default="{}",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("conversation_expression_plans") as batch:
        batch.drop_column("generation_metadata")
