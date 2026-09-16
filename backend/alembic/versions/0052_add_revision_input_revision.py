"""将 PersonWorld 纠正的用户输入版本与 SSE 会话版本分离。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0052_add_revision_input_revision"
down_revision: str | None = "0051_add_branch_runtime_input_revision"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {
        item["name"] for item in sa.inspect(bind).get_columns("person_world_revision_sessions")
    }
    if "input_revision" not in columns:
        op.add_column(
            "person_world_revision_sessions",
            sa.Column("input_revision", sa.Integer(), nullable=False, server_default="1"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    columns = {
        item["name"] for item in sa.inspect(bind).get_columns("person_world_revision_sessions")
    }
    if "input_revision" in columns:
        with op.batch_alter_table("person_world_revision_sessions") as batch:
            batch.drop_column("input_revision")
