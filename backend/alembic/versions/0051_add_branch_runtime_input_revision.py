"""为 Runtime 分支增加持久化输入版本。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0051_add_branch_runtime_input_revision"
down_revision: str | None = "0050_add_unified_agent_runtime"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns("branches")}
    if "runtime_input_revision" not in columns:
        op.add_column(
            "branches",
            sa.Column(
                "runtime_input_revision",
                sa.Integer(),
                nullable=False,
                server_default="1",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns("branches")}
    if "runtime_input_revision" in columns:
        with op.batch_alter_table("branches") as batch:
            batch.drop_column("runtime_input_revision")
