"""为节点增加本地生成的展示摘要。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0022_add_event_display_summaries"
down_revision: str | None = "0021_include_origin_event_in_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "event_nodes",
        sa.Column("display_summary", sa.Text(), nullable=True),
    )
    op.add_column(
        "event_nodes",
        sa.Column(
            "summary_status",
            sa.String(length=32),
            nullable=False,
            server_default="pending",
        ),
    )
    op.add_column(
        "event_nodes",
        sa.Column("summary_model", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("event_nodes", "summary_model")
    op.drop_column("event_nodes", "summary_status")
    op.drop_column("event_nodes", "display_summary")
