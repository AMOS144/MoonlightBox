"""让新分支基础历史包含所选节点的完整对话。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021_include_origin_event_in_baseline"
down_revision: str | None = "0020_quarantine_legacy_branches"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "branch_baseline_manifests",
        sa.Column(
            "boundary_inclusive",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("branch_baseline_manifests", "boundary_inclusive")
