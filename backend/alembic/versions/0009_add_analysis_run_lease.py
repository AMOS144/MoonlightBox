"""为分析运行增加可恢复的数据库 lease。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_add_analysis_run_lease"
down_revision: str | Sequence[str] | None = "0008_add_analysis_revision_run_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analysis_runs") as batch:
        batch.add_column(sa.Column("lease_owner", sa.String(255), nullable=True))
        batch.add_column(sa.Column("lease_token", sa.String(36), nullable=True))
        batch.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_index("ix_analysis_runs_lease_token", ["lease_token"])
        batch.create_check_constraint(
            "ck_analysis_runs_lease_fields",
            "(lease_owner IS NULL AND lease_token IS NULL "
            "AND lease_expires_at IS NULL) OR "
            "(lease_owner IS NOT NULL AND length(trim(lease_owner)) > 0 "
            "AND lease_token IS NOT NULL AND length(trim(lease_token)) > 0 "
            "AND lease_expires_at IS NOT NULL)",
        )


def downgrade() -> None:
    with op.batch_alter_table("analysis_runs") as batch:
        batch.drop_constraint("ck_analysis_runs_lease_fields", type_="check")
        batch.drop_index("ix_analysis_runs_lease_token")
        batch.drop_column("lease_expires_at")
        batch.drop_column("lease_token")
        batch.drop_column("lease_owner")
