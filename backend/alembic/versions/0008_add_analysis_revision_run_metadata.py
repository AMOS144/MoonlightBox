"""为事件修订增加模型与分析运行审计字段。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_add_analysis_revision_run_metadata"
down_revision: str | Sequence[str] | None = "0007_create_analysis_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analysis_revisions") as batch:
        batch.add_column(sa.Column("model", sa.String(255), nullable=True))
        batch.add_column(sa.Column("run_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_analysis_revisions_run_id",
            "analysis_runs",
            ["run_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index("ix_analysis_revisions_run_id", ["run_id"])


def downgrade() -> None:
    with op.batch_alter_table("analysis_revisions") as batch:
        batch.drop_index("ix_analysis_revisions_run_id")
        batch.drop_constraint(
            "fk_analysis_revisions_run_id",
            type_="foreignkey",
        )
        batch.drop_column("run_id")
        batch.drop_column("model")
