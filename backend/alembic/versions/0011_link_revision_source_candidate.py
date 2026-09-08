"""为事件修订关联准确的来源候选。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_link_revision_source_candidate"
down_revision: str | Sequence[str] | None = "0010_add_job_lease_and_dedupe"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analysis_revisions") as batch:
        batch.add_column(sa.Column("candidate_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_analysis_revisions_candidate_id",
            "event_candidates",
            ["candidate_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index("ix_analysis_revisions_candidate_id", ["candidate_id"])


def downgrade() -> None:
    with op.batch_alter_table("analysis_revisions") as batch:
        batch.drop_index("ix_analysis_revisions_candidate_id")
        batch.drop_constraint(
            "fk_analysis_revisions_candidate_id",
            type_="foreignkey",
        )
        batch.drop_column("candidate_id")
