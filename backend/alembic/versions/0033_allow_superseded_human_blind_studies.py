"""Allow newer candidate studies to retire stale open blind tests."""

from collections.abc import Sequence

from alembic import op

revision: str = "0033_allow_superseded_human_blind_studies"
down_revision: str | None = "0032_add_media_semantic_annotations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("human_blind_studies") as batch:
        batch.drop_constraint("ck_human_blind_studies_status", type_="check")
        batch.create_check_constraint(
            "ck_human_blind_studies_status",
            "status IN ('open', 'passed', 'failed', 'superseded')",
        )


def downgrade() -> None:
    op.execute(
        "UPDATE human_blind_studies SET status = 'failed' "
        "WHERE status = 'superseded'"
    )
    with op.batch_alter_table("human_blind_studies") as batch:
        batch.drop_constraint("ck_human_blind_studies_status", type_="check")
        batch.create_check_constraint(
            "ck_human_blind_studies_status",
            "status IN ('open', 'passed', 'failed')",
        )
