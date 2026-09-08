"""Run isolated candidate branches through authoritative two-stage cognition."""

from collections.abc import Sequence

from alembic import op

revision: str = "0034_add_candidate_preview_mode"
down_revision: str | None = "0033_allow_superseded_human_blind_studies"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("branches") as batch:
        batch.drop_constraint("ck_branches_subject_agent_mode", type_="check")
        batch.create_check_constraint(
            "ck_branches_subject_agent_mode",
            "subject_agent_mode IN ('shadow', 'preview', 'active')",
        )
    op.execute(
        "UPDATE branches SET subject_agent_mode = 'preview' "
        "WHERE generation_policy_version = 'subject-v2-candidate-v1' "
        "AND subject_agent_mode = 'shadow'"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE branches SET subject_agent_mode = 'shadow' "
        "WHERE subject_agent_mode = 'preview'"
    )
    with op.batch_alter_table("branches") as batch:
        batch.drop_constraint("ck_branches_subject_agent_mode", type_="check")
        batch.create_check_constraint(
            "ck_branches_subject_agent_mode",
            "subject_agent_mode IN ('shadow', 'active')",
        )
