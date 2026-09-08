"""增加主体认知 Agent 影子验收与分支级激活门槛。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0026_subject_agent_activation"
down_revision: str | None = "0025_unique_cycle_trigger"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("branches") as batch:
        batch.add_column(
            sa.Column(
                "subject_agent_mode",
                sa.String(16),
                nullable=False,
                server_default="shadow",
            )
        )
        batch.create_check_constraint(
            "ck_branches_subject_agent_mode",
            "subject_agent_mode IN ('shadow', 'active')",
        )

    op.create_table(
        "subject_agent_acceptance_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("branch_id", sa.String(36), nullable=False),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column(
            "structure_extraction_success_rate",
            sa.Float(),
            nullable=False,
        ),
        sa.Column("fact_safety_rate", sa.Float(), nullable=False),
        sa.Column("expression_decision_accuracy", sa.Float(), nullable=False),
        sa.Column("p95_cognition_latency_ms", sa.Float(), nullable=False),
        sa.Column("direct_lora_p95", sa.Float(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column(
            "failure_reasons",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["branch_id", "project_id"],
            ["branches.id", "branches.project_id"],
            name="fk_subject_agent_acceptance_reports_branch_project",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint(
            "sample_count >= 0",
            name="ck_subject_agent_acceptance_reports_sample_count",
        ),
    )
    op.create_index(
        "ix_subject_agent_acceptance_reports_latest",
        "subject_agent_acceptance_reports",
        ["project_id", "branch_id", "model_version_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_subject_agent_acceptance_reports_latest",
        table_name="subject_agent_acceptance_reports",
    )
    op.drop_table("subject_agent_acceptance_reports")
    with op.batch_alter_table("branches") as batch:
        batch.drop_constraint(
            "ck_branches_subject_agent_mode",
            type_="check",
        )
        batch.drop_column("subject_agent_mode")
