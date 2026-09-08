"""增加时间轴确认记录和模型启用状态。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_add_timeline_confirmations"
down_revision: str | Sequence[str] | None = "0012_add_v3_event_contract"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "timeline_confirmations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "import_id",
            sa.String(36),
            sa.ForeignKey("import_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "analysis_run_id",
            sa.String(36),
            sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("confirmation_fingerprint", sa.String(64), nullable=False),
        sa.Column("active_event_ids", sa.JSON(), nullable=False),
        sa.Column("rejected_event_ids", sa.JSON(), nullable=False),
        sa.Column("event_revision_snapshots", sa.JSON(), nullable=False),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("training_job_id", sa.String(36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ux_timeline_confirmations_fingerprint",
        "timeline_confirmations",
        ["confirmation_fingerprint"],
        unique=True,
    )
    op.create_index(
        "ix_timeline_confirmations_project_id",
        "timeline_confirmations",
        ["project_id"],
    )
    op.create_index(
        "ix_timeline_confirmations_import_id",
        "timeline_confirmations",
        ["import_id"],
    )
    op.create_index(
        "ix_timeline_confirmations_analysis_run_id",
        "timeline_confirmations",
        ["analysis_run_id"],
    )

    with op.batch_alter_table("model_versions") as batch:
        batch.add_column(
            sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch.add_column(sa.Column("timeline_confirmation_id", sa.String(36), nullable=True))
        batch.add_column(sa.Column("training_job_id", sa.String(36), nullable=True))
        batch.add_column(
            sa.Column("training_config", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.create_foreign_key(
            "fk_model_versions_timeline_confirmation",
            "timeline_confirmations",
            ["timeline_confirmation_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index("ix_model_versions_active", ["active"])
        batch.create_index(
            "ix_model_versions_timeline_confirmation_id",
            ["timeline_confirmation_id"],
        )
        batch.create_index("ix_model_versions_training_job_id", ["training_job_id"])


def downgrade() -> None:
    with op.batch_alter_table("model_versions") as batch:
        batch.drop_index("ix_model_versions_training_job_id")
        batch.drop_index("ix_model_versions_timeline_confirmation_id")
        batch.drop_index("ix_model_versions_active")
        batch.drop_constraint(
            "fk_model_versions_timeline_confirmation",
            type_="foreignkey",
        )
        batch.drop_column("training_config")
        batch.drop_column("training_job_id")
        batch.drop_column("timeline_confirmation_id")
        batch.drop_column("active")

    op.drop_index(
        "ix_timeline_confirmations_analysis_run_id",
        table_name="timeline_confirmations",
    )
    op.drop_index(
        "ix_timeline_confirmations_import_id",
        table_name="timeline_confirmations",
    )
    op.drop_index(
        "ix_timeline_confirmations_project_id",
        table_name="timeline_confirmations",
    )
    op.drop_index(
        "ux_timeline_confirmations_fingerprint",
        table_name="timeline_confirmations",
    )
    op.drop_table("timeline_confirmations")
