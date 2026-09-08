"""创建分析运行与事件候选表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_create_analysis_runs"
down_revision: str | Sequence[str] | None = "0006_create_branches"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "uq_import_sources_id_project_id",
        "import_sources",
        ["id", "project_id"],
        unique=True,
    )
    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("import_id", sa.String(36), nullable=False),
        sa.Column("analysis_version", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(128), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("config_fingerprint", sa.String(64), nullable=False),
        sa.Column("window_ids", sa.JSON(), nullable=False),
        sa.Column("window_manifest_fingerprint", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("total_windows", sa.Integer(), nullable=False),
        sa.Column("completed_windows", sa.Integer(), nullable=False),
        sa.Column("checkpoint", sa.Integer(), nullable=False),
        sa.Column("error_category", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["import_id", "project_id"],
            ["import_sources.id", "import_sources.project_id"],
            ondelete="CASCADE",
            name="fk_analysis_runs_import_project",
        ),
        sa.UniqueConstraint(
            "import_id",
            "analysis_version",
            "prompt_version",
            "model",
            "config_fingerprint",
            name="uq_analysis_runs_identity",
        ),
        sa.CheckConstraint(
            "total_windows >= 0",
            name="ck_analysis_runs_total_windows",
        ),
        sa.CheckConstraint(
            "json_type(window_ids) = 'array' AND json_array_length(window_ids) = total_windows",
            name="ck_analysis_runs_window_manifest",
        ),
        sa.CheckConstraint(
            "completed_windows >= 0 AND completed_windows <= total_windows",
            name="ck_analysis_runs_completed_windows",
        ),
        sa.CheckConstraint(
            "checkpoint >= 0 AND checkpoint <= total_windows",
            name="ck_analysis_runs_checkpoint",
        ),
        sa.CheckConstraint(
            "completed_windows = checkpoint",
            name="ck_analysis_runs_progress_match",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'interrupted', 'failed', 'succeeded', 'cancelled')",
            name="ck_analysis_runs_status",
        ),
        sa.CheckConstraint(
            "(status IN ('succeeded', 'failed', 'cancelled') "
            "AND completed_at IS NOT NULL) OR "
            "(status IN ('queued', 'running', 'interrupted') "
            "AND completed_at IS NULL)",
            name="ck_analysis_runs_completion_time",
        ),
        sa.CheckConstraint(
            "status != 'succeeded' OR completed_windows = total_windows",
            name="ck_analysis_runs_succeeded_progress",
        ),
        sa.CheckConstraint(
            "(status IN ('failed', 'interrupted') "
            "AND error_category IS NOT NULL "
            "AND length(trim(error_category)) > 0 "
            "AND error_message IS NOT NULL "
            "AND length(trim(error_message)) > 0) OR "
            "(status NOT IN ('failed', 'interrupted') "
            "AND error_category IS NULL AND error_message IS NULL)",
            name="ck_analysis_runs_error_fields",
        ),
    )
    op.create_index("ix_analysis_runs_project_id", "analysis_runs", ["project_id"])
    op.create_index("ix_analysis_runs_import_id", "analysis_runs", ["import_id"])
    op.create_index("ix_analysis_runs_status", "analysis_runs", ["status"])

    op.create_table(
        "event_candidates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("window_id", sa.String(255), nullable=False),
        sa.Column("candidate_key", sa.String(255), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("review_payload", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["analysis_runs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "run_id",
            "window_id",
            "candidate_key",
            name="uq_event_candidates_window_key",
        ),
        sa.CheckConstraint(
            "status IN ('pending_review', 'accepted', 'rejected')",
            name="ck_event_candidates_status",
        ),
        sa.CheckConstraint(
            "(status = 'rejected' AND rejection_reason IS NOT NULL "
            "AND length(trim(rejection_reason)) > 0) OR "
            "(status != 'rejected' AND rejection_reason IS NULL)",
            name="ck_event_candidates_rejection_reason",
        ),
    )
    op.create_index("ix_event_candidates_run_id", "event_candidates", ["run_id"])
    op.create_index(
        "ix_event_candidates_run_status",
        "event_candidates",
        ["run_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_event_candidates_run_status", "event_candidates")
    op.drop_index("ix_event_candidates_run_id", "event_candidates")
    op.drop_table("event_candidates")
    op.drop_index("ix_analysis_runs_status", "analysis_runs")
    op.drop_index("ix_analysis_runs_import_id", "analysis_runs")
    op.drop_index("ix_analysis_runs_project_id", "analysis_runs")
    op.drop_table("analysis_runs")
    op.drop_index("uq_import_sources_id_project_id", "import_sources")
