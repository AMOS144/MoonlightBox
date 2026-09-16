"""记录 Runtime Cycle 的本地诊断轨迹。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0042_add_runtime_cycle_traces"
down_revision: str | None = "0041_retire_legacy_runtime"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "runtime_cycle_traces",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("project_id", sa.String(length=36), nullable=False),
        sa.Column("branch_id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("cycle_key", sa.String(length=180), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("stage", sa.String(length=48), nullable=False),
        sa.Column("trigger_event_ids", sa.JSON(), nullable=False),
        sa.Column("wakeup_ids", sa.JSON(), nullable=False),
        sa.Column("virtual_now", sa.DateTime(timezone=True), nullable=False),
        sa.Column("packet", sa.JSON(), nullable=False),
        sa.Column("director_steps", sa.JSON(), nullable=False),
        sa.Column("actor_steps", sa.JSON(), nullable=False),
        sa.Column("director_decision", sa.JSON(), nullable=True),
        sa.Column("committed_decision", sa.JSON(), nullable=True),
        sa.Column("outcome", sa.JSON(), nullable=False),
        sa.Column("error_code", sa.String(length=80), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["branch_id"], ["branches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_runtime_cycle_traces_branch_started",
        "runtime_cycle_traces",
        ["branch_id", "started_at"],
    )
    op.create_index("ix_runtime_cycle_traces_job", "runtime_cycle_traces", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_runtime_cycle_traces_job", table_name="runtime_cycle_traces")
    op.drop_index("ix_runtime_cycle_traces_branch_started", table_name="runtime_cycle_traces")
    op.drop_table("runtime_cycle_traces")
