"""为后台任务增加租约与原子幂等键。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_add_job_lease_and_dedupe"
down_revision: str | Sequence[str] | None = "0009_add_analysis_run_lease"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 旧版 running 任务没有租约，无法证明仍有存活 Worker，先安全中断。
    op.execute(
        sa.text(
            "UPDATE jobs "
            "SET status = 'interrupted', "
            "error_code = 'worker_interrupted', "
            "error_message = 'Worker 升级后恢复遗留运行任务' "
            "WHERE status = 'running'"
        )
    )
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("dedupe_key", sa.String(255), nullable=True))
        batch.add_column(sa.Column("worker_token", sa.String(36), nullable=True))
        batch.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_index("ux_jobs_dedupe_key", ["dedupe_key"], unique=True)
        batch.create_check_constraint(
            "ck_jobs_lease_fields",
            "(status = 'running' "
            "AND worker_token IS NOT NULL "
            "AND length(trim(worker_token)) > 0 "
            "AND lease_expires_at IS NOT NULL) OR "
            "(status != 'running' "
            "AND worker_token IS NULL "
            "AND lease_expires_at IS NULL)",
        )


def downgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("ck_jobs_lease_fields", type_="check")
        batch.drop_index("ux_jobs_dedupe_key")
        batch.drop_column("lease_expires_at")
        batch.drop_column("worker_token")
        batch.drop_column("dedupe_key")
