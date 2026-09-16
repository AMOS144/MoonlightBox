"""隔离分支起点调查状态，不迁移或重置已有分支心理状态。"""

import sqlalchemy as sa
from alembic import op

revision = "0058_director_initialization"
down_revision = "0057_job_cancellation_ack"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "runtime_initializations",
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("work", sa.JSON(), nullable=False),
        sa.Column("error_code", sa.String(120), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("runtime_initializations")
