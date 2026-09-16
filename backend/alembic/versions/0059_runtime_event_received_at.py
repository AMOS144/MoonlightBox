"""接入顺序独立于事件发生时间；不新增时间片或邮箱实体。"""

import sqlalchemy as sa
from alembic import op

revision = "0059_runtime_event_received_at"
down_revision = "0058_director_initialization"
branch_labels = None
depends_on = None


def upgrade():
    columns = {c["name"]: c for c in sa.inspect(op.get_bind()).get_columns("runtime_events")}
    if "received_at" not in columns:
        op.add_column("runtime_events", sa.Column("received_at", sa.DateTime(timezone=True)))
    # 历史队列未保存接入时刻，仅以发生时刻回填；新输入记录真实接入时间。
    op.execute("UPDATE runtime_events SET received_at = occurred_at WHERE received_at IS NULL")
    if columns.get("received_at", {}).get("nullable", True):
        with op.batch_alter_table("runtime_events") as batch:
            batch.alter_column(
                "received_at", nullable=False, existing_type=sa.DateTime(timezone=True)
            )


def downgrade():
    with op.batch_alter_table("runtime_events") as batch:
        batch.drop_column("received_at")
