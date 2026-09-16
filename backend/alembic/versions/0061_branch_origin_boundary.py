"""分支起点独立于事件和训练；旧引用与已有时钟原样保留。"""

import sqlalchemy as sa
from alembic import op

revision = "0061_branch_origin_boundary"
down_revision = "0060_node_investigations"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("branches") as batch:
        batch.alter_column("origin_event_id", existing_type=sa.String(36), nullable=True)
        batch.alter_column("model_version_id", existing_type=sa.String(36), nullable=True)
        batch.add_column(sa.Column("origin_boundary", sa.JSON(), nullable=True))


def downgrade():
    # 有新式分支时不能恢复旧的非空约束，更不能为了降级删数据。
    connection = op.get_bind()
    if connection.execute(
        sa.text(
            "SELECT count(*) FROM branches "
            "WHERE origin_event_id IS NULL OR model_version_id IS NULL"
        )
    ).scalar():
        raise RuntimeError("存在无旧引用的分支，不能安全降级")
    with op.batch_alter_table("branches") as batch:
        batch.drop_column("origin_boundary")
        batch.alter_column("origin_event_id", existing_type=sa.String(36), nullable=False)
        batch.alter_column("model_version_id", existing_type=sa.String(36), nullable=False)
