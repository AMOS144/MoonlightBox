"""新节点调查工作台；历史事件及分支不改写。"""

import sqlalchemy as sa
from alembic import op

revision = "0060_node_investigations"
down_revision = "0059_runtime_event_received_at"
branch_labels = None
depends_on = None


def upgrade():
    if "node_investigations" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "node_investigations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("dataset_version", sa.String(64), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "dataset_version"),
    )


def downgrade():
    op.drop_table("node_investigations")
