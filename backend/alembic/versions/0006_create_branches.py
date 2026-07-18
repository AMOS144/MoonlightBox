"""创建平行时间分支及消息表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_create_branches"
down_revision: str | Sequence[str] | None = "0005_create_model_versions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "branches",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("origin_event_id", sa.String(36), nullable=False),
        sa.Column("model_version_id", sa.String(36), nullable=False),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("origin_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("state_snapshot", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["origin_event_id"], ["event_nodes.id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
    )
    op.create_index("ix_branches_project_id", "branches", ["project_id"])
    op.create_table(
        "branch_messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("branch_id", sa.String(36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("generation_metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["branch_id"], ["branches.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("branch_id", "sequence"),
    )
    op.create_index(
        "ix_branch_messages_branch_id", "branch_messages", ["branch_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_branch_messages_branch_id", "branch_messages")
    op.drop_table("branch_messages")
    op.drop_index("ix_branches_project_id", "branches")
    op.drop_table("branches")
