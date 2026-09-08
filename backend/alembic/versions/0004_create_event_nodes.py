"""创建事件节点及修订历史表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_create_event_nodes"
down_revision: str | Sequence[str] | None = "0003_create_import_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "event_nodes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("start_message_id", sa.String(255), nullable=False),
        sa.Column("end_message_id", sa.String(255), nullable=False),
        sa.Column("before_state", sa.Text(), nullable=False),
        sa.Column("after_state", sa.Text(), nullable=False),
        sa.Column("emotion_labels", sa.JSON(), nullable=False),
        sa.Column("topic", sa.Text(), nullable=False),
        sa.Column("conflict_level", sa.Integer(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("evidence_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_event_nodes_project_id", "event_nodes", ["project_id"])
    op.create_table(
        "analysis_revisions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("event_id", sa.String(36), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("action_reason", sa.Text(), nullable=False),
        sa.Column("analysis_version", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["event_nodes.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("event_id", "revision_number"),
    )
    op.create_index("ix_analysis_revisions_event_id", "analysis_revisions", ["event_id"])


def downgrade() -> None:
    op.drop_index("ix_analysis_revisions_event_id", "analysis_revisions")
    op.drop_table("analysis_revisions")
    op.drop_index("ix_event_nodes_project_id", "event_nodes")
    op.drop_table("event_nodes")
