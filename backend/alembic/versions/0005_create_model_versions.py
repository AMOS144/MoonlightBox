"""创建模型版本表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_create_model_versions"
down_revision: str | Sequence[str] | None = "0004_create_event_nodes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("base_model", sa.String(255), nullable=False),
        sa.Column("adapter_path", sa.String(500), nullable=False),
        sa.Column("dataset_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("recommended", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_model_versions_project_id", "model_versions", ["project_id"])


def downgrade() -> None:
    op.drop_index("ix_model_versions_project_id", "model_versions")
    op.drop_table("model_versions")
