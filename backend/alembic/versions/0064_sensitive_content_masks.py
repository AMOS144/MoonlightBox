"""项目级敏感内容遮罩注册表：云端风控拒绝后定位到的消息跨流程复用。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0064_sensitive_content_masks"
down_revision: str | None = "0063_probabilistic_life_cursor"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if "sensitive_content_masks" in sa.inspect(op.get_bind()).get_table_names():
        return
    op.create_table(
        "sensitive_content_masks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("message_ref", sa.String(64), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "message_ref"),
    )
    op.create_index(
        "ix_sensitive_content_masks_project_id", "sensitive_content_masks", ["project_id"]
    )


def downgrade() -> None:
    bind = op.get_bind()
    if "sensitive_content_masks" not in sa.inspect(bind).get_table_names():
        return
    op.drop_index("ix_sensitive_content_masks_project_id", "sensitive_content_masks")
    op.drop_table("sensitive_content_masks")
