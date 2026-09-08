"""增加分支升级与生成策略字段。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016_add_branch_upgrade_fields"
down_revision: str | None = "0015_add_media_assets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("branches") as batch_op:
        batch_op.add_column(
            sa.Column(
                "lifecycle_status",
                sa.String(length=16),
                nullable=False,
                server_default="active",
            )
        )
        batch_op.add_column(
            sa.Column(
                "replacement_branch_id",
                sa.String(length=36),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "generation_policy_version",
                sa.String(length=32),
                nullable=False,
                server_default="legacy",
            )
        )
        batch_op.create_foreign_key(
            "fk_branch_replacement",
            "branches",
            ["replacement_branch_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_branches_replacement_branch_id",
            ["replacement_branch_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("branches") as batch_op:
        batch_op.drop_index("ix_branches_replacement_branch_id")
        batch_op.drop_constraint("fk_branch_replacement", type_="foreignkey")
        batch_op.drop_column("generation_policy_version")
        batch_op.drop_column("replacement_branch_id")
        batch_op.drop_column("lifecycle_status")
