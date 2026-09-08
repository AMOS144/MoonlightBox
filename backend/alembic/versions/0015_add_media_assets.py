"""增加项目媒体资产和消息媒体引用。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_add_media_assets"
down_revision: str | Sequence[str] | None = "0014_add_branch_message_turn_fields"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "media_assets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("relative_path", sa.String(500), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("source_key", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "sha256", name="uq_media_asset_project_hash"),
    )
    op.create_index("ix_media_assets_project_id", "media_assets", ["project_id"])
    op.create_index("ix_media_assets_kind", "media_assets", ["kind"])
    with op.batch_alter_table("participants") as batch:
        batch.add_column(sa.Column("avatar_asset_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_participants_avatar_asset",
            "media_assets",
            ["avatar_asset_id"],
            ["id"],
            ondelete="SET NULL",
        )
    with op.batch_alter_table("messages") as batch:
        batch.add_column(sa.Column("media_asset_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_messages_media_asset",
            "media_assets",
            ["media_asset_id"],
            ["id"],
            ondelete="SET NULL",
        )
    with op.batch_alter_table("branch_messages") as batch:
        batch.add_column(sa.Column("type", sa.String(16), nullable=False, server_default="text"))
        batch.add_column(sa.Column("media_asset_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_branch_messages_media_asset",
            "media_assets",
            ["media_asset_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("branch_messages") as batch:
        batch.drop_constraint("fk_branch_messages_media_asset", type_="foreignkey")
        batch.drop_column("media_asset_id")
        batch.drop_column("type")
    with op.batch_alter_table("messages") as batch:
        batch.drop_constraint("fk_messages_media_asset", type_="foreignkey")
        batch.drop_column("media_asset_id")
    with op.batch_alter_table("participants") as batch:
        batch.drop_constraint("fk_participants_avatar_asset", type_="foreignkey")
        batch.drop_column("avatar_asset_id")
    op.drop_index("ix_media_assets_kind", table_name="media_assets")
    op.drop_index("ix_media_assets_project_id", table_name="media_assets")
    op.drop_table("media_assets")
