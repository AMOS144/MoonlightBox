"""Add auditable local semantic annotations for reusable media."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0032_add_media_semantic_annotations"
down_revision: str | None = "0031_repair_media_asset_kinds"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "media_semantic_annotations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "asset_id",
            sa.String(36),
            sa.ForeignKey("media_assets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("modality", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("transcript", sa.Text(), nullable=False, server_default=""),
        sa.Column("ocr_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("safety_tags", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("source_model", sa.String(255), nullable=False, server_default=""),
        sa.Column("source_version", sa.String(64), nullable=False, server_default=""),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("reusable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "reuse_decision", sa.String(16), nullable=False, server_default="pending"
        ),
        sa.Column("failure_code", sa.String(64), nullable=True),
        sa.Column("review_source", sa.String(64), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("asset_id", name="uq_media_semantic_annotation_asset"),
        sa.CheckConstraint(
            "status IN ('pending', 'succeeded', 'failed', 'needs_review')",
            name="ck_media_semantic_annotations_status",
        ),
        sa.CheckConstraint(
            "reuse_decision IN ('pending', 'approved', 'blocked')",
            name="ck_media_semantic_annotations_reuse",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_media_semantic_annotations_confidence",
        ),
    )
    op.create_index(
        "ix_media_semantic_annotations_project_id",
        "media_semantic_annotations",
        ["project_id"],
    )
    op.create_index(
        "ix_media_semantic_annotations_asset_id",
        "media_semantic_annotations",
        ["asset_id"],
    )
    op.create_index(
        "ix_media_semantic_annotations_modality",
        "media_semantic_annotations",
        ["modality"],
    )
    op.create_index(
        "ix_media_semantic_annotations_status",
        "media_semantic_annotations",
        ["status"],
    )


def downgrade() -> None:
    op.drop_table("media_semantic_annotations")
