"""Repair legacy generic image kinds using linked message and MIME evidence."""

from collections.abc import Sequence

from alembic import op

revision: str = "0031_repair_media_asset_kinds"
down_revision: str | None = "0030_human_blind_acceptance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Older generic importers persisted every attachment as kind=image. MIME
    # is authoritative for audio/video; linked message kind distinguishes a
    # sticker from a normal image without inspecting private asset contents.
    op.execute(
        "UPDATE media_assets SET kind = 'audio' "
        "WHERE lower(mime_type) LIKE 'audio/%'"
    )
    op.execute(
        "UPDATE media_assets SET kind = 'video' "
        "WHERE lower(mime_type) LIKE 'video/%'"
    )
    op.execute(
        "UPDATE media_assets SET kind = 'sticker' "
        "WHERE id IN ("
        "SELECT media_asset_id FROM messages "
        "WHERE kind = 'sticker' AND media_asset_id IS NOT NULL"
        ")"
    )
    op.execute(
        "UPDATE media_assets SET kind = 'image' "
        "WHERE id IN ("
        "SELECT media_asset_id FROM messages "
        "WHERE kind = 'image' AND media_asset_id IS NOT NULL"
        ") AND kind != 'sticker'"
    )


def downgrade() -> None:
    # This is a lossless data repair, not a schema contract. Restoring the
    # known-wrong generic kind would make existing media unusable again.
    pass
