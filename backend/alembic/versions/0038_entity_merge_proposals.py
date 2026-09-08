from collections.abc import Sequence
import moonlightbox.world.models  # noqa: F401
from alembic import op
from moonlightbox.db import Base

revision: str = "0038_entity_merge_proposals"
down_revision: str | None = "0037_create_participant_aliases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    Base.metadata.tables["entity_merge_proposals"].create(bind=op.get_bind(), checkfirst=True)

def downgrade() -> None:
    op.drop_table("entity_merge_proposals")
