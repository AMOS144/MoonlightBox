"""Archive deterministic participant aliases found in source message structure."""

from collections.abc import Sequence

import moonlightbox.imports.models  # noqa: F401
import moonlightbox.projects.models  # noqa: F401
from alembic import op
from moonlightbox.db import Base

revision: str = "0037_create_participant_aliases"
down_revision: str | None = "0036_create_lightrag_world"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    Base.metadata.tables["participant_aliases"].create(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    op.drop_table("participant_aliases")
