"""Create LightRAG world graph, bundle mapping, and person profile tables."""

from collections.abc import Sequence

import moonlightbox.imports.models  # noqa: F401
import moonlightbox.projects.models  # noqa: F401
import moonlightbox.world.models  # noqa: F401
from alembic import op
from moonlightbox.db import Base

revision: str = "0036_create_lightrag_world"
down_revision: str | None = "0035_create_spatial_place_graph"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = (
    "world_graph_versions",
    "conversation_bundles",
    "conversation_bundle_messages",
    "person_world_profiles",
)


def upgrade() -> None:
    bind = op.get_bind()
    for table_name in TABLES:
        Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)


def downgrade() -> None:
    for table_name in reversed(TABLES):
        op.drop_table(table_name)
