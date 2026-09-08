"""Create chat spatial analysis and personal place graph tables."""

from collections.abc import Sequence

import moonlightbox.imports.models  # noqa: F401
import moonlightbox.projects.models  # noqa: F401
import moonlightbox.spatial.models  # noqa: F401
from alembic import op
from moonlightbox.db import Base

revision: str = "0035_create_spatial_place_graph"
down_revision: str | None = "0034_add_candidate_preview_mode"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLES = (
    "spatial_analysis_runs",
    "spatial_conversation_episodes",
    "spatial_episode_candidates",
    "spatial_analysis_bundles",
    "place_mentions",
    "place_candidates",
    "place_entities",
    "place_aliases",
    "place_resolutions",
    "visit_observations",
    "visit_episodes",
    "place_graph_snapshots",
    "place_graph_nodes",
    "place_graph_edges",
    "place_provider_enrichments",
)


def upgrade() -> None:
    bind = op.get_bind()
    for table_name in TABLES:
        Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)


def downgrade() -> None:
    for table_name in reversed(TABLES):
        op.drop_table(table_name)

