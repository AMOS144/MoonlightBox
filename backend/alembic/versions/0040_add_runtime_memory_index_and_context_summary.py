"""补齐 Runtime v1 的记忆索引边界和上下文摘要表。"""

from collections.abc import Sequence

from alembic import op
from moonlightbox.db import Base
from moonlightbox.runtime_v1 import db_models  # noqa: F401

revision: str = "0040_add_runtime_memory_index_and_context_summary"
down_revision: str | None = "0039_create_runtime_v1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    for table_name in ("runtime_memory_index_versions", "runtime_context_summaries"):
        Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for table_name in ("runtime_context_summaries", "runtime_memory_index_versions"):
        Base.metadata.tables[table_name].drop(bind=bind, checkfirst=True)
