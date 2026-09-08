"""创建 Runtime v1 的独立事件、时钟、计划、状态和记忆表。

本迁移尚未发布，因此直接从 ORM 元数据建表；Runtime 表不复用旧 cognition
或 BranchStateVersion 表，避免两套状态在运行时互相覆盖。
"""

from collections.abc import Sequence

from alembic import op
from moonlightbox.db import Base
from moonlightbox.runtime_v1 import db_models  # noqa: F401

revision: str = "0039_create_runtime_v1"
down_revision: str | None = "0038_entity_merge_proposals"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    for table_name in (
        "runtime_events",
        "runtime_life_states",
        "runtime_day_plans",
        "runtime_memory_records",
        "runtime_origin_snapshots",
        "runtime_clocks",
        "runtime_wakeups",
        "runtime_life_events",
    ):
        Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    for table_name in (
        "runtime_life_events",
        "runtime_wakeups",
        "runtime_clocks",
        "runtime_origin_snapshots",
        "runtime_memory_records",
        "runtime_day_plans",
        "runtime_life_states",
        "runtime_events",
    ):
        Base.metadata.tables[table_name].drop(bind=bind, checkfirst=True)
