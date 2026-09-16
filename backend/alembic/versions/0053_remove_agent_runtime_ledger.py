"""移除已由 Phoenix 替代的统一 Agent 持久化调用账本。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0053_remove_agent_runtime_ledger"
down_revision: str | None = "0052_add_revision_input_revision"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TABLES = (
    "agent_trace_events",
    "agent_context_snapshots",
    "agent_tool_invocations",
    "agent_steps",
    "agent_runs",
)


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    # 先删除所有引用 agent_runs 的子表，兼容 SQLite 的外键约束。
    for table in _TABLES:
        if table in existing:
            op.drop_table(table)


def downgrade() -> None:
    # 不恢复旧的自建调用链账本。若需要历史排障，应从 Phoenix 的已导出 Trace 中读取。
    pass
