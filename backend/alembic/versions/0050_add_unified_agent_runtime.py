"""已废弃的统一 Agent 调用账本迁移占位。"""

from collections.abc import Sequence

revision: str = "0050_add_unified_agent_runtime"
down_revision: str | None = "0049_add_atomic_claim_structural_fields"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 新安装不再创建这套表；旧安装由 0053 负责删除已存在的历史表。
    pass


def downgrade() -> None:
    pass
