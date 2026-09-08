"""隔离未使用主体人格 V2 协议创建的旧分支。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020_quarantine_legacy_branches"
down_revision: str | None = "0019_subject_persona_v2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE branches
            SET lifecycle_status = 'archived'
            WHERE lifecycle_status = 'active'
              AND generation_policy_version != 'subject-v2-actor-v1'
            """
        )
    )


def downgrade() -> None:
    # 隔离状态可能已被后续人工升级，降级时不自动恢复旧分支写入能力。
    pass
