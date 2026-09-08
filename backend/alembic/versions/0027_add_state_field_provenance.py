"""为持续人格状态增加逐字段证据和置信度。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0027_state_field_provenance"
down_revision: str | None = "0026_subject_agent_activation"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("branch_state_versions") as batch:
        batch.add_column(
            sa.Column(
                "field_evidence",
                sa.JSON(),
                nullable=False,
                server_default="{}",
            )
        )
        batch.add_column(
            sa.Column(
                "field_confidence",
                sa.JSON(),
                nullable=False,
                server_default="{}",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("branch_state_versions") as batch:
        batch.drop_column("field_confidence")
        batch.drop_column("field_evidence")
