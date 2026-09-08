"""保证同一分支中的触发事件只创建一个认知周期。"""

from collections.abc import Sequence

from alembic import op

revision: str = "0025_unique_cycle_trigger"
down_revision: str | None = "0024_subject_cognitive_agent"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("cognitive_cycles") as batch:
        batch.create_unique_constraint(
            "ux_cognitive_cycles_branch_trigger",
            ["branch_id", "trigger_event_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("cognitive_cycles") as batch:
        batch.drop_constraint(
            "ux_cognitive_cycles_branch_trigger",
            type_="unique",
        )
