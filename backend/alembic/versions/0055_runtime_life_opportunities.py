"""生活机会恢复状态和不可变日程版本，不迁移或覆盖历史事实。"""

from alembic import op

revision = "0055_runtime_life_opportunities"
down_revision = "0054_add_person_world_v3"
branch_labels = None
depends_on = None


def upgrade():
    from moonlightbox.runtime_v1.life_events.models import DayPlanVersionRow, LifeOpportunityRow

    for table in (LifeOpportunityRow.__table__, DayPlanVersionRow.__table__):
        table.create(op.get_bind(), checkfirst=True)


def downgrade():
    from moonlightbox.runtime_v1.life_events.models import DayPlanVersionRow, LifeOpportunityRow

    for table in (DayPlanVersionRow.__table__, LifeOpportunityRow.__table__):
        table.drop(op.get_bind(), checkfirst=True)
