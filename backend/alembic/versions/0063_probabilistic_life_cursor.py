"""生活推进固定节拍；保留旧经历和旧任务现场。"""

import sqlalchemy as sa
from alembic import op

revision = "0063_probabilistic_life_cursor"
down_revision = "0062_node_profile_publications"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    columns = {c["name"] for c in inspector.get_columns("runtime_life_opportunities")}
    with op.batch_alter_table("runtime_life_opportunities") as batch:
        if "check_key" not in columns:
            batch.add_column(sa.Column("check_key", sa.String(120), nullable=True))
            batch.create_unique_constraint("uq_life_check", ["branch_id", "check_key"])
        if "kind" not in columns:
            batch.add_column(
                sa.Column("kind", sa.String(24), nullable=False, server_default="legacy")
            )
    from moonlightbox.runtime_v1.life_events.models import LifeScheduleCursor

    LifeScheduleCursor.__table__.create(op.get_bind(), checkfirst=True)
    # 未激活旧抽样被替代；活跃工作暂停待显式恢复，不丢草稿、不伪造新强度。
    op.execute(
        "UPDATE runtime_life_opportunities SET status='superseded' "
        "WHERE kind='legacy' AND status='scheduled'"
    )
    op.execute(
        "UPDATE runtime_life_opportunities SET status='blocked', "
        "error_code='legacy_life_work_requires_review' WHERE kind='legacy' AND status='active'"
    )


def downgrade():
    raise RuntimeError("生活节拍和历史任务不能无损回退；请恢复迁移前备份")
