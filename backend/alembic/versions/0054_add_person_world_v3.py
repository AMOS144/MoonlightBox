"""保留 v1/v2 历史，为 v3 人物画像增加独立规范载荷。"""

import sqlalchemy as sa
from alembic import op

revision = "0054_add_person_world_v3"
down_revision = "0053_remove_agent_runtime_ledger"
branch_labels = None
depends_on = None


def upgrade():
    for table in ("person_world_profiles", "person_world_profile_drafts"):
        # 早期建表迁移引用了 ORM metadata：全新库可能已经包含当前字段。
        # 兼容全新安装、旧库升级及 SQLite 部分执行后的安全重试。
        columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}
        if "profile_v3" not in columns:
            op.add_column(table, sa.Column("profile_v3", sa.JSON(), nullable=False, server_default="{}"))


def downgrade():
    for table in ("person_world_profile_drafts", "person_world_profiles"):
        op.drop_column(table, "profile_v3")
