"""为七栏目 PersonWorldProfile v2 增加规范 JSON 投影。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0046_add_person_world_profile_v2"
down_revision: str | None = "0045_remove_claim_verifier"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    for table_name in ("person_world_profiles", "person_world_profile_drafts"):
        columns = {item["name"] for item in sa.inspect(bind).get_columns(table_name)}
        if "profile_v2" not in columns:
            op.add_column(
                table_name,
                sa.Column("profile_v2", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
            )
        if "profile_schema_version" not in columns:
            op.add_column(
                table_name,
                sa.Column(
                    "profile_schema_version",
                    sa.String(24),
                    nullable=False,
                    server_default=sa.text("'v1'"),
                ),
            )
        if "investigation_report" not in columns:
            op.add_column(
                table_name,
                sa.Column(
                    "investigation_report",
                    sa.JSON(),
                    nullable=False,
                    server_default=sa.text("'{}'"),
                ),
            )


def downgrade() -> None:
    for table_name in ("person_world_profile_drafts", "person_world_profiles"):
        columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns(table_name)}
        with op.batch_alter_table(table_name) as batch:
            if "investigation_report" in columns:
                batch.drop_column("investigation_report")
            if "profile_schema_version" in columns:
                batch.drop_column("profile_schema_version")
            if "profile_v2" in columns:
                batch.drop_column("profile_v2")
