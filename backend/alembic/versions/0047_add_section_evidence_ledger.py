"""持久化 PersonWorld 的栏目任务、独立证据账本和证据身份快照。"""

from collections.abc import Sequence

import moonlightbox.imports.models  # noqa: F401
import moonlightbox.world.models  # noqa: F401
import sqlalchemy as sa
from alembic import op
from moonlightbox.db import Base

revision: str = "0047_add_section_evidence_ledger"
down_revision: str | None = "0046_add_person_world_profile_v2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    for table_name in ("person_world_section_tasks", "section_evidence_ledger_entries"):
        Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)

    columns = {item["name"] for item in sa.inspect(bind).get_columns("world_evidence")}
    additions = (
        ("timestamp", sa.Column("timestamp", sa.DateTime(timezone=True), nullable=True)),
        ("participant_id", sa.Column("participant_id", sa.String(36), nullable=True)),
        ("participant_name", sa.Column("participant_name", sa.String(500), nullable=True)),
        ("participant_role", sa.Column("participant_role", sa.String(32), nullable=True)),
    )
    for name, column in additions:
        if name not in columns:
            op.add_column("world_evidence", column)


def downgrade() -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns("world_evidence")}
    with op.batch_alter_table("world_evidence") as batch:
        for name in ("participant_role", "participant_name", "participant_id", "timestamp"):
            if name in columns:
                batch.drop_column(name)
    for table_name in ("section_evidence_ledger_entries", "person_world_section_tasks"):
        if table_name in sa.inspect(bind).get_table_names():
            op.drop_table(table_name)
