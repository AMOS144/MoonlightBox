"""为 PersonWorld 原子事实补充栏目和事实类型的结构索引。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0049_add_atomic_claim_structural_fields"
down_revision: str | None = "0048_add_revision_context_protocol"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns("atomic_world_claims")}
    if "primary_domain" not in columns:
        op.add_column(
            "atomic_world_claims",
            sa.Column("primary_domain", sa.String(80), nullable=False, server_default="unknown"),
        )
    if "fact_type" not in columns:
        op.add_column(
            "atomic_world_claims",
            sa.Column("fact_type", sa.String(80), nullable=False, server_default="unknown"),
        )
    if bind.dialect.name == "sqlite":
        op.execute(
            "UPDATE atomic_world_claims SET primary_domain = "
            "CASE WHEN instr(profile_section, '.') > 0 "
            "THEN substr(profile_section, 1, instr(profile_section, '.') - 1) "
            "ELSE profile_section END WHERE primary_domain = 'unknown'"
        )
    else:
        op.execute(
            "UPDATE atomic_world_claims SET primary_domain = "
            "split_part(profile_section, '.', 1) WHERE primary_domain = 'unknown'"
        )
    op.execute(
        "UPDATE atomic_world_claims SET fact_type = predicate WHERE fact_type = 'unknown'"
    )
    indexes = {item["name"] for item in sa.inspect(bind).get_indexes("atomic_world_claims")}
    if "ix_atomic_world_claims_graph_domain_fact" not in indexes:
        op.create_index(
            "ix_atomic_world_claims_graph_domain_fact",
            "atomic_world_claims",
            ["graph_version_id", "primary_domain", "fact_type"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    indexes = {item["name"] for item in sa.inspect(bind).get_indexes("atomic_world_claims")}
    if "ix_atomic_world_claims_graph_domain_fact" in indexes:
        op.drop_index("ix_atomic_world_claims_graph_domain_fact", "atomic_world_claims")
    columns = {item["name"] for item in sa.inspect(bind).get_columns("atomic_world_claims")}
    with op.batch_alter_table("atomic_world_claims") as batch:
        if "fact_type" in columns:
            batch.drop_column("fact_type")
        if "primary_domain" in columns:
            batch.drop_column("primary_domain")
