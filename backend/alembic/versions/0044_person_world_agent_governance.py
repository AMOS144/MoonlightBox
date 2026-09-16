"""增加 PersonWorldAgent、纠正审核和图谱版本治理数据结构。"""

from collections.abc import Sequence

import moonlightbox.imports.models  # noqa: F401
import moonlightbox.projects.models  # noqa: F401
import moonlightbox.world.models  # noqa: F401
import sqlalchemy as sa
from alembic import op
from moonlightbox.db import Base

revision: str = "0044_person_world_agent_governance"
down_revision: str | None = "0043_add_day_plan_agent_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


NEW_TABLES = (
    "person_world_agent_runs",
    "world_evidence",
    "atomic_world_claims",
    "world_claim_verifications",
    "person_world_profile_drafts",
    "person_world_revision_sessions",
    "person_world_revision_messages",
    "world_corrections",
    "world_graph_change_sets",
    "world_change_approvals",
    "world_graph_operation_logs",
    "world_publications",
)


def _create_legacy_claim_verification_table() -> None:
    """保留 0044 当时的表结构，使全新数据库仍能完整回放迁移链。"""

    op.create_table(
        "world_claim_verifications",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "claim_id",
            sa.String(36),
            sa.ForeignKey("atomic_world_claims.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("verdict", sa.String(32), nullable=False),
        sa.Column("subject_verdict", sa.String(32), nullable=False),
        sa.Column("temporal_verdict", sa.String(32), nullable=False),
        sa.Column("speech_act_verdict", sa.String(32), nullable=False),
        sa.Column("recurrence_verdict", sa.String(32), nullable=False),
        sa.Column("supporting_evidence_ids", sa.JSON(), nullable=False),
        sa.Column("contradicting_evidence_ids", sa.JSON(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("verifier_version", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("claim_id", name="uq_world_claim_verification_claim"),
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    graph_columns = {column["name"] for column in inspector.get_columns("world_graph_versions")}
    graph_additions = (
        ("parent_version_id", sa.Column("parent_version_id", sa.String(36), nullable=True)),
        (
            "revision",
            sa.Column("revision", sa.Integer(), nullable=False, server_default=sa.text("1")),
        ),
        (
            "correction_head_hash",
            sa.Column(
                "correction_head_hash",
                sa.String(64),
                nullable=False,
                server_default=sa.text("''"),
            ),
        ),
        ("change_set_id", sa.Column("change_set_id", sa.String(36), nullable=True)),
        ("published_at", sa.Column("published_at", sa.DateTime(timezone=True), nullable=True)),
        ("superseded_at", sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True)),
    )
    for name, column in graph_additions:
        if name not in graph_columns:
            op.add_column("world_graph_versions", column)

    profile_columns = {column["name"] for column in inspector.get_columns("person_world_profiles")}
    if "agent_run_id" not in profile_columns:
        op.add_column(
            "person_world_profiles",
            sa.Column("agent_run_id", sa.String(36), nullable=True),
        )
    if not {"verification_summary", "generation_summary"} & profile_columns:
        op.add_column(
            "person_world_profiles",
            sa.Column(
                "verification_summary",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            ),
        )

    for table_name in NEW_TABLES:
        if table_name == "world_claim_verifications":
            if table_name not in sa.inspect(bind).get_table_names():
                _create_legacy_claim_verification_table()
        else:
            Base.metadata.tables[table_name].create(bind=bind, checkfirst=True)


def downgrade() -> None:
    for table_name in reversed(NEW_TABLES):
        op.drop_table(table_name)
    # SQLite 不能逐列删除带自引用外键的 parent_version_id；batch 会一次性
    # 重建表，并保留其余唯一约束和索引。
    with op.batch_alter_table("person_world_profiles") as batch:
        batch.drop_column("verification_summary")
        batch.drop_column("agent_run_id")
    with op.batch_alter_table("world_graph_versions") as batch:
        for column_name in (
            "superseded_at",
            "published_at",
            "change_set_id",
            "correction_head_hash",
            "revision",
            "parent_version_id",
        ):
            batch.drop_column(column_name)
