"""增加分支基础历史清单、事件快照和基础状态。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018_add_branch_baseline_history"
down_revision: str | None = "0017_add_branch_continual_memory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("branches") as batch:
        batch.add_column(sa.Column("origin_import_id", sa.String(36), nullable=True))
        batch.add_column(
            sa.Column("origin_boundary_message_id", sa.String(36), nullable=True)
        )
        batch.add_column(sa.Column("baseline_job_id", sa.String(36), nullable=True))
        batch.add_column(
            sa.Column(
                "baseline_status",
                sa.String(16),
                nullable=False,
                server_default="preparing",
            )
        )
        batch.add_column(
            sa.Column("baseline_error_code", sa.String(64), nullable=True)
        )
        batch.add_column(
            sa.Column("baseline_error_message", sa.Text(), nullable=True)
        )
        batch.add_column(
            sa.Column(
                "baseline_ready_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch.create_foreign_key(
            "fk_branches_origin_import",
            "import_sources",
            ["origin_import_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_foreign_key(
            "fk_branches_origin_boundary_message",
            "messages",
            ["origin_boundary_message_id"],
            ["id"],
            ondelete="RESTRICT",
        )
        batch.create_foreign_key(
            "fk_branches_baseline_job",
            "jobs",
            ["baseline_job_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_check_constraint(
            "ck_branches_baseline_status",
            "baseline_status IN ('preparing', 'ready', 'failed')",
        )

    op.create_table(
        "branch_baseline_manifests",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "import_id",
            sa.String(36),
            sa.ForeignKey("import_sources.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "origin_event_id",
            sa.String(36),
            sa.ForeignKey("event_nodes.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "boundary_message_id",
            sa.String(36),
            sa.ForeignKey("messages.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("boundary_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("boundary_source_id", sa.String(100), nullable=False),
        sa.Column("message_count", sa.Integer(), nullable=False),
        sa.Column("event_snapshot_count", sa.Integer(), nullable=False),
        sa.Column("message_digest", sa.String(64), nullable=False),
        sa.Column("event_digest", sa.String(64), nullable=False),
        sa.Column("index_fingerprint", sa.String(64), nullable=False),
        sa.Column("recent_tail_message_ids", sa.JSON(), nullable=False),
        sa.Column("protocol_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_branch_baseline_manifests_project",
        "branch_baseline_manifests",
        ["project_id"],
    )
    op.create_index(
        "ix_branch_baseline_manifests_import",
        "branch_baseline_manifests",
        ["import_id"],
    )

    op.create_table(
        "branch_baseline_event_snapshots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "manifest_id",
            sa.String(36),
            sa.ForeignKey("branch_baseline_manifests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_event_id", sa.String(36), nullable=False),
        sa.Column("source_revision_id", sa.String(36), nullable=True),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("evidence_message_ids", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("manifest_id", "source_event_id"),
    )
    op.create_index(
        "ix_branch_baseline_event_snapshots_manifest",
        "branch_baseline_event_snapshots",
        ["manifest_id"],
    )

    op.create_table(
        "branch_baseline_states",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "manifest_id",
            sa.String(36),
            sa.ForeignKey("branch_baseline_manifests.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("persona_state", sa.JSON(), nullable=False),
        sa.Column("relationship_state", sa.JSON(), nullable=False),
        sa.Column("emotional_tendency", sa.JSON(), nullable=False),
        sa.Column("user_model", sa.JSON(), nullable=False),
        sa.Column("historical_belief_ids", sa.JSON(), nullable=False),
        sa.Column("evidence_message_ids", sa.JSON(), nullable=False),
        sa.Column("evidence_event_snapshot_ids", sa.JSON(), nullable=False),
        sa.Column("local_proposal", sa.JSON(), nullable=False),
        sa.Column("review_result", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    with op.batch_alter_table("branches") as batch:
        batch.add_column(
            sa.Column("baseline_manifest_id", sa.String(36), nullable=True)
        )
        batch.create_foreign_key(
            "fk_branches_baseline_manifest",
            "branch_baseline_manifests",
            ["baseline_manifest_id"],
            ["id"],
            ondelete="SET NULL",
        )

    with op.batch_alter_table("branch_state_versions") as batch:
        batch.add_column(
            sa.Column("baseline_manifest_id", sa.String(36), nullable=True)
        )
        batch.add_column(sa.Column("baseline_state_id", sa.String(36), nullable=True))
        batch.create_foreign_key(
            "fk_state_baseline_manifest",
            "branch_baseline_manifests",
            ["baseline_manifest_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_foreign_key(
            "fk_state_baseline_state",
            "branch_baseline_states",
            ["baseline_state_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table("branch_state_versions") as batch:
        batch.drop_constraint("fk_state_baseline_state", type_="foreignkey")
        batch.drop_constraint("fk_state_baseline_manifest", type_="foreignkey")
        batch.drop_column("baseline_state_id")
        batch.drop_column("baseline_manifest_id")
    with op.batch_alter_table("branches") as batch:
        batch.drop_constraint("fk_branches_baseline_manifest", type_="foreignkey")
        batch.drop_column("baseline_manifest_id")
    op.drop_table("branch_baseline_states")
    op.drop_index(
        "ix_branch_baseline_event_snapshots_manifest",
        table_name="branch_baseline_event_snapshots",
    )
    op.drop_table("branch_baseline_event_snapshots")
    op.drop_index(
        "ix_branch_baseline_manifests_import",
        table_name="branch_baseline_manifests",
    )
    op.drop_index(
        "ix_branch_baseline_manifests_project",
        table_name="branch_baseline_manifests",
    )
    op.drop_table("branch_baseline_manifests")
    with op.batch_alter_table("branches") as batch:
        batch.drop_constraint("ck_branches_baseline_status", type_="check")
        batch.drop_constraint("fk_branches_baseline_job", type_="foreignkey")
        batch.drop_constraint(
            "fk_branches_origin_boundary_message", type_="foreignkey"
        )
        batch.drop_constraint("fk_branches_origin_import", type_="foreignkey")
        batch.drop_column("baseline_ready_at")
        batch.drop_column("baseline_error_message")
        batch.drop_column("baseline_error_code")
        batch.drop_column("baseline_status")
        batch.drop_column("baseline_job_id")
        batch.drop_column("origin_boundary_message_id")
        batch.drop_column("origin_import_id")
