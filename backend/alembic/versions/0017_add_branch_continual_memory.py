"""增加分支持续人格记忆。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017_add_branch_continual_memory"
down_revision: str | None = "0016_add_branch_upgrade_fields"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "identity_kernels",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("evidence_message_ids", sa.JSON(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_identity_kernels_project_id",
        "identity_kernels",
        ["project_id"],
    )

    op.create_table(
        "branch_memory_episodes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_turn_id", sa.String(36), nullable=False),
        sa.Column("assistant_turn_id", sa.String(36), nullable=False),
        sa.Column("user_content", sa.Text(), nullable=False),
        sa.Column("assistant_bubbles", sa.JSON(), nullable=False),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("episode_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("processing_status", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "branch_id",
            "user_turn_id",
            "assistant_turn_id",
            name="ux_branch_memory_episode_turns",
        ),
    )
    op.create_index(
        "ix_branch_memory_episodes_branch_status",
        "branch_memory_episodes",
        ["branch_id", "processing_status"],
    )

    op.create_table(
        "branch_memory_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("subject", sa.String(128), nullable=False),
        sa.Column("predicate", sa.String(128), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("invalidated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "supersedes_id",
            sa.String(36),
            sa.ForeignKey("branch_memory_items.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_episode_ids", sa.JSON(), nullable=False),
        sa.Column("source_item_ids", sa.JSON(), nullable=False),
        sa.Column("lineage_hash", sa.String(64), nullable=False),
        sa.Column("review_status", sa.String(32), nullable=False),
        sa.CheckConstraint(
            "kind IN ('fact', 'experience', 'self_narrative', 'belief', 'reflection')",
            name="ck_branch_memory_items_kind",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_branch_memory_items_confidence",
        ),
        sa.CheckConstraint(
            "importance >= 1 AND importance <= 10",
            name="ck_branch_memory_items_importance",
        ),
    )
    op.create_index(
        "ix_branch_memory_items_branch_valid",
        "branch_memory_items",
        ["branch_id", "review_status", "valid_to"],
    )
    op.create_index(
        "ix_branch_memory_items_lineage",
        "branch_memory_items",
        ["branch_id", "lineage_hash"],
    )

    op.create_table(
        "branch_belief_evidence",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "belief_id",
            sa.String(36),
            sa.ForeignKey("branch_memory_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "episode_id",
            sa.String(36),
            sa.ForeignKey("branch_memory_episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("stance", sa.String(16), nullable=False),
        sa.Column("source_role", sa.String(32), nullable=False),
        sa.Column("evidence_type", sa.String(64), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "belief_id",
            "episode_id",
            "stance",
            name="ux_branch_belief_episode_stance",
        ),
        sa.CheckConstraint(
            "stance IN ('support', 'oppose')",
            name="ck_branch_belief_evidence_stance",
        ),
    )
    op.create_index(
        "ix_branch_belief_evidence_branch",
        "branch_belief_evidence",
        ["branch_id"],
    )

    op.create_table(
        "branch_state_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "previous_version_id",
            sa.String(36),
            sa.ForeignKey("branch_state_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("persona_state", sa.JSON(), nullable=False),
        sa.Column("relationship_state", sa.JSON(), nullable=False),
        sa.Column("user_model", sa.JSON(), nullable=False),
        sa.Column("emotional_tendency", sa.JSON(), nullable=False),
        sa.Column("active_belief_ids", sa.JSON(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("source_episode_ids", sa.JSON(), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rolled_back_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("branch_id", "version"),
    )
    op.create_index(
        "ix_branch_state_versions_current",
        "branch_state_versions",
        ["branch_id", "is_current"],
    )

    op.create_table(
        "branch_reflection_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "trigger_episode_id",
            sa.String(36),
            sa.ForeignKey("branch_memory_episodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("input_item_ids", sa.JSON(), nullable=False),
        sa.Column("input_importance_sum", sa.Float(), nullable=False),
        sa.Column("local_proposal", sa.JSON(), nullable=True),
        sa.Column("review_result", sa.JSON(), nullable=True),
        sa.Column("output_item_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_branch_reflection_runs_branch_status",
        "branch_reflection_runs",
        ["branch_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_branch_reflection_runs_branch_status",
        table_name="branch_reflection_runs",
    )
    op.drop_table("branch_reflection_runs")
    op.drop_index(
        "ix_branch_state_versions_current",
        table_name="branch_state_versions",
    )
    op.drop_table("branch_state_versions")
    op.drop_index(
        "ix_branch_belief_evidence_branch",
        table_name="branch_belief_evidence",
    )
    op.drop_table("branch_belief_evidence")
    op.drop_index(
        "ix_branch_memory_items_lineage",
        table_name="branch_memory_items",
    )
    op.drop_index(
        "ix_branch_memory_items_branch_valid",
        table_name="branch_memory_items",
    )
    op.drop_table("branch_memory_items")
    op.drop_index(
        "ix_branch_memory_episodes_branch_status",
        table_name="branch_memory_episodes",
    )
    op.drop_table("branch_memory_episodes")
    op.drop_index(
        "ix_identity_kernels_project_id",
        table_name="identity_kernels",
    )
    op.drop_table("identity_kernels")
