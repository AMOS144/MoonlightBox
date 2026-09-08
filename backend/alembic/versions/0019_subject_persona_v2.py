"""增加主体人格 V2 与分支会话 Actor。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019_subject_persona_v2"
down_revision: str | None = "0018_add_branch_baseline_history"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("identity_kernels") as batch:
        batch.add_column(
            sa.Column("field_evidence", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.add_column(
            sa.Column("field_confidence", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.add_column(sa.Column("acceptance_report_id", sa.String(64), nullable=True))

    with op.batch_alter_table("branch_state_versions") as batch:
        batch.add_column(
            sa.Column(
                "contested_belief_ids", sa.JSON(), nullable=False, server_default="[]"
            )
        )
        batch.add_column(
            sa.Column("current_goals", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.add_column(
            sa.Column("current_concerns", sa.JSON(), nullable=False, server_default="{}")
        )
        batch.add_column(sa.Column("memory_cutoff_version", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("rollback_of_version_id", sa.String(36), nullable=True)
        )
        batch.create_foreign_key(
            "fk_state_rollback_version",
            "branch_state_versions",
            ["rollback_of_version_id"],
            ["id"],
            ondelete="SET NULL",
        )

    with op.batch_alter_table("branch_memory_items") as batch:
        batch.add_column(
            sa.Column(
                "verification_status",
                sa.String(32),
                nullable=False,
                server_default="inferred",
            )
        )
        batch.add_column(sa.Column("claim_key", sa.String(255), nullable=True))
        batch.add_column(
            sa.Column("stance", sa.String(16), nullable=False, server_default="support")
        )
        batch.add_column(sa.Column("state_version_id", sa.String(36), nullable=True))
        batch.add_column(
            sa.Column(
                "root_episode_hashes", sa.JSON(), nullable=False, server_default="[]"
            )
        )
        batch.create_foreign_key(
            "fk_memory_state_version",
            "branch_state_versions",
            ["state_version_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index(
            "ix_branch_memory_items_claim",
            ["branch_id", "claim_key", "valid_to"],
        )

    op.create_table(
        "conversation_actor_states",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("attention_focus", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("shared_ground", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("open_sequences", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column(
            "expression_intentions", sa.JSON(), nullable=False, server_default="[]"
        ),
        sa.Column(
            "approach_motivation", sa.JSON(), nullable=False, server_default="{}"
        ),
        sa.Column("inhibition", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column(
            "observed_message_sequence", sa.Integer(), nullable=False, server_default="-1"
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default="idle"),
        sa.Column("user_typing_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_review_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_owner", sa.String(64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_conversation_actor_states_next_review",
        "conversation_actor_states",
        ["next_review_at"],
    )

    op.create_table(
        "conversation_expression_plans",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("trigger_sequence_from", sa.Integer(), nullable=False),
        sa.Column("trigger_sequence_to", sa.Integer(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=False),
        sa.Column("state_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="planned"),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("proactive", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_conversation_expression_plans_branch_status",
        "conversation_expression_plans",
        ["branch_id", "status"],
    )

    with op.batch_alter_table("branch_messages") as batch:
        batch.add_column(sa.Column("client_message_id", sa.String(64), nullable=True))
        batch.add_column(sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True))
        batch.add_column(sa.Column("expression_plan_id", sa.String(36), nullable=True))
        batch.add_column(sa.Column("actor_intent", sa.String(64), nullable=True))
        batch.add_column(
            sa.Column("is_proactive", sa.Boolean(), nullable=False, server_default=sa.false())
        )
        batch.create_unique_constraint(
            "ux_branch_messages_client_id",
            ["branch_id", "client_message_id"],
        )
        batch.create_foreign_key(
            "fk_branch_messages_expression_plan",
            "conversation_expression_plans",
            ["expression_plan_id"],
            ["id"],
            ondelete="SET NULL",
        )

    op.create_table(
        "conversation_pending_bubbles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "plan_id",
            sa.String(36),
            sa.ForeignKey("conversation_expression_plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "branch_id",
            sa.String(36),
            sa.ForeignKey("branches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("message_type", sa.String(16), nullable=False, server_default="text"),
        sa.Column(
            "media_asset_id",
            sa.String(36),
            sa.ForeignKey("media_assets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("delay_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("send_not_before", sa.DateTime(timezone=True), nullable=True),
        sa.Column("basis_sequence", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "sent_message_id",
            sa.String(36),
            sa.ForeignKey("branch_messages.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_conversation_pending_bubbles_plan_status",
        "conversation_pending_bubbles",
        ["plan_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_conversation_pending_bubbles_plan_status",
        table_name="conversation_pending_bubbles",
    )
    op.drop_table("conversation_pending_bubbles")
    with op.batch_alter_table("branch_messages") as batch:
        batch.drop_constraint(
            "fk_branch_messages_expression_plan", type_="foreignkey"
        )
        batch.drop_constraint("ux_branch_messages_client_id", type_="unique")
        batch.drop_column("is_proactive")
        batch.drop_column("actor_intent")
        batch.drop_column("expression_plan_id")
        batch.drop_column("observed_at")
        batch.drop_column("client_message_id")
    op.drop_index(
        "ix_conversation_expression_plans_branch_status",
        table_name="conversation_expression_plans",
    )
    op.drop_table("conversation_expression_plans")
    op.drop_index(
        "ix_conversation_actor_states_next_review",
        table_name="conversation_actor_states",
    )
    op.drop_table("conversation_actor_states")
    with op.batch_alter_table("branch_memory_items") as batch:
        batch.drop_index("ix_branch_memory_items_claim")
        batch.drop_constraint("fk_memory_state_version", type_="foreignkey")
        batch.drop_column("root_episode_hashes")
        batch.drop_column("state_version_id")
        batch.drop_column("stance")
        batch.drop_column("claim_key")
        batch.drop_column("verification_status")
    with op.batch_alter_table("branch_state_versions") as batch:
        batch.drop_constraint("fk_state_rollback_version", type_="foreignkey")
        batch.drop_column("rollback_of_version_id")
        batch.drop_column("memory_cutoff_version")
        batch.drop_column("current_concerns")
        batch.drop_column("current_goals")
        batch.drop_column("contested_belief_ids")
    with op.batch_alter_table("identity_kernels") as batch:
        batch.drop_column("acceptance_report_id")
        batch.drop_column("field_confidence")
        batch.drop_column("field_evidence")
