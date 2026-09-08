"""增加主体认知 Agent 持久化数据层。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024_subject_cognitive_agent"
down_revision: str | None = "0023_add_assistant_typing_timeout"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _audit_columns() -> list[sa.Column[object]]:
    return [
        sa.Column("evidence", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "model_protocol_version",
            sa.String(64),
            nullable=False,
            server_default="subject-cognition-v1",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    ]


def _branch_scope_foreign_key(name: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["branch_id", "project_id"],
        ["branches.id", "branches.project_id"],
        name=name,
        ondelete="CASCADE",
    )


def upgrade() -> None:
    op.create_index(
        "ux_branches_id_project_id",
        "branches",
        ["id", "project_id"],
        unique=True,
    )

    op.create_table(
        "perception_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("branch_id", sa.String(36), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(128), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("visible_through", sa.DateTime(timezone=True), nullable=False),
        *_audit_columns(),
        _branch_scope_foreign_key("fk_perception_events_branch_project"),
        sa.UniqueConstraint(
            "branch_id",
            "idempotency_key",
            name="ux_perception_events_branch_idempotency",
        ),
        sa.UniqueConstraint("id", "branch_id", name="ux_perception_events_id_branch"),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_perception_events_confidence",
        ),
    )
    op.create_index(
        "ix_perception_events_branch_occurred",
        "perception_events",
        ["branch_id", "occurred_at"],
    )
    op.create_index(
        "ix_perception_events_branch_type",
        "perception_events",
        ["branch_id", "event_type"],
    )

    op.create_table(
        "agent_goals",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("branch_id", sa.String(36), nullable=False),
        sa.Column("goal_type", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("priority", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column(
            "supporting_evidence", sa.JSON(), nullable=False, server_default="[]"
        ),
        sa.Column(
            "opposing_evidence", sa.JSON(), nullable=False, server_default="[]"
        ),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("review_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        *_audit_columns(),
        _branch_scope_foreign_key("fk_agent_goals_branch_project"),
        sa.UniqueConstraint("id", "branch_id", name="ux_agent_goals_id_branch"),
        sa.CheckConstraint(
            "status IN ('active', 'suspended', 'completed', 'abandoned', 'conflicted')",
            name="ck_agent_goals_status",
        ),
    )
    op.create_index(
        "ix_agent_goals_branch_status", "agent_goals", ["branch_id", "status"]
    )
    op.create_index("ix_agent_goals_review", "agent_goals", ["status", "review_at"])

    op.create_table(
        "private_cognition_notes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("branch_id", sa.String(36), nullable=False),
        sa.Column("trigger_event_id", sa.String(36), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "subjective_feelings", sa.JSON(), nullable=False, server_default="{}"
        ),
        sa.Column("attention_target", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("desired_actions", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("memory_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        *_audit_columns(),
        _branch_scope_foreign_key("fk_private_cognition_notes_branch_project"),
        sa.ForeignKeyConstraint(
            ["trigger_event_id", "branch_id"],
            ["perception_events.id", "perception_events.branch_id"],
            name="fk_private_cognition_notes_trigger_branch",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "id", "branch_id", name="ux_private_cognition_notes_id_branch"
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_private_cognition_notes_confidence",
        ),
    )
    op.create_index(
        "ix_private_cognition_notes_branch_created",
        "private_cognition_notes",
        ["branch_id", "created_at"],
    )

    op.create_table(
        "mental_state_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("branch_id", sa.String(36), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("previous_version_id", sa.String(36), nullable=True),
        sa.Column("source_cycle_id", sa.String(36), nullable=True),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        *_audit_columns(),
        _branch_scope_foreign_key("fk_mental_state_versions_branch_project"),
        sa.ForeignKeyConstraint(
            ["previous_version_id", "branch_id"],
            ["mental_state_versions.id", "mental_state_versions.branch_id"],
            name="fk_mental_state_versions_previous_branch",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "branch_id",
            "version",
            name="ux_mental_state_versions_branch_version",
        ),
        sa.UniqueConstraint(
            "id", "branch_id", name="ux_mental_state_versions_id_branch"
        ),
    )
    op.create_index(
        "ux_mental_state_versions_current_branch",
        "mental_state_versions",
        ["branch_id"],
        unique=True,
        sqlite_where=sa.text("is_current = 1"),
        postgresql_where=sa.text("is_current"),
    )
    op.create_index(
        "ix_mental_state_versions_branch_created",
        "mental_state_versions",
        ["branch_id", "created_at"],
    )

    op.create_table(
        "cognitive_cycles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("branch_id", sa.String(36), nullable=False),
        sa.Column("trigger_event_id", sa.String(36), nullable=False),
        sa.Column("input_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("starting_state_version_id", sa.String(36), nullable=True),
        sa.Column("memory_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("goal_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("private_note_id", sa.String(36), nullable=True),
        sa.Column("structured_changes", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("final_decision", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        *_audit_columns(),
        _branch_scope_foreign_key("fk_cognitive_cycles_branch_project"),
        sa.ForeignKeyConstraint(
            ["trigger_event_id", "branch_id"],
            ["perception_events.id", "perception_events.branch_id"],
            name="fk_cognitive_cycles_trigger_branch",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["starting_state_version_id", "branch_id"],
            ["mental_state_versions.id", "mental_state_versions.branch_id"],
            name="fk_cognitive_cycles_start_state_branch",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["private_note_id", "branch_id"],
            ["private_cognition_notes.id", "private_cognition_notes.branch_id"],
            name="fk_cognitive_cycles_note_branch",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("id", "branch_id", name="ux_cognitive_cycles_id_branch"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'invalidated', 'failed')",
            name="ck_cognitive_cycles_status",
        ),
    )
    op.create_index(
        "ix_cognitive_cycles_branch_status",
        "cognitive_cycles",
        ["branch_id", "status"],
    )
    op.create_index(
        "ix_cognitive_cycles_trigger", "cognitive_cycles", ["trigger_event_id"]
    )

    with op.batch_alter_table("mental_state_versions") as batch:
        batch.create_foreign_key(
            "fk_mental_state_versions_cycle_branch",
            "cognitive_cycles",
            ["source_cycle_id", "branch_id"],
            ["id", "branch_id"],
            ondelete="RESTRICT",
        )

    op.create_table(
        "agent_intentions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("branch_id", sa.String(36), nullable=False),
        sa.Column("intention_type", sa.String(64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("goal_id", sa.String(36), nullable=True),
        sa.Column("trigger_event_id", sa.String(36), nullable=False),
        sa.Column("expression_plan", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        *_audit_columns(),
        _branch_scope_foreign_key("fk_agent_intentions_branch_project"),
        sa.ForeignKeyConstraint(
            ["goal_id", "branch_id"],
            ["agent_goals.id", "agent_goals.branch_id"],
            name="fk_agent_intentions_goal_branch",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["trigger_event_id", "branch_id"],
            ["perception_events.id", "perception_events.branch_id"],
            name="fk_agent_intentions_event_branch",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'suspended', 'fulfilled', 'cancelled')",
            name="ck_agent_intentions_status",
        ),
    )
    op.create_index(
        "ix_agent_intentions_branch_status",
        "agent_intentions",
        ["branch_id", "status"],
    )

    op.create_table(
        "agent_wakeups",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("project_id", sa.String(36), nullable=False),
        sa.Column("branch_id", sa.String(36), nullable=False),
        sa.Column("wake_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("goal_id", sa.String(36), nullable=True),
        sa.Column("event_id", sa.String(36), nullable=True),
        sa.Column("idempotency_key", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="scheduled"),
        sa.Column(
            "model_version_id",
            sa.String(36),
            sa.ForeignKey("model_versions.id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        *_audit_columns(),
        _branch_scope_foreign_key("fk_agent_wakeups_branch_project"),
        sa.ForeignKeyConstraint(
            ["goal_id", "branch_id"],
            ["agent_goals.id", "agent_goals.branch_id"],
            name="fk_agent_wakeups_goal_branch",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["event_id", "branch_id"],
            ["perception_events.id", "perception_events.branch_id"],
            name="fk_agent_wakeups_event_branch",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "branch_id",
            "idempotency_key",
            name="ux_agent_wakeups_branch_idempotency",
        ),
        sa.CheckConstraint(
            "status IN ('scheduled', 'claimed', 'completed', 'cancelled', 'expired')",
            name="ck_agent_wakeups_status",
        ),
    )
    op.create_index(
        "ix_agent_wakeups_due", "agent_wakeups", ["status", "wake_at"]
    )
    op.create_index(
        "ix_agent_wakeups_branch_status",
        "agent_wakeups",
        ["branch_id", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_wakeups_branch_status", table_name="agent_wakeups")
    op.drop_index("ix_agent_wakeups_due", table_name="agent_wakeups")
    op.drop_table("agent_wakeups")
    op.drop_index(
        "ix_agent_intentions_branch_status", table_name="agent_intentions"
    )
    op.drop_table("agent_intentions")
    with op.batch_alter_table("mental_state_versions") as batch:
        batch.drop_constraint(
            "fk_mental_state_versions_cycle_branch", type_="foreignkey"
        )
    op.drop_index("ix_cognitive_cycles_trigger", table_name="cognitive_cycles")
    op.drop_index(
        "ix_cognitive_cycles_branch_status", table_name="cognitive_cycles"
    )
    op.drop_table("cognitive_cycles")
    op.drop_index(
        "ix_mental_state_versions_branch_created",
        table_name="mental_state_versions",
    )
    op.drop_index(
        "ux_mental_state_versions_current_branch",
        table_name="mental_state_versions",
    )
    op.drop_table("mental_state_versions")
    op.drop_index(
        "ix_private_cognition_notes_branch_created",
        table_name="private_cognition_notes",
    )
    op.drop_table("private_cognition_notes")
    op.drop_index("ix_agent_goals_review", table_name="agent_goals")
    op.drop_index("ix_agent_goals_branch_status", table_name="agent_goals")
    op.drop_table("agent_goals")
    op.drop_index(
        "ix_perception_events_branch_type", table_name="perception_events"
    )
    op.drop_index(
        "ix_perception_events_branch_occurred", table_name="perception_events"
    )
    op.drop_table("perception_events")
    op.drop_index("ux_branches_id_project_id", table_name="branches")
