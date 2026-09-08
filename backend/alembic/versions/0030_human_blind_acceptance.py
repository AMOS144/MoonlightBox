"""Add persistent human blind acceptance studies."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0030_human_blind_acceptance"
down_revision: str | None = "0029_server_side_bubble_delivery"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "human_blind_studies",
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
        ),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("minimum_ratings", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("minimum_preference", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("valid_rating_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("candidate_preference_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("report", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('open', 'passed', 'failed')", name="ck_human_blind_studies_status"
        ),
    )
    op.create_index("ix_human_blind_studies_project_id", "human_blind_studies", ["project_id"])
    op.create_index(
        "ix_human_blind_studies_model_version_id", "human_blind_studies", ["model_version_id"]
    )
    op.create_index("ix_human_blind_studies_status", "human_blind_studies", ["status"])
    op.create_table(
        "human_blind_cases",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "study_id",
            sa.String(36),
            sa.ForeignKey("human_blind_studies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("context", sa.JSON(), nullable=False),
        sa.Column("human_reply", sa.Text(), nullable=False),
        sa.Column("candidate_reply", sa.Text(), nullable=False),
        sa.Column("candidate_option", sa.String(1), nullable=False),
        sa.Column("order_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_human_blind_cases_study_id", "human_blind_cases", ["study_id"])
    op.create_table(
        "human_blind_ratings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "study_id",
            sa.String(36),
            sa.ForeignKey("human_blind_studies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "case_id",
            sa.String(36),
            sa.ForeignKey("human_blind_cases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rater_key", sa.String(128), nullable=False),
        sa.Column("choice", sa.String(8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("case_id", "rater_key", name="ux_human_blind_case_rater"),
        sa.CheckConstraint("choice IN ('a', 'b', 'tie')", name="ck_human_blind_ratings_choice"),
    )
    op.create_index("ix_human_blind_ratings_study_id", "human_blind_ratings", ["study_id"])
    op.create_index("ix_human_blind_ratings_case_id", "human_blind_ratings", ["case_id"])


def downgrade() -> None:
    op.drop_table("human_blind_ratings")
    op.drop_table("human_blind_cases")
    op.drop_table("human_blind_studies")
