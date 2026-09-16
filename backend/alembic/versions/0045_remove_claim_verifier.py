"""删除 PersonWorld 的独立 Claim Verifier 数据表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0045_remove_claim_verifier"
down_revision: str | None = "0044_person_world_agent_governance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if "world_claim_verifications" in sa.inspect(bind).get_table_names():
        op.drop_table("world_claim_verifications")
    _rename_column("atomic_world_claims", "verification_status", "admission_status")
    _rename_column("person_world_profiles", "verification_summary", "generation_summary")
    _rename_column(
        "person_world_profile_drafts",
        "verification_summary",
        "generation_summary",
    )


def downgrade() -> None:
    bind = op.get_bind()
    _rename_column("atomic_world_claims", "admission_status", "verification_status")
    _rename_column("person_world_profiles", "generation_summary", "verification_summary")
    _rename_column(
        "person_world_profile_drafts",
        "generation_summary",
        "verification_summary",
    )
    if "world_claim_verifications" not in sa.inspect(bind).get_table_names():
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


def _rename_column(table_name: str, old_name: str, new_name: str) -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns(table_name)}
    if old_name not in columns:
        return
    with op.batch_alter_table(table_name) as batch:
        if new_name in columns:
            batch.drop_column(old_name)
        else:
            batch.alter_column(old_name, new_column_name=new_name)
