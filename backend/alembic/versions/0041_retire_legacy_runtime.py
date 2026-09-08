"""退役旧数字人循环，仅保留 Runtime v1 的持久化边界。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0041_retire_legacy_runtime"
down_revision: str | None = "0040_add_runtime_memory_index_and_context_summary"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _drop_if_present(name: str) -> None:
    if _has_table(name):
        op.drop_table(name)


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _constraint_names(table_name: str, constraint_type: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    if constraint_type == "foreignkey":
        constraints = inspector.get_foreign_keys(table_name)
    elif constraint_type == "check":
        constraints = inspector.get_check_constraints(table_name)
    else:
        raise ValueError(f"不支持的约束类型: {constraint_type}")
    return {constraint["name"] for constraint in constraints if constraint.get("name")}


def _index_names(table_name: str) -> set[str]:
    return {index["name"] for index in sa.inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    # 只迁移旧系统中明确通过审核、且尚未失效的条目。pending/review/reflection
    # 不是 Runtime 事实，不能因裁剪而被误提升为 confirmed。
    if _has_table("branch_memory_items") and _has_table("runtime_memory_records"):
        bind.execute(
            sa.text(
                """
                INSERT INTO runtime_memory_records
                    (id, scope, branch_id, snapshot_id, subject, predicate, object,
                     summary, status, source_ids, confidence, valid_from, valid_to,
                     supersedes_id, created_at)
                SELECT old.id, 'branch', old.branch_id, NULL, old.subject, old.predicate,
                       substr(old.object, 1, 500), substr(old.content, 1, 2000),
                       'confirmed', old.source_episode_ids, old.confidence,
                       old.valid_from, old.valid_to, old.supersedes_id, old.created_at
                FROM branch_memory_items AS old
                WHERE old.review_status = 'approved'
                  AND old.invalidated_at IS NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM runtime_memory_records AS target WHERE target.id = old.id
                  )
                """
            )
        )

    # branch_messages 仍为 Runtime 的对话日志，但旧 ConversationActor 的计划关联
    # 已不存在，先移除外键列再删除计划表。
    if _has_table("branch_messages"):
        message_columns = _columns("branch_messages")
        message_foreign_keys = _constraint_names("branch_messages", "foreignkey")
        with op.batch_alter_table("branch_messages") as batch:
            if "fk_branch_messages_expression_plan" in message_foreign_keys:
                batch.drop_constraint("fk_branch_messages_expression_plan", type_="foreignkey")
            if "expression_plan_id" in message_columns:
                batch.drop_column("expression_plan_id")
            if "actor_intent" in message_columns:
                batch.drop_column("actor_intent")

    # branches 仍保留主键、起点、模型、标题、时间和生命周期；其余均属于旧
    # Baseline / state snapshot / subject-agent 体系。
    if _has_table("branches"):
        branch_columns = _columns("branches")
        branch_checks = _constraint_names("branches", "check")
        branch_foreign_keys = _constraint_names("branches", "foreignkey")
        branch_indexes = _index_names("branches")
        with op.batch_alter_table("branches") as batch:
            for name in ("ck_branches_subject_agent_mode", "ck_branches_baseline_status"):
                if name in branch_checks:
                    batch.drop_constraint(name, type_="check")
            for name in (
                "fk_branch_replacement",
                "fk_branches_origin_import",
                "fk_branches_origin_boundary_message",
                "fk_branches_baseline_job",
                "fk_branches_baseline_manifest",
            ):
                if name in branch_foreign_keys:
                    batch.drop_constraint(name, type_="foreignkey")
            if "ix_branches_replacement_branch_id" in branch_indexes:
                batch.drop_index("ix_branches_replacement_branch_id")
            for name in (
                "state_snapshot",
                "replacement_branch_id",
                "generation_policy_version",
                "origin_import_id",
                "origin_boundary_message_id",
                "baseline_manifest_id",
                "baseline_job_id",
                "baseline_status",
                "baseline_error_code",
                "baseline_error_message",
                "baseline_ready_at",
                "subject_agent_mode",
            ):
                if name in branch_columns:
                    batch.drop_column(name)

    # 表按依赖从叶子到根删除；不删除 identity_kernels，它是 LoRA 训练产物，
    # 已由 personas.models 接管并被 Runtime StyleService 只读使用。
    for table in (
        "conversation_pending_bubbles",
        "conversation_expression_plans",
        "conversation_actor_states",
        "subject_agent_acceptance_reports",
        "agent_wakeups",
        "agent_intentions",
        "cognitive_cycles",
        "private_cognition_notes",
        "mental_state_versions",
        "agent_goals",
        "perception_events",
        "branch_reflection_runs",
        "branch_belief_evidence",
        "branch_state_versions",
        "branch_memory_items",
        "branch_memory_episodes",
        "branch_baseline_states",
        "branch_baseline_event_snapshots",
        "branch_baseline_manifests",
    ):
        _drop_if_present(table)


def downgrade() -> None:
    raise RuntimeError("旧数字人运行时已退役，不能在不恢复备份的情况下回滚")
