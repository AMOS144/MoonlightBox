from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def test_subject_persona_v2_schema_contains_truth_state_and_actor_fields(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.actor_models import (
        ConversationActorState,
        ConversationExpressionPlan,
        ConversationPendingBubble,
    )
    from moonlightbox.branches.continuity_models import (
        BranchMemoryItem,
        BranchStateVersion,
        IdentityKernel,
    )
    from moonlightbox.branches.models import BranchMessage

    model_columns = {
        IdentityKernel: {"field_evidence", "field_confidence", "acceptance_report_id"},
        BranchMemoryItem: {
            "verification_status",
            "claim_key",
            "stance",
            "state_version_id",
            "root_episode_hashes",
        },
        BranchStateVersion: {
            "current_goals",
            "current_concerns",
            "contested_belief_ids",
            "memory_cutoff_version",
            "rollback_of_version_id",
        },
        BranchMessage: {
            "client_message_id",
            "observed_at",
            "expression_plan_id",
            "actor_intent",
            "is_proactive",
        },
    }
    for model, expected in model_columns.items():
        assert expected.issubset(model.__table__.columns.keys())
    assert ConversationActorState.__tablename__ == "conversation_actor_states"
    assert ConversationExpressionPlan.__tablename__ == "conversation_expression_plans"
    assert ConversationPendingBubble.__tablename__ == "conversation_pending_bubbles"


def test_0019_subject_persona_v2_migration_is_reversible(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'subject-persona-v2.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "0019_subject_persona_v2")
    inspector = inspect(create_engine(database_url))
    assert {
        "conversation_actor_states",
        "conversation_expression_plans",
        "conversation_pending_bubbles",
    }.issubset(inspector.get_table_names())
    memory_columns = {
        column["name"] for column in inspector.get_columns("branch_memory_items")
    }
    assert {"verification_status", "claim_key", "state_version_id"}.issubset(
        memory_columns
    )

    command.downgrade(config, "0018_add_branch_baseline_history")
    downgraded = inspect(create_engine(database_url))
    assert "conversation_actor_states" not in downgraded.get_table_names()


def test_head_migration_quarantines_legacy_active_branches(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'legacy-quarantine.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "0019_subject_persona_v2")
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO branches (
                    id, project_id, origin_event_id, model_version_id,
                    title, origin_time, state_snapshot, lifecycle_status,
                    generation_policy_version, baseline_status, created_at
                ) VALUES (
                    'legacy-branch', 'project-1', 'event-1', 'model-1',
                    '旧分支', '2026-01-01', '{}', 'active',
                    'hybrid-v1', 'ready', '2026-01-01'
                )
                """
            )
        )

    command.upgrade(config, "head")

    with engine.connect() as connection:
        status = connection.scalar(
            text(
                "SELECT lifecycle_status FROM branches WHERE id = 'legacy-branch'"
            )
        )
    assert status == "archived"
