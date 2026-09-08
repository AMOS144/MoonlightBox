import importlib
from collections.abc import Callable
from typing import Any

import pytest
from sqlalchemy import ForeignKeyConstraint, Index, UniqueConstraint


def _load_models() -> Any:
    try:
        return importlib.import_module("moonlightbox.agent.models")
    except ModuleNotFoundError:
        pytest.fail("主体认知 Agent ORM 模块尚未实现")


def _foreign_key_columns(model: type[object]) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    return {
        (
            tuple(constraint.column_keys),
            tuple(element.target_fullname for element in constraint.elements),
        )
        for constraint in model.__table__.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    }


def _unique_columns(model: type[object]) -> set[tuple[str, ...]]:
    return {
        tuple(column.name for column in constraint.columns)
        for constraint in model.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }


def _default(model: type[object], column_name: str) -> object:
    default = model.__table__.columns[column_name].default
    assert default is not None
    value: object | Callable[[], object] = default.arg
    return value() if callable(value) else value


def test_models_use_project_scoped_branch_foreign_keys() -> None:
    models = _load_models()
    from moonlightbox.branches.models import Branch

    assert ("id", "project_id") in _unique_columns(Branch)
    model_types = (
        models.PerceptionEvent,
        models.CognitiveCycle,
        models.PrivateCognitionNote,
        models.MentalStateVersion,
        models.AgentGoal,
        models.AgentIntention,
        models.AgentWakeup,
    )

    for model in model_types:
        assert (
            ("branch_id", "project_id"),
            ("branches.id", "branches.project_id"),
        ) in _foreign_key_columns(model)


def test_perception_event_has_branch_idempotency_contract() -> None:
    models = _load_models()
    event = models.PerceptionEvent

    assert ("branch_id", "idempotency_key") in _unique_columns(event)
    assert {
        "event_type",
        "occurred_at",
        "source",
        "confidence",
        "idempotency_key",
        "visible_through",
        "evidence",
        "schema_version",
        "model_protocol_version",
        "created_at",
    }.issubset(event.__table__.columns.keys())
    assert _default(event, "confidence") == 1.0
    assert _default(event, "schema_version") == 1


def test_mental_state_versions_form_branch_local_version_chain() -> None:
    models = _load_models()
    state = models.MentalStateVersion

    assert ("branch_id", "version") in _unique_columns(state)
    assert (
        ("previous_version_id", "branch_id"),
        ("mental_state_versions.id", "mental_state_versions.branch_id"),
    ) in _foreign_key_columns(state)
    assert {
        "state",
        "source_cycle_id",
        "evidence",
        "is_current",
        "model_version_id",
        "model_protocol_version",
    }.issubset(state.__table__.columns.keys())
    current_indexes = {
        index.name
        for index in state.__table__.indexes
        if isinstance(index, Index) and index.unique
    }
    assert "ux_mental_state_versions_current_branch" in current_indexes
    assert _default(state, "is_current") is True


def test_cognitive_cycle_links_trigger_state_note_and_audit_payloads() -> None:
    models = _load_models()
    cycle = models.CognitiveCycle
    foreign_keys = _foreign_key_columns(cycle)

    assert (
        ("trigger_event_id", "branch_id"),
        ("perception_events.id", "perception_events.branch_id"),
    ) in foreign_keys
    assert (
        ("starting_state_version_id", "branch_id"),
        ("mental_state_versions.id", "mental_state_versions.branch_id"),
    ) in foreign_keys
    assert (
        ("private_note_id", "branch_id"),
        ("private_cognition_notes.id", "private_cognition_notes.branch_id"),
    ) in foreign_keys
    assert {
        "input_cutoff_at",
        "memory_ids",
        "goal_ids",
        "structured_changes",
        "final_decision",
        "status",
        "model_version_id",
        "model_protocol_version",
        "created_at",
        "started_at",
        "completed_at",
    }.issubset(cycle.__table__.columns.keys())
    assert _default(cycle, "status") == "pending"


def test_goal_intention_note_and_wakeup_defaults_and_evidence() -> None:
    models = _load_models()

    assert _default(models.AgentGoal, "status") == "active"
    assert _default(models.AgentIntention, "status") == "active"
    assert _default(models.AgentWakeup, "status") == "scheduled"
    assert _default(models.PrivateCognitionNote, "confidence") == 1.0
    assert ("branch_id", "idempotency_key") in _unique_columns(models.AgentWakeup)

    expected_columns = {
        models.AgentGoal: {
            "goal_type",
            "content",
            "source",
            "priority",
            "supporting_evidence",
            "opposing_evidence",
            "review_at",
        },
        models.AgentIntention: {
            "intention_type",
            "content",
            "goal_id",
            "trigger_event_id",
            "expression_plan",
            "evidence",
        },
        models.PrivateCognitionNote: {
            "content",
            "subjective_feelings",
            "attention_target",
            "desired_actions",
            "trigger_event_id",
            "memory_ids",
            "evidence",
        },
        models.AgentWakeup: {
            "wake_at",
            "reason",
            "goal_id",
            "event_id",
            "idempotency_key",
            "evidence",
        },
    }
    for model, columns in expected_columns.items():
        assert columns.issubset(model.__table__.columns.keys())


def test_api_import_registers_cognitive_models_in_shared_metadata() -> None:
    import moonlightbox.api as api
    from moonlightbox.db import Base

    assert api.app is not None
    assert {
        "perception_events",
        "cognitive_cycles",
        "private_cognition_notes",
        "mental_state_versions",
        "agent_goals",
        "agent_intentions",
        "agent_wakeups",
    }.issubset(Base.metadata.tables)


def test_optional_links_preserve_non_nullable_branch_scope_on_parent_delete() -> None:
    models = _load_models()
    constraint_names = {
        "fk_agent_intentions_goal_branch",
        "fk_agent_wakeups_goal_branch",
        "fk_agent_wakeups_event_branch",
    }
    constraints = {
        constraint.name: constraint
        for model in (models.AgentIntention, models.AgentWakeup)
        for constraint in model.__table__.constraints
        if isinstance(constraint, ForeignKeyConstraint)
        and constraint.name in constraint_names
    }

    assert constraints.keys() == constraint_names
    assert all(constraint.ondelete == "RESTRICT" for constraint in constraints.values())
