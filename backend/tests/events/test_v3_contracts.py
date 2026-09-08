from datetime import UTC, datetime

import pytest
from moonlightbox.events.schemas import (
    EventNodeRead,
    EventScoreComponents,
    V2EventScoreComponents,
    V3ReviewedEvent,
)
from moonlightbox.events.v3_types import (
    RELATIONSHIP_EVENT_TYPES,
    SHARED_EXPERIENCE_EVENT_TYPES,
    is_valid_event_type,
)
from pydantic import ValidationError


def _event_node_read_payload() -> dict[str, object]:
    return {
        "id": "event-1",
        "project_id": "project-1",
        "lane": "shared_experience",
        "event_status": "confirmed",
        "type": "travel",
        "title": "一起去旅行",
        "summary": "共同完成了一次长途旅行",
        "start_message_id": "m1",
        "end_message_id": "m9",
        "started_at": datetime(2026, 7, 1, tzinfo=UTC),
        "ended_at": datetime(2026, 7, 8, tzinfo=UTC),
        "source_lanes": ["shared_experience"],
        "before_state": None,
        "after_state": None,
        "evidence_ids": ["m1", "m9"],
        "status": "active",
        "created_at": datetime(2026, 7, 9, tzinfo=UTC),
    }


def _v3_reviewed_payload() -> dict[str, object]:
    return {
        "lane": "shared_experience",
        "event_status": "occurred",
        "type": "travel",
        "title": "旅行",
        "summary": "一起出行",
        "start_message_id": "m1",
        "end_message_id": "m2",
        "started_at": None,
        "ended_at": None,
        "source_lanes": ["shared_experience"],
        "before_state": None,
        "after_state": None,
        "evidence_ids": ["m1"],
    }


def test_v3_event_type_catalogs_match_the_lane_contract() -> None:
    assert RELATIONSHIP_EVENT_TYPES == frozenset(
        {
            "relationship_started",
            "intimacy_increased",
            "commitment",
            "boundary_change",
            "conflict",
            "distancing",
            "reconciliation",
            "separation",
            "reconnection",
        }
    )
    assert SHARED_EXPERIENCE_EVENT_TYPES == frozenset(
        {
            "date",
            "outing",
            "travel",
            "celebration",
            "gift",
            "family_social",
            "support_care",
            "shared_project",
            "important_plan",
            "life_milestone",
        }
    )
    assert is_valid_event_type("relationship", "commitment")
    assert is_valid_event_type("shared_experience", "travel")
    assert not is_valid_event_type("relationship", "travel")
    assert not is_valid_event_type("shared_experience", "conflict")


def test_shared_experience_read_allows_empty_states_and_v3_scores() -> None:
    node = EventNodeRead.model_validate(
        {
            **_event_node_read_payload(),
            "score_components": {
                "event_significance": 0.91,
                "relationship_impact": 0.72,
                "evidence_quality": 0.88,
                "persistence": 0.76,
                "type_support": 0.95,
                "model_confidence": 0.84,
            },
        }
    )

    assert isinstance(node.score_components, EventScoreComponents)
    assert node.before_state is None
    assert node.after_state is None
    assert node.score_components.model_dump() == {
        "event_significance": 0.91,
        "relationship_impact": 0.72,
        "evidence_quality": 0.88,
        "persistence": 0.76,
        "type_support": 0.95,
        "model_confidence": 0.84,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"lane": "unknown"},
        {"event_status": "pending"},
    ],
)
def test_event_node_read_rejects_invalid_lane_or_status(
    changes: dict[str, str],
) -> None:
    payload = _event_node_read_payload()
    payload.update(changes)

    with pytest.raises(ValidationError):
        EventNodeRead.model_validate(payload)


def test_event_node_read_rejects_empty_relationship_states() -> None:
    payload = _event_node_read_payload()
    payload.update(
        {
            "lane": "relationship",
            "type": "commitment",
            "source_lanes": ["relationship"],
        }
    )

    with pytest.raises(ValidationError, match="关系事件必须同时提供"):
        EventNodeRead.model_validate(payload)


def test_event_node_read_accepts_empty_shared_experience_states() -> None:
    node = EventNodeRead.model_validate(_event_node_read_payload())

    assert node.before_state is None
    assert node.after_state is None


@pytest.mark.parametrize(
    "changes",
    [
        {
            "lane": "relationship",
            "type": "travel",
            "before_state": "关系稳定",
            "after_state": "关系稳定",
            "source_lanes": ["relationship"],
        },
        {
            "lane": "shared_experience",
            "type": "conflict",
            "source_lanes": ["shared_experience"],
        },
    ],
)
def test_event_node_read_rejects_v3_type_from_another_lane(
    changes: dict[str, object],
) -> None:
    payload = _event_node_read_payload()
    payload.update(changes)

    with pytest.raises(ValidationError, match="事件类型与 lane 不匹配"):
        EventNodeRead.model_validate(payload)


@pytest.mark.parametrize(
    ("model_name", "source_lanes"),
    [
        ("read", []),
        ("read", ["relationship"]),
        ("read", ["shared_experience", "unknown"]),
        ("reviewed", []),
        ("reviewed", ["relationship"]),
        ("reviewed", ["shared_experience", "unknown"]),
    ],
)
def test_v3_contract_rejects_empty_or_primary_lane_missing_sources(
    model_name: str,
    source_lanes: list[str],
) -> None:
    payload = _event_node_read_payload() if model_name == "read" else _v3_reviewed_payload()
    payload["source_lanes"] = source_lanes

    model = EventNodeRead if model_name == "read" else V3ReviewedEvent
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize("model_name", ["read", "reviewed"])
def test_v3_contract_allows_cross_lane_sources(model_name: str) -> None:
    payload = _event_node_read_payload() if model_name == "read" else _v3_reviewed_payload()
    payload["source_lanes"] = ["relationship", "shared_experience"]

    model = EventNodeRead if model_name == "read" else V3ReviewedEvent
    validated = model.model_validate(payload)

    assert validated.source_lanes == ["relationship", "shared_experience"]


def test_score_component_public_names_match_v3_and_v2_contracts() -> None:
    assert set(EventScoreComponents.model_fields) == {
        "event_significance",
        "relationship_impact",
        "evidence_quality",
        "persistence",
        "type_support",
        "model_confidence",
    }
    assert set(V2EventScoreComponents.model_fields) == {
        "state_change_strength",
        "persistence",
        "evidence_quality",
        "model_confidence",
    }


def test_v3_score_component_rejects_v2_shape() -> None:
    with pytest.raises(ValidationError):
        EventScoreComponents(
            state_change_strength=0.88,
            persistence=0.77,
            evidence_quality=0.66,
            model_confidence=0.55,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"lane": "unknown"},
        {"event_status": "pending"},
        {"lane": "relationship", "type": "travel"},
        {"lane": "shared_experience", "type": "conflict"},
    ],
)
def test_v3_reviewed_event_rejects_invalid_lane_status_or_type(
    changes: dict[str, str],
) -> None:
    payload = _v3_reviewed_payload()
    payload.update(changes)

    with pytest.raises(ValidationError):
        V3ReviewedEvent.model_validate(payload)


def test_relationship_reviewed_event_requires_both_states() -> None:
    with pytest.raises(ValidationError, match="关系事件必须同时提供"):
        V3ReviewedEvent(
            lane="relationship",
            event_status="occurred",
            type="commitment",
            title="确认关系",
            summary="双方确认长期关系",
            start_message_id="m1",
            end_message_id="m2",
            started_at=None,
            ended_at=None,
            source_lanes=["relationship"],
            before_state=None,
            after_state="进入稳定关系",
            evidence_ids=["m1", "m2"],
        )


def test_v2_event_read_remains_compatible_with_four_score_dimensions() -> None:
    node = EventNodeRead(
        id="event-v2",
        project_id="project-1",
        type="cold_war",
        start_message_id="m1",
        end_message_id="m2",
        before_state="仍在沟通",
        after_state="停止回复",
        emotion_labels=["失望"],
        topic="冲突",
        conflict_level=4,
        importance=0.9,
        reason="回复突然中断",
        evidence_ids=["m1", "m2"],
        status="active",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        score_components={
            "state_change_strength": 0.88,
            "persistence": 0.77,
            "evidence_quality": 0.66,
            "model_confidence": 0.55,
        },
    )

    assert node.lane == "relationship"
    assert node.event_status == "occurred"
    assert isinstance(node.score_components, V2EventScoreComponents)
    assert node.score_components.model_dump() == {
        "state_change_strength": 0.88,
        "persistence": 0.77,
        "evidence_quality": 0.66,
        "model_confidence": 0.55,
    }


@pytest.mark.parametrize("legacy_type", ["cold_war", "long_pause"])
def test_v2_event_read_allows_legacy_types_outside_v3_catalog(
    legacy_type: str,
) -> None:
    payload = _event_node_read_payload()
    payload.update(
        {
            "lane": "relationship",
            "type": legacy_type,
            "source_lanes": ["relationship"],
            "before_state": "仍在沟通",
            "after_state": "停止回复",
        }
    )

    node = EventNodeRead.model_validate(payload)

    assert node.type == legacy_type


@pytest.mark.parametrize("legacy_type", ["cold_war", "long_pause"])
def test_shared_experience_read_rejects_legacy_v2_type(
    legacy_type: str,
) -> None:
    payload = _event_node_read_payload()
    payload["type"] = legacy_type

    with pytest.raises(ValidationError, match="事件类型与 lane 不匹配"):
        EventNodeRead.model_validate(payload)
