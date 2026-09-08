from typing import Literal

type EventLane = Literal["relationship", "shared_experience"]
type EventStatus = Literal["occurred", "confirmed"]

RELATIONSHIP_EVENT_TYPES = frozenset(
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

SHARED_EXPERIENCE_EVENT_TYPES = frozenset(
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
ALL_V3_EVENT_TYPES = RELATIONSHIP_EVENT_TYPES | SHARED_EXPERIENCE_EVENT_TYPES

EVENT_TYPES_BY_LANE: dict[EventLane, frozenset[str]] = {
    "relationship": RELATIONSHIP_EVENT_TYPES,
    "shared_experience": SHARED_EXPERIENCE_EVENT_TYPES,
}


def event_types_for_lane(lane: EventLane) -> frozenset[str]:
    """返回指定事件通道允许的类型集合。"""

    return EVENT_TYPES_BY_LANE[lane]


def is_valid_event_type(lane: str, event_type: str) -> bool:
    """判断事件类型是否属于指定通道。"""

    if lane == "relationship":
        return event_type in RELATIONSHIP_EVENT_TYPES
    if lane == "shared_experience":
        return event_type in SHARED_EXPERIENCE_EVENT_TYPES
    return False


def validate_event_type(lane: EventLane, event_type: str) -> str:
    """校验事件类型与通道匹配，并返回原事件类型。"""

    if not is_valid_event_type(lane, event_type):
        raise ValueError(f"事件类型 {event_type!r} 不属于 {lane!r} 通道")
    return event_type
