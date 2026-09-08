from datetime import UTC, datetime, timedelta
from typing import Literal

from moonlightbox.branches.memory_policy import is_current_state_question
from moonlightbox.branches.models import Branch

SituationalSource = Literal[
    "external_observation",
    "user_configured",
    "agent_action",
    "historical_replay",
]

_ALLOWED_SLOTS = {
    "activity",
    "location",
    "availability",
    "physical_state",
    "emotion",
    "environment",
    "current_facts",
}
_TRUSTED_SOURCES = {
    "external_observation",
    "user_configured",
    "agent_action",
    "historical_replay",
}
_MAX_TTL = timedelta(hours=24)
_SOURCE_PRIORITY = {
    "historical_replay": 1,
    "user_configured": 2,
    "external_observation": 3,
    "agent_action": 3,
}


def set_situational_state(
    branch: Branch,
    *,
    values: dict[str, str],
    source: SituationalSource,
    evidence_ids: tuple[str, ...],
    observed_at: datetime,
    valid_until: datetime,
    confidence: float = 1.0,
) -> dict[str, object]:
    """Persist trusted, expiring working state without turning it into memory."""

    if source not in _TRUSTED_SOURCES:
        raise ValueError("短期情境状态来源不可信")
    if not evidence_ids:
        raise ValueError("短期情境状态必须包含证据")
    if not 0 <= confidence <= 1:
        raise ValueError("短期情境状态置信度必须在 0 到 1 之间")
    if observed_at.tzinfo is None or valid_until.tzinfo is None:
        raise ValueError("短期情境状态时间必须包含时区")
    observed = observed_at.astimezone(UTC)
    expires = valid_until.astimezone(UTC)
    if expires <= observed or expires - observed > _MAX_TTL:
        raise ValueError("短期情境状态有效期必须在 0 到 24 小时之间")
    normalized = {
        key: value.strip()
        for key, value in values.items()
        if key in _ALLOWED_SLOTS and isinstance(value, str) and value.strip()
    }
    if not normalized:
        raise ValueError("短期情境状态没有有效槽位")
    stored = branch.state_snapshot.get("situational_state")
    slots = (
        dict(stored.get("slots", {}))
        if isinstance(stored, dict)
        and stored.get("schema_version") == "situational-state-v2"
        and isinstance(stored.get("slots"), dict)
        else {}
    )
    for key, value in normalized.items():
        candidate = {
            "value": value,
            "source": source,
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
            "confidence": confidence,
            "observed_at": observed.isoformat(),
            "valid_until": expires.isoformat(),
        }
        existing = slots.get(key)
        if not isinstance(existing, dict) or _candidate_wins(candidate, existing):
            slots[key] = candidate
    payload: dict[str, object] = {
        "schema_version": "situational-state-v2",
        "slots": slots,
        "values": {
            key: slot["value"]
            for key, slot in slots.items()
            if isinstance(slot, dict) and isinstance(slot.get("value"), str)
        },
    }
    branch.state_snapshot = {
        **branch.state_snapshot,
        "situational_state": payload,
    }
    return payload


def active_situational_state(
    snapshot: dict[str, object],
    *,
    now: datetime | None = None,
) -> dict[str, object] | None:
    raw = snapshot.get("situational_state")
    if not isinstance(raw, dict):
        return None
    if raw.get("schema_version") == "situational-state-v2":
        return _active_v2(raw, now=now)
    if raw.get("source") not in _TRUSTED_SOURCES:
        return None
    values = raw.get("values")
    evidence_ids = raw.get("evidence_ids")
    if not isinstance(values, dict) or not isinstance(evidence_ids, list) or not evidence_ids:
        return None
    try:
        observed = datetime.fromisoformat(str(raw["observed_at"]))
        expires = datetime.fromisoformat(str(raw["valid_until"]))
    except (KeyError, TypeError, ValueError):
        return None
    if observed.tzinfo is None or expires.tzinfo is None:
        return None
    current = (now or datetime.now(UTC)).astimezone(UTC)
    if current < observed.astimezone(UTC) or current >= expires.astimezone(UTC):
        return None
    normalized = {
        str(key): value.strip()
        for key, value in values.items()
        if key in _ALLOWED_SLOTS and isinstance(value, str) and value.strip()
    }
    if not normalized:
        return None
    return {**raw, "values": normalized}


def _active_v2(
    raw: dict[str, object],
    *,
    now: datetime | None,
) -> dict[str, object] | None:
    raw_slots = raw.get("slots")
    if not isinstance(raw_slots, dict):
        return None
    current = (now or datetime.now(UTC)).astimezone(UTC)
    slots: dict[str, dict[str, object]] = {}
    for key, slot in raw_slots.items():
        if key not in _ALLOWED_SLOTS or not isinstance(slot, dict):
            continue
        value = slot.get("value")
        source = slot.get("source")
        evidence_ids = slot.get("evidence_ids")
        confidence = slot.get("confidence")
        if (
            not isinstance(value, str)
            or not value.strip()
            or source not in _TRUSTED_SOURCES
            or not isinstance(evidence_ids, list)
            or not evidence_ids
            or not isinstance(confidence, int | float)
            or not 0 <= float(confidence) <= 1
        ):
            continue
        try:
            observed = datetime.fromisoformat(str(slot["observed_at"]))
            expires = datetime.fromisoformat(str(slot["valid_until"]))
        except (KeyError, TypeError, ValueError):
            continue
        if observed.tzinfo is None or expires.tzinfo is None:
            continue
        if current < observed.astimezone(UTC) or current >= expires.astimezone(UTC):
            continue
        slots[str(key)] = {**slot, "value": value.strip()}
    if not slots:
        return None
    return {
        "schema_version": "situational-state-v2",
        "slots": slots,
        "values": {key: slot["value"] for key, slot in slots.items()},
    }


def _candidate_wins(
    candidate: dict[str, object],
    existing: dict[str, object],
) -> bool:
    candidate_time = datetime.fromisoformat(str(candidate["observed_at"])).astimezone(UTC)
    try:
        existing_time = datetime.fromisoformat(str(existing["observed_at"])).astimezone(UTC)
    except (KeyError, TypeError, ValueError):
        return True
    if candidate_time != existing_time:
        return candidate_time > existing_time
    candidate_priority = _SOURCE_PRIORITY.get(str(candidate.get("source")), 0)
    existing_priority = _SOURCE_PRIORITY.get(str(existing.get("source")), 0)
    if candidate_priority != existing_priority:
        return candidate_priority > existing_priority
    return float(candidate.get("confidence", 0)) >= float(existing.get("confidence", 0))


def natural_situational_context(state: dict[str, object] | None) -> tuple[str, ...]:
    if state is None:
        return ()
    values = state.get("values")
    if not isinstance(values, dict):
        return ()
    labels = {
        "activity": "当前活动",
        "location": "当前位置",
        "availability": "当前是否方便联系",
        "physical_state": "当前身体状态",
        "emotion": "当前情绪",
        "environment": "当前外部环境",
        "current_facts": "当前相关事实",
    }
    return tuple(
        f"{labels[key]}：{values[key]}"
        for key in labels
        if isinstance(values.get(key), str) and values[key]
    )


def current_state_content_draft(
    state: dict[str, object] | None,
    question: str,
) -> str | None:
    """Resolve factual state content before the persona LoRA styles it."""

    if state is None:
        return None
    if not is_current_state_question(question):
        return None
    values = state.get("values")
    if not isinstance(values, dict):
        return None
    compact = "".join(question.split())
    routes = (
        (("电话", "方便", "忙不忙", "忙吗"), "availability"),
        (("在哪", "哪里", "位置"), "location"),
        (("天气", "下雨", "下雪", "环境"), "environment"),
        (("心情", "开心", "难过", "生气"), "emotion"),
        (("累", "舒服", "身体"), "physical_state"),
        (("干嘛", "做什么", "忙什么", "吃", "睡", "上班", "下班"), "activity"),
        (("多少", "几盒", "几份", "几个"), "current_facts"),
    )
    for markers, slot in routes:
        value = values.get(slot)
        if any(marker in compact for marker in markers) and isinstance(value, str):
            return value.strip() or None
    fallback = values.get("current_facts")
    return fallback.strip() if isinstance(fallback, str) and fallback.strip() else None
