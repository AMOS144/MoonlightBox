from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest


def test_trusted_situational_state_is_active_only_inside_its_ttl() -> None:
    from moonlightbox.branches.situational_state import (
        active_situational_state,
        set_situational_state,
    )

    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    branch = SimpleNamespace(state_snapshot={})
    payload = set_situational_state(
        branch,  # type: ignore[arg-type]
        values={"activity": "正在开会", "unknown": "不能写入"},
        source="external_observation",
        evidence_ids=("observation-1",),
        observed_at=now,
        valid_until=now + timedelta(hours=1),
    )

    assert payload["values"] == {"activity": "正在开会"}
    assert active_situational_state(branch.state_snapshot, now=now) == payload
    assert (
        active_situational_state(
            branch.state_snapshot,
            now=now + timedelta(hours=1),
        )
        is None
    )


@pytest.mark.parametrize(
    ("evidence_ids", "duration"),
    (((), timedelta(hours=1)), (("e-1",), timedelta(hours=25))),
)
def test_situational_state_rejects_missing_evidence_or_excessive_ttl(
    evidence_ids: tuple[str, ...],
    duration: timedelta,
) -> None:
    from moonlightbox.branches.situational_state import set_situational_state

    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    branch = SimpleNamespace(state_snapshot={})

    with pytest.raises(ValueError):
        set_situational_state(
            branch,  # type: ignore[arg-type]
            values={"location": "公司"},
            source="user_configured",
            evidence_ids=evidence_ids,
            observed_at=now,
            valid_until=now + duration,
        )


def test_situational_slots_merge_and_expire_independently() -> None:
    from moonlightbox.branches.situational_state import (
        active_situational_state,
        set_situational_state,
    )

    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    branch = SimpleNamespace(state_snapshot={})
    set_situational_state(
        branch,  # type: ignore[arg-type]
        values={"location": "公司"},
        source="external_observation",
        evidence_ids=("location-1",),
        observed_at=now,
        valid_until=now + timedelta(hours=4),
    )
    set_situational_state(
        branch,  # type: ignore[arg-type]
        values={"activity": "正在开会"},
        source="agent_action",
        evidence_ids=("action-1",),
        observed_at=now + timedelta(minutes=5),
        valid_until=now + timedelta(minutes=35),
    )

    active = active_situational_state(
        branch.state_snapshot,
        now=now + timedelta(minutes=10),
    )
    assert active is not None
    assert active["values"] == {"location": "公司", "activity": "正在开会"}
    later = active_situational_state(
        branch.state_snapshot,
        now=now + timedelta(hours=1),
    )
    assert later is not None
    assert later["values"] == {"location": "公司"}


def test_older_observation_cannot_overwrite_newer_slot() -> None:
    from moonlightbox.branches.situational_state import (
        active_situational_state,
        set_situational_state,
    )

    now = datetime(2026, 8, 5, 12, tzinfo=UTC)
    branch = SimpleNamespace(state_snapshot={})
    set_situational_state(
        branch,  # type: ignore[arg-type]
        values={"location": "公司"},
        source="external_observation",
        evidence_ids=("new",),
        observed_at=now,
        valid_until=now + timedelta(hours=2),
    )
    set_situational_state(
        branch,  # type: ignore[arg-type]
        values={"location": "家"},
        source="historical_replay",
        evidence_ids=("old",),
        observed_at=now - timedelta(hours=1),
        valid_until=now + timedelta(hours=1),
    )

    active = active_situational_state(branch.state_snapshot, now=now)
    assert active is not None
    assert active["values"] == {"location": "公司"}


def test_current_state_draft_selects_only_the_asked_trusted_slot() -> None:
    from moonlightbox.branches.situational_state import current_state_content_draft

    state = {
        "values": {
            "availability": "并不方便电话",
            "location": "在公司",
            "activity": "正在开会",
        }
    }

    assert current_state_content_draft(state, "现在可以电话吗") == "并不方便电话"
    assert current_state_content_draft(state, "你在哪里呀") == "在公司"
    assert current_state_content_draft(state, "你在干嘛") == "正在开会"
    assert current_state_content_draft(state, "你还爱我吗") is None
