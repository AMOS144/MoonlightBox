from datetime import datetime, timedelta

import pytest
from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.imports.types import MessageKind

BASE_TIME = datetime(2026, 7, 18, 12, 0)


def make_message(
    source_id: str,
    content: str,
    *,
    minutes: int = 0,
) -> NormalizedMessage:
    return NormalizedMessage(
        source_id=source_id,
        timestamp=BASE_TIME + timedelta(minutes=minutes),
        sender="甲",
        kind=MessageKind.TEXT,
        content=content,
    )


def test_split_sessions_uses_only_gap_as_boundary() -> None:
    from moonlightbox.events.windowing import split_sessions

    sessions = split_sessions(
        [
            make_message("m1", "一", minutes=0),
            make_message("m2", "二", minutes=30),
            make_message("m3", "三", minutes=151),
            make_message("m4", "四", minutes=160),
        ],
        gap=timedelta(hours=2),
    )

    assert [[message.source_id for message in session.messages] for session in sessions] == [
        ["m1", "m2"],
        ["m3", "m4"],
    ]


def test_build_windows_honors_budget_boundary_and_metadata() -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions(
        [
            make_message("m1", "甲乙"),
            make_message("m2", "一二三", minutes=1),
            make_message("m3", "末", minutes=2),
        ],
        gap=timedelta(hours=1),
    )

    windows = build_analysis_windows(sessions, character_budget=5)

    assert [[message.source_id for message in window.messages] for window in windows] == [
        ["m1", "m2"],
        ["m3"],
    ]
    assert windows[0].start_message_id == "m1"
    assert windows[0].end_message_id == "m2"
    assert windows[0].character_count == 5
    assert windows == build_analysis_windows(sessions, character_budget=5)
    assert windows[0].window_id != windows[1].window_id


def test_character_budget_counts_only_content_characters() -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions(
        [
            make_message("long-metadata-id", "甲 乙"),
            make_message("another-long-id", "四五", minutes=1),
        ],
        gap=timedelta(hours=1),
    )

    windows = build_analysis_windows(sessions, character_budget=5)

    assert len(windows) == 1
    assert windows[0].character_count == len("甲 乙") + len("四五")


def test_build_windows_repeats_configured_overlap_without_losing_order() -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions(
        [
            make_message("m1", "一二三"),
            make_message("m2", "四五六", minutes=1),
            make_message("m3", "七八九", minutes=2),
            make_message("m4", "十十一", minutes=3),
        ],
        gap=timedelta(hours=1),
    )

    windows = build_analysis_windows(
        sessions,
        character_budget=6,
        overlap_messages=1,
    )

    assert [[message.source_id for message in window.messages] for window in windows] == [
        ["m1", "m2"],
        ["m2", "m3"],
        ["m3", "m4"],
    ]


def test_windows_never_cross_session_boundaries() -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions(
        [
            make_message("event-start", "我们需要谈谈"),
            make_message("event-follow-up", "昨天的问题还没解决", minutes=180),
        ],
        gap=timedelta(hours=2),
    )

    windows = build_analysis_windows(sessions, character_budget=100)

    assert len(sessions) == 2
    assert [[message.source_id for message in window.messages] for window in windows] == [
        ["event-start"],
        ["event-follow-up"],
    ]


def test_empty_input_produces_no_sessions_or_windows() -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions([], gap=timedelta(hours=1))

    assert sessions == []
    assert build_analysis_windows(sessions, character_budget=10) == []


def test_single_oversized_message_forms_its_own_window() -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions(
        [make_message("oversized", "超" * 20)],
        gap=timedelta(hours=1),
    )

    windows = build_analysis_windows(
        sessions,
        character_budget=5,
        overlap_messages=3,
    )

    assert len(windows) == 1
    assert windows[0].start_message_id == "oversized"
    assert windows[0].end_message_id == "oversized"
    assert windows[0].character_count == 20


@pytest.mark.parametrize("overlap_messages", [1, 100])
def test_overlap_is_capped_to_zero_for_single_message_windows(overlap_messages: int) -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions(
        [
            make_message("m1", "甲"),
            make_message("m2", "乙", minutes=1),
        ],
        gap=timedelta(hours=1),
    )

    windows = build_analysis_windows(
        sessions,
        character_budget=1,
        overlap_messages=overlap_messages,
    )

    assert [[message.source_id for message in window.messages] for window in windows] == [
        ["m1"],
        ["m2"],
    ]


def test_near_full_overlap_keeps_window_references_linear() -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions(
        [make_message(f"m{index}", "字", minutes=index) for index in range(2000)],
        gap=timedelta(hours=1),
    )

    windows = build_analysis_windows(
        sessions,
        character_budget=1000,
        overlap_messages=999,
    )

    assert len(windows) == 3
    assert sum(len(window.messages) for window in windows) == 3000


def test_window_ids_remain_unique_and_stable_when_source_ids_repeat() -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions(
        [
            make_message("duplicate", "甲"),
            make_message("duplicate", "乙", minutes=1),
            make_message("duplicate", "丙", minutes=2),
        ],
        gap=timedelta(hours=1),
    )

    first = build_analysis_windows(sessions, character_budget=1)
    second = build_analysis_windows(sessions, character_budget=1)

    assert len({window.window_id for window in first}) == 3
    assert [window.window_id for window in first] == [window.window_id for window in second]


def test_messages_are_ordered_stably_by_timestamp_and_source_id() -> None:
    from moonlightbox.events.windowing import build_analysis_windows, split_sessions

    sessions = split_sessions(
        [
            make_message("m3", "三", minutes=2),
            make_message("m2", "二", minutes=1),
            make_message("m1", "一", minutes=1),
        ],
        gap=timedelta(hours=1),
    )

    windows = build_analysis_windows(sessions, character_budget=10)

    assert [message.source_id for message in sessions[0].messages] == ["m1", "m2", "m3"]
    assert [message.source_id for message in windows[0].messages] == ["m1", "m2", "m3"]


def test_persistence_context_starts_after_candidate_then_adds_three_sessions() -> None:
    from moonlightbox.events.windowing import (
        build_analysis_windows,
        get_persistence_context,
        split_sessions,
    )

    sessions = split_sessions(
        [
            make_message("m0", "零", minutes=0),
            make_message("m1", "一", minutes=1),
            make_message("m2", "二", minutes=2),
            make_message("m3", "三", minutes=120),
            make_message("m4", "四", minutes=240),
            make_message("m5", "五", minutes=360),
            make_message("m6", "六", minutes=480),
        ],
        gap=timedelta(hours=1),
    )
    candidate = build_analysis_windows(sessions, character_budget=2)[0]

    context = get_persistence_context(candidate, sessions)

    assert [message.source_id for message in context.remaining_session_messages] == ["m2"]
    assert [session.messages[0].source_id for session in context.following_sessions] == [
        "m3",
        "m4",
        "m5",
    ]
    assert [message.source_id for message in context.messages] == ["m2", "m3", "m4", "m5"]


@pytest.mark.parametrize("gap", [timedelta(0), timedelta(seconds=-1)])
def test_split_sessions_rejects_non_positive_gap(gap: timedelta) -> None:
    from moonlightbox.events.windowing import split_sessions

    with pytest.raises(ValueError, match="gap"):
        split_sessions([make_message("m1", "甲")], gap=gap)


@pytest.mark.parametrize(
    ("character_budget", "overlap_messages"),
    [(0, 0), (-1, 0), (1, -1)],
)
def test_build_windows_rejects_invalid_configuration(
    character_budget: int,
    overlap_messages: int,
) -> None:
    from moonlightbox.events.windowing import build_analysis_windows

    with pytest.raises(ValueError):
        build_analysis_windows(
            [],
            character_budget=character_budget,
            overlap_messages=overlap_messages,
        )


@pytest.mark.parametrize("max_sessions", [-1, 4])
def test_persistence_context_rejects_invalid_session_count(max_sessions: int) -> None:
    from moonlightbox.events.windowing import (
        build_analysis_windows,
        get_persistence_context,
        split_sessions,
    )

    sessions = split_sessions(
        [make_message("m1", "甲")],
        gap=timedelta(hours=1),
    )
    candidate = build_analysis_windows(sessions, character_budget=1)[0]

    with pytest.raises(ValueError, match="max_sessions"):
        get_persistence_context(candidate, sessions, max_sessions=max_sessions)
