from datetime import datetime, timedelta

from moonlightbox.imports.types import ImportedMessage, MessageKind


def _message(
    source_id: str,
    sender: str,
    second: int,
    content: str | None = None,
    *,
    kind: MessageKind = MessageKind.TEXT,
    raw: dict[str, object] | None = None,
) -> ImportedMessage:
    return ImportedMessage(
        source_id=source_id,
        sender=sender,
        timestamp=datetime(2026, 1, 1) + timedelta(seconds=second),
        kind=kind,
        content=content or source_id,
        raw=raw or {},
    )


def test_turn_builder_ends_on_sender_change_and_long_same_sender_gap() -> None:
    from moonlightbox.training.turns import build_conversation_turns

    messages = [
        _message("m1", "我", 0),
        _message("m2", "她", 1),
        _message("m3", "她", 2),
        _message("m4", "我", 3),
        _message("m5", "她", 4),
        _message("m6", "她", 70),
    ]

    turns, profiles = build_conversation_turns(messages)

    assert [(turn.speaker, len(turn.bubbles)) for turn in turns] == [
        ("我", 1),
        ("她", 2),
        ("我", 1),
        ("她", 1),
        ("她", 1),
    ]
    assert turns[1].bubbles[1].delay_ms == 1000
    assert [bubble.delay_ms for turn in turns for bubble in turn.bubbles] == [
        0,
        1000,
        1000,
        1000,
        1000,
        66_000,
    ]
    assert 3 <= profiles["她"].threshold_seconds <= 180


def test_turn_builder_learns_sender_specific_threshold() -> None:
    from moonlightbox.training.turns import build_conversation_turns

    messages = [
        _message("a1", "她", 0),
        _message("a2", "她", 10),
        _message("x1", "我", 20),
        _message("a3", "她", 30),
        _message("a4", "她", 42),
        _message("x2", "我", 50),
        _message("a5", "她", 60),
        _message("a6", "她", 71),
    ]

    _turns, profiles = build_conversation_turns(messages)

    assert profiles["她"].sample_count == 3
    assert profiles["她"].threshold_seconds > 12


def test_turn_builder_preserves_more_than_four_consecutive_bubbles() -> None:
    from moonlightbox.training.turns import build_conversation_turns

    messages = [_message(f"m{index}", "她", index) for index in range(5)]

    turns, _profiles = build_conversation_turns(messages)

    assert [len(turn.bubbles) for turn in turns] == [5]
    assert [bubble.source_id for bubble in turns[0].bubbles] == [
        "m0",
        "m1",
        "m2",
        "m3",
        "m4",
    ]
    assert [bubble.delay_ms for bubble in turns[0].bubbles] == [0, 1000, 1000, 1000, 1000]


def test_turn_builder_preserves_text_sticker_emoji_and_full_delay() -> None:
    from moonlightbox.training.turns import build_conversation_turns

    messages = [
        _message("m1", "她", 0, "啊啊啊！！！🥺"),
        _message(
            "m2",
            "她",
            240,
            "[动画表情]",
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "sticker<&>"},
        ),
        _message(
            "m3",
            "她",
            241,
            "<msg><emoji md5='abc'/></msg>",
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "emoji-1"},
        ),
    ]

    turns, _profiles = build_conversation_turns(messages)

    assert [bubble.content for turn in turns for bubble in turn.bubbles] == [
        "啊啊啊！！！🥺",
        "[动画表情]",
        "<msg><emoji md5='abc'/></msg>",
    ]
    assert [bubble.asset_id for turn in turns for bubble in turn.bubbles] == [
        None,
        "sticker<&>",
        "emoji-1",
    ]
    assert [bubble.kind for turn in turns for bubble in turn.bubbles] == [
        "text",
        "sticker",
        "emoji",
    ]
    assert turns[1].bubbles[0].timestamp == messages[1].timestamp


def test_turn_builder_keeps_input_order_when_timestamps_are_equal() -> None:
    from moonlightbox.training.turns import build_conversation_turns

    messages = [
        _message("z-first", "她", 0),
        _message("a-second", "她", 0),
    ]

    turns, _profiles = build_conversation_turns(messages)

    assert turns[0].source_ids == ("z-first", "a-second")
