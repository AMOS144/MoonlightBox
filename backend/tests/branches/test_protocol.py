import pytest


def test_reply_protocol_accepts_text_and_sticker_bubbles() -> None:
    from moonlightbox.branches.replies import parse_reply_turn

    turn = parse_reply_turn(
        '{"bubbles":['
        '{"type":"text","content":"抱抱你","delay_ms":0},'
        '{"type":"sticker","asset_id":"asset-1","delay_ms":800}'
        "]}"
    )

    assert turn.bubbles[0].type == "text"
    assert turn.bubbles[1].type == "sticker"
    assert turn.bubbles[1].asset_id == "asset-1"


def test_sticker_bubble_requires_asset_id() -> None:
    from moonlightbox.branches.replies import ReplyStructureError, parse_reply_turn

    with pytest.raises(ReplyStructureError):
        parse_reply_turn('{"bubbles":[{"type":"sticker","delay_ms":0}]}')


def test_reply_protocol_accepts_training_compact_protocol() -> None:
    from moonlightbox.branches.replies import parse_reply_turn

    turn = parse_reply_turn(
        "<bubble>原样&lt;/bubble&gt;&amp;🥺</bubble><delay>60000</delay>"
        "<sticker>asset-1</sticker><delay>800</delay>"
    )

    assert [(bubble.type, bubble.content, bubble.asset_id) for bubble in turn.bubbles] == [
        ("text", "原样</bubble>&🥺", None),
        ("sticker", None, "asset-1"),
    ]
    assert turn.bubbles[1].delay_ms == 800
