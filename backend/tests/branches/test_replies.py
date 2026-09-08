import pytest


def test_reply_parser_accepts_and_clamps_multiple_bubbles() -> None:
    from moonlightbox.branches.replies import parse_reply_turn

    turn = parse_reply_turn(
        '{"bubbles":[{"content":"我刚洗完澡","delay_ms":0},'
        '{"content":"你呢笨笨","delay_ms":999999}]}'
    )

    assert [bubble.content for bubble in turn.bubbles] == [
        "我刚洗完澡",
        "你呢笨笨",
    ]
    assert turn.bubbles[1].delay_ms == 5000


@pytest.mark.parametrize(
    "payload",
    [
        '{"bubbles":[]}',
        '{"bubbles":[{"content":"","delay_ms":0}]}',
        '{"bubbles":[{"content":"重复","delay_ms":0},{"content":"重复","delay_ms":1}]}',
    ],
)
def test_reply_parser_rejects_invalid_or_repeated_bubbles(payload: str) -> None:
    from moonlightbox.branches.replies import ReplyStructureError, parse_reply_turn

    with pytest.raises(ReplyStructureError):
        parse_reply_turn(payload)


def test_repetition_detector_rejects_repeated_phrases() -> None:
    from moonlightbox.branches.replies import ReplyStructureError, parse_reply_turn

    with pytest.raises(ReplyStructureError):
        parse_reply_turn(
            '{"bubbles":[{"content":"我猜你可能在等我，'
            '我猜你可能在等我，我猜你可能在等我","delay_ms":0}]}'
        )


def test_reply_parser_extracts_json_after_thinking_residue() -> None:
    from moonlightbox.branches.replies import parse_reply_turn

    turn = parse_reply_turn(
        '.GridView\n\n</think>\n\n{"bubbles":[{"content":"笨笨，我想你了","delay_ms":0}]}'
    )

    assert [bubble.content for bubble in turn.bubbles] == ["笨笨，我想你了"]


def test_reply_parser_preserves_compact_protocol_emoji_kind() -> None:
    from moonlightbox.branches.replies import parse_reply_turn

    turn = parse_reply_turn(
        '<sticker kind="emoji">emoji-smile</sticker><delay>0</delay>'
    )

    assert turn.bubbles[0].type == "emoji"
    assert turn.bubbles[0].asset_id == "emoji-smile"


def test_persona_text_parser_maps_natural_lines_to_application_bubbles() -> None:
    from moonlightbox.branches.replies import parse_persona_text_turn

    turn = parse_persona_text_turn("嗯\n咋啦")

    assert [bubble.content for bubble in turn.bubbles] == ["嗯", "咋啦"]
    assert [bubble.delay_ms for bubble in turn.bubbles] == [0, 800]


def test_persona_text_parser_preserves_real_multi_bubble_rhythm() -> None:
    from moonlightbox.branches.replies import parse_persona_text_turn

    turn = parse_persona_text_turn("\n".join(f"短句{index}" for index in range(10)))

    assert len(turn.bubbles) == 10


def test_persona_text_parser_preserves_person_specific_catchphrase_verbatim() -> None:
    from moonlightbox.branches.replies import parse_persona_text_turn

    turn = parse_persona_text_turn("入\n哎呀入！\n此笨入在做什么呢")

    assert [bubble.content for bubble in turn.bubbles] == [
        "入",
        "哎呀入！",
        "此笨入在做什么呢",
    ]
    assert turn.normalization is None


def test_persona_text_parser_rejects_markdown_media_hallucination() -> None:
    from moonlightbox.branches.replies import ReplyStructureError, parse_persona_text_turn

    with pytest.raises(ReplyStructureError, match="Markdown"):
        parse_persona_text_turn("![](https://example.com/fake.png)")
    with pytest.raises(ReplyStructureError, match="Markdown"):
        parse_persona_text_turn("![](https://example.com/truncated")


def test_reply_parser_normalizes_mixed_compact_without_rewriting_text() -> None:
    from moonlightbox.branches.replies import parse_reply_turn

    turn = parse_reply_turn(
        "assistant: <bubble>  原样文字  </bubble>"
        "标签间文字"
        "<sticker>future-id</sticker>",
        allowed_sticker_ids=("allowed",),
        normalize_compact=True,
    )

    assert [bubble.content for bubble in turn.bubbles] == [
        "  原样文字  ",
        "标签间文字",
    ]
    assert turn.normalization == "mixed-compact-v1"
    assert turn.attempted_invalid_sticker_ids == ("future-id",)
    assert turn.raw_output.startswith("assistant:")


def test_compact_normalization_never_accepts_json_fallback() -> None:
    from moonlightbox.branches.replies import ReplyStructureError, parse_reply_turn

    with pytest.raises(ReplyStructureError, match="结构残片"):
        parse_reply_turn(
            '{"bubbles":[{"content":"不能把 JSON 当 compact"}]}',
            normalize_compact=True,
        )


def test_exact_legacy_sticker_lines_keep_allowed_and_audit_out_of_top_k_ids() -> None:
    from moonlightbox.branches.replies import parse_reply_turn

    reply = parse_reply_turn(
        "[sticker资产:allowed]\n[sticker资产:future-id]",
        allowed_sticker_ids=("allowed",),
        normalize_compact=True,
    )

    assert [bubble.asset_id for bubble in reply.bubbles] == ["allowed"]
    assert reply.attempted_invalid_sticker_ids == ("future-id",)
    assert reply.normalization == "mixed-compact-v1"
