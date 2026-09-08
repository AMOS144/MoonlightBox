import pytest


def test_style_transfer_prompt_includes_real_examples_without_copying_facts() -> None:
    from moonlightbox.training.bubble_protocol import (
        persona_style_transfer_instruction,
    )

    prompt = persona_style_transfer_instruction(
        ("对方：这该如何选择呢\n本人：啊呀 / 请尽力吧！",)
    )

    assert "这该如何选择呢" in prompt
    assert "称呼、口癖、句长" in prompt
    assert "不得照抄其中的具体事实" in prompt


def test_compact_bubble_protocol_round_trips_escaped_content() -> None:
    from moonlightbox.training.bubble_protocol import (
        ProtocolBubble,
        parse_bubble_protocol,
        serialize_bubble_protocol,
    )

    bubbles = (
        ProtocolBubble(kind="text", value="原样</bubble>&🥺", delay_ms=60_000),
        ProtocolBubble(kind="sticker", value="asset<&>", delay_ms=0),
    )

    serialized = serialize_bubble_protocol(bubbles)

    assert serialized == (
        "<bubble>原样&lt;/bubble&gt;&amp;🥺</bubble><delay>60000</delay>"
        "<sticker>asset&lt;&amp;&gt;</sticker><delay>0</delay>"
    )
    assert parse_bubble_protocol(serialized) == bubbles


def test_compact_protocol_round_trips_emoji_asset_kind_and_keeps_legacy_default() -> None:
    from moonlightbox.training.bubble_protocol import (
        ProtocolBubble,
        parse_bubble_protocol,
        serialize_bubble_protocol,
    )

    emoji = ProtocolBubble(kind="emoji", value="emoji-1", delay_ms=300)

    assert serialize_bubble_protocol((emoji,)) == (
        '<sticker kind="emoji">emoji-1</sticker><delay>300</delay>'
    )
    assert parse_bubble_protocol(
        '<sticker kind="emoji">emoji-1</sticker><delay>300</delay>'
    ) == (emoji,)
    assert parse_bubble_protocol(
        "<sticker>legacy-1</sticker><delay>0</delay>"
    ) == (ProtocolBubble(kind="sticker", value="legacy-1", delay_ms=0),)


def test_compact_protocol_instruction_is_single_source_and_has_no_copyable_placeholder() -> None:
    from moonlightbox.training.bubble_protocol import compact_protocol_instruction

    forbidden = compact_protocol_instruction(())
    allowed = compact_protocol_instruction(("real-a", "real-b"))

    assert '文字气泡 ::= "<bubble>" 文本 "</bubble>"' in forbidden
    assert '延迟 ::= "<delay>" 非负整数 "</delay>"' in forbidden
    assert "本轮禁止输出 sticker" in forbidden
    assert "资产ID" not in forbidden
    assert "asset_id" not in forbidden
    assert "real-a、real-b" in allowed
    assert "本轮禁止输出 sticker" not in allowed


def test_allowed_sticker_parser_uses_last_protocol_override() -> None:
    from moonlightbox.training.bubble_protocol import (
        allowed_sticker_ids_from_prompt,
    )

    assert allowed_sticker_ids_from_prompt(
        "旧提示：本轮禁止输出 sticker。\n"
        "新提示：本轮 sticker 仅允许以下资产 ID：new-a、new-b；禁止输出其他 sticker。"
    ) == ("new-a", "new-b")
    assert allowed_sticker_ids_from_prompt(
        "旧提示：本轮 sticker 仅允许以下资产 ID：old；禁止输出其他 sticker。\n"
        "新提示：本轮禁止输出 sticker。"
    ) == ()


@pytest.mark.parametrize(
    "content",
    [
        "游离文本<bubble>你好</bubble><delay>0</delay>",
        "<unknown>你好</unknown><delay>0</delay>",
        "<bubble><sticker>嵌套</sticker></bubble><delay>0</delay>",
        "<bubble>你好</bubble><delay>abc</delay>",
        "<bubble>你好</bubble><delay>-1</delay>",
        "<delay>0</delay><bubble>你好</bubble>",
        "<bubble>你好</bubble>尾部文本<delay>0</delay>",
        '<sticker kind="gif">asset</sticker><delay>0</delay>',
        '<sticker kind="emoji" extra="x">asset</sticker><delay>0</delay>',
        '<bubble kind="emoji">你好</bubble><delay>0</delay>',
    ],
)
def test_compact_bubble_protocol_rejects_noncanonical_structure(content: str) -> None:
    from moonlightbox.training.bubble_protocol import parse_bubble_protocol

    with pytest.raises(ValueError):
        parse_bubble_protocol(content)


def test_plain_text_lines_are_safely_normalized_without_rewriting() -> None:
    from moonlightbox.training.bubble_protocol import normalize_plain_text_lines

    assert normalize_plain_text_lines("  第一行原样  \n\n第二行") == (
        "  第一行原样  ",
        "第二行",
    )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   \n\t",
        "半截<bubble",
        '半截{"bubbles":',
        "```text\n自然文本\n```",
        "带有}结构",
        "[sticker资产:越界ID]",
    ],
)
def test_plain_text_normalization_rejects_empty_or_protocol_like_content(
    content: str,
) -> None:
    from moonlightbox.training.bubble_protocol import normalize_plain_text_lines

    with pytest.raises(ValueError):
        normalize_plain_text_lines(content)


def test_mixed_compact_normalizer_preserves_text_and_recovers_missing_delays() -> None:
    from moonlightbox.training.bubble_protocol import (
        ProtocolBubble,
        normalize_mixed_compact_protocol,
    )

    normalized = normalize_mixed_compact_protocol(
        "```text\n"
        "assistant: <bubble>  第一条原样  </bubble>\n"
        "标签之间的文字\n"
        "<bubble>第二条</bubble><delay>350</delay>\n"
        "<sticker>allowed</sticker>\n"
        "```",
        allowed_sticker_ids=("allowed",),
    )

    assert normalized.bubbles == (
        ProtocolBubble(kind="text", value="  第一条原样  ", delay_ms=0),
        ProtocolBubble(kind="text", value="标签之间的文字", delay_ms=0),
        ProtocolBubble(kind="text", value="第二条", delay_ms=350),
        ProtocolBubble(kind="sticker", value="allowed", delay_ms=0),
    )
    assert normalized.attempted_invalid_sticker_ids == ()


def test_mixed_compact_normalizer_deletes_only_invalid_sticker_bubble() -> None:
    from moonlightbox.training.bubble_protocol import (
        ProtocolBubble,
        normalize_mixed_compact_protocol,
    )

    normalized = normalize_mixed_compact_protocol(
        "<bubble>保留原文字</bubble>"
        "<sticker>future-id</sticker><delay>10</delay>",
        allowed_sticker_ids=("allowed",),
    )

    assert normalized.bubbles == (
        ProtocolBubble(kind="text", value="保留原文字", delay_ms=0),
    )
    assert normalized.attempted_invalid_sticker_ids == ("future-id",)


def test_mixed_compact_normalizer_accepts_exact_legacy_sticker_line_only() -> None:
    from moonlightbox.training.bubble_protocol import (
        ProtocolBubble,
        normalize_mixed_compact_protocol,
    )

    normalized = normalize_mixed_compact_protocol(
        "<bubble>保留文字</bubble>\n"
        "[sticker资产:allowed_ID-1]\n"
        "末尾文字",
        allowed_sticker_ids=("allowed_ID-1",),
    )

    assert normalized.bubbles == (
        ProtocolBubble(kind="text", value="保留文字", delay_ms=0),
        ProtocolBubble(kind="sticker", value="allowed_ID-1", delay_ms=0),
        ProtocolBubble(kind="text", value="末尾文字", delay_ms=0),
    )
    assert normalized.attempted_invalid_sticker_ids == ()


def test_mixed_compact_normalizer_drops_out_of_top_k_legacy_sticker_line() -> None:
    from moonlightbox.training.bubble_protocol import (
        ProtocolBubble,
        normalize_mixed_compact_protocol,
    )

    normalized = normalize_mixed_compact_protocol(
        "<bubble>保留文字</bubble>\n[sticker资产:future-id]",
        allowed_sticker_ids=("allowed",),
    )

    assert normalized.bubbles == (
        ProtocolBubble(kind="text", value="保留文字", delay_ms=0),
    )
    assert normalized.attempted_invalid_sticker_ids == ("future-id",)


def test_mixed_compact_normalizer_rejects_invalid_sticker_only_output() -> None:
    from moonlightbox.training.bubble_protocol import (
        CompactProtocolNormalizationError,
        normalize_mixed_compact_protocol,
    )

    with pytest.raises(
        CompactProtocolNormalizationError,
        match="删除越界 sticker 后回复为空",
    ) as captured:
        normalize_mixed_compact_protocol(
            "<sticker>future-id</sticker>",
            allowed_sticker_ids=("allowed",),
        )

    assert captured.value.attempted_invalid_sticker_ids == ("future-id",)


@pytest.mark.parametrize(
    "content",
    [
        "[bubble]无人陪我打王者[/bubble][delay]0[/delay]",
        "前缀[sticker资产:asset-id]",
        "[sticker资产:asset-id]后缀",
        " [sticker资产:asset-id]",
        "[sticker资产:asset.id]",
        "[unknown:asset-id]",
        "[[sticker资产:asset-id]]",
        "<bubble>半截",
        "<bubble><sticker>嵌套</sticker></bubble>",
        "<unknown>未知</unknown>",
        "<script>alert(1)</script>",
        '{"bubbles":[{"content":"JSON 残片"}]}',
        "```text\n```inner\n文字\n```\n```",
    ],
)
def test_mixed_compact_normalizer_rejects_unsafe_or_unknown_structures(
    content: str,
) -> None:
    from moonlightbox.training.bubble_protocol import normalize_mixed_compact_protocol

    with pytest.raises(ValueError):
        normalize_mixed_compact_protocol(content, allowed_sticker_ids=())


def test_mixed_compact_normalizer_moves_leading_delay_to_sticker() -> None:
    from moonlightbox.training.bubble_protocol import (
        ProtocolBubble,
        normalize_mixed_compact_protocol,
    )

    normalized = normalize_mixed_compact_protocol(
        "<delay>500</delay>\n<sticker>asset-1</sticker>",
        allowed_sticker_ids=("asset-1",),
    )

    assert normalized.bubbles == (ProtocolBubble("sticker", "asset-1", 500),)
