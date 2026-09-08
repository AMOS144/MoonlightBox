from types import SimpleNamespace

from moonlightbox.agent.persona_examples import (
    AuthenticDialogueExample,
    extract_authentic_dialogue_examples,
    rank_authentic_dialogue_examples,
)


def test_extract_examples_keeps_real_multi_bubble_structure() -> None:
    rows = [
        (SimpleNamespace(kind="text", content="这些牛仔裤分不清"), "self"),
        (SimpleNamespace(kind="text", content="这该如何选择呢"), "self"),
        (SimpleNamespace(kind="text", content="啊呀"), "target"),
        (SimpleNamespace(kind="text", content="请尽力吧！"), "target"),
        (SimpleNamespace(kind="text", content="我非常想你"), "self"),
        (SimpleNamespace(kind="text", content="我也很想你 笨入"), "target"),
    ]

    examples = extract_authentic_dialogue_examples(rows)  # type: ignore[arg-type]

    assert examples[0].other_messages == (
        "这些牛仔裤分不清",
        "这该如何选择呢",
    )
    assert examples[0].person_messages == ("啊呀", "请尽力吧！")


def test_rank_examples_prefers_matching_situation_then_adds_recent_style() -> None:
    examples = (
        AuthenticDialogueExample(("宝宝你在干嘛" ,), ("小入",)),
        AuthenticDialogueExample(("这该如何选择呢",), ("啊呀", "请尽力吧！")),
        AuthenticDialogueExample(("我非常想你",), ("我也很想你 笨入",)),
    )

    selected = rank_authentic_dialogue_examples(
        examples,
        "该如何是好呢\n非常焦虑",
        limit=2,
    )

    assert selected[0] == examples[1]
    assert len(selected) == 2


def test_extract_examples_drops_corrupted_and_single_glyph_bubbles() -> None:
    rows = [
        (SimpleNamespace(kind="text", content="非常纠结"), "self"),
        (SimpleNamespace(kind="text", content="딨ﴯⴠ]⠀\ue85b늉ŝ儀ⴖ"), "target"),
        (SimpleNamespace(kind="text", content="入"), "target"),
        (SimpleNamespace(kind="text", content="要不去看海"), "target"),
    ]

    examples = extract_authentic_dialogue_examples(rows)  # type: ignore[arg-type]

    assert len(examples) == 1
    assert examples[0].person_messages == ("要不去看海",)


def test_rank_examples_has_no_unrelated_recency_fallback() -> None:
    examples = (
        AuthenticDialogueExample(("牛仔裤尺码怎么选",), ("你试一下呀",)),
        AuthenticDialogueExample(("今天吃了什么",), ("吃面",)),
    )

    selected = rank_authentic_dialogue_examples(
        examples,
        "旅行目的地非常纠结",
        limit=6,
    )

    assert selected == ()
