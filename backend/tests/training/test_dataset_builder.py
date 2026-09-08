import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from moonlightbox.imports.types import ImportedMessage, MessageKind


def make_message(
    source_id: str,
    sender: str,
    content: str,
    minute: int,
    *,
    kind: MessageKind = MessageKind.TEXT,
    raw: dict[str, object] | None = None,
) -> ImportedMessage:
    return ImportedMessage(
        source_id=source_id,
        timestamp=datetime(2026, 1, 1, 12) + timedelta(minutes=minute),
        sender=sender,
        kind=kind,
        content=content,
        raw=raw or {},
    )


def test_dataset_excludes_messages_after_historical_cutoff() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "我", "今天见面吗", 0),
        make_message("m2", "她", "好呀", 1),
        make_message("m3", "我", "后来我们分手了", 60),
        make_message("m4", "她", "再见", 61),
    ]

    examples = DatasetBuilder(context_turns=4).build(
        messages,
        target_sender="她",
        cutoff=datetime(2026, 1, 1, 12, 30),
    )

    assert len(examples) == 1
    assert examples[0].source_ids == ["m1", "m2"]
    assert all("分手" not in turn.content for turn in examples[0].messages)


def test_training_system_condition_matches_private_chat_runtime_without_meta_roleplay() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    examples = DatasetBuilder().build(
        [
            make_message("m1", "我", "在干嘛", 0),
            make_message("m2", "她", "不知道说啥", 1),
        ],
        target_sender="她",
        cutoff=datetime(2026, 1, 2),
    )

    system = examples[0].messages[0].content
    assert "私人微信聊天" in system
    assert "正在学习复刻" not in system
    assert "正在扮演" not in system
    assert "assistant" not in system
    assert "历史截止到" not in system
    assert "<delay>0</delay>" not in system


def test_dataset_preserves_unusual_persona_catchphrase_verbatim() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    examples = DatasetBuilder(plain_text_supervision=True).build(
        [
            make_message("m1", "我", "在干嘛", 0),
            make_message("m2", "她", "入", 1),
            make_message("m3", "她", "此笨入已消失", 1),
        ],
        target_sender="她",
        cutoff=datetime(2026, 1, 2),
    )

    assert examples[0].messages[-1].content == "入\n此笨入已消失"


def test_dataset_export_uses_temporal_split_and_reproducible_manifest(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "我", "第一问", 0),
        make_message("m2", "她", "第一答", 1),
        make_message("m3", "我", "第二问", 60),
        make_message("m4", "她", "第二答", 61),
    ]
    cutoff = datetime(2026, 1, 2)
    builder = DatasetBuilder(context_turns=1)
    examples = builder.build(messages, target_sender="她", cutoff=cutoff)

    first = builder.write_mlx_dataset(examples, tmp_path, "她", cutoff)
    second = builder.write_mlx_dataset(examples, tmp_path, "她", cutoff)
    train = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8"))
    test = json.loads((tmp_path / "test.jsonl").read_text(encoding="utf-8"))

    assert train["metadata"]["source_ids"] == ["m1", "m2"]
    assert (tmp_path / "valid.jsonl").read_text(encoding="utf-8") == ""
    assert test["metadata"]["source_ids"] == ["m3", "m4"]
    assert first.dataset_hash == second.dataset_hash


def test_recency_weighting_duplicates_only_recent_train_rows_and_is_audited(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        message
        for index in range(10)
        for message in (
            make_message(f"q{index}", "我", f"问题{index}", index * 60),
            make_message(f"a{index}", "她", f"回答{index}", index * 60 + 1),
        )
    ]
    cutoff = datetime(2026, 1, 2, 12)
    builder = DatasetBuilder(
        context_turns=1,
        recency_window_days=1,
        recency_multiplier=2,
    )
    examples = builder.build(messages, target_sender="她", cutoff=cutoff)

    manifest = builder.write_mlx_dataset(examples, tmp_path, "她", cutoff)
    rows = {
        split: [
            json.loads(line)
            for line in (tmp_path / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for split in ("train", "valid", "test")
    }

    assert len(rows["train"]) == 16
    assert len(rows["valid"]) == 1
    assert len(rows["test"]) == 1
    assert len({tuple(row["metadata"]["source_ids"]) for row in rows["train"]}) == 8
    assert manifest.source_example_count == 10
    assert manifest.example_count == 18
    assert manifest.recency_oversampled_count == 8
    assert manifest.recency_window_days == 1
    assert manifest.recency_multiplier == 2


def test_consecutive_target_messages_form_one_multi_bubble_example() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "我", "在干嘛", 0),
        make_message("m2", "她", "刚洗完澡", 1),
        make_message("m3", "她", "你呢笨笨", 1),
    ]

    examples = DatasetBuilder(context_turns=4).build(
        messages,
        target_sender="她",
        cutoff=datetime(2026, 1, 2),
    )
    payload = ET.fromstring(f"<turn>{examples[0].messages[-1].content}</turn>")

    assert len(examples) == 1
    assert [(item.tag, item.text) for item in payload] == [
        ("bubble", "刚洗完澡"),
        ("delay", "60000"),
        ("bubble", "你呢笨笨"),
        ("delay", "0"),
    ]
    assert examples[0].source_ids == ["m1", "m2", "m3"]


def test_unpredictable_response_delay_is_bucketed_for_language_supervision() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    examples = DatasetBuilder().build(
        [
            make_message("m1", "我", "在吗", 0),
            make_message("m2", "她", "刚看到", 228),
        ],
        "她",
        datetime(2026, 1, 2),
    )

    assert examples[0].messages[-1].content == (
        "<bubble>刚看到</bubble><delay>60000</delay>"
    )


def test_compact_protocol_escapes_text_and_asset_ids_deterministically() -> None:
    from moonlightbox.training.dataset_builder import (
        DatasetBuilder,
        parse_assistant_protocol,
    )

    messages = [
        make_message("m1", "我", "说点什么", 0),
        make_message("m2", "她", "原样</bubble>&🥺", 1),
        make_message(
            "m3",
            "她",
            "[动画表情]",
            1,
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "asset<&>", "asset_kind": "emoji"},
        ),
    ]

    examples = DatasetBuilder().build(messages, "她", datetime(2026, 1, 2))
    target = examples[0].messages[-1].content
    payload = ET.fromstring(f"<turn>{target}</turn>")

    assert target == (
        "<bubble>原样&lt;/bubble&gt;&amp;🥺</bubble><delay>60000</delay>"
    )
    assert [(item.tag, item.text) for item in payload] == [
        ("bubble", "原样</bubble>&🥺"),
        ("delay", "60000"),
    ]
    assert [
        (bubble.kind, bubble.value, bubble.delay_ms)
        for bubble in parse_assistant_protocol(target)
    ] == [
        ("text", "原样</bubble>&🥺", 60_000),
    ]
    assert {item.tag for item in payload} == {"bubble", "delay"}


def test_dataset_filters_xml_forbidden_control_characters_with_audit_reason() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    builder = DatasetBuilder()
    examples = builder.build(
        [
            make_message("m1", "我", "能看到吗", 0),
            make_message("m2", "她", "损坏\x01文本", 1),
        ],
        "她",
        datetime(2026, 1, 2),
    )

    assert examples == []
    assert builder.audit.filtered_target_message_count == 1
    assert builder.audit.filter_reason_counts == {"xml_forbidden_control": 1}


def test_compact_protocol_rejects_emoji_tag() -> None:
    from moonlightbox.training.dataset_builder import parse_assistant_protocol

    with pytest.raises(ValueError, match="标签顺序"):
        parse_assistant_protocol("<emoji>asset-1</emoji><delay>0</delay>")


def test_private_use_font_residue_is_excluded_from_persona_supervision() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    builder = DatasetBuilder(plain_text_supervision=True)
    examples = builder.build(
        [
            make_message("u1", "我", "在吗", 0),
            make_message("t1", "她", "真人文字\ue85b字体残片", 1),
        ],
        "她",
        datetime(2026, 1, 2),
    )

    assert examples == []
    assert builder.audit.filter_reason_counts == {"private_use_unicode": 1}


def test_plain_text_supervision_contains_only_observed_persona_text() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("u1", "我", "在吗", 0),
        make_message("t1", "她", "嗯", 1),
        make_message("t2", "她", "咋啦", 2),
    ]
    examples = DatasetBuilder(plain_text_supervision=True).build(
        messages,
        "她",
        datetime(2026, 1, 2),
    )

    assert "每条独占一行。" in examples[-1].messages[0].content
    targets = [example.messages[-1].content for example in examples]
    assert targets == ["嗯", "咋啦"]
    assert examples[0].messages[-2].content == "在吗"
    assert examples[1].messages[-2].content == "（此刻没有新的用户消息）"
    assert "现在没有新的对方消息" in examples[1].messages[0].content
    assert all("<bubble>" not in target for target in targets)
    assert all("<delay>" not in target for target in targets)


def test_plain_text_target_retains_full_multimodal_behavior_supervision() -> None:
    from moonlightbox.imports.types import ImportedMessage, MessageKind
    from moonlightbox.training.dataset_builder import DatasetBuilder

    started = datetime(2026, 1, 1)
    examples = DatasetBuilder(plain_text_supervision=True).build(
        [
            ImportedMessage("u1", started, "我", MessageKind.TEXT, "抱抱", {}),
            ImportedMessage(
                "t1",
                started + timedelta(seconds=1),
                "她",
                MessageKind.TEXT,
                "来啦",
                {},
            ),
            ImportedMessage(
                "t2",
                started + timedelta(seconds=2),
                "她",
                MessageKind.STICKER,
                "[动画表情]",
                {"media_asset_id": "hug"},
            ),
        ],
        "她",
        datetime(2026, 1, 2),
    )

    assert examples[0].messages[-1].content == "来啦"
    assert [
        (item.kind, item.value, item.delay_ms)
        for item in examples[0].observed_bubbles
    ] == [
        ("text", "来啦", 1_000),
        ("sticker", "hug", 1_000),
    ]


def test_style_transfer_supervision_preserves_exact_target_and_split_group(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("u1", "我", "第一问", 0),
        make_message("t1", "她", "不去", 1),
        make_message("t2", "她", "今天不想动！", 1),
        make_message("u2", "我", "第二问", 60),
        make_message("t3", "她", "嗯嗯", 61),
        make_message("u3", "我", "第三问", 120),
        make_message("t4", "她", "咋啦", 121),
    ]
    builder = DatasetBuilder(
        plain_text_supervision=True,
        style_transfer_ratio=1.0,
    )
    examples = builder.build(messages, "她", datetime(2026, 1, 2))

    transfers = [item for item in examples if item.training_task == "style_transfer"]
    conversations = [item for item in examples if item.training_task == "conversation"]
    assert len(transfers) == len(conversations) == 3
    assert transfers[0].messages[-1].content == conversations[0].messages[-1].content
    assert transfers[0].messages[-2].content == "内容草稿：\n不去，今天不想动"
    assert "聊天文字本身" in transfers[0].messages[0].content

    formal = builder.build(
        [
            make_message("u4", "我", "怎么了", 180),
            make_message("t5", "她", "啊呀", 181),
            make_message("t6", "她", "是否非常想我呢", 181),
        ],
        "她",
        datetime(2026, 1, 2),
    )
    formal_transfer = next(
        item for item in formal if item.training_task == "style_transfer"
    )
    assert formal_transfer.messages[-2].content == "内容草稿：\n是不是很想我呢"

    manifest = builder.write_mlx_dataset(examples, tmp_path, "她", datetime(2026, 1, 2))
    split_groups: dict[tuple[str, ...], set[str]] = {}
    for split in ("train", "valid", "test"):
        for line in (tmp_path / f"{split}.jsonl").read_text().splitlines():
            payload = json.loads(line)
            key = tuple(payload["metadata"]["source_ids"])
            split_groups.setdefault(key, set()).add(split)
    assert all(len(splits) == 1 for splits in split_groups.values())
    assert manifest.regular_example_count == 3
    assert manifest.augmented_example_count == 3
    assert manifest.example_count == 6


def test_plain_text_supervision_removes_quoted_partner_payload() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    examples = DatasetBuilder(plain_text_supervision=True).build(
        [
            make_message("u1", "我", "原问题", 0),
            make_message(
                "t1",
                "她",
                '自己的回复\n> AMOS：<msg><img aeskey="secret"></img></msg>',
                1,
            ),
        ],
        "她",
        datetime(2026, 1, 2),
    )

    assert examples[0].messages[-1].content == "自己的回复"
    assert "AMOS" not in examples[0].messages[-1].content
    assert "aeskey" not in examples[0].messages[-1].content


def test_every_supported_target_message_is_covered_even_without_user_predecessor() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "她", "先说一句", 0),
        make_message("m2", "她", "隔很久再说", 10),
        make_message("m3", "我", "收到", 11),
        make_message("m4", "她", "最后一句", 12),
    ]
    builder = DatasetBuilder()

    examples = builder.build(messages, "她", datetime(2026, 1, 2))

    assert [example.source_ids[-1] for example in examples] == ["m1", "m2", "m4"]
    assert builder.audit.target_message_count == 3
    assert builder.audit.covered_target_message_count == 3
    assert builder.audit.filtered_target_message_count == 0


def test_prompt_context_is_capped_to_latest_twelve_turns() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message(
            f"m{index}",
            "我" if index % 2 == 0 else "她",
            f"第{index}句",
            index,
        )
        for index in range(16)
    ]

    examples = DatasetBuilder(context_turns=20).build(
        messages,
        "她",
        datetime(2026, 1, 2),
    )
    latest = examples[-1]

    assert len(latest.messages[1:-1]) == 12
    assert "第3句" in latest.messages[1].content
    assert all("第2句" not in message.content for message in latest.messages[1:-1])


def test_consecutive_bubble_coverage_uses_only_multi_message_target_groups() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "我", "开始", 0),
        make_message("m2", "她", "单独回复", 1),
        make_message("m3", "我", "继续", 2),
        make_message("m4", "她", "连续一", 3),
        make_message("m5", "她", "[图片]", 3, kind=MessageKind.IMAGE),
        make_message(
            "m6",
            "她",
            "[动画表情]",
            3,
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "asset-1"},
        ),
        make_message("m7", "我", "结束", 4),
        make_message("m8", "她", "另一条单独回复", 5),
    ]
    builder = DatasetBuilder()

    builder.build(messages, "她", datetime(2026, 1, 2))

    assert builder.audit.consecutive_bubble_coverage_rate == 2 / 3


def test_consecutive_bubble_coverage_is_one_without_consecutive_target_group() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "我", "第一问", 0),
        make_message("m2", "她", "第一答", 1),
        make_message("m3", "我", "第二问", 2),
        make_message("m4", "她", "第二答", 3),
    ]
    builder = DatasetBuilder()

    builder.build(messages, "她", datetime(2026, 1, 2))

    assert builder.audit.consecutive_bubble_coverage_rate == 1.0


def test_grounding_policy_examples_are_separate_from_identity_evidence() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    examples = DatasetBuilder().grounding_policy_examples(
        persona="洪欣羽",
        cutoff=datetime(2026, 5, 12),
        copies=2,
    )

    assert len(examples) == 10
    assert all(example.kind == "policy_augmented" for example in examples)
    assert all(
        all(source_id.startswith("policy:") for source_id in example.source_ids)
        for example in examples
    )
    assert any("抱抱" in example.messages[-1].content for example in examples)


def test_export_writes_independent_temporal_splits_and_excludes_augmented_targets(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.dataset_builder import (
        ConfirmedEventContext,
        DatasetBuilder,
    )

    messages = [
        make_message(
            f"m{index}",
            "我" if index % 2 == 0 else "她",
            f"消息{index}",
            index,
        )
        for index in range(20)
    ]
    builder = DatasetBuilder(maximum_event_ratio=1.0)
    regular = builder.build(messages, "她", datetime(2026, 1, 2))
    augmented = builder.augment_with_events(
        regular,
        [
            ConfirmedEventContext(
                event_id="event-1",
                title="节点标题",
                summary="节点摘要不得成为训练目标",
                lane="shared_experience",
                event_status="confirmed",
                evidence_ids=("m0", "m1"),
            )
        ],
    )
    all_examples = [
        *augmented,
        *builder.grounding_policy_examples(
            persona="她",
            cutoff=datetime(2026, 1, 2),
            copies=1,
        ),
    ]

    manifest = builder.write_mlx_dataset(
        all_examples,
        tmp_path,
        "她",
        datetime(2026, 1, 2),
    )
    splits = {
        name: [
            json.loads(line)
            for line in (tmp_path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for name in ("train", "valid", "test")
    }

    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 8,
        "valid": 1,
        "test": 1,
    }
    assert all(
        row["metadata"]["kind"] == "chat"
        for rows in splits.values()
        for row in rows
    )
    assert max(row["metadata"]["target_at"] for row in splits["train"]) < (
        splits["valid"][0]["metadata"]["target_at"]
    )
    assert splits["valid"][0]["metadata"]["target_at"] < (
        splits["test"][0]["metadata"]["target_at"]
    )
    assert manifest.split_counts == {"train": 8, "valid": 1, "test": 1}
    assert manifest.filtered_count == 6
    assert manifest.filter_reason_counts["policy_augmented"] == 5
    assert manifest.filter_reason_counts["event_augmented"] == 1
    assert manifest.protocol_version == "text-only-private-chat-v7"


def test_manifest_audits_target_filtering_and_bubble_kind_coverage(tmp_path: Path) -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "我", "看看", 0),
        make_message("m2", "她", "原文！！！", 1),
        make_message("m3", "她", "[图片]", 1, kind=MessageKind.IMAGE),
        make_message("m4", "她", "[动画表情]", 1, kind=MessageKind.STICKER),
        make_message(
            "m5",
            "她",
            "[动画表情]",
            1,
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "emoji-asset", "asset_kind": "emoji"},
        ),
    ]
    builder = DatasetBuilder()
    examples = builder.build(messages, "她", datetime(2026, 1, 2))

    manifest = builder.write_mlx_dataset(
        examples,
        tmp_path,
        "她",
        datetime(2026, 1, 2),
    )

    assert manifest.target_message_count == 4
    assert manifest.covered_target_message_count == 2
    assert manifest.filtered_target_message_count == 2
    assert manifest.filter_reason_counts == {
        "missing_asset_id": 1,
        "unsupported_kind:image": 1,
    }
    assert manifest.consecutive_bubble_coverage_rate == 0.5
    assert manifest.text_coverage_count == 1
    assert manifest.emoji_coverage_count == 1
    assert manifest.sticker_coverage_count == 0
    assert manifest.split_time_boundaries["train"]["start"] == examples[0].target_at.isoformat()


def test_each_training_prompt_only_lists_assets_seen_before_target() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [
        make_message("m1", "我", "先来一个", 0),
        make_message(
            "m2",
            "她",
            "[动画表情]",
            1,
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "first"},
        ),
        make_message("m3", "我", "再来一个", 2),
        make_message(
            "m4",
            "她",
            "[动画表情]",
            3,
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "first"},
        ),
        make_message(
            "m5",
            "她",
            "[动画表情]",
            3,
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "future"},
        ),
        make_message("m6", "我", "只回文字", 4),
        make_message("m7", "她", "好的", 5),
    ]

    examples = DatasetBuilder().build(messages, "她", datetime(2026, 1, 2))

    assert len(examples) == 1
    text_prompt = examples[0].messages[0].content
    assert "first" not in text_prompt
    assert "future" not in text_prompt
    assert examples[0].allowed_sticker_ids == ("first", "future")
    assert examples[0].sticker_first_seen is False
    assert examples[0].seen_target_sticker_ids == ()
    assert examples[0].first_seen_target_sticker_ids == ()
    assert all(
        asset_id not in message.content
        for asset_id in ("first", "future")
        for message in examples[0].messages[1:-1]
    )
    assert all(
        "<sticker>" not in message.content
        for message in examples[0].messages[1:-1]
    )
    assert all(
        "[sticker资产:" not in message.content
        for example in examples
        for message in example.messages[1:-1]
    )


def test_language_context_drops_repeated_stickers_and_hides_asset_ids() -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder

    messages = [make_message("u1", "我", "看这个", 0)]
    messages.extend(
        make_message(
            f"s{index}",
            "她",
            "[动画表情]",
            1,
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "private-asset-id"},
        )
        for index in range(100)
    )
    messages.extend(
        [
            make_message("u2", "我", "怎么啦", 2),
            make_message("a1", "她", "没事", 3),
        ]
    )

    examples = DatasetBuilder().build(messages, "她", datetime(2026, 1, 2))

    assert len(examples) == 1
    context = "\n".join(message.content for message in examples[0].messages[:-1])
    assert "<sticker>" not in context
    assert "private-asset-id" not in context
    assert all(message.content for message in examples[0].messages)
