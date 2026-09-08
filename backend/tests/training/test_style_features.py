import json
from datetime import datetime, timedelta
from pathlib import Path

from moonlightbox.imports.types import ImportedMessage, MessageKind


def test_extracts_chinese_lexical_and_structural_features() -> None:
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    turns = [
        StyleTurn(
            bubbles=(
                StyleBubble(text="笨笨，你在干嘛呀呀"),
                StyleBubble(text="哈哈哈！！", delay_ms=1_200),
            )
        ),
        StyleTurn(bubbles=(StyleBubble(text="笨笨，快来"),)),
    ]

    profile = extract_style_features(turns)

    assert profile["schema_version"] == "moonlightbox.style-features.v4"
    assert {"text": "笨笨", "count": 2} in profile["lexical"]["address_terms"]
    assert any(item["text"] == "笨笨" for item in profile["lexical"]["frequent_phrases"])
    assert any(item["pattern"] == "呀呀" for item in profile["lexical"]["repetition_patterns"])
    assert profile["structural"]["bubble_length"]["count"] == 3
    assert profile["structural"]["bubbles_per_turn"]["p90"] == 2.0
    assert {"text": "！！", "count": 1} in profile["structural"]["punctuation_combinations"]
    assert profile["structural"]["terminal_period_omission_rate"] == 1.0
    assert profile["structural"]["delays_ms"]["count"] == 3
    assert "values" not in profile["structural"]["delays_ms"]


def test_extracts_all_six_pragmatic_behaviors() -> None:
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    texts = [
        "好呀，我知道了",
        "那你呢？",
        "笨蛋哈哈哈",
        "抱抱，别难过",
        "不要，我不想",
        "对了，换个话题",
    ]

    pragmatic = extract_style_features(
        [StyleTurn(bubbles=(StyleBubble(text=text),)) for text in texts]
    )["pragmatic"]

    assert pragmatic["counts"] == {
        "回应": 1,
        "追问": 1,
        "调侃": 1,
        "安慰": 1,
        "拒绝": 1,
        "转移话题": 1,
    }
    assert pragmatic["rates"] == {name: 1 / 6 for name in pragmatic["counts"]}


def test_extracts_emoji_text_and_sticker_context_features() -> None:
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    profile = extract_style_features(
        [
            StyleTurn(
                previous_text="今天好累",
                bubbles=(
                    StyleBubble(text="想你🥺"),
                    StyleBubble(
                        kind="sticker",
                        asset_id="sticker-hug",
                        delay_ms=800,
                    ),
                ),
            ),
            StyleTurn(
                previous_text="抱一下",
                bubbles=(StyleBubble(kind="sticker", asset_id="sticker-hug"),),
            ),
        ]
    )

    assert {"emoji": "🥺", "count": 1} in profile["media"]["emoji_frequencies"]
    assert profile["media"]["emoji_text_combinations"] == [
        {"emoji": "🥺", "text": "想你", "count": 1}
    ]
    assert profile["media"]["sticker_asset_frequencies"] == [
        {"asset_id": "sticker-hug", "count": 2}
    ]
    assert profile["media"]["sticker_contexts"]["sticker-hug"] == {
        "previous_text": [
            {"text": "今天好累", "count": 1},
            {"text": "抱一下", "count": 1},
        ],
        "same_turn_text": [{"text": "想你🥺", "count": 1}],
    }


def test_emoji_assets_are_separate_from_sticker_assets() -> None:
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    profile = extract_style_features(
        [
            StyleTurn(
                previous_text="给你看看",
                bubbles=(
                    StyleBubble(text="这个"),
                    StyleBubble(kind="emoji", asset_id="emoji-smile"),
                    StyleBubble(kind="sticker", asset_id="sticker-hug"),
                ),
            )
        ]
    )

    assert profile["media"]["emoji_asset_frequencies"] == [
        {"asset_id": "emoji-smile", "count": 1}
    ]
    assert profile["media"]["emoji_asset_contexts"]["emoji-smile"] == {
        "previous_text": [{"text": "给你看看", "count": 1}],
        "same_turn_text": [{"text": "这个", "count": 1}],
    }
    assert profile["media"]["sticker_asset_frequencies"] == [
        {"asset_id": "sticker-hug", "count": 1}
    ]
    assert "emoji-smile" not in profile["media"]["sticker_contexts"]


def test_emoji_characters_are_extracted_as_unicode_grapheme_clusters() -> None:
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    profile = extract_style_features(
        [
            StyleTurn(
                bubbles=(
                    StyleBubble(text="爱你❤️"),
                    StyleBubble(text="一家人👨‍👩‍👧‍👦"),
                    StyleBubble(text="赞👍🏽"),
                    StyleBubble(text="中国🇨🇳"),
                )
            )
        ]
    )

    assert profile["media"]["emoji_frequencies"] == [
        {"emoji": "❤️", "count": 1},
        {"emoji": "🇨🇳", "count": 1},
        {"emoji": "👍🏽", "count": 1},
        {"emoji": "👨‍👩‍👧‍👦", "count": 1},
    ]
    assert profile["media"]["emoji_text_combinations"] == [
        {"emoji": "❤️", "text": "爱你", "count": 1},
        {"emoji": "🇨🇳", "text": "中国", "count": 1},
        {"emoji": "👍🏽", "text": "赞", "count": 1},
        {"emoji": "👨‍👩‍👧‍👦", "text": "一家人", "count": 1},
    ]


def test_common_bmp_emoji_are_detected_without_misclassifying_chinese() -> None:
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    profile = extract_style_features(
        [
            StyleTurn(
                bubbles=(
                    StyleBubble(text="⌚⏰◾〰️㊗️普通中文，。"),
                )
            )
        ]
    )

    assert {
        item["emoji"]
        for item in profile["media"]["emoji_frequencies"]
    } == {"⌚", "⏰", "◾", "〰️", "㊗️"}
    assert {
        item["text"]
        for item in profile["media"]["emoji_text_combinations"]
    } == {"普通中文，。"}


def test_bounded_frequency_caps_internal_cardinality_deterministically() -> None:
    from moonlightbox.training.style_features import BoundedFrequency

    first = BoundedFrequency[str](capacity=8)
    second = BoundedFrequency[str](capacity=8)
    values = [f"唯一值-{index}" for index in range(100)] + ["常见项"] * 200
    for value in values:
        first.add(value)
        second.add(value)

    assert first.snapshot() == second.snapshot()
    assert len(first) <= 8
    assert "常见项" in first
    assert first["常见项"] == 200
    assert first.audit()["internal_capacity"] == 8
    assert first.audit()["dropped_unique_count"] > 0
    assert first.audit()["approximate"] is True
    assert first.audit()["count_semantics"] == "lower-bound-v1"

    unique_only = BoundedFrequency[str](capacity=8)
    for index in range(100):
        unique_only.add(f"只出现一次-{index}")
    assert max(unique_only.values()) == 1


def test_high_cardinality_features_are_bounded_and_auditable() -> None:
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    turns = [
        StyleTurn(
            previous_text=f"上下文{index}",
            bubbles=(
                StyleBubble(text=f"独立短语{index}", delay_ms=index * 100),
                StyleBubble(kind="sticker", asset_id=f"asset-{index}"),
            ),
        )
        for index in range(600)
    ] + [
        StyleTurn(bubbles=(StyleBubble(text="真正重复"),))
        for _index in range(10)
    ]

    profile = extract_style_features(turns)

    assert "values" not in profile["structural"]["bubble_length"]
    assert "values" not in profile["structural"]["delays_ms"]
    assert profile["structural"]["delays_ms"]["histogram"]
    assert len(profile["media"]["sticker_asset_frequencies"]) == 64
    asset_audit = profile["media"]["ranking_audit"]["sticker_asset_frequencies"]
    assert asset_audit["total_count"] == 600
    assert asset_audit["tracked_unique"] <= asset_audit["internal_capacity"]
    assert asset_audit["dropped_unique_count"] > 0
    assert asset_audit["approximate"] is True
    context_audit = profile["media"]["ranking_audit"]["sticker_contexts"]
    assert context_audit["tracked_assets"] <= context_audit["asset_internal_capacity"]
    assert context_audit["dropped_asset_unique_count"] > 0
    assert context_audit["count_semantics"] == "lower-bound-v1"
    assert context_audit["maximum_asset_error"] > 0
    assert profile["lexical"]["phrase_semantics"] == "punctuation-delimited-segments-v1"
    assert profile["lexical"]["ranking_audit"]["frequent_phrases"]["limit"] == 32
    assert profile["lexical"]["frequent_phrases"] == [
        {"text": "真正重复", "count": 10}
    ]


def test_all_p90_fields_use_nearest_rank() -> None:
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )
    from moonlightbox.training.style_profile import build_style_profile

    unified = extract_style_features(
        [
            StyleTurn(bubbles=(StyleBubble(text="一"),)),
            StyleTurn(bubbles=(StyleBubble(text="一二"),)),
        ]
    )
    legacy = build_style_profile(["一", "一二"])

    assert unified["structural"]["bubble_length"]["p90"] == 2.0
    assert legacy["p90_length"] == 2.0


def test_legacy_ending_ranking_is_bounded_auditable_and_stable() -> None:
    from moonlightbox.training.style_profile import build_style_profile

    texts = [f"回复{index:02d}" for index in range(20)]
    first = build_style_profile(texts)
    second = build_style_profile(list(reversed(texts)))

    assert first["common_endings"] == second["common_endings"]
    assert len(first["common_endings"]) == 8
    assert first["compatibility_ranking_audit"]["common_endings"] == {
        "total_count": 20,
        "total_unique": 20,
        "returned_count": 8,
        "limit": 8,
        "truncated": True,
    }


def test_unobserved_ai_register_is_rejected_but_observed_language_is_allowed() -> None:
    from moonlightbox.training.style_profile import build_style_profile, style_violations

    profile = build_style_profile(["不知道这是啥", "并没有", "照顾好自己"])

    assert "我理解你的感受" in profile["forbidden_unobserved_ai_register"]
    assert "我不能参与这样的对话" in profile["forbidden_unobserved_ai_register"]
    assert "照顾好自己" not in profile["forbidden_unobserved_ai_register"]
    assert style_violations("我理解你的感受", profile) == ["我理解你的感受"]
    assert style_violations("我不能参与这样的对话", profile) == [
        "我不能参与这样的对话"
    ]
    assert style_violations("照顾好自己", profile) == []


def test_empty_short_and_sticker_only_inputs_are_deterministic() -> None:
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    empty = extract_style_features([])
    sticker_only = [
        StyleTurn(bubbles=(StyleBubble(kind="sticker", asset_id="asset-1"),)),
        StyleTurn(bubbles=(StyleBubble(text=""),)),
    ]
    first = extract_style_features(sticker_only)
    second = extract_style_features(sticker_only)

    assert empty["sample_count"] == 0
    assert empty["structural"]["terminal_period_omission_rate"] == 0.0
    assert first == second
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second,
        ensure_ascii=False,
        sort_keys=True,
    )


def _message(
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


def test_manifest_persists_versioned_style_profile_and_loads_legacy(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.dataset_builder import DatasetBuilder, DatasetManifest

    messages = [
        _message("u1", "我", "在吗", 0),
        _message("a1", "她", "在呀🥺", 1),
        _message(
            "a2",
            "她",
            "[动画表情]",
            1,
            kind=MessageKind.STICKER,
            raw={"media_asset_id": "asset-1"},
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
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    legacy_payload = dict(payload)
    legacy_payload.pop("style_profile")

    assert manifest.style_profile["schema_version"] == "moonlightbox.style-features.v4"
    assert payload["style_profile"] == manifest.style_profile
    assert DatasetManifest.from_dict(legacy_payload).style_profile == {}


def test_manifest_loads_real_five_field_legacy_payload() -> None:
    from moonlightbox.training.dataset_builder import DatasetManifest

    manifest = DatasetManifest.from_dict(
        {
            "dataset_hash": "legacy-hash",
            "example_count": 12,
            "cutoff": "2025-01-01T00:00:00",
            "target_sender": "她",
            "context_turns": 8,
        }
    )

    assert manifest.regular_example_count == 12
    assert manifest.augmented_example_count == 0
    assert manifest.protocol_version == "legacy-text-v1"
    assert manifest.target_message_count == 12
    assert manifest.covered_target_message_count == 12
    assert manifest.filtered_count == 0
    assert manifest.consecutive_bubble_coverage_rate == 1.0
    assert manifest.text_coverage_count == 12
    assert manifest.split_counts == {"train": 12, "valid": 0, "test": 0}
    assert manifest.style_profile == {}


def test_dataset_export_derives_profile_from_current_examples_not_previous_build(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.dataset_builder import (
        ChatTurn,
        DatasetBuilder,
        TrainingExample,
    )

    builder = DatasetBuilder()
    builder.build(
        [_message("old", "她", "很长很长的旧回复", 0)],
        "她",
        datetime(2026, 1, 2),
    )
    current = [
        TrainingExample(
            messages=[
                ChatTurn(role="user", content="在吗"),
                ChatTurn(
                    role="assistant",
                    content="<bubble>短</bubble><delay>0</delay>",
                ),
            ],
            source_ids=["u1", "a1"],
            target_at=datetime(2026, 1, 1, 12),
        )
    ]

    manifest = builder.write_mlx_dataset(
        current,
        tmp_path,
        "她",
        datetime(2026, 1, 2),
    )

    assert manifest.style_profile["average_length"] == 1.0
    assert manifest.style_profile["structural"]["bubble_length"]["max"] == 1


def test_identity_kernel_profile_is_structured_and_backward_compatible() -> None:
    from datetime import UTC

    from moonlightbox.branches.identity import (
        EvidenceBackedIdentityKernelBuilder,
        IdentityKernelProposal,
    )
    from moonlightbox.training.bubble_protocol import (
        ProtocolBubble,
        serialize_bubble_protocol,
    )
    from moonlightbox.training.dataset_builder import ChatTurn, TrainingExample

    legacy = IdentityKernelProposal.model_validate(
        {
            "persona": "她",
            "values": [],
            "stable_preferences": [],
            "relationship_boundaries": [],
            "language_patterns": [],
            "typical_reactions": [],
        }
    )
    example = TrainingExample(
        messages=[
            ChatTurn(role="user", content="在吗"),
            ChatTurn(
                role="assistant",
                content=serialize_bubble_protocol(
                    (
                        ProtocolBubble(kind="text", value="在呀🥺", delay_ms=500),
                        ProtocolBubble(kind="emoji", value="asset-1", delay_ms=0),
                    )
                ),
            ),
        ],
        source_ids=["u1", "a1"],
        target_at=datetime.now(UTC),
    )

    proposal = EvidenceBackedIdentityKernelBuilder().build(
        persona="她",
        examples=[example],
        event_contexts=[],
    )

    assert legacy.style_profile == {}
    assert proposal.style_profile["schema_version"] == "moonlightbox.style-features.v4"
    assert isinstance(proposal.style_profile["structural"], dict)
    assert proposal.style_profile["media"]["emoji_asset_frequencies"] == [
        {"asset_id": "asset-1", "count": 1}
    ]
    assert proposal.style_profile["media"]["sticker_asset_frequencies"] == []
    assert all(
        "schema_version" not in message.content
        for message in example.messages
        if message.role == "assistant"
    )
