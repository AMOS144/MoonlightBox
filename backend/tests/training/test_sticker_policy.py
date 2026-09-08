from datetime import UTC, datetime, timedelta

import pytest


def _usage(
    asset_id: str | None,
    minute: int,
    *,
    project_id: str = "project-1",
    target_id: str = "target-1",
    kind: str = "sticker",
    text: str = "",
    linked: bool = True,
    is_target: bool = True,
) -> object:
    from moonlightbox.training.sticker_policy import StickerHistoryItem

    return StickerHistoryItem(
        message_id=f"m-{project_id}-{target_id}-{minute}-{asset_id}",
        project_id=project_id,
        target_id=target_id,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minute),
        kind=kind,
        text=text,
        asset_id=asset_id,
        asset_project_id=project_id if linked else "other-project",
        asset_kind="sticker" if linked else "image",
        is_target=is_target,
    )


def test_policy_isolates_project_target_cutoff_branch_time_and_linked_assets() -> None:
    from moonlightbox.training.sticker_policy import build_sticker_policy

    items = [
        _usage(None, 0, kind="text", text="抱抱"),
        _usage("valid", 1),
        _usage("other-project", 2, project_id="project-2"),
        _usage("other-target", 3, target_id="target-2"),
        _usage("unlinked", 4, linked=False),
        _usage("future-cutoff", 20),
        _usage("future-branch", 12),
    ]

    policy = build_sticker_policy(
        items,
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
        branch_time=datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
    )

    assert [asset.asset_id for asset in policy.assets] == ["valid"]
    assert policy.boundary["effective_cutoff"] == "2026-01-01T00:10:00+00:00"
    assert policy.asset_summary == {
        "linked_sticker_count": 1,
        "sticker_event_count": 1,
    }


def test_held_out_sticker_cases_use_real_media_labels_not_text_only_dataset_rows() -> None:
    from moonlightbox.training.sticker_policy import build_sticker_evaluation_cases

    items = [
        _usage(None, 0, kind="text", text="抱抱", is_target=False),
        _usage("hug", 1),
        _usage(None, 2, kind="text", text="我到了", is_target=False),
        _usage(None, 3, kind="text", text="好", is_target=True),
        _usage("outside", 20),
        _usage("wrong-project", 4, project_id="project-2"),
    ]

    cases = build_sticker_evaluation_cases(
        items,
        project_id="project-1",
        target_id="target-1",
        start=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        end=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
        split="valid",
    )

    assert [case.actual_asset_id for case in cases] == ["hug", None]
    assert cases[0].context.current_text == "抱抱"
    assert cases[1].context.current_text == "我到了"
    assert all(case.split == "valid" for case in cases)


def test_explicit_sticker_request_can_trigger_without_lowering_global_threshold() -> None:
    from moonlightbox.training.sticker_policy import (
        StickerContext,
        build_sticker_policy,
        should_send_sticker,
    )

    policy = build_sticker_policy(
        [
            _usage(None, 0, kind="text", text="抱抱", is_target=False),
            _usage("hug", 1),
        ],
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
        modality_threshold=1.1,
    )

    assert should_send_sticker(
        policy,
        StickerContext(current_text="发个表情包", recent=()),
    ) is True
    assert should_send_sticker(
        policy,
        StickerContext(current_text="讨论数据库迁移", recent=()),
    ) is False


def test_asset_ranking_does_not_decide_whether_to_send_sticker() -> None:
    from moonlightbox.training.sticker_policy import (
        StickerContext,
        build_sticker_policy,
        rank_stickers,
    )

    items = [
        _usage(None, 0, kind="text", text="抱抱"),
        _usage("hug", 1),
        _usage(None, 2, kind="text", text="抱抱"),
        _usage("hug", 3),
        _usage(None, 4, kind="text", text="早上好"),
        _usage("popular", 5),
        _usage(None, 6, kind="text", text="早上好"),
        _usage("popular", 7),
        _usage(None, 8, kind="text", text="早上好"),
        _usage("popular", 9),
    ]
    policy = build_sticker_policy(
        items,
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
        modality_threshold=1.1,
    )
    unrelated = StickerContext(current_text="讨论数据库迁移", recent=())
    relevant = StickerContext(current_text="抱抱", recent=())

    assert rank_stickers(policy, unrelated, top_k=2) == ()
    assert [item.asset_id for item in rank_stickers(policy, relevant, top_k=2)] == [
        "hug"
    ]


def test_ranking_is_deterministic_top_k_explainable_and_penalizes_repetition() -> None:
    from moonlightbox.training.sticker_policy import (
        RecentStickerBubble,
        StickerContext,
        build_sticker_policy,
        rank_stickers,
    )

    items = [
        _usage(None, 0, kind="text", text="晚安"),
        _usage("a", 1),
        _usage(None, 2, kind="text", text="晚安"),
        _usage("a", 3),
        _usage(None, 4, kind="text", text="晚安"),
        _usage("b", 5),
        _usage(None, 6, kind="text", text="晚安"),
        _usage("c", 7),
    ]
    policy = build_sticker_policy(
        items,
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
    )
    context = StickerContext(
        current_text="晚安",
        recent=(RecentStickerBubble(kind="sticker", asset_id="a"),),
    )

    first = rank_stickers(policy, context, top_k=2)
    second = rank_stickers(policy, context, top_k=2)

    assert first == second
    assert [item.asset_id for item in first] == ["c", "b"]
    assert all(item.reasons for item in first)
    assert first[0].score >= first[1].score


def test_unknown_sticker_id_is_rejected_against_current_top_k() -> None:
    from moonlightbox.branches.replies import ReplyStructureError, parse_reply_turn

    with pytest.raises(ReplyStructureError, match="Top-K"):
        parse_reply_turn(
            '{"bubbles":[{"type":"sticker","asset_id":"unknown","delay_ms":0}]}',
            allowed_sticker_ids=("known",),
        )


def test_sticker_metrics_cover_modality_ranking_failures_repetition_and_coverage() -> None:
    from moonlightbox.training.sticker_policy import evaluate_sticker_predictions

    metrics = evaluate_sticker_predictions(
        predictions=(("a", "b"), (), ("a",), ("a",)),
        actual=("b", None, None, "a"),
        known_asset_ids=("a", "b", "c"),
    )

    assert metrics["modality_precision"] == pytest.approx(2 / 3)
    assert metrics["modality_recall"] == 1.0
    assert metrics["modality_f1"] == pytest.approx(0.8)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["ineffective_rate"] == pytest.approx(1 / 3)
    assert metrics["consecutive_repeat_rate"] == pytest.approx(1 / 3)
    assert metrics["coverage"] == pytest.approx(2 / 3)


def test_legacy_or_missing_policy_metadata_disables_stickers() -> None:
    from moonlightbox.training.sticker_policy import StickerPolicy

    assert StickerPolicy.from_metadata({}).enabled is False
    assert StickerPolicy.from_metadata({"version": "legacy-v1"}).enabled is False


def test_dataset_builder_leaves_asset_prediction_to_sticker_policy() -> None:
    from moonlightbox.imports.types import ImportedMessage, MessageKind
    from moonlightbox.training.dataset_builder import DatasetBuilder

    started = datetime(2026, 1, 1, tzinfo=UTC)
    messages = [
        ImportedMessage("m1", started, "我", MessageKind.TEXT, "抱抱", {}),
        ImportedMessage(
            "m2",
            started + timedelta(minutes=1),
            "她",
            MessageKind.STICKER,
            "[动画表情]",
            {"media_asset_id": "hug"},
        ),
        ImportedMessage(
            "m3",
            started + timedelta(minutes=2),
            "她",
            MessageKind.TEXT,
            "好啦",
            {},
        ),
    ]

    examples = DatasetBuilder().build(
        messages,
        target_sender="她",
        cutoff=started + timedelta(days=1),
    )

    assert len(examples) == 1
    assert examples[0].messages[-1].content == (
        "<bubble>好啦</bubble><delay>60000</delay>"
    )
    assert all(
        "<sticker>hug</sticker>" not in example.messages[-1].content
        for example in examples
    )


def test_context_only_v2_uses_character_ngrams_and_historical_negative_turns() -> None:
    from moonlightbox.training.sticker_policy import (
        STICKER_POLICY_VERSION,
        StickerContext,
        build_sticker_policy,
        rank_stickers,
    )

    items = [
        _usage(None, 0, kind="text", text="今天真的好累", is_target=False),
        _usage("comfort", 1),
        _usage(None, 2, kind="text", text="数据库迁移完成", is_target=False),
        _usage(None, 3, kind="text", text="知道了"),
        _usage(None, 4, kind="text", text="今天特别累呀", is_target=False),
        _usage("comfort", 5),
    ]
    policy = build_sticker_policy(
        items,
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert STICKER_POLICY_VERSION == "context-only-sticker-v2"
    assert policy.positive_context_count == 2
    assert policy.negative_context_count >= 1
    related = StickerContext(current_text="我今天有点累了", recent=())
    unrelated = StickerContext(current_text="数据库索引建好了", recent=())
    assert [item.asset_id for item in rank_stickers(policy, related)] == ["comfort"]
    assert rank_stickers(policy, unrelated) == ()


def test_v2_uses_history_idf_knn_and_asset_context_top_n() -> None:
    from moonlightbox.training.sticker_policy import (
        StickerContext,
        build_sticker_policy,
        rank_stickers,
    )

    items = [
        _usage(None, 0, kind="text", text="晚安宝宝", is_target=False),
        _usage("sleep", 1),
        _usage(None, 2, kind="text", text="晚安亲亲", is_target=False),
        _usage("sleep", 3),
        _usage(None, 4, kind="text", text="数据库晚安迁移", is_target=False),
        _usage("noise", 5),
        _usage(None, 6, kind="text", text="数据库索引", is_target=False),
        _usage(None, 7, kind="text", text="收到"),
    ]

    policy = build_sticker_policy(
        items,
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
    )
    ranked = rank_stickers(policy, StickerContext(current_text="晚安宝贝", recent=()))

    assert policy.parameters["similarity"] == "char-2-3gram-tfidf-cosine"
    assert policy.parameters["modality_k"] == 3
    assert policy.parameters["asset_context_top_n"] == 3
    assert all(asset.contexts for asset in policy.assets)
    assert ranked[0].asset_id == "sleep"
    assert "TF-IDF" in ranked[0].reasons[0]


def test_offline_tuning_accepts_valid_only_and_reports_seen_metrics() -> None:
    from moonlightbox.training.sticker_policy import (
        StickerContext,
        StickerEvaluationCase,
        build_sticker_policy,
        tune_sticker_policy_on_valid,
    )

    items = [
        _usage(None, 0, kind="text", text="抱抱我", is_target=False),
        _usage("hug", 1),
        _usage(None, 2, kind="text", text="普通文字", is_target=False),
        _usage(None, 3, kind="text", text="好的"),
    ]
    policy = build_sticker_policy(
        items,
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
    )
    valid_cases = (
        StickerEvaluationCase(
            split="valid",
            context=StickerContext(current_text="想抱抱", recent=()),
            actual_asset_id="hug",
        ),
        StickerEvaluationCase(
            split="valid",
            context=StickerContext(current_text="数据库", recent=()),
            actual_asset_id=None,
        ),
    )

    tuned, metrics = tune_sticker_policy_on_valid(
        policy,
        valid_cases,
        parameter_grid=(
            {"modality_threshold": 0.4, "minimum_similarity": 0.05},
            {"modality_threshold": 0.9, "minimum_similarity": 0.8},
        ),
    )

    assert tuned.parameters["tuned_on_split"] == "valid"
    assert tuned.parameters["threshold_source"] == (
        "valid-only-confidence-lower-bound-v1"
    )
    assert tuned.parameters["minimum_modality_f1"] <= metrics["modality_f1"]
    assert metrics["seen_positive_recall_at_5"] == 1.0
    assert metrics["seen_positive_mrr"] == 1.0
    assert metrics["modality_f1"] > 0.0
    assert metrics["coverage"] > 0.0
    with pytest.raises(ValueError, match="valid"):
        tune_sticker_policy_on_valid(
            policy,
            (
                StickerEvaluationCase(
                    split="test",
                    context=StickerContext(current_text="抱抱", recent=()),
                    actual_asset_id="hug",
                ),
            ),
            parameter_grid=({"modality_threshold": 0.5},),
        )

    from moonlightbox.training.sticker_policy import evaluate_sticker_policy

    test_cases = tuple(
        StickerEvaluationCase(
            split="test",
            context=case.context,
            actual_asset_id=case.actual_asset_id,
        )
        for case in valid_cases
    )
    frozen_metrics = evaluate_sticker_policy(
        tuned,
        test_cases,
        expected_split="test",
    )
    assert frozen_metrics["modality_f1"] >= 0.0
    with pytest.raises(ValueError, match="test"):
        evaluate_sticker_policy(tuned, valid_cases, expected_split="test")


def test_v2_semantic_embedder_recovers_related_context_without_char_overlap() -> None:
    from moonlightbox.training.sticker_policy import (
        StickerContext,
        build_sticker_policy,
        rank_stickers,
    )

    class MeaningEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [
                [1.0, 0.0] if any(word in text for word in ("累坏", "疲惫")) else [0.0, 1.0]
                for text in texts
            ]

    items = [
        _usage(None, 0, kind="text", text="我已经累坏啦", is_target=False),
        _usage("comfort", 1),
    ]
    policy = build_sticker_policy(
        items,
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
        embedder=MeaningEmbedder(),
    )

    ranked = rank_stickers(
        policy,
        StickerContext(current_text="今天疲惫不堪", recent=()),
    )

    assert policy.parameters["semantic_status"] == "available"
    assert policy.assets[0].semantic_contexts[0].text_hash
    assert ranked[0].asset_id == "comfort"
    assert "语义余弦" in ranked[0].reasons[0]


def test_semantic_policy_embeds_only_retained_asset_contexts() -> None:
    from moonlightbox.training.sticker_policy import build_sticker_policy

    class RecordingEmbedder:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.batches.append(texts)
            return [[1.0, 0.0] for _text in texts]

    items = []
    minute = 0
    for asset_index in range(10):
        for context_index in range(12):
            items.extend(
                (
                    _usage(
                        None,
                        minute,
                        kind="text",
                        text=f"资产 {asset_index} 上下文 {context_index}",
                        is_target=False,
                    ),
                    _usage(f"asset-{asset_index}", minute + 1),
                )
            )
            minute += 2
    items.extend(
        _usage(
            None,
            100 + index,
            kind="text",
            text=f"目标文字 {index}",
        )
        for index in range(100)
    )
    embedder = RecordingEmbedder()

    policy = build_sticker_policy(
        items,
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
        embedder=embedder,
    )

    assert len(embedder.batches) == 1
    assert len(embedder.batches[0]) == 64
    assert all(len(asset.semantic_contexts) <= 8 for asset in policy.assets)


def test_v2_embedding_failure_fails_closed_to_text_features_and_reports_status() -> None:
    from moonlightbox.training.sticker_policy import build_sticker_policy

    class BrokenEmbedder:
        def embed(self, _texts: list[str]) -> list[list[float]]:
            raise RuntimeError("本地模型不可用")

    policy = build_sticker_policy(
        [
            _usage(None, 0, kind="text", text="抱抱", is_target=False),
            _usage("hug", 1),
        ],
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
        embedder=BrokenEmbedder(),
    )

    assert policy.enabled is True
    assert policy.parameters["semantic_status"] == "unavailable"
    assert policy.parameters["semantic_failure"] == "RuntimeError"


def test_recent_asset_sequence_recurrence_changes_asset_ranking() -> None:
    from moonlightbox.training.sticker_policy import (
        RecentStickerBubble,
        StickerContext,
        build_sticker_policy,
        rank_stickers,
    )

    items = [
        _usage("lead", 0),
        _usage("follow", 1),
        _usage("other", 2),
        _usage("noise", 3),
    ]
    policy = build_sticker_policy(
        items,
        project_id="project-1",
        target_id="target-1",
        cutoff=datetime(2026, 1, 2, tzinfo=UTC),
        branch_time=datetime(2026, 1, 2, tzinfo=UTC),
        minimum_context_hits=0,
    )
    context = StickerContext(
        current_text="",
        recent=(RecentStickerBubble(kind="sticker", asset_id="lead", sender="target"),),
    )

    follow = next(asset for asset in policy.assets if asset.asset_id == "follow")
    assert follow.feature_counts["recent-assets:lead"] == 1
    assert rank_stickers(policy, context)[0].asset_id == "follow"
