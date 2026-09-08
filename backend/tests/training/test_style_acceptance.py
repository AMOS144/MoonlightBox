from __future__ import annotations

import json

import pytest
from moonlightbox.training.style_features import StyleBubble, StyleTurn


def _turn(text: str) -> StyleTurn:
    return StyleTurn(bubbles=(StyleBubble(text=text),))


def test_blind_ab_is_reproducible_and_hides_labels() -> None:
    from moonlightbox.training.style_acceptance import BlindABCase, run_blind_ab

    seen: list[tuple[str, str]] = []

    def judge(context: tuple[str, ...], left: str, right: str) -> str:
        assert context == ("最近忙吗",)
        assert "candidate" not in left + right
        assert "human" not in left + right
        seen.append((left, right))
        return "left"

    cases = [
        BlindABCase("a", ("最近忙吗",), "候选一", "真人一"),
        BlindABCase("b", ("最近忙吗",), "候选二", "真人二"),
    ]
    first = run_blind_ab(cases, judge, seed=17)
    first_order = list(seen)
    seen.clear()
    second = run_blind_ab(cases, judge, seed=17)

    assert seen == first_order
    assert first == second
    assert first.valid_count == 2
    assert 0 <= first.confidence_interval[0] <= first.preference_rate
    assert first.preference_rate <= first.confidence_interval[1] <= 1


def test_blind_ab_fails_closed_on_judge_failure_or_too_few_samples() -> None:
    from moonlightbox.training.style_acceptance import BlindABCase, run_blind_ab

    case = BlindABCase("a", ("上下文",), "候选", "真人")

    def broken(_context: tuple[str, ...], _left: str, _right: str) -> str:
        raise RuntimeError("评审不可用")

    report = run_blind_ab([case], broken, seed=1, minimum_valid=2)

    assert report.passed is False
    assert report.invalid_count == 1
    assert "有效评审不足" in report.failure_reasons


def test_speaker_identifier_never_fits_test_and_scores_target_style() -> None:
    from moonlightbox.training.style_acceptance import SpeakerSample, SpeakerStyleIdentifier

    identifier = SpeakerStyleIdentifier(ngram_size=2)
    identifier.fit(
        [
            SpeakerSample("train", "target", "好呀宝宝，抱抱你"),
            SpeakerSample("valid", "target", "嗯嗯，知道啦"),
            SpeakerSample("train", "other", "收到，我将按计划处理。"),
            SpeakerSample("valid", "base", "您好，请问有什么可以帮助您？"),
        ]
    )
    result = identifier.evaluate(
        [
            SpeakerSample("test", "target", "好呀，抱抱"),
            SpeakerSample("test", "other", "我将按计划处理"),
        ],
        positive_speaker="target",
    )

    assert result.accuracy == 1.0
    assert result.mean_positive_probability > 0.5
    assert 0 <= result.brier_score <= 1
    with pytest.raises(ValueError, match="test"):
        identifier.fit([SpeakerSample("test", "target", "泄漏文本")])


def test_distribution_distance_is_versioned_bounded_and_uses_unified_features() -> None:
    from moonlightbox.training.style_acceptance import compare_style_distributions

    same = compare_style_distributions([_turn("好呀")], [_turn("好呀")])
    far = compare_style_distributions(
        [_turn("这是一段非常非常长而且正式的回复。")],
        [_turn("好")],
    )

    assert same.version.startswith("moonlightbox.style-distance.")
    assert same.overall_distance == 0
    assert 0 <= far.overall_distance <= 1
    assert far.distances["length"] > same.distances["length"]
    assert "emoji" in far.distances
    assert "sticker" in far.distances


def test_memorization_uses_train_only_and_is_length_aware() -> None:
    from moonlightbox.training.style_acceptance import (
        MemorizationIndex,
        MemorizationSource,
    )

    index = MemorizationIndex(
        [
            MemorizationSource("train", "train-short", "好"),
            MemorizationSource("train", "train-long", "今天下班以后我们去公园散步吧"),
        ],
        ngram_size=3,
        threshold=0.8,
    )

    assert index.check("好").passed is True
    copied = index.check("今天下班以后我们去公园散步吧")
    assert copied.passed is False
    assert copied.nearest_source_hash == "train-long"
    with pytest.raises(ValueError, match="test"):
        MemorizationIndex([MemorizationSource("test", "forbidden", "不能进入索引")])


def test_acceptance_report_serializes_auditable_hashes_without_raw_text() -> None:
    from moonlightbox.training.style_acceptance import AcceptanceJSONReport

    report = AcceptanceJSONReport(
        dataset={"test_hash": "abc", "split_counts": {"test": 2}},
        model_ids={"candidate": "c", "base": "b", "active": "a"},
        seed=7,
        gates={"style": {"passed": False}},
        comparisons={"candidate": {"score": 0.4}},
        failure_reasons=("风格门槛失败",),
        sample_audit=({"case_id": "x", "raw_output_hash": "def", "preview": "抱抱…"},),
        passed=False,
    )
    payload = json.loads(report.to_json())

    assert payload["seed"] == 7
    assert payload["passed"] is False
    assert payload["sample_audit"][0]["raw_output_hash"] == "def"


def test_held_out_generation_accepts_only_test_and_preserves_raw_output() -> None:
    from moonlightbox.training.style_acceptance import (
        HeldOutCase,
        generate_held_out_candidates,
    )

    raw = "<bubble>原始本地输出</bubble><delay>0</delay>"
    seen_contexts: list[tuple[dict[str, str], ...]] = []

    def generate(context: tuple[dict[str, str], ...]) -> str:
        seen_contexts.append(context)
        return raw

    result = generate_held_out_candidates(
        [
            HeldOutCase(
                case_id="test-1",
                split="test",
                context=({"role": "user", "content": "真实上下文"},),
                human_target="真人回复",
                source_hash="source",
            )
        ],
        generate,
    )

    assert seen_contexts == [({"role": "user", "content": "真实上下文"},)]
    assert result[0].raw_output == raw
    assert result[0].text == "原始本地输出"
    with pytest.raises(ValueError, match="test"):
        generate_held_out_candidates(
            [
                HeldOutCase(
                    "train-1",
                    "train",
                    (),
                    "训练回复",
                    "train-source",
                )
            ],
            generate,
        )
