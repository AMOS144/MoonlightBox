import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(case_id: str, target: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "训练系统提示"},
            {"role": "user", "content": f"问题-{case_id}"},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "source_ids": [case_id],
            "target_at": "2026-01-01T00:00:00",
            "kind": "chat",
            "event_id": None,
        },
    }


def test_reply_protocol_resolves_compact_only_for_high_fidelity_version() -> None:
    from moonlightbox.training.acceptance_retry import resolve_reply_protocol

    assert (
        resolve_reply_protocol({"reply_protocol_version": "high-fidelity-compact-bubble-v2"})
        == "compact"
    )
    assert resolve_reply_protocol({"reply_protocol_version": "typed-bubble-v1"}) == ("legacy_json")
    assert resolve_reply_protocol({}) == "legacy_json"


def test_held_out_corpus_uses_test_only_and_reports_deterministic_hash(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.acceptance_retry import load_acceptance_corpus

    _write_rows(
        tmp_path / "train.jsonl",
        [_row("train-1", "<bubble>训练回复</bubble><delay>0</delay>")],
    )
    _write_rows(
        tmp_path / "valid.jsonl",
        [_row("valid-1", "<bubble>验证回复</bubble><delay>0</delay>")],
    )
    test_rows = [
        _row(
            f"test-{index}",
            f"<bubble>真人回复{index}</bubble><delay>0</delay>",
        )
        for index in range(8)
    ]
    _write_rows(tmp_path / "test.jsonl", test_rows)

    first = load_acceptance_corpus(tmp_path, sample_count=5)
    second = load_acceptance_corpus(tmp_path, sample_count=5)

    assert first.sample_hash == second.sample_hash
    assert len(first.held_out_cases) == 5
    assert all(case.split == "test" for case in first.held_out_cases)
    assert not any("训练回复" in case.human_target for case in first.held_out_cases)
    assert {sample.split for sample in first.speaker_fit_samples} == {
        "train",
        "valid",
    }


def test_corpus_rejects_sticker_policy_that_crosses_train_valid_boundary(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.acceptance_retry import load_acceptance_corpus
    from moonlightbox.training.sticker_policy import StickerPolicy

    _write_rows(tmp_path / "train.jsonl", [_row("train", "<bubble>训练</bubble><delay>0</delay>")])
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", "<bubble>验证</bubble><delay>0</delay>")])
    _write_rows(
        tmp_path / "test.jsonl",
        [_row("test", "<bubble>测试</bubble><delay>0</delay>")],
    )
    leaked = StickerPolicy(
        enabled=True,
        boundary={"effective_cutoff": "2026-01-02T00:00:00+00:00"},
    )

    with pytest.raises(ValueError, match="train/valid"):
        load_acceptance_corpus(
            tmp_path,
            sample_count=1,
            minimum_sample_count=1,
            sticker_policy=leaked,
        )


def test_corpus_accepts_real_repeated_bubbles_without_runtime_deduplication(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.acceptance_retry import load_acceptance_corpus

    repeated = (
        "<bubble>哈</bubble><delay>0</delay>"
        "<bubble>哈</bubble><delay>1000</delay>"
    )
    _write_rows(tmp_path / "train.jsonl", [_row("train", repeated)])
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", repeated)])
    _write_rows(tmp_path / "test.jsonl", [_row("test", repeated)])

    corpus = load_acceptance_corpus(tmp_path, sample_count=1, minimum_sample_count=1)

    assert len(corpus.human_style_turns[0].bubbles) == 2


def test_corpus_preserves_plain_text_target_above_runtime_bubble_limit(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.acceptance_retry import load_acceptance_corpus

    target = "一\n二\n三\n四\n五\n六"
    _write_rows(tmp_path / "train.jsonl", [_row("train", target)])
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", target)])
    _write_rows(tmp_path / "test.jsonl", [_row("test", target)])

    corpus = load_acceptance_corpus(
        tmp_path,
        sample_count=1,
        minimum_sample_count=1,
    )

    assert [bubble.text for bubble in corpus.human_style_turns[0].bubbles] == [
        "一",
        "二",
        "三",
        "四",
        "五",
        "六",
    ]


def test_corpus_uses_deterministic_test_ratio_strata_and_prefers_retrievable_stickers(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.acceptance_retry import load_acceptance_corpus

    _write_rows(
        tmp_path / "train.jsonl",
        [
            _row(
                "train-known",
                "<sticker>known</sticker><delay>0</delay>",
            )
        ],
    )
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", "<bubble>好</bubble><delay>0</delay>")])
    test_rows = [
        _row(
            f"positive-known-{index}",
            "<sticker>known</sticker><delay>0</delay>",
        )
        for index in range(10)
    ]
    test_rows.extend(
        _row(
            f"positive-unseen-{index}",
            "<sticker>unseen</sticker><delay>0</delay>",
        )
        for index in range(10)
    )
    test_rows.extend(
        _row(f"negative-{index}", "<bubble>文字</bubble><delay>0</delay>")
        for index in range(45)
    )
    _write_rows(tmp_path / "test.jsonl", test_rows)

    first = load_acceptance_corpus(tmp_path, sample_count=20)
    second = load_acceptance_corpus(tmp_path, sample_count=20)

    assert first.sample_hash == second.sample_hash
    assert len(first.held_out_cases) == 20
    assert sum(item is not None for item in first.actual_sticker_ids) == 6
    assert set(item for item in first.actual_sticker_ids if item is not None) == {"known"}
    assert first.sticker_coverage["ground_truth_event_count"] == 20
    assert first.sticker_coverage["historical_seen_event_count"] == 10
    assert first.sticker_coverage["unseen_ground_truth_event_count"] == 10


def test_corpus_fails_closed_when_requested_strata_are_too_small(tmp_path: Path) -> None:
    from moonlightbox.training.acceptance_retry import load_acceptance_corpus

    _write_rows(tmp_path / "train.jsonl", [_row("train", "<bubble>训练</bubble><delay>0</delay>")])
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", "<bubble>验证</bubble><delay>0</delay>")])
    _write_rows(
        tmp_path / "test.jsonl",
        [_row(f"test-{index}", "<bubble>测试</bubble><delay>0</delay>") for index in range(19)],
    )

    with pytest.raises(ValueError, match="至少需要 20 条"):
        load_acceptance_corpus(tmp_path, sample_count=20)


def test_compact_held_out_prompt_lists_only_case_allowed_ids_without_placeholder(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
    from moonlightbox.training.acceptance_retry import (
        evaluate_held_out_model,
        load_acceptance_corpus,
    )
    from moonlightbox.training.sticker_policy import (
        StickerAssetProfile,
        StickerPolicy,
    )

    class CapturingGenerator:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            self.prompts.append(system_prompt)
            return GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="好", delay_ms=0),),
                raw_output="<bubble>好</bubble><delay>0</delay>",
            )

    _write_rows(
        tmp_path / "train.jsonl",
        [_row("train", "<sticker>allowed-one</sticker><delay>0</delay>")],
    )
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", "<bubble>验证</bubble><delay>0</delay>")])
    _write_rows(
        tmp_path / "test.jsonl",
        [
            _row("test-sticker", "<sticker>globally-known</sticker><delay>0</delay>"),
            *[
                _row(f"test-{index}", "<bubble>真人</bubble><delay>0</delay>")
                for index in range(19)
            ],
        ],
    )
    corpus = load_acceptance_corpus(tmp_path, sample_count=20)
    policy = StickerPolicy(
        enabled=True,
        project_id="project",
        target_id="target",
        boundary={"effective_cutoff": datetime(2026, 1, 1, tzinfo=UTC).isoformat()},
        parameters={
            "top_k": 5,
            "minimum_context_hits": 1,
            "modality_threshold": 0.5,
            "repetition_penalty": 5.0,
        },
        assets=(
            StickerAssetProfile(
                asset_id="allowed-one",
                usage_count=1,
                latest_used_at=datetime(2026, 1, 1, tzinfo=UTC).isoformat(),
                feature_counts={"text:问题test0": 1},
            ),
        ),
    )
    generator = CapturingGenerator()

    evaluate_held_out_model(
        corpus,
        generator,
        model_id="candidate",
        base_model="base",
        adapter_path="adapter",
        reply_protocol="compact",
        sticker_policy=policy,
        allowed_sticker_ids_by_case={
            case.case_id: (("allowed-one",) if index == 0 else ())
            for index, case in enumerate(corpus.held_out_cases)
        },
    )

    assert "资产ID" not in "".join(generator.prompts)
    assert "allowed-one" in generator.prompts[0]
    assert all("本轮禁止输出 sticker" in prompt for prompt in generator.prompts[1:])


def test_invalid_sticker_is_checked_against_each_case_top_k(tmp_path: Path) -> None:
    from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
    from moonlightbox.training.acceptance_retry import (
        evaluate_held_out_model,
        load_acceptance_corpus,
    )

    class InvalidGenerator:
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            return GeneratedReplyTurn(
                bubbles=(
                    GeneratedBubble(
                        content=None,
                        delay_ms=0,
                        type="sticker",
                        asset_id="globally-known",
                    ),
                ),
                raw_output="<sticker>globally-known</sticker><delay>0</delay>",
            )

    _write_rows(
        tmp_path / "train.jsonl",
        [_row("train", "<sticker>globally-known</sticker><delay>0</delay>")],
    )
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", "<bubble>验证</bubble><delay>0</delay>")])
    _write_rows(
        tmp_path / "test.jsonl",
        [
            _row("test-sticker", "<sticker>globally-known</sticker><delay>0</delay>"),
            *[
                _row(f"test-{index}", "<bubble>真人</bubble><delay>0</delay>")
                for index in range(19)
            ],
        ],
    )
    corpus = load_acceptance_corpus(tmp_path, sample_count=20)

    report = evaluate_held_out_model(
        corpus,
        InvalidGenerator(),
        model_id="candidate",
        base_model="base",
        adapter_path="adapter",
        reply_protocol="compact",
        allowed_sticker_ids_by_case={case.case_id: () for case in corpus.held_out_cases},
    )

    assert report.parse_failures == 20
    assert report.sticker["invalid_asset_rate"] == 0.0
    assert report.sticker["attempted_invalid_rate"] == 1.0
    assert report.sticker["attempted_invalid_asset_count"] == 20
    assert report.passed is False


def test_invalid_sticker_bubble_is_deleted_without_hiding_attempt(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
    from moonlightbox.training.acceptance_retry import (
        evaluate_held_out_model,
        load_acceptance_corpus,
    )

    class MixedGenerator:
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            return GeneratedReplyTurn(
                bubbles=(
                    GeneratedBubble(content="保留文字", delay_ms=0),
                    GeneratedBubble(
                        content=None,
                        delay_ms=0,
                        type="sticker",
                        asset_id="future-id",
                    ),
                ),
                raw_output=(
                    "<bubble>保留文字</bubble>"
                    "<sticker>future-id</sticker>"
                ),
            )

    _write_rows(
        tmp_path / "train.jsonl",
        [_row("train", "<sticker>allowed</sticker><delay>0</delay>")],
    )
    _write_rows(
        tmp_path / "valid.jsonl",
        [_row("valid", "<bubble>验证</bubble><delay>0</delay>")],
    )
    _write_rows(
        tmp_path / "test.jsonl",
        [_row("test", "<sticker>allowed</sticker><delay>0</delay>")],
    )
    corpus = load_acceptance_corpus(
        tmp_path,
        sample_count=1,
        minimum_sample_count=1,
    )
    case = corpus.held_out_cases[0]

    report = evaluate_held_out_model(
        corpus,
        MixedGenerator(),
        model_id="candidate",
        base_model="base",
        adapter_path="adapter",
        reply_protocol="compact",
        allowed_sticker_ids_by_case={case.case_id: ("allowed",)},
    )

    assert report.parse_failures == 0
    assert report.sample_audit[0]["attempted_invalid_sticker_ids"] == ["future-id"]
    assert report.sticker["invalid_asset_rate"] == 0.0
    assert report.sticker["attempted_invalid_asset_count"] == 1


def test_baseline_partial_parse_failure_still_reports_observed_style_for_comparison(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
    from moonlightbox.training.acceptance_retry import (
        evaluate_held_out_model,
        load_acceptance_corpus,
    )

    class OneFailureGenerator:
        def __init__(self) -> None:
            self.index = 0

        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            self.index += 1
            if self.index == 1:
                raise RuntimeError("格式失败")
            return GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="好", delay_ms=0),),
                raw_output="<bubble>好</bubble><delay>0</delay>",
            )

    _write_rows(tmp_path / "train.jsonl", [_row("train", "<bubble>训练</bubble><delay>0</delay>")])
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", "<bubble>验证</bubble><delay>0</delay>")])
    _write_rows(
        tmp_path / "test.jsonl",
        [_row(f"test-{index}", "<bubble>真人</bubble><delay>0</delay>") for index in range(20)],
    )
    corpus = load_acceptance_corpus(tmp_path, sample_count=20)

    report = evaluate_held_out_model(
        corpus,
        OneFailureGenerator(),
        model_id="base",
        base_model="base",
        adapter_path="",
        reply_protocol="compact",
    )

    assert report.parse_failures == 1
    assert report.style["overall_distance"] >= 0.0
    assert report.speaker["sample_count"] == 19
    assert report.passed is False


def test_held_out_failure_audit_keeps_hash_excerpt_and_reason(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.acceptance_retry import (
        evaluate_held_out_model,
        load_acceptance_corpus,
    )
    from moonlightbox.training.model_acceptance import ModelOutputStructureError

    class BrokenGenerator:
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> object:
            raise ModelOutputStructureError(
                ("<bubble>半截", "x" * 240),
                reasons=("半截标签", "未知标签"),
                attempted_invalid_sticker_ids=(("future-id",), ()),
            )

    _write_rows(
        tmp_path / "train.jsonl",
        [_row("train", "<bubble>训练</bubble><delay>0</delay>")],
    )
    _write_rows(
        tmp_path / "valid.jsonl",
        [_row("valid", "<bubble>验证</bubble><delay>0</delay>")],
    )
    _write_rows(
        tmp_path / "test.jsonl",
        [_row("test", "<bubble>真人</bubble><delay>0</delay>")],
    )
    corpus = load_acceptance_corpus(
        tmp_path,
        sample_count=1,
        minimum_sample_count=1,
    )

    report = evaluate_held_out_model(
        corpus,
        BrokenGenerator(),  # type: ignore[arg-type]
        model_id="candidate",
        base_model="base",
        adapter_path="adapter",
        reply_protocol="compact",
    )

    attempts = report.sample_audit[0]["attempts"]
    assert report.sample_audit[0]["reason"] == "候选模型输出结构无效"
    assert attempts[0]["raw_output_hash"]
    assert attempts[0]["raw_excerpt"] == "<bubble>半截"
    assert attempts[0]["reason"] == "半截标签"
    assert attempts[0]["attempted_invalid_sticker_ids"] == ("future-id",)
    assert len(attempts[1]["raw_excerpt"]) == 200


def test_held_out_reports_raw_compliance_separately_from_safe_normalization(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
    from moonlightbox.training.acceptance_retry import (
        evaluate_held_out_model,
        load_acceptance_corpus,
    )

    class NormalizedGenerator:
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            return GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="自然回复", delay_ms=0),),
                raw_output="自然回复",
                normalization="plain-text-lines-v1",
            )

    _write_rows(tmp_path / "train.jsonl", [_row("train", "<bubble>训练</bubble><delay>0</delay>")])
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", "<bubble>验证</bubble><delay>0</delay>")])
    _write_rows(tmp_path / "test.jsonl", [_row("test", "<bubble>真人</bubble><delay>0</delay>")])
    corpus = load_acceptance_corpus(
        tmp_path,
        sample_count=1,
        minimum_sample_count=1,
    )

    report = evaluate_held_out_model(
        corpus,
        NormalizedGenerator(),
        model_id="candidate",
        base_model="base",
        adapter_path="adapter",
        reply_protocol="compact",
    )

    assert report.parse_failures == 0
    assert report.raw_format_compliance == 0.0
    assert report.normalization_rate == 1.0


def test_held_out_keeps_text_only_when_lora_does_not_choose_sticker(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
    from moonlightbox.training.acceptance_retry import (
        evaluate_held_out_model,
        load_acceptance_corpus,
    )

    class TextOnlyGenerator:
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            return GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="抱抱你", delay_ms=0),),
                raw_output="<bubble>抱抱你</bubble><delay>0</delay>",
            )

    _write_rows(
        tmp_path / "train.jsonl",
        [_row("train", "<sticker>hug</sticker><delay>0</delay>")],
    )
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", "<bubble>验证</bubble><delay>0</delay>")])
    _write_rows(
        tmp_path / "test.jsonl",
        [_row("test", "<sticker>hug</sticker><delay>0</delay>")],
    )
    corpus = load_acceptance_corpus(
        tmp_path,
        sample_count=1,
        minimum_sample_count=1,
    )
    case = corpus.held_out_cases[0]

    report = evaluate_held_out_model(
        corpus,
        TextOnlyGenerator(),
        model_id="candidate",
        base_model="base",
        adapter_path="adapter",
        reply_protocol="compact",
        allowed_sticker_ids_by_case={case.case_id: ("hug",)},
    )

    assert report.sample_audit[0]["policy_appended"] is False
    assert report.sticker["metrics"]["modality_recall"] == 0.0
    assert report.sticker["invalid_asset_rate"] == 0.0


def test_held_out_does_not_append_sticker_when_policy_returns_empty(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
    from moonlightbox.training.acceptance_retry import (
        evaluate_held_out_model,
        load_acceptance_corpus,
    )

    class TextOnlyGenerator:
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            return GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="好", delay_ms=0),),
                raw_output="<bubble>好</bubble><delay>0</delay>",
            )

    _write_rows(tmp_path / "train.jsonl", [_row("train", "<bubble>训练</bubble><delay>0</delay>")])
    _write_rows(tmp_path / "valid.jsonl", [_row("valid", "<bubble>验证</bubble><delay>0</delay>")])
    _write_rows(tmp_path / "test.jsonl", [_row("test", "<bubble>真人</bubble><delay>0</delay>")])
    corpus = load_acceptance_corpus(
        tmp_path,
        sample_count=1,
        minimum_sample_count=1,
    )
    case = corpus.held_out_cases[0]

    report = evaluate_held_out_model(
        corpus,
        TextOnlyGenerator(),
        model_id="candidate",
        base_model="base",
        adapter_path="adapter",
        reply_protocol="compact",
        allowed_sticker_ids_by_case={case.case_id: ()},
    )

    assert report.sample_audit[0]["policy_appended"] is False
    assert report.sample_audit[0]["prompt_excerpt"] == "问题-test"
    assert report.sample_audit[0]["human_excerpt"] == "真人"
    assert report.sample_audit[0]["generated_excerpt"] == "好"
    assert report.sticker["invalid_asset_rate"] == 0.0
    assert report.sticker["attempted_invalid_rate"] == 0.0


def test_memorization_text_excludes_policy_controlled_asset_ids() -> None:
    from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
    from moonlightbox.training.acceptance_retry import _reply_memorization_text

    reply = GeneratedReplyTurn(
        bubbles=(
            GeneratedBubble(content="这是模型原样文字", delay_ms=0),
            GeneratedBubble(
                content=None,
                delay_ms=0,
                type="sticker",
                asset_id="historical-asset-id",
            ),
        ),
        raw_output="这是模型原样文字",
        policy_appended=True,
    )

    assert _reply_memorization_text(reply) == "这是模型原样文字"


def test_sticker_repetition_uses_each_case_history_not_batch_adjacency() -> None:
    from moonlightbox.training.acceptance_retry import _sticker_metrics

    independent = _sticker_metrics(
        (("same",), ("same",)),
        (None, None),
        (),
        (),
        previous_sticker_ids=(None, None),
    )
    repeated_from_history = _sticker_metrics(
        (("same",),),
        (None,),
        (),
        (),
        previous_sticker_ids=("same",),
    )

    assert independent["consecutive_repeat_rate"] == 0.0
    assert repeated_from_history["consecutive_repeat_rate"] == 1.0
