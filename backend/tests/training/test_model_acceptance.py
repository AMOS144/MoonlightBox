import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from moonlightbox.branches.context import ContextPacket
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
from moonlightbox.branches.reviewer import ReviewResult


class FakeGenerator:
    def generate(
        self,
        _base_model: str,
        _adapter_path: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="抱抱你", delay_ms=0),),
            raw_output="",
        )


class SeedTrackingGenerator(FakeGenerator):
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)


class FakeReviewer:
    def review(self, _packet: ContextPacket, draft: GeneratedReplyTurn) -> ReviewResult:
        return ReviewResult(verdict="approve", reasons=(), reply=draft)


class ConstantEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _text in texts]


def test_plain_text_dataset_reference_preserves_more_than_runtime_bubble_limit() -> None:
    from moonlightbox.training.model_acceptance import _parse_dataset_reply

    reply = _parse_dataset_reply("一\n二\n三\n四\n五\n六")

    assert [bubble.content for bubble in reply.bubbles] == [
        "一",
        "二",
        "三",
        "四",
        "五",
        "六",
    ]


def test_fixed_acceptance_cases_use_stable_per_case_sampling_seeds() -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    generator = SeedTrackingGenerator()
    runner = ModelAcceptanceRunner(generator, FakeReviewer(), fixture)

    runner.run(
        base_model="base",
        adapter_path="adapter-a",
        persona="目标人物",
        cutoff="2026-01-01T00:00:00+00:00",
    )
    first = list(generator.seeds)
    generator.seeds.clear()
    runner.run(
        base_model="base",
        adapter_path="adapter-b",
        persona="目标人物",
        cutoff="2026-01-01T00:00:00+00:00",
    )

    assert generator.seeds == first
    case_count = len(json.loads(fixture.read_text(encoding="utf-8"))["cases"])
    assert len(first) == case_count
    assert len(set(first)) == case_count


def test_memory_causal_acceptance_switches_decision_without_style_drift() -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    class EvidenceFollowingGenerator(FakeGenerator):
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            system_prompt: str,
            messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            context = system_prompt + "\n" + "\n".join(
                message["content"] for message in messages
            )
            marker = next(
                item
                for item in (
                    "韩国",
                    "日本",
                    "火锅",
                    "寿司",
                    "电话",
                    "视频",
                    "修复",
                    "距离",
                )
                    if item in context
                )
            return GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content=f"{marker}呀", delay_ms=0),),
                raw_output="",
            )

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    metrics = ModelAcceptanceRunner(
        EvidenceFollowingGenerator(), FakeReviewer(), fixture
    ).evaluate_memory_causality(
        base_model="base",
        adapter_path="adapter",
        persona="目标人物",
    )

    assert metrics["memory_causal_variant_accuracy"] == 1.0
    assert metrics["memory_causal_grounding_rate"] == 1.0
    assert metrics["memory_causal_switch_rate"] == 1.0
    assert metrics["memory_causal_style_consistency"] == 1.0


def test_memory_causal_acceptance_keeps_non_compiled_memory_failures_visible() -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    metrics = ModelAcceptanceRunner(
        FakeGenerator(), FakeReviewer(), fixture
    ).evaluate_memory_causality(
        base_model="base",
        adapter_path="adapter",
        persona="目标人物",
    )

    assert metrics["memory_causal_variant_accuracy"] == 0.25
    assert metrics["memory_causal_switch_rate"] == 0.25


def test_fixed_acceptance_preserves_raw_structure_failure_audit() -> None:
    from moonlightbox.training.model_acceptance import (
        ModelAcceptanceRunner,
        ModelOutputStructureError,
    )

    class BrokenGenerator(FakeGenerator):
        def generate(self, *_args: object, **_kwargs: object) -> GeneratedReplyTurn:
            raise ModelOutputStructureError(
                ("not-protocol", "still-not-protocol"),
                reasons=("缺少 bubble", "重试仍缺少 bubble"),
            )

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    report = ModelAcceptanceRunner(BrokenGenerator(), FakeReviewer(), fixture).run(
        base_model="base",
        adapter_path="adapter",
        persona="目标人物",
        cutoff="2026-01-01T00:00:00+00:00",
    )

    assert report.structure_failures == report.case_count
    assert len(report.structure_failure_audit) == report.case_count
    assert report.structure_failure_audit[0]["attempts"][0]["raw_excerpt"] == "not-protocol"


class UnsafeGenerator(FakeGenerator):
    def generate(
        self,
        _base_model: str,
        _adapter_path: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="我今天也累", delay_ms=0),),
            raw_output="",
        )


class RewritingReviewer:
    def review(self, _packet: ContextPacket, _draft: GeneratedReplyTurn) -> ReviewResult:
        safe = GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="抱抱你", delay_ms=0),),
            raw_output="",
        )
        return ReviewResult(verdict="rewrite", reasons=("删除编造事实",), reply=safe)


class UnlearnedStyleGenerator(FakeGenerator):
    def generate(
        self,
        _base_model: str,
        _adapter_path: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="好哒", delay_ms=0),),
            raw_output="",
        )


class MultiBubbleStyleGenerator(FakeGenerator):
    def generate(
        self,
        _base_model: str,
        _adapter_path: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        return GeneratedReplyTurn(
            bubbles=(
                GeneratedBubble(content="第一条", delay_ms=0),
                GeneratedBubble(content="第二条", delay_ms=800),
            ),
            raw_output="",
        )


class AlternatingPunctuationGenerator(FakeGenerator):
    def __init__(self) -> None:
        self._index = 0

    def generate(
        self,
        _base_model: str,
        _adapter_path: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        content = "第一句。" if self._index == 0 else "第二句"
        self._index += 1
        return GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content=content, delay_ms=0),),
            raw_output="",
        )


class CompactEmojiGenerator(FakeGenerator):
    def generate(
        self,
        _base_model: str,
        _adapter_path: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        from moonlightbox.branches.replies import parse_reply_turn

        return parse_reply_turn('<sticker kind="emoji">emoji-smile</sticker><delay>0</delay>')


def test_valid_style_evaluation_is_deterministic_and_never_reads_test(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    rows = []
    for index in range(24):
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "必须使用高保真紧凑气泡协议"},
                    {"role": "assistant", "content": "历史里的自然回复"},
                    {"role": "user", "content": f"问题{index}"},
                    {
                        "role": "assistant",
                        "content": (
                            "<bubble>抱抱你</bubble><delay>0</delay>"
                            + (
                                "<bubble>抱抱你</bubble><delay>100</delay>"
                                if index == 0
                                else ""
                            )
                        ),
                    },
                ],
                    "metadata": {
                        "source_ids": [f"valid-{index}"],
                        "conversation_mode": (
                            "proactive" if index < 4 else "responsive"
                        ),
                    },
            }
        )
    (tmp_path / "valid.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (tmp_path / "test.jsonl").write_text("not-json-and-must-not-be-read\n", encoding="utf-8")
    runner = ModelAcceptanceRunner(
        FakeGenerator(),
        FakeReviewer(),
        fixture,
        semantic_embedder=ConstantEmbedder(),
    )

    first = runner.evaluate_valid_style(
        data_dir=tmp_path,
        base_model="base",
        adapter_path="adapter",
        persona="她",
        sample_count=20,
        style_profile={"forbidden_unobserved_ai_register": ["抱抱"]},
    )
    second = runner.evaluate_valid_style(
        data_dir=tmp_path,
        base_model="base",
        adapter_path="adapter",
        persona="她",
        sample_count=20,
        style_profile={"forbidden_unobserved_ai_register": ["抱抱"]},
    )

    assert first == second
    assert first["valid_sample_count"] == 20
    assert first["valid_conversation_mode"] == "responsive"
    assert first["valid_structure_failures"] == 0
    assert first["paired_response_similarity"] == 1.0
    assert first["paired_similarity_method"] == "bge-small-zh-v1.5-cosine-v1"
    assert first["valid_register_violations"] == 20
    assert len(json.loads(str(first["valid_register_violation_audit"]))) == 20
    assert first["valid_sample_hash"]
    assert first["style_distance"] >= 0.0
    assert first["speaker_probability"] > 0.0
    assert first["human_oracle_speaker_probability"] > 0.0
    assert 0.0 <= first["speaker_probability_alignment"] <= 1.0
    assert first["paired_response_similarity"] > 0.999
    assert first["composite_fidelity"] > 0.0
    assert first["memory_poison_copy_rate"] == 0.0
    assert json.loads(str(first["memory_poison_copy_audit"])) == []
    assert first["low_authority_prompt_leak_rate"] == 0.0
    assert json.loads(str(first["low_authority_prompt_leak_audit"])) == []


def test_style_transfer_acceptance_measures_persona_expression_not_content_guessing(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    class DraftRewriter(FakeGenerator):
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            draft = messages[-1]["content"].removeprefix("内容草稿：\n")
            return GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content=draft + "呀", delay_ms=0),),
                raw_output=draft + "呀",
            )

    rows = [
        {
            "messages": [
                {"role": "system", "content": "改写并只输出聊天文字本身"},
                {"role": "user", "content": f"内容草稿：\n今晚不去{index}"},
                {"role": "assistant", "content": f"今晚不去{index}呀"},
            ],
            "metadata": {
                "source_ids": [f"style-{index}"],
                "conversation_mode": "responsive",
                "training_task": "style_transfer",
            },
        }
        for index in range(20)
    ]
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"

    metrics = ModelAcceptanceRunner(
        DraftRewriter(),
        FakeReviewer(),
        fixture,
        semantic_embedder=ConstantEmbedder(),
    ).evaluate_valid_style(
        data_dir=tmp_path,
        base_model="base",
        adapter_path="adapter",
        persona="她",
        sample_count=20,
        training_task="style_transfer",
    )

    assert metrics["valid_training_task"] == "style_transfer"
    assert metrics["excluded_unanswerable_current_state_count"] == 0
    assert metrics["valid_grounding_failures"] == 0
    assert metrics["paired_response_similarity"] == 1.0


def test_valid_style_never_feeds_held_out_current_state_answer_to_model(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    class CapturingGenerator(FakeGenerator):
        def __init__(self) -> None:
            self.inputs: list[str] = []

        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            system_prompt: str,
            messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            self.inputs.append(
                system_prompt + "\n" + "\n".join(item["content"] for item in messages)
            )
            return super().generate("", "", "", [])

    ordinary_rows = [
        {
            "messages": [
                {"role": "system", "content": "私人聊天"},
                {"role": "user", "content": f"普通问题{index}"},
                {"role": "assistant", "content": "普通回答"},
            ],
            "metadata": {"source_ids": [f"ordinary-{index}"]},
        }
        for index in range(20)
    ]
    current_state_row = {
        "messages": [
            {"role": "system", "content": "私人聊天"},
            {"role": "user", "content": "你现在在干嘛"},
            {"role": "assistant", "content": "正在洗澡的秘密答案"},
        ],
        "metadata": {"source_ids": ["current-state"]},
    }
    rows = [*ordinary_rows, current_state_row]
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    generator = CapturingGenerator()
    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"

    metrics = ModelAcceptanceRunner(
        generator,
        FakeReviewer(),
        fixture,
        semantic_embedder=ConstantEmbedder(),
    ).evaluate_valid_style(
        data_dir=tmp_path,
        base_model="base",
        adapter_path="adapter",
        persona="她",
        sample_count=20,
    )

    assert metrics["excluded_unanswerable_current_state_count"] == 1
    assert len(generator.inputs) == 20
    assert all("正在洗澡的秘密答案" not in item for item in generator.inputs)


def test_human_blind_cases_use_base_content_then_persona_style_without_target_leak(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    class TwoStageGenerator(FakeGenerator):
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, list[dict[str, str]]]] = []

        def generate(
            self,
            _base_model: str,
            adapter_path: str,
            system_prompt: str,
            messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            self.calls.append((adapter_path, system_prompt, messages))
            visible = system_prompt + json.dumps(messages, ensure_ascii=False)
            assert "真人保留答案" not in visible
            text = "我觉得先等等" if not adapter_path else "我觉得先等等\n入"
            return GeneratedReplyTurn(
                bubbles=tuple(
                    GeneratedBubble(content=item, delay_ms=0)
                    for item in text.splitlines()
                ),
                raw_output=text,
            )

    row = {
        "messages": [
            {"role": "system", "content": "私人聊天；只输出聊天文字本身"},
            {"role": "user", "content": "他还没回我"},
            {"role": "assistant", "content": "真人保留答案"},
        ],
        "metadata": {
            "source_ids": ["blind-1"],
            "conversation_mode": "responsive",
            "training_task": "conversation",
        },
    }
    (tmp_path / "test.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    generator = TwoStageGenerator()
    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"

    cases = ModelAcceptanceRunner(
        generator,
        FakeReviewer(),
        fixture,
    ).build_human_blind_cases(
        data_dir=tmp_path,
        base_model="base",
        adapter_path="persona-adapter",
        sample_count=1,
    )

    assert cases[0]["human_reply"] == "真人保留答案"
    assert cases[0]["candidate_reply"] == "我觉得先等等\n入"
    assert [call[0] for call in generator.calls] == ["", "persona-adapter"]
    assert generator.calls[1][2] == [
        {"role": "user", "content": "内容草稿：\n我觉得先等等"}
    ]


def test_valid_style_appends_full_compact_instruction_to_legacy_system(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    class CapturingGenerator(FakeGenerator):
        def __init__(self) -> None:
            self.system_prompts: list[str] = []

        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            self.system_prompts.append(system_prompt)
            return super().generate("", "", system_prompt, [])

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "必须使用高保真紧凑气泡协议"},
                {"role": "user", "content": f"问题{index}"},
                {
                    "role": "assistant",
                    "content": "<bubble>好</bubble><delay>0</delay>",
                },
            ],
            "metadata": {
                "source_ids": [f"valid-{index}"],
                "allowed_sticker_ids": [],
            },
        }
        for index in range(20)
    ]
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    generator = CapturingGenerator()

    ModelAcceptanceRunner(generator, FakeReviewer(), fixture).evaluate_valid_style(
        data_dir=tmp_path,
        base_model="base",
        adapter_path="adapter",
        persona="她",
        sample_count=20,
    )

    assert len(generator.system_prompts) == 20
    assert all("文字气泡 ::=" in prompt for prompt in generator.system_prompts)
    assert all("本轮禁止输出 sticker" in prompt for prompt in generator.system_prompts)


def test_valid_style_records_raw_output_diagnostics_for_structure_failures(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.model_acceptance import (
        ModelAcceptanceRunner,
        ModelOutputStructureError,
    )

    class BrokenGenerator(FakeGenerator):
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            raise ModelOutputStructureError(
                ("普通自然文本", "x" * 240),
                reasons=("缺少协议标签", "回复结构无效"),
            )

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    rows = [
        {
            "messages": [
                {"role": "system", "content": "旧系统提示"},
                {"role": "user", "content": f"问题{index}"},
                {
                    "role": "assistant",
                    "content": "<bubble>好</bubble><delay>0</delay>",
                },
            ],
            "metadata": {"source_ids": [f"valid-{index}"]},
        }
        for index in range(20)
    ]
    for split in ("train", "valid"):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    metrics = ModelAcceptanceRunner(
        BrokenGenerator(),
        FakeReviewer(),
        fixture,
    ).evaluate_valid_style(
        data_dir=tmp_path,
        base_model="base",
        adapter_path="adapter",
        persona="她",
        sample_count=20,
    )

    audit = json.loads(str(metrics["valid_structure_failure_audit"]))
    assert metrics["valid_structure_failures"] == 20
    assert len(audit) == 20
    assert len(audit[0]["attempts"]) == 2
    assert audit[0]["attempts"][0]["raw_output_hash"]
    assert audit[0]["attempts"][0]["raw_excerpt"] == "普通自然文本"
    assert audit[0]["attempts"][0]["reason"] == "缺少协议标签"
    assert len(audit[0]["attempts"][1]["raw_excerpt"]) == 200


def test_actual_model_acceptance_requires_generated_cases_to_pass() -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    report = ModelAcceptanceRunner(
        FakeGenerator(),
        FakeReviewer(),
        fixture,
    ).run(
        base_model="base",
        adapter_path="adapter",
        persona="小肥入",
        cutoff="2026-01-01T00:00:00",
    )

    assert report.passed is True
    assert report.passed_count == report.case_count


def test_reviewer_rewrite_cannot_hide_unsafe_raw_model_output() -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    report = ModelAcceptanceRunner(
        UnsafeGenerator(),
        RewritingReviewer(),
        fixture,
    ).run(
        base_model="base",
        adapter_path="adapter",
        persona="洪欣羽",
        cutoff="2026-01-01T00:00:00",
    )

    assert report.passed is False
    assert report.raw_output_failures > 0


def test_normalization_cannot_hide_forbidden_raw_model_text() -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    class RawUnsafeGenerator(FakeGenerator):
        def generate(
            self,
            _base_model: str,
            _adapter_path: str,
            _system_prompt: str,
            _messages: list[dict[str, str]],
        ) -> GeneratedReplyTurn:
            return GeneratedReplyTurn(
                bubbles=(GeneratedBubble(content="抱抱你", delay_ms=0),),
                raw_output="<bubble>我今天也累</bubble>",
                normalization="mixed-compact-v1",
            )

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    report = ModelAcceptanceRunner(
        RawUnsafeGenerator(),
        FakeReviewer(),
        fixture,
    ).run(
        base_model="base",
        adapter_path="adapter",
        persona="洪欣羽",
        cutoff="2026-01-01T00:00:00",
    )

    assert report.passed is False
    assert report.raw_output_failures > 0


def test_acceptance_style_uses_final_reviewer_output() -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    baseline = extract_style_features(
        [StyleTurn(bubbles=(StyleBubble(text="抱抱你"),))]
    )
    report = ModelAcceptanceRunner(
        MultiBubbleStyleGenerator(),
        RewritingReviewer(),
        fixture,
    ).run(
        base_model="base",
        adapter_path="adapter",
        persona="洪欣羽",
        cutoff="2026-01-01T00:00:00",
        style_profile=baseline,
    )

    assert report.style_feature_failures == 0
    assert report.passed is True


def test_acceptance_rejects_unlearned_style_even_if_reviewer_approves() -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    report = ModelAcceptanceRunner(
        UnlearnedStyleGenerator(),
        FakeReviewer(),
        fixture,
    ).run(
        base_model="base",
        adapter_path="adapter",
        persona="洪欣羽",
        cutoff="2026-01-01T00:00:00",
        style_profile={"forbidden_unobserved_markers": ["哒"]},
    )

    assert report.passed is False
    assert report.raw_output_failures == report.case_count


def test_acceptance_uses_unified_structural_features_from_nested_baseline() -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    fixture = Path(__file__).parents[1] / "fixtures" / "conversation_acceptance_v1.json"
    baseline = extract_style_features(
        [
            StyleTurn(bubbles=(StyleBubble(text="短回复"),)),
            StyleTurn(bubbles=(StyleBubble(text="也很短"),)),
        ]
    )
    report = ModelAcceptanceRunner(
        MultiBubbleStyleGenerator(),
        FakeReviewer(),
        fixture,
    ).run(
        base_model="base",
        adapter_path="adapter",
        persona="洪欣羽",
        cutoff="2026-01-01T00:00:00",
        style_profile={"style_profile": baseline},
    )

    assert report.passed is False
    assert report.style_feature_failures == 1
    assert report.raw_output_failures == 0
    assert any(
        difference.feature == "structural.bubbles_per_turn.max"
        for difference in report.style_feature_differences
    )


def test_acceptance_compares_style_distribution_after_all_cases(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner
    from moonlightbox.training.style_features import (
        StyleBubble,
        StyleTurn,
        extract_style_features,
    )

    fixture = tmp_path / "aggregate-cases.json"
    fixture.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "case-1",
                        "history": [],
                        "current_user_content": "第一问",
                    },
                    {
                        "id": "case-2",
                        "history": [],
                        "current_user_content": "第二问",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    baseline = extract_style_features(
        [
            StyleTurn(bubbles=(StyleBubble(text="第一句。"),)),
            StyleTurn(bubbles=(StyleBubble(text="第二句"),)),
        ]
    )

    report = ModelAcceptanceRunner(
        AlternatingPunctuationGenerator(),
        FakeReviewer(),
        fixture,
    ).run(
        base_model="base",
        adapter_path="adapter",
        persona="洪欣羽",
        cutoff="2026-01-01T00:00:00",
        style_profile=baseline,
    )

    assert report.passed is True
    assert report.style_feature_failures == 0
    assert report.style_feature_sample_count == 2


def test_protocol_emoji_kind_reaches_acceptance_style_features(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.model_acceptance import ModelAcceptanceRunner

    fixture = tmp_path / "emoji-case.json"
    fixture.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "emoji-case",
                        "history": [],
                        "current_user_content": "发个表情",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = ModelAcceptanceRunner(
        CompactEmojiGenerator(),
        FakeReviewer(),
        fixture,
    ).run(
        base_model="base",
        adapter_path="adapter",
        persona="洪欣羽",
        cutoff="2026-01-01T00:00:00",
    )

    assert report.candidate_style_profile["media"]["emoji_asset_frequencies"] == [
        {"asset_id": "emoji-smile", "count": 1}
    ]
    assert report.candidate_style_profile["media"]["sticker_asset_frequencies"] == []


def test_activation_requires_semantic_and_style_hard_gates() -> None:
    from moonlightbox.training.model_acceptance import (
        AcceptanceMetric,
        ModelGateSnapshot,
        evaluate_activation_gates,
    )

    candidate = ModelGateSnapshot(
        model_id="candidate",
        semantic={
            "fixed_regression": AcceptanceMetric(1.0),
            "structure": AcceptanceMetric(1.0),
            "future_isolation": AcceptanceMetric(1.0),
            "unsupported_fact": AcceptanceMetric(1.0),
        },
        style={"blind_preference": AcceptanceMetric(0.7)},
    )
    base = ModelGateSnapshot(
        model_id="base",
        semantic={},
        style={"blind_preference": AcceptanceMetric(0.4)},
    )
    active = ModelGateSnapshot(
        model_id="active",
        semantic={},
        style={"blind_preference": AcceptanceMetric(0.5)},
    )

    passed = evaluate_activation_gates(candidate, base, active)
    failed = evaluate_activation_gates(
        ModelGateSnapshot(
            model_id="candidate",
            semantic=candidate.semantic,
            style={"blind_preference": AcceptanceMetric(0.49)},
        ),
        base,
        active,
    )

    assert passed.passed is True
    assert passed.semantic_passed is True
    assert passed.style_passed is True
    assert failed.passed is False
    assert failed.semantic_passed is True
    assert failed.style_passed is False


def test_candidate_must_beat_base_and_active_with_tolerance() -> None:
    from moonlightbox.training.model_acceptance import (
        AcceptanceMetric,
        ModelGateSnapshot,
        evaluate_activation_gates,
    )

    candidate = ModelGateSnapshot(
        "candidate",
        semantic={
            "fixed_regression": AcceptanceMetric(1.0),
            "structure": AcceptanceMetric(1.0),
            "future_isolation": AcceptanceMetric(1.0),
            "unsupported_fact": AcceptanceMetric(1.0),
        },
        style={"speaker_probability": AcceptanceMetric(0.70, tolerance=0.01)},
    )
    base = ModelGateSnapshot("base", {}, {"speaker_probability": AcceptanceMetric(0.60)})
    active = ModelGateSnapshot("active", {}, {"speaker_probability": AcceptanceMetric(0.705)})

    report = evaluate_activation_gates(candidate, base, active)

    assert report.passed is False
    assert any("active" in reason for reason in report.failure_reasons)


def test_task4_metrics_are_style_hard_gates() -> None:
    from moonlightbox.training.model_acceptance import (
        AcceptanceMetric,
        ModelGateSnapshot,
        evaluate_activation_gates,
    )

    semantic = {
        "fixed_regression": AcceptanceMetric(1.0),
        "structure": AcceptanceMetric(1.0),
        "future_isolation": AcceptanceMetric(1.0),
        "unsupported_fact": AcceptanceMetric(1.0),
    }
    style = {
        "modality_f1": AcceptanceMetric(0.8, minimum=0.7),
        "sticker_recall_at_k": AcceptanceMetric(0.9, minimum=0.8),
        "sticker_mrr": AcceptanceMetric(0.8, minimum=0.7),
        "emoji_distance": AcceptanceMetric(0.1, maximum=0.2, higher_is_better=False),
        "repetition_rate": AcceptanceMetric(0.1, maximum=0.2, higher_is_better=False),
        "invalid_asset_rate": AcceptanceMetric(0.0, maximum=0.0, higher_is_better=False),
    }
    candidate = ModelGateSnapshot("candidate", semantic, style)
    base = ModelGateSnapshot("base", {}, {name: metric for name, metric in style.items()})
    active = ModelGateSnapshot("active", {}, {name: metric for name, metric in style.items()})

    assert evaluate_activation_gates(candidate, base, active).passed is True
    failing_style = dict(style)
    failing_style["invalid_asset_rate"] = AcceptanceMetric(
        0.01,
        maximum=0.0,
        higher_is_better=False,
    )
    assert (
        evaluate_activation_gates(
            ModelGateSnapshot("candidate", semantic, failing_style),
            base,
            active,
        ).passed
        is False
    )


def test_task4_evaluation_metrics_are_mapped_into_style_gates() -> None:
    from moonlightbox.training.model_acceptance import build_task4_style_metrics

    metrics = build_task4_style_metrics(
        {
            "modality_f1": 0.8,
            "recall_at_k": 0.9,
            "mrr": 0.7,
            "consecutive_repeat_rate": 0.1,
        },
        emoji_distance=0.12,
        invalid_asset_rate=0.0,
        calibrated_minimums={
            "modality_f1": 0.61,
            "sticker_recall_at_k": 0.52,
            "sticker_mrr": 0.43,
        },
    )

    assert metrics["modality_f1"].value == 0.8
    assert metrics["sticker_recall_at_k"].value == 0.9
    assert metrics["sticker_mrr"].value == 0.7
    assert metrics["emoji_distance"].higher_is_better is False
    assert metrics["repetition_rate"].value == 0.1
    assert metrics["invalid_asset_rate"].maximum == 0.0
    assert metrics["modality_f1"].minimum == 0.61
    assert metrics["sticker_recall_at_k"].minimum == 0.52
    assert metrics["sticker_mrr"].minimum == 0.43


def test_fixed_semantic_full_score_may_equal_active() -> None:
    from moonlightbox.training.model_acceptance import (
        AcceptanceMetric,
        ModelGateSnapshot,
        evaluate_activation_gates,
    )

    semantic = {
        name: AcceptanceMetric(1.0)
        for name in (
            "fixed_regression",
            "structure",
            "future_isolation",
            "unsupported_fact",
        )
    }
    candidate = ModelGateSnapshot(
        "candidate",
        semantic,
        {
            "style_distance": AcceptanceMetric(
                0.2,
                tolerance=0.02,
                higher_is_better=False,
            ),
            "speaker_probability": AcceptanceMetric(0.8, tolerance=0.02),
        },
    )
    base = ModelGateSnapshot(
        "base",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.3, higher_is_better=False),
            "speaker_probability": AcceptanceMetric(0.79),
        },
    )
    active = ModelGateSnapshot(
        "active",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.25, higher_is_better=False),
            "speaker_probability": AcceptanceMetric(0.79),
        },
    )

    report = evaluate_activation_gates(candidate, base, active)

    assert report.passed is True
    assert report.semantic_passed is True


def test_style_requires_one_strict_improvement_and_allows_other_metrics_within_tolerance() -> None:
    from moonlightbox.training.model_acceptance import (
        AcceptanceMetric,
        ModelGateSnapshot,
        evaluate_activation_gates,
    )

    semantic = {
        name: AcceptanceMetric(1.0)
        for name in (
            "fixed_regression",
            "structure",
            "future_isolation",
            "unsupported_fact",
        )
    }
    candidate = ModelGateSnapshot(
        "candidate",
        semantic,
        {
            "style_distance": AcceptanceMetric(
                0.20,
                tolerance=0.02,
                higher_is_better=False,
            ),
            "speaker_probability": AcceptanceMetric(0.70, tolerance=0.02),
        },
    )
    base = ModelGateSnapshot(
        "base",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.30, higher_is_better=False),
            "speaker_probability": AcceptanceMetric(0.71),
        },
    )
    active = ModelGateSnapshot(
        "active",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.25, higher_is_better=False),
            "speaker_probability": AcceptanceMetric(0.71),
        },
    )

    assert evaluate_activation_gates(candidate, base, active).passed is True
    equal = ModelGateSnapshot(
        "candidate",
        semantic,
        {
            "style_distance": AcceptanceMetric(
                0.30,
                tolerance=0.02,
                higher_is_better=False,
            ),
            "speaker_probability": AcceptanceMetric(0.71, tolerance=0.02),
        },
    )
    assert evaluate_activation_gates(equal, base, active).style_passed is False


def test_retry_instruction_follows_reply_protocol() -> None:
    from moonlightbox.training.model_acceptance import retry_format_instruction

    assert "紧凑气泡协议" in retry_format_instruction("compact")
    assert "气泡 JSON" not in retry_format_instruction("compact")
    assert "[bubble]" in retry_format_instruction("compact")
    assert "[sticker资产:" in retry_format_instruction("compact")
    assert "不得替换" in retry_format_instruction("compact", ("allowed",))
    assert "气泡 JSON" in retry_format_instruction("legacy_json")


def test_composite_fidelity_must_improve_both_baselines_with_component_limits() -> None:
    from moonlightbox.training.model_acceptance import (
        AcceptanceMetric,
        ModelGateSnapshot,
        evaluate_activation_gates,
    )

    semantic = {
        name: AcceptanceMetric(1.0)
        for name in (
            "fixed_regression",
            "structure",
            "future_isolation",
            "unsupported_fact",
        )
    }
    candidate = ModelGateSnapshot(
        "candidate",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.20),
            "speaker_probability_alignment": AcceptanceMetric(0.75),
            "paired_response_similarity": AcceptanceMetric(0.55),
            "style_distance_threshold": AcceptanceMetric(
                0.20,
                maximum=0.35,
                higher_is_better=False,
            ),
            "low_authority_prompt_leak_rate": AcceptanceMetric(
                0.0,
                maximum=0.0,
                higher_is_better=False,
            ),
        },
    )
    base = ModelGateSnapshot(
        "base",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.30),
            "speaker_probability_alignment": AcceptanceMetric(0.65),
            "paired_response_similarity": AcceptanceMetric(0.30),
            "low_authority_prompt_leak_rate": AcceptanceMetric(0.0),
        },
    )
    active = ModelGateSnapshot(
        "active",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.25),
            "speaker_probability_alignment": AcceptanceMetric(0.70),
            "paired_response_similarity": AcceptanceMetric(0.40),
            "low_authority_prompt_leak_rate": AcceptanceMetric(0.0),
        },
    )

    report = evaluate_activation_gates(candidate, base, active)

    assert report.passed is True
    fidelity = report.comparisons["style"]["composite_fidelity"]
    assert fidelity["weights"] == {
        "speaker_probability_alignment": 0.45,
        "style_similarity": 0.25,
        "paired_response_similarity": 0.30,
    }
    assert fidelity["candidate"] == pytest.approx(0.7025)
    assert fidelity["delta_active"] == pytest.approx(0.08)
    assert fidelity["minimum_improvement"] == 0.005

    poisoned_style = dict(candidate.style)
    poisoned_style["low_authority_prompt_leak_rate"] = AcceptanceMetric(
        0.05,
        maximum=0.0,
        higher_is_better=False,
    )
    poisoned = evaluate_activation_gates(
        ModelGateSnapshot("candidate", semantic, poisoned_style),
        base,
        active,
    )
    assert poisoned.passed is False
    assert any(
        "low_authority_prompt_leak_rate" in reason
        for reason in poisoned.failure_reasons
    )

    regressed = ModelGateSnapshot(
        "candidate",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.34),
            "speaker_probability_alignment": AcceptanceMetric(0.58),
            "paired_response_similarity": AcceptanceMetric(0.10),
            "style_distance_threshold": AcceptanceMetric(
                0.34,
                maximum=0.35,
                higher_is_better=False,
            ),
        },
    )
    failed = evaluate_activation_gates(regressed, base, active)
    assert failed.style_passed is False
    assert any("alignment" in reason for reason in failed.failure_reasons)


def test_composite_style_requires_calibrated_person_identity_alignment_floor() -> None:
    from moonlightbox.training.model_acceptance import (
        AcceptanceMetric,
        ModelGateSnapshot,
        evaluate_activation_gates,
    )

    semantic = {
        name: AcceptanceMetric(1.0)
        for name in (
            "fixed_regression",
            "structure",
            "future_isolation",
            "unsupported_fact",
        )
    }
    candidate = ModelGateSnapshot(
        "candidate",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.01),
            "speaker_probability_alignment": AcceptanceMetric(0.59),
            "paired_response_similarity": AcceptanceMetric(0.40),
        },
    )
    weak = ModelGateSnapshot(
        "weak",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.30),
            "speaker_probability_alignment": AcceptanceMetric(0.10),
            "paired_response_similarity": AcceptanceMetric(0.0),
        },
    )

    report = evaluate_activation_gates(candidate, weak, weak)

    assert report.style_passed is False
    assert (
        report.comparisons["style"]["speaker_probability_alignment"][
            "calibrated_minimum"
        ]
        == 0.60
    )
    assert any("校准下限" in reason for reason in report.failure_reasons)


def test_composite_style_rejects_contextually_wrong_reply_below_semantic_floor() -> None:
    from moonlightbox.training.model_acceptance import (
        AcceptanceMetric,
        ModelGateSnapshot,
        evaluate_activation_gates,
    )

    semantic = {
        name: AcceptanceMetric(1.0)
        for name in (
            "fixed_regression",
            "structure",
            "future_isolation",
            "unsupported_fact",
        )
    }
    candidate = ModelGateSnapshot(
        "candidate",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.05),
            "speaker_probability_alignment": AcceptanceMetric(0.80),
            "paired_response_similarity": AcceptanceMetric(0.49),
        },
    )
    baseline = ModelGateSnapshot(
        "baseline",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.20),
            "speaker_probability_alignment": AcceptanceMetric(0.70),
            "paired_response_similarity": AcceptanceMetric(0.45),
        },
    )

    report = evaluate_activation_gates(candidate, baseline, baseline)

    assert report.style_passed is False
    assert (
        report.comparisons["style"]["paired_response_similarity"][
            "calibrated_minimum"
        ]
        == 0.50
    )
    assert any("真实回复语义下限" in reason for reason in report.failure_reasons)


def test_mlx_generator_loads_same_model_once_and_releases_before_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import MlxPathReplyGenerator

    loaded: list[tuple[str, str | None]] = []
    generated_models: list[object] = []

    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

    def fake_load(base_model: str, *, adapter_path: str | None = None) -> tuple[object, object]:
        model = object()
        loaded.append((base_model, adapter_path))
        return model, FakeTokenizer()

    def fake_generate(model: object, *_args: object, **_kwargs: object) -> str:
        generated_models.append(model)
        return "<bubble>好</bubble><delay>0</delay>"

    fake_mlx_lm = SimpleNamespace(load=fake_load, generate=fake_generate)
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    generator = MlxPathReplyGenerator()
    for _ in range(20):
        generator.generate("base", "adapter-a", "紧凑气泡协议", [])
    generator.generate("base", "adapter-b", "紧凑气泡协议", [])

    assert loaded == [("base", "adapter-a"), ("base", "adapter-b")]
    assert len(set(map(id, generated_models[:20]))) == 1
    assert generated_models[20] is not generated_models[0]


def test_mlx_acceptance_uses_same_human_style_sampling_as_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import MlxPathReplyGenerator

    captured: dict[str, dict[str, float | int]] = {}

    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: "<bubble>好</bubble><delay>0</delay>",
    )
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **kwargs: captured.setdefault("sampler", kwargs),
        make_repetition_penalty=lambda **kwargs: captured.setdefault(
            "repetition", kwargs
        ),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    MlxPathReplyGenerator().generate("base", "adapter", "紧凑气泡协议", [])

    assert captured["sampler"] == {"temp": 0.65, "top_p": 0.9, "min_p": 0.02}
    assert captured["repetition"] == {"penalty": 1.02, "context_size": 128}


def test_mlx_generator_enforces_sticker_top_k_from_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import (
        MlxPathReplyGenerator,
        ModelOutputStructureError,
    )

    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: (
            '<sticker kind="sticker">future-id</sticker><delay>0</delay>'
        ),
    )
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    with pytest.raises(RuntimeError, match="结构") as captured:
        MlxPathReplyGenerator().generate(
            "base",
            "adapter",
            "紧凑气泡协议\n本轮禁止输出 sticker。",
            [],
        )

    assert isinstance(captured.value, ModelOutputStructureError)
    assert captured.value.attempted_invalid_sticker_ids == (
        "future-id",
        "future-id",
        "future-id",
    )


def test_mlx_generator_does_not_replace_invalid_sticker_with_policy_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import (
        MlxPathReplyGenerator,
        ModelOutputStructureError,
    )

    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: "[sticker资产:future-id]",
    )
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    with pytest.raises(ModelOutputStructureError) as captured:
        MlxPathReplyGenerator().generate(
            "base",
            "adapter",
            "紧凑气泡协议\n"
            "本轮 sticker 仅允许以下资产 ID：policy-top1、other；"
            "禁止输出其他 sticker。",
            [],
        )

    assert captured.value.attempted_invalid_sticker_ids == (
        "future-id",
        "future-id",
        "future-id",
    )


def test_mlx_generator_retries_by_strengthening_system_not_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import MlxPathReplyGenerator

    class FakeTokenizer:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            **_kwargs: object,
        ) -> str:
            self.calls.append([dict(item) for item in messages])
            return "prompt"

    tokenizer = FakeTokenizer()
    outputs = iter(
        (
            "[bubble]非法[/bubble]",
            "<bubble>合法回复</bubble><delay>0</delay>",
        )
    )
    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), tokenizer),
        generate=lambda *_args, **_kwargs: next(outputs),
    )
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    reply = MlxPathReplyGenerator().generate(
        "base",
        "adapter",
        "紧凑气泡协议",
        [{"role": "user", "content": "原始问题"}],
    )

    assert [bubble.content for bubble in reply.bubbles] == ["合法回复"]
    assert [item["role"] for item in tokenizer.calls[1]] == ["system", "user"]
    assert "严禁使用 [bubble]" in tokenizer.calls[1][0]["content"]
    assert tokenizer.calls[1][1]["content"] == "原始问题"


def test_mlx_generator_normalizes_mixed_output_and_records_invalid_sticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import MlxPathReplyGenerator

    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: (
            "assistant: <bubble>  保留原文  </bubble>"
            "<sticker>future-id</sticker>"
        ),
    )
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    reply = MlxPathReplyGenerator().generate(
        "base",
        "adapter",
        "紧凑气泡协议\n本轮 sticker 仅允许以下资产 ID：known；禁止输出其他 sticker。",
        [],
    )

    assert [bubble.content for bubble in reply.bubbles] == ["  保留原文  "]
    assert reply.attempted_invalid_sticker_ids == ("future-id",)
    assert reply.normalization == "mixed-compact-v1"
    assert all(bubble.asset_id != "known" for bubble in reply.bubbles)


def test_mlx_generator_uses_last_protocol_override_for_legacy_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import MlxPathReplyGenerator

    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: (
            '{"bubbles":[{"content":"旧 active 正常输出","delay_ms":0}]}'
        ),
    )
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    reply = MlxPathReplyGenerator().generate(
        "base",
        "adapter",
        "历史提示包含紧凑气泡协议\n只输出合法统一气泡 JSON。本轮禁止输出 sticker。",
        [],
    )

    assert [bubble.content for bubble in reply.bubbles] == ["旧 active 正常输出"]
    assert reply.normalization is None


def test_mlx_generator_normalizes_plain_text_after_two_compact_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import MlxPathReplyGenerator

    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

    outputs = iter(("第一行\n\n第二行", "第一行\n\n第二行"))
    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: next(outputs),
    )
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    reply = MlxPathReplyGenerator().generate(
        "base",
        "adapter",
        "紧凑气泡协议",
        [],
    )

    assert [bubble.content for bubble in reply.bubbles] == ["第一行", "第二行"]
    assert [bubble.delay_ms for bubble in reply.bubbles] == [0, 0]
    assert reply.normalization == "mixed-compact-v1"


def test_mlx_generator_never_normalizes_partial_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import MlxPathReplyGenerator

    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: "半截<bubble",
    )
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    with pytest.raises(RuntimeError, match="结构"):
        MlxPathReplyGenerator().generate(
            "base",
            "adapter",
            "紧凑气泡协议",
            [],
        )


def test_mlx_generator_uses_safe_first_attempt_when_retry_is_partial_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moonlightbox.training.model_acceptance import MlxPathReplyGenerator

    class FakeTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            return "prompt"

    outputs = iter(("第一行\n第二行", "半截<bubble"))
    fake_mlx_lm = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: next(outputs),
    )
    fake_sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    reply = MlxPathReplyGenerator().generate(
        "base",
        "adapter",
        "紧凑气泡协议",
        [],
    )

    assert [bubble.content for bubble in reply.bubbles] == ["第一行", "第二行"]
    assert reply.raw_output == "第一行\n第二行"
    assert reply.normalization == "mixed-compact-v1"


def test_calibrated_sticker_test_metrics_must_improve_active_pipeline() -> None:
    from moonlightbox.training.model_acceptance import (
        AcceptanceMetric,
        ModelGateSnapshot,
        evaluate_activation_gates,
    )

    semantic = {
        name: AcceptanceMetric(1.0)
        for name in (
            "fixed_regression",
            "structure",
            "future_isolation",
            "unsupported_fact",
        )
    }
    candidate_style = {
        "style_distance": AcceptanceMetric(0.2),
        "speaker_probability": AcceptanceMetric(0.8),
        "speaker_probability_alignment": AcceptanceMetric(0.8),
        "paired_response_similarity": AcceptanceMetric(0.55),
        "modality_f1": AcceptanceMetric(0.6, minimum=0.5),
        "sticker_recall_at_k": AcceptanceMetric(0.6, minimum=0.5),
        "sticker_mrr": AcceptanceMetric(0.6, minimum=0.5),
        "invalid_asset_rate": AcceptanceMetric(
            0.0,
            maximum=0.0,
            higher_is_better=False,
        ),
    }
    base = ModelGateSnapshot(
        "base",
        semantic,
        {
            "style_distance": AcceptanceMetric(0.3),
            "speaker_probability": AcceptanceMetric(0.7),
            "speaker_probability_alignment": AcceptanceMetric(0.7),
            "paired_response_similarity": AcceptanceMetric(0.3),
        },
    )
    active_style = {
        **candidate_style,
        "style_distance": AcceptanceMetric(0.25),
        "speaker_probability": AcceptanceMetric(0.79),
        "paired_response_similarity": AcceptanceMetric(0.50),
    }
    active = ModelGateSnapshot("active", semantic, active_style)

    equal = evaluate_activation_gates(
        ModelGateSnapshot("candidate", semantic, candidate_style),
        base,
        active,
    )
    improved_style = dict(candidate_style)
    improved_style["sticker_mrr"] = AcceptanceMetric(0.61, minimum=0.5)
    improved = evaluate_activation_gates(
        ModelGateSnapshot("candidate", semantic, improved_style),
        base,
        active,
    )

    assert equal.style_passed is False
    assert any("active pipeline" in reason for reason in equal.failure_reasons)
    assert improved.style_passed is True
