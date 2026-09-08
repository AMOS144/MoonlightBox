from pathlib import Path
from types import SimpleNamespace

import pytest
from moonlightbox.db import Database
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class FakeTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        self.messages = messages
        return "已格式化的聊天提示"


def test_mlx_generator_uses_model_chat_template(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from moonlightbox.branches.mlx_generation import DatabaseMlxGenerator

    database = Database(f"sqlite:///{tmp_path / 'generation.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="推理测试"))
        session.flush()
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="/models/qwen",
                adapter_path="/models/adapter",
                dataset_hash="hash",
                metrics={},
            )
        )
        session.commit()

    tokenizer = FakeTokenizer()
    generated: dict[str, object] = {}

    def generate(
        _model: object,
        _tokenizer: object,
        *,
        prompt: str,
        max_tokens: int,
        sampler: object,
        logits_processors: list[object],
        verbose: bool,
    ) -> str:
        generated.update(
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=verbose,
        )
        return "<bubble>正常回复</bubble><delay>0</delay>"

    runtime = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), tokenizer),
        generate=generate,
    )
    sampler = object()
    repetition_penalty = object()
    sampling: dict[str, float] = {}
    repetition: dict[str, float | int] = {}

    def make_sampler(**kwargs: float) -> object:
        sampling.update(kwargs)
        return sampler

    sample_utils = SimpleNamespace(
        make_sampler=make_sampler,
        make_repetition_penalty=lambda **kwargs: (
            repetition.update(kwargs) or repetition_penalty
        ),
    )
    monkeypatch.setattr(
        "moonlightbox.branches.mlx_generation.importlib.import_module",
        lambda name: sample_utils if name == "mlx_lm.sample_utils" else runtime,
    )

    result = DatabaseMlxGenerator(database).generate(
        "model-1",
        "只能使用当前时间点之前的信息",
        [
            {"role": "user", "content": "之前的话"},
            {
                "role": "assistant",
                "content": '{"bubbles":[{"content":"之前的回复","delay_ms":0}]}',
            },
            {"role": "user", "content": "你在干啥 笨笨"},
        ],
    )

    assert [bubble.content for bubble in result.bubbles] == ["正常回复"]
    assert len(tokenizer.messages) == 4
    assert tokenizer.messages[0]["role"] == "system"
    assert "只能使用当前时间点之前的信息" in tokenizer.messages[0]["content"]
    assert tokenizer.messages[1]["content"] == "之前的话"
    assert tokenizer.messages[-1]["content"] == "你在干啥 笨笨"
    assert generated["prompt"] == "已格式化的聊天提示"
    assert generated["max_tokens"] == 160
    assert generated["sampler"] is sampler
    assert generated["logits_processors"] == [repetition_penalty]
    assert sampling == {"temp": 0.65, "top_p": 0.9, "min_p": 0.02}
    assert repetition == {"penalty": 1.02, "context_size": 128}


def test_runtime_generator_reuses_loaded_model_for_same_version(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from moonlightbox.branches.mlx_generation import DatabaseMlxGenerator

    database = Database(f"sqlite:///{tmp_path / 'generation-cache.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="推理缓存测试"))
        session.flush()
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="/models/qwen",
                adapter_path="/models/adapter",
                dataset_hash="hash",
                metrics={},
            )
        )
        session.commit()

    tokenizer = FakeTokenizer()
    load_count = 0

    def load(*_args: object, **_kwargs: object) -> tuple[object, FakeTokenizer]:
        nonlocal load_count
        load_count += 1
        return object(), tokenizer

    runtime = SimpleNamespace(
        load=load,
        generate=lambda *_args, **_kwargs: (
            "<bubble>正常回复</bubble><delay>0</delay>"
        ),
    )
    sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "moonlightbox.branches.mlx_generation.importlib.import_module",
        lambda name: sample_utils if name == "mlx_lm.sample_utils" else runtime,
    )

    generator = DatabaseMlxGenerator(database)
    generator.generate("model-1", "系统提示", [])
    generator.generate("model-1", "系统提示", [])

    assert load_count == 1


def test_v17_runtime_rewrites_draft_and_rejects_negation_flip(tmp_path: Path) -> None:
    from moonlightbox.branches.mlx_generation import DatabaseMlxGenerator

    database = Database(f"sqlite:///{tmp_path / 'style-transfer.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="两阶段风格迁移"))
        session.flush()
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="/models/qwen",
                adapter_path="/models/adapter",
                dataset_hash="hash",
                metrics={},
                training_config={
                    "training_protocol_version": (
                        "persona-plain-text-private-chat-v17-style-transfer"
                    ),
                    "runtime_style_transfer_enabled": True,
                },
            )
        )
        session.commit()

    outputs = iter(("我今天不想去", "我今天想去", "我今天不想去\n啊呀"))
    calls: list[list[dict[str, str]]] = []

    class Runtime:
        def generate_raw(self, **kwargs: object) -> str:
            calls.append(kwargs["messages"])  # type: ignore[arg-type]
            return next(outputs)

    reply = DatabaseMlxGenerator(database, runtime=Runtime()).generate(
        "model-1",
        "私人微信聊天\n只输出聊天文字本身，多条消息换行。",
        [{"role": "user", "content": "今天去吗"}],
    )

    assert [bubble.content for bubble in reply.bubbles] == ["我今天不想去", "啊呀"]
    assert len(calls) == 3
    assert calls[1][-1]["content"] == "内容草稿：\n我今天不想去"


def test_explicit_style_transfer_is_not_rewritten_a_second_time(tmp_path: Path) -> None:
    from moonlightbox.branches.mlx_generation import DatabaseMlxGenerator
    from moonlightbox.training.bubble_protocol import (
        persona_style_transfer_instruction,
    )

    database = Database(f"sqlite:///{tmp_path / 'single-style-transfer.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="单次风格迁移"))
        session.flush()
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="/models/qwen",
                adapter_path="/models/adapter",
                dataset_hash="hash",
                metrics={},
                training_config={"runtime_style_transfer_enabled": True},
            )
        )
        session.commit()

    calls: list[list[dict[str, str]]] = []

    class Runtime:
        def generate_raw(self, **kwargs: object) -> str:
            calls.append(kwargs["messages"])  # type: ignore[arg-type]
            return "啊呀\n别急，我们一起想想去哪里好"

    reply = DatabaseMlxGenerator(database, runtime=Runtime()).generate(
        "model-1",
        persona_style_transfer_instruction(("对方：该如何是好\n本人：啊呀",)),
        [{"role": "user", "content": "内容草稿：\n别急，我们一起想想去哪里好。"}],
    )

    assert [bubble.content for bubble in reply.bubbles] == [
        "啊呀",
        "别急，我们一起想想去哪里好",
    ]
    assert len(calls) == 1


def test_style_transfer_rejects_person_referent_swap() -> None:
    from moonlightbox.branches.mlx_generation import (
        rewrite_preserves_hard_semantics,
    )

    draft = "因为你抖音说你下班了我没来得及回"

    assert rewrite_preserves_hard_semantics(
        draft,
        "因为你抖音说你下班了，我没来得及回呀",
    )
    assert not rewrite_preserves_hard_semantics(
        draft,
        "因为我抖音说你下班了我没来得及回",
    )


def test_runtime_generator_resamples_unobserved_ai_register(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from moonlightbox.agent.mlx_inference import SharedMlxModelRuntime
    from moonlightbox.branches.continuity_models import IdentityKernel
    from moonlightbox.branches.mlx_generation import DatabaseMlxGenerator

    database = Database(f"sqlite:///{tmp_path / 'style-retry.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="风格重采样测试"))
        session.flush()
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="/models/qwen",
                adapter_path="/models/adapter",
                dataset_hash="hash",
                metrics={},
            )
        )
        session.flush()
        session.add(
            IdentityKernel(
                id="kernel-1",
                project_id="project-1",
                model_version_id="model-1",
                content={
                    "style_profile": {
                        "forbidden_unobserved_ai_register": ["我理解你的感受"],
                    }
                },
                evidence_message_ids=["message-1"],
                field_evidence={},
                field_confidence={},
                content_hash="style-kernel-hash",
                locked_at=datetime.now(UTC),
            )
        )
        session.commit()

    tokenizer = FakeTokenizer()
    outputs = iter(
        (
            "<bubble>我理解你的感受</bubble><delay>0</delay>",
            "<bubble>不知道说啥</bubble><delay>0</delay>",
        )
    )
    prompts: list[list[dict[str, str]]] = []

    class Runtime:
        def load(self, *_args: object, **_kwargs: object) -> tuple[object, FakeTokenizer]:
            return object(), tokenizer

        def generate(self, *_args: object, **_kwargs: object) -> str:
            prompts.append(list(tokenizer.messages))
            return next(outputs)

    sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    raw_runtime = SharedMlxModelRuntime(
        mlx_runtime=Runtime(),
        sample_utils=sample_utils,
    )

    reply = DatabaseMlxGenerator(database, runtime=raw_runtime).generate(
        "model-1",
        "像本人一样聊天",
        [{"role": "user", "content": "今天好累"}],
    )

    assert [bubble.content for bubble in reply.bubbles] == ["不知道说啥"]
    assert len(prompts) == 2
    assert "我理解你的感受" in prompts[1][0]["content"]


def test_runtime_generator_retries_empty_invalid_sticker_then_keeps_safe_text(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from moonlightbox.branches.mlx_generation import DatabaseMlxGenerator

    database = Database(f"sqlite:///{tmp_path / 'generation.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="推理测试"))
        session.flush()
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="/models/qwen",
                adapter_path="/models/adapter",
                dataset_hash="hash",
                metrics={},
            )
        )
        session.commit()

    tokenizer = FakeTokenizer()
    outputs = iter(
        (
            "<sticker>future-id</sticker>",
            "<bubble>  保留原文  </bubble><sticker>future-id</sticker>",
        )
    )
    generated_count = 0

    def generate(*_args: object, **_kwargs: object) -> str:
        nonlocal generated_count
        generated_count += 1
        return next(outputs)

    runtime = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), tokenizer),
        generate=generate,
    )
    sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "moonlightbox.branches.mlx_generation.importlib.import_module",
        lambda name: sample_utils if name == "mlx_lm.sample_utils" else runtime,
    )

    reply = DatabaseMlxGenerator(database).generate(
        "model-1",
        "紧凑气泡协议\n"
        "本轮 sticker 仅允许以下资产 ID：allowed；禁止输出其他 sticker。",
        [{"role": "user", "content": "继续"}],
    )

    assert generated_count == 2
    assert [bubble.content for bubble in reply.bubbles] == ["  保留原文  "]
    assert reply.attempted_invalid_sticker_ids == ("future-id",)
    assert all(bubble.asset_id != "allowed" for bubble in reply.bubbles)
    assert [item["role"] for item in tokenizer.messages] == ["system", "user"]


def test_runtime_generator_does_not_replace_invalid_sticker_with_policy_candidate(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    from moonlightbox.branches.generation import GenerationFailedError
    from moonlightbox.branches.mlx_generation import DatabaseMlxGenerator

    database = Database(f"sqlite:///{tmp_path / 'generation.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="推理测试"))
        session.flush()
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="/models/qwen",
                adapter_path="/models/adapter",
                dataset_hash="hash",
                metrics={},
            )
        )
        session.commit()

    tokenizer = FakeTokenizer()
    runtime = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), tokenizer),
        generate=lambda *_args, **_kwargs: "[sticker资产:future-id]",
    )
    sample_utils = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "moonlightbox.branches.mlx_generation.importlib.import_module",
        lambda name: sample_utils if name == "mlx_lm.sample_utils" else runtime,
    )

    with pytest.raises(GenerationFailedError):
        DatabaseMlxGenerator(database).generate(
            "model-1",
            "紧凑气泡协议\n"
            "本轮 sticker 仅允许以下资产 ID：policy-top1、other；"
            "禁止输出其他 sticker。",
            [{"role": "user", "content": "继续"}],
        )
