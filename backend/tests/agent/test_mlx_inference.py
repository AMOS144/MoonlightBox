import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import sleep
from types import SimpleNamespace

import pytest
from moonlightbox.agent.inference import (
    InferencePreemptedError,
    InferencePriority,
    PersonaInferenceRequest,
)
from moonlightbox.agent.mlx_inference import (
    DatabasePersonaInferenceBackend,
    SharedMlxModelRuntime,
)
from moonlightbox.branches.generation import GenerationFailedError
from moonlightbox.branches.replies import ReplyStructureError
from moonlightbox.db import Database
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return json.dumps(messages, ensure_ascii=False)


def _database(tmp_path: Path) -> Database:
    database = Database(f"sqlite:///{tmp_path / 'persona.db'}")
    ModelVersion.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="共享推理"))
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
    return database


def _request(
    request_type: str,
    payload: dict[str, object],
) -> PersonaInferenceRequest:
    return PersonaInferenceRequest(
        request_id=f"{request_type}-1",
        request_type=request_type,
        priority=(
            InferencePriority.REALTIME
            if request_type == "reply"
            else InferencePriority.OFFLINE
        ),
        deadline=datetime.now(UTC) + timedelta(seconds=5),
        model_version_id="model-1",
        payload=payload,
    )


def test_reply_uses_persona_adapter_but_cognition_uses_clean_base_model(
    tmp_path: Path,
) -> None:
    load_count = 0
    outputs = iter(
        (
            "<bubble>你好</bubble><delay>0</delay>",
            json.dumps(
                {
                    "private_content": "我想先观察",
                    "subjective_feelings": {"calm": 0.8},
                    "attention_target": {"topic": "user"},
                    "desired_actions": [],
                    "expression_decision": {
                        "express": False,
                        "content": None,
                        "reason": "暂不打扰",
                    },
                    "suggested_next_wakeup": None,
                    "structured_changes": {},
                    "confidence": 0.9,
                },
                ensure_ascii=False,
            ),
        )
    )

    loaded: list[tuple[str, str | None]] = []

    def load(
        model_path: str, *, adapter_path: str | None = None
    ) -> tuple[object, FakeTokenizer]:
        nonlocal load_count
        assert model_path == "/models/qwen"
        loaded.append((model_path, adapter_path))
        load_count += 1
        return object(), FakeTokenizer()

    mlx = SimpleNamespace(load=load, generate=lambda *_args, **_kwargs: next(outputs))
    sampling = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    runtime = SharedMlxModelRuntime(mlx_runtime=mlx, sample_utils=sampling)
    backend = DatabasePersonaInferenceBackend(_database(tmp_path), runtime)

    reply = backend.infer(
        _request(
            "reply",
            {"system_prompt": "系统", "messages": [{"role": "user", "content": "在吗"}]},
        )
    )
    cognition = backend.infer(
        _request(
            "cognition",
            {
                "project_id": "project-1",
                "branch_id": "branch-1",
                "trigger_event": {},
                "deadline": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
                "current_mental_state": {},
                "goals": [],
                "relevant_context": [],
            },
        )
    )

    assert load_count == 2
    assert loaded == [
        ("/models/qwen", "/models/adapter"),
        ("/models/qwen", None),
    ]
    assert reply.output["bubbles"][0]["content"] == "你好"
    assert cognition.output["private_content"] == "我想先观察"


def test_cognition_fails_closed_after_two_invalid_structures(tmp_path: Path) -> None:
    mlx = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: "<bubble>这只是聊天气泡</bubble>",
    )
    sampling = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    backend = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        SharedMlxModelRuntime(mlx_runtime=mlx, sample_utils=sampling),
    )

    with pytest.raises(GenerationFailedError, match="有效认知结构"):
        backend.infer(
            _request(
                "cognition",
                {
                    "project_id": "project-1",
                    "branch_id": "branch-1",
                    "trigger_event": {},
                    "deadline": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
                    "current_mental_state": {},
                    "goals": [],
                    "relevant_context": [],
                },
            )
        )


def test_authoritative_cognition_retries_when_expression_decision_breaks_policy(
    tmp_path: Path,
) -> None:
    outputs = iter(
        (
            {
                "private_content": "先不回",
                "subjective_feelings": {},
                "attention_target": {},
                "desired_actions": [],
                "expression_decision": {
                    "express": False,
                    "content": None,
                },
                "suggested_next_wakeup": None,
                "structured_changes": {},
                "confidence": 1,
            },
            {
                "private_content": "正常回应",
                "subjective_feelings": {},
                "attention_target": {},
                "desired_actions": [],
                "expression_decision": {
                    "express": True,
                    "content": "我在呢",
                },
                "suggested_next_wakeup": None,
                "structured_changes": {},
                "confidence": 1,
            },
        )
    )
    calls: list[dict[str, object]] = []

    class Runtime:
        def generate_raw(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return json.dumps(next(outputs), ensure_ascii=False)

    result = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "cognition",
            {
                "shadow_mode": False,
                "expression_required": True,
                "trigger_event": {"event_type": "user_message"},
            },
        )
    )

    assert result.output["expression_decision"]["express"] is True
    assert result.output["expression_decision"]["content"] == "我在呢"
    assert len(calls) == 2
    assert "本轮必须回复" in calls[0]["messages"][0]["content"]


def test_shadow_cognition_keeps_unstructured_note_without_external_action(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class Runtime:
        def generate_raw(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return "我暂时不想打扰她，先等等看。"

    backend = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    )

    result = backend.infer(
        _request(
            "cognition",
            {
                "shadow_mode": True,
                "trigger_event": {"event_type": "elapsed_time"},
            },
        )
    )

    assert result.output["private_content"] == "我暂时不想打扰她，先等等看。"
    assert result.output["expression_decision"] == {
        "express": False,
        "content": None,
        "reason": "影子认知尚未结构化",
    }
    assert result.output["structured_changes"] == {
        "extraction_status": "pending"
    }
    assert calls[0]["max_tokens"] == 256


def test_cognition_adds_an_unambiguous_two_sided_transcript(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    output = {
        "private_content": "对方在追问我刚才的感叹号",
        "subjective_feelings": {},
        "attention_target": {},
        "desired_actions": [],
        "expression_decision": {
            "express": True,
            "content": "我就是有点惊讶",
        },
        "suggested_next_wakeup": None,
        "structured_changes": {},
        "confidence": 0.9,
    }

    class Runtime:
        def generate_raw(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return json.dumps(output, ensure_ascii=False)

    DatabasePersonaInferenceBackend(_database(tmp_path), Runtime()).infer(
        _request(
            "cognition",
            {
                "shadow_mode": False,
                "expression_required": True,
                "relevant_context": [
                    {
                        "event_type": "agent_expression",
                        "evidence": {"content": "！"},
                    },
                    {
                        "event_type": "user_message",
                        "evidence": {"content": "什么意思呢"},
                    },
                ],
            },
        )
    )

    prompt_payload = json.loads(calls[0]["messages"][1]["content"])
    assert prompt_payload["conversation_transcript"] == [
        {"speaker": "数字人自己（你）", "content": "！"},
        {"speaker": "聊天对方（用户）", "content": "什么意思呢"},
    ]


def test_stream_generation_stops_at_chunk_boundary_without_partial_result() -> None:
    generated_chunks = 0
    cancel_checks = 0

    def stream_generate(*_args: object, **_kwargs: object) -> object:
        nonlocal generated_chunks
        while True:
            generated_chunks += 1
            yield SimpleNamespace(text=f"片段{generated_chunks}")

    def should_cancel() -> bool:
        nonlocal cancel_checks
        cancel_checks += 1
        return cancel_checks >= 3

    mlx = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: pytest.fail("存在流式 API 时不得调用同步生成"),
        stream_generate=stream_generate,
    )
    sampling = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    runtime = SharedMlxModelRuntime(mlx_runtime=mlx, sample_utils=sampling)

    with pytest.raises(InferencePreemptedError):
        runtime.generate_raw(
            base_model="/models/qwen",
            adapter_path="/models/adapter",
            messages=[{"role": "user", "content": "继续"}],
            max_tokens=10,
            should_cancel=should_cancel,
        )

    assert generated_chunks < 10


def test_stream_generation_does_not_pass_sync_only_verbose_argument() -> None:
    received: dict[str, object] = {}

    def stream_generate(*_args: object, **kwargs: object) -> object:
        received.update(kwargs)
        yield SimpleNamespace(text="完成")

    mlx = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: pytest.fail("不得调用同步生成"),
        stream_generate=stream_generate,
    )
    sampling = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )

    result = SharedMlxModelRuntime(
        mlx_runtime=mlx,
        sample_utils=sampling,
    ).generate_raw(
        base_model="/models/qwen",
        adapter_path="/models/adapter",
        messages=[{"role": "user", "content": "测试"}],
        max_tokens=8,
    )

    assert result == "完成"
    assert "verbose" not in received


def test_stream_generation_stops_when_deadline_expires() -> None:
    def stream_generate(*_args: object, **_kwargs: object) -> object:
        yield SimpleNamespace(text="部分")
        sleep(0.02)
        yield SimpleNamespace(text="结果")

    mlx = SimpleNamespace(
        load=lambda *_args, **_kwargs: (object(), FakeTokenizer()),
        generate=lambda *_args, **_kwargs: pytest.fail("存在流式 API 时不得调用同步生成"),
        stream_generate=stream_generate,
    )
    sampling = SimpleNamespace(
        make_sampler=lambda **_kwargs: object(),
        make_repetition_penalty=lambda **_kwargs: object(),
    )
    runtime = SharedMlxModelRuntime(mlx_runtime=mlx, sample_utils=sampling)

    with pytest.raises(InferencePreemptedError):
        runtime.generate_raw(
            base_model="/models/qwen",
            adapter_path="/models/adapter",
            messages=[{"role": "user", "content": "继续"}],
            max_tokens=10,
            deadline=datetime.now(UTC) + timedelta(milliseconds=5),
        )


def test_database_backend_forwards_cancellation_and_deadline_to_all_paths(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    cognition = json.dumps(
        {
            "private_content": "继续观察",
            "subjective_feelings": {},
            "attention_target": {},
            "desired_actions": [],
            "expression_decision": {"express": False},
            "suggested_next_wakeup": None,
        },
        ensure_ascii=False,
    )
    outputs = iter(("<bubble>收到</bubble><delay>0</delay>", cognition))

    class CapturingRuntime:
        def generate_raw(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return next(outputs)

    backend = DatabasePersonaInferenceBackend(_database(tmp_path), CapturingRuntime())

    def should_cancel() -> bool:
        return False

    reply_request = _request(
        "reply",
        {"system_prompt": "系统", "messages": [{"role": "user", "content": "在吗"}]},
    )
    backend.infer(
        reply_request,
        should_cancel=should_cancel,
    )
    cognition_request = _request("cognition", {"trigger_event": {}})
    backend.infer(cognition_request, should_cancel=should_cancel)

    assert [call["should_cancel"] for call in calls] == [should_cancel, should_cancel]
    assert [call["deadline"] for call in calls] == [
        reply_request.deadline,
        cognition_request.deadline,
    ]


def test_fused_reply_keeps_private_partition_out_of_public_bubbles(
    tmp_path: Path,
) -> None:
    private_content = "我想保持距离，<bubble>绝不能发给用户</bubble>"

    class Runtime:
        def generate_raw(self, **_kwargs: object) -> str:
            return json.dumps(
                {
                    "private_cognition": {
                        "private_content": private_content,
                        "subjective_feelings": {"警惕": 0.7},
                        "attention_target": {"对象": "用户语气"},
                        "desired_actions": [{"type": "wait"}],
                        "suggested_next_wakeup": None,
                        "structured_changes": {},
                        "confidence": 0.9,
                    },
                    "expression_decision": {
                        "express": True,
                        "reason": "只表达边界",
                    },
                    "public_compact_bubbles": "<bubble>我现在不太想聊</bubble>",
                },
                ensure_ascii=False,
            )

    result = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "fused_reply",
            {
                "system_prompt": "系统",
                "messages": [{"role": "user", "content": "继续说"}],
                "allowed_sticker_ids": [],
            },
        )
    )

    assert result.output["private_cognition"]["private_content"] == private_content
    assert [bubble["content"] for bubble in result.output["reply"]["bubbles"]] == [
        "我现在不太想聊"
    ]
    assert "绝不能发给用户" not in json.dumps(result.output["reply"], ensure_ascii=False)


def test_fused_reply_silence_never_parses_private_content_as_bubble(
    tmp_path: Path,
) -> None:
    class Runtime:
        def generate_raw(self, **_kwargs: object) -> str:
            return json.dumps(
                {
                    "private_cognition": {
                        "private_content": "<bubble>私密想法</bubble>",
                        "subjective_feelings": {},
                        "attention_target": {},
                        "desired_actions": [],
                        "suggested_next_wakeup": None,
                    },
                    "expression_decision": {"express": False, "reason": "保持沉默"},
                    "public_compact_bubbles": None,
                },
                ensure_ascii=False,
            )

    result = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "fused_reply",
            {"system_prompt": "系统", "messages": [], "allowed_sticker_ids": []},
        )
    )

    assert result.output["expression_decision"]["express"] is False
    assert result.output["reply"] is None


def test_fused_reply_requests_a_short_private_judgment_for_low_latency(
    tmp_path: Path,
) -> None:
    prompts: list[str] = []

    class Runtime:
        def generate_raw(self, **kwargs: object) -> str:
            messages = kwargs["messages"]
            assert isinstance(messages, list)
            prompts.append(messages[0]["content"])
            return json.dumps(
                {
                        "p": "先回应",
                        "e": True,
                        "o": "<bubble>好</bubble>",
                },
                ensure_ascii=False,
            )

    DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "fused_reply",
            {"system_prompt": "系统", "messages": [], "allowed_sticker_ids": []},
        )
    )

    assert "p为少于15字" in prompts[0]
    assert "绝不公开" in prompts[0]


def test_fused_reply_accepts_strict_compact_public_output_without_retry(
    tmp_path: Path,
) -> None:
    calls = 0

    class Runtime:
        def generate_raw(self, **_kwargs: object) -> str:
            nonlocal calls
            calls += 1
            return "<bubble>在呢</bubble><delay>0</delay>"

    result = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "fused_reply",
            {"system_prompt": "系统", "messages": [], "allowed_sticker_ids": []},
        )
    )

    assert calls == 1
    assert result.output["private_cognition"]["private_content"] == "决定回应"
    assert result.output["expression_decision"]["express"] is True
    assert result.output["reply"]["bubbles"][0]["content"] == "在呢"


def test_fused_reply_normalizes_concise_private_fields_without_leaking_them(
    tmp_path: Path,
) -> None:
    class Runtime:
        def generate_raw(self, **_kwargs: object) -> str:
            return json.dumps(
                {
                    "private_cognition": {
                        "private_content": "我想先确认她的意思",
                        "subjective_feelings": "好奇和谨慎",
                        "attention_target": "用户的意图",
                        "desired_actions": ["先观察", "再回应"],
                        "suggested_next_wakeup": "等待用户进一步说明",
                    },
                    "expression_decision": {"express": False},
                    "public_compact_bubbles": None,
                },
                ensure_ascii=False,
            )

    result = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "fused_reply",
            {"system_prompt": "系统", "messages": [], "allowed_sticker_ids": []},
        )
    )

    private = result.output["private_cognition"]
    assert private["subjective_feelings"] == {"summary": "好奇和谨慎"}
    assert private["attention_target"] == {"summary": "用户的意图"}
    assert private["desired_actions"] == [
        {"content": "先观察"},
        {"content": "再回应"},
    ]
    assert private["suggested_next_wakeup"] is None
    assert result.output["reply"] is None


def test_fused_reply_normalizes_nonsemantic_container_noise_with_bounded_budget(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class Runtime:
        def generate_raw(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return json.dumps(
                {
                    "private_cognition": {
                        "private_content": "先自然回应",
                        "subjective_feelings": {},
                        "attention_target": {},
                        "desired_actions": [],
                        "suggested_next_wakeup": 15,
                        "structured_changes": [],
                    },
                    "expression_decision": {"express": True},
                    "public_compact_bubbles": "<bubble>好呀</bubble>",
                },
                ensure_ascii=False,
            )

    result = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "fused_reply",
            {"system_prompt": "系统", "messages": [], "allowed_sticker_ids": []},
        )
    )

    assert result.output["private_cognition"]["suggested_next_wakeup"] is None
    assert result.output["private_cognition"]["structured_changes"] == {}
    assert calls[0]["max_tokens"] == 96


def test_fused_reply_accepts_compact_private_decision_public_protocol(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class Runtime:
        def generate_raw(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return json.dumps(
                {
                    "private": "想自然回应",
                    "express": True,
                    "public": "<bubble>好呀</bubble>",
                },
                ensure_ascii=False,
            )

    result = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "fused_reply",
            {"system_prompt": "系统", "messages": [], "allowed_sticker_ids": []},
        )
    )

    assert result.output["private_cognition"]["private_content"] == "想自然回应"
    assert result.output["expression_decision"]["express"] is True
    assert result.output["reply"]["bubbles"][0]["content"] == "好呀"
    assert calls[0]["max_tokens"] == 96


def test_fused_reply_retries_once_after_invalid_structure(tmp_path: Path) -> None:
    outputs = iter(
        (
            "无效结构",
            json.dumps(
                {
                    "private": "想回应",
                    "express": True,
                    "public": "<bubble>收到</bubble>",
                },
                ensure_ascii=False,
            ),
        )
    )
    calls: list[dict[str, object]] = []

    class Runtime:
        def generate_raw(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return next(outputs)

    result = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "fused_reply",
            {"system_prompt": "系统", "messages": [], "allowed_sticker_ids": []},
        )
    )

    assert result.output["reply"]["bubbles"][0]["content"] == "收到"
    assert len(calls) == 2
    assert "输出结构无效" in calls[1]["messages"][0]["content"]


def test_fused_reply_allows_final_protocol_retry(tmp_path: Path) -> None:
    outputs = iter(
        (
            "无效结构一",
            "无效结构二",
            '{"private":"想回应","express":true,"public":"<bubble>好</bubble>"}',
        )
    )

    class Runtime:
        def generate_raw(self, **_kwargs: object) -> str:
            return next(outputs)

    result = DatabasePersonaInferenceBackend(
        _database(tmp_path),
        Runtime(),
    ).infer(
        _request(
            "fused_reply",
            {"system_prompt": "系统", "messages": [], "allowed_sticker_ids": []},
        )
    )

    assert result.output["reply"]["bubbles"][0]["content"] == "好"


@pytest.mark.parametrize(
    "public_output",
    [
        None,
        "<sticker>not-allowed</sticker>",
        "<bubble></bubble>",
    ],
)
def test_fused_reply_express_true_requires_valid_compact_public_bubble(
    tmp_path: Path,
    public_output: str | None,
) -> None:
    class Runtime:
        def generate_raw(self, **_kwargs: object) -> str:
            return json.dumps(
                {
                    "private_cognition": {
                        "private_content": "想回应",
                        "subjective_feelings": {},
                        "attention_target": {},
                        "desired_actions": [],
                        "suggested_next_wakeup": None,
                    },
                    "expression_decision": {"express": True},
                    "public_compact_bubbles": public_output,
                },
                ensure_ascii=False,
            )

    backend = DatabasePersonaInferenceBackend(_database(tmp_path), Runtime())

    with pytest.raises((GenerationFailedError, ReplyStructureError)):
        backend.infer(
            _request(
                "fused_reply",
                {
                    "system_prompt": "系统",
                    "messages": [],
                    "allowed_sticker_ids": ["allowed"],
                },
            )
        )
