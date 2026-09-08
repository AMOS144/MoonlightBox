import io
import json
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError

import pytest
from moonlightbox.agent.inference import InferencePriority
from moonlightbox.agent.inference_client import PersonaInferenceClient
from moonlightbox.agent.types import CognitionRequest
from moonlightbox.branches.generation import GeneratorUnavailableError


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode()

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_client_deserializes_reply_and_sends_unique_realtime_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def urlopen(request: object, *, timeout: float) -> FakeResponse:
        captured.extend((request, timeout))
        return FakeResponse(
            {
                "request_id": "ignored-by-client",
                "request_type": "reply",
                "output": {
                    "bubbles": [
                        {
                            "type": "text",
                            "content": "你好",
                            "delay_ms": 0,
                            "asset_id": None,
                        }
                    ],
                    "raw_output": "<bubble>你好</bubble>",
                },
            }
        )

    monkeypatch.setattr("moonlightbox.agent.inference_client.urlopen", urlopen)
    client = PersonaInferenceClient("http://127.0.0.1:8765", "token", timeout=3)
    reply = client.generate("model-1", "系统", [{"role": "user", "content": "在吗"}])

    request = captured[0]
    body = json.loads(request.data)
    assert request.full_url.endswith("/v1/inference/reply")
    assert request.headers["Authorization"] == "Bearer token"
    assert body["priority"] == InferencePriority.REALTIME
    assert body["request_id"]
    assert [bubble.content for bubble in reply.bubbles] == ["你好"]
    assert reply.raw_output == "<bubble>你好</bubble>"


def test_client_forwards_deterministic_sampling_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def urlopen(request: object, *, timeout: float) -> FakeResponse:
        del timeout
        captured.append(request)
        return FakeResponse(
            {
                "request_id": "reply",
                "request_type": "reply",
                "output": {
                    "bubbles": [{"type": "text", "content": "好", "delay_ms": 0}]
                },
            }
        )

    monkeypatch.setattr("moonlightbox.agent.inference_client.urlopen", urlopen)
    client = PersonaInferenceClient("http://127.0.0.1:8765", "token")
    client.set_seed(17)

    client.generate("model-1", "系统", [{"role": "user", "content": "在吗"}])

    assert json.loads(captured[0].data)["payload"]["sampling_seed"] == 17


def test_client_rewrites_trusted_content_draft_through_persona_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def urlopen(request: object, *, timeout: float) -> FakeResponse:
        del timeout
        captured.append(request)
        return FakeResponse(
            {
                "request_id": "rewrite",
                "request_type": "reply",
                "output": {
                    "bubbles": [
                        {"type": "text", "content": "我今晚不去呀", "delay_ms": 0}
                    ]
                },
            }
        )

    monkeypatch.setattr("moonlightbox.agent.inference_client.urlopen", urlopen)
    rewritten = PersonaInferenceClient(
        "http://127.0.0.1:8765",
        "token",
    ).rewrite_content_draft("model-1", "我今晚不去")

    assert rewritten is not None
    assert rewritten.bubbles[0].content == "我今晚不去呀"
    body = json.loads(captured[0].data)
    assert body["payload"]["messages"] == [
        {"role": "user", "content": "内容草稿：\n我今晚不去"}
    ]
    assert "聊天文字本身" in body["payload"]["system_prompt"]


def test_client_uses_example_backed_rhythm_when_lora_copies_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "moonlightbox.agent.inference_client.urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "request_id": "rewrite",
                "request_type": "reply",
                "output": {
                    "bubbles": [
                        {
                            "type": "text",
                            "content": "别急，我们一起想想去哪里好。",
                            "delay_ms": 0,
                        }
                    ]
                },
            }
        ),
    )

    rewritten = PersonaInferenceClient(
        "http://127.0.0.1:8765",
        "token",
    ).rewrite_content_draft_with_examples(
        "model-1",
        "别急，我们一起想想去哪里好。",
        ("对方：这该如何选择呢\n本人：哎呀 / 现在很多男女同款",),
    )

    assert rewritten is not None
    assert [bubble.content for bubble in rewritten.bubbles] == [
        "哎呀",
        "别急",
        "我们一起想想去哪里好",
    ]


def test_client_deserializes_cognition_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "moonlightbox.agent.inference_client.urlopen",
        lambda *_args, **_kwargs: FakeResponse(
            {
                "request_id": "request-2",
                "request_type": "cognition",
                "output": {
                    "private_content": "我有点担心她",
                    "subjective_feelings": {"worry": 0.6},
                    "attention_target": {"topic": "sleep"},
                    "desired_actions": [{"type": "wait"}],
                    "expression_decision": {
                        "express": False,
                        "content": None,
                        "reason": "现在先不打扰",
                    },
                    "suggested_next_wakeup": None,
                    "structured_changes": {},
                    "confidence": 0.8,
                },
            }
        ),
    )
    request = CognitionRequest(
        project_id="project-1",
        branch_id="branch-1",
        trigger_event={"type": "silence"},
        deadline=datetime.now(UTC) + timedelta(seconds=5),
        current_mental_state={"mood": "calm"},
        goals=(),
        relevant_context=(),
    )

    draft = PersonaInferenceClient(
        "http://127.0.0.1:8765", "token"
    ).generate(request, model_version_id="model-1")

    assert draft.private_content == "我有点担心她"
    assert draft.expression_decision.express is False
    assert draft.confidence == 0.8


def test_client_uses_remote_runtime_director_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def urlopen(request: object, *, timeout: float) -> FakeResponse:
        del timeout
        captured.append(request)
        return FakeResponse(
            {
                "request_id": "director-1",
                "request_type": "runtime_director",
                "output": {
                    "content": '{"action":"wait","private_reason":"证据不足"}'
                },
            }
        )

    monkeypatch.setattr("moonlightbox.agent.inference_client.urlopen", urlopen)
    content = PersonaInferenceClient(
        "http://persona-runtime:8765", "shared-token"
    ).generate_runtime_director(
        "model-1", "Director 系统提示", [{"role": "user", "content": "上下文"}]
    )

    assert content.startswith('{"action":"wait"')
    request = captured[0]
    assert request.full_url.endswith("/v1/inference/runtime-director")
    assert json.loads(request.data)["request_type"] == "runtime_director"


def test_client_uses_runtime_tokenizer_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def urlopen(request: object, *, timeout: float) -> FakeResponse:
        del timeout
        captured.append(request)
        return FakeResponse(
            {
                "request_id": "token-count",
                "request_type": "runtime_token_count",
                "output": {"count": 42},
            }
        )

    monkeypatch.setattr("moonlightbox.agent.inference_client.urlopen", urlopen)
    count = PersonaInferenceClient("http://127.0.0.1:8765", "token").count_runtime_director_tokens(
        "model-1", "<runtime_context>{}</runtime_context>"
    )

    request = captured[0]
    body = json.loads(request.data)
    assert request.full_url.endswith("/v1/inference/runtime-token-count")
    assert body["request_type"] == "runtime_token_count"
    assert count == 42


def test_one_client_uses_same_authenticated_protocol_for_reply_and_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def urlopen(request: object, *, timeout: float) -> FakeResponse:
        captured.append(request)
        if request.full_url.endswith("/v1/inference/reply"):
            return FakeResponse(
                {
                    "request_id": "reply",
                    "request_type": "reply",
                    "output": {
                        "bubbles": [
                            {"type": "text", "content": "收到", "delay_ms": 0}
                        ]
                    },
                }
            )
        return FakeResponse(
            {
                "request_id": "cognition",
                "request_type": "cognition",
                "output": {
                    "private_content": "我先想想",
                    "subjective_feelings": {},
                    "attention_target": {},
                    "desired_actions": [],
                    "expression_decision": {"express": False},
                    "suggested_next_wakeup": None,
                },
            }
        )

    monkeypatch.setattr("moonlightbox.agent.inference_client.urlopen", urlopen)
    client = PersonaInferenceClient("http://persona-runtime:8765", "shared-token")
    client.generate("model-1", "系统", [{"role": "user", "content": "你好"}])
    client.generate(
        CognitionRequest(
            project_id="project-1",
            branch_id="branch-1",
            model_version_id="model-1",
            trigger_event={"event_type": "user_message"},
            deadline=datetime.now(UTC) + timedelta(seconds=5),
            current_mental_state={},
            goals=(),
            relevant_context=(),
        )
    )

    assert [request.full_url for request in captured] == [
        "http://persona-runtime:8765/v1/inference/reply",
        "http://persona-runtime:8765/v1/inference/cognition",
    ]
    assert all(
        request.headers["Authorization"] == "Bearer shared-token"
        for request in captured
    )
    assert [json.loads(request.data)["model_version_id"] for request in captured] == [
        "model-1",
        "model-1",
    ]
    assert json.loads(captured[1].data)["payload"]["shadow_mode"] is True


def test_client_converts_http_error_to_safe_existing_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> FakeResponse:
        raise HTTPError(
            "http://127.0.0.1",
            503,
            "secret server detail",
            {},
            io.BytesIO(b'{"detail":"private path"}'),
        )

    monkeypatch.setattr("moonlightbox.agent.inference_client.urlopen", fail)
    client = PersonaInferenceClient("http://127.0.0.1:8765", "token")

    with pytest.raises(GeneratorUnavailableError, match="人格推理服务不可用") as caught:
        client.generate("model-1", "系统", [])
    assert "private path" not in str(caught.value)


def test_client_generate_fused_uses_realtime_endpoint_and_deserializes_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def urlopen(request: object, *, timeout: float) -> FakeResponse:
        captured.extend((request, timeout))
        return FakeResponse(
            {
                "request_id": "fused-1",
                "request_type": "fused_reply",
                "output": {
                    "private_cognition": {
                        "private_content": "我先不回应",
                        "subjective_feelings": {"犹豫": 0.5},
                        "attention_target": {},
                        "desired_actions": [{"type": "wait"}],
                        "suggested_next_wakeup": None,
                    },
                    "expression_decision": {
                        "express": False,
                        "reason": "还想再观察",
                    },
                    "reply": None,
                },
            }
        )

    monkeypatch.setattr("moonlightbox.agent.inference_client.urlopen", urlopen)
    turn = PersonaInferenceClient(
        "http://persona-runtime:8765",
        "shared-token",
    ).generate_fused(
        "model-1",
        "系统",
        [{"role": "user", "content": "在吗"}],
        allowed_sticker_ids=("sticker-1",),
    )

    request = captured[0]
    body = json.loads(request.data)
    assert request.full_url == "http://persona-runtime:8765/v1/inference/fused-reply"
    assert body["request_type"] == "fused_reply"
    assert body["priority"] == InferencePriority.REALTIME
    assert body["payload"]["allowed_sticker_ids"] == ["sticker-1"]
    assert turn.cognition.private_content == "我先不回应"
    assert turn.expression_decision.express is False
    assert turn.reply is None
