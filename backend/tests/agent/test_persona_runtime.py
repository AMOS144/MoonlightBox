from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from moonlightbox.agent.inference import PersonaInferenceRequest, PersonaInferenceResult
from moonlightbox.config import Settings
from moonlightbox.persona_runtime import create_persona_runtime_app
from pydantic import SecretStr


class FakeBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[PersonaInferenceRequest] = []

    def infer(self, request: PersonaInferenceRequest) -> PersonaInferenceResult:
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("不应暴露的模型路径 /private/model")
        return PersonaInferenceResult(
            request_id=request.request_id,
            request_type=request.request_type,
            output={"bubbles": [{"type": "text", "content": "你好", "delay_ms": 0}]},
        )


def _settings() -> Settings:
    return Settings.model_construct(
        persona_inference_token=SecretStr("test-token"),
        persona_inference_timeout_seconds=2.0,
    )


def _reply_payload() -> dict[str, object]:
    return {
        "request_id": "request-1",
        "request_type": "reply",
        "priority": 0,
        "deadline": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(),
        "model_version_id": "model-1",
        "payload": {"system_prompt": "系统", "messages": []},
    }


def test_runtime_requires_bearer_token_and_returns_structured_result() -> None:
    backend = FakeBackend()
    with TestClient(create_persona_runtime_app(_settings(), backend=backend)) as client:
        assert client.post("/v1/inference/reply", json=_reply_payload()).status_code == 401
        response = client.post(
            "/v1/inference/reply",
            json=_reply_payload(),
            headers={"Authorization": "Bearer test-token"},
        )
        health = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "request_id": "request-1",
        "request_type": "reply",
        "output": {"bubbles": [{"type": "text", "content": "你好", "delay_ms": 0}]},
    }
    assert health.json() == {"status": "ok"}
    assert backend.requests[0].priority == 0


def test_linux_runtime_default_backend_starts_without_macos_gate(tmp_path) -> None:
    """Linux 默认后端延迟加载模型，健康检查不依赖 Apple MLX。"""

    settings = Settings.model_construct(
        database_url=f"sqlite:///{tmp_path / 'persona-runtime.db'}",
        persona_inference_token=SecretStr("test-token"),
        persona_inference_timeout_seconds=2.0,
        persona_device="cpu",
        persona_load_in_4bit=False,
    )
    with TestClient(create_persona_runtime_app(settings)) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_runtime_hides_backend_exception_details() -> None:
    with TestClient(
        create_persona_runtime_app(_settings(), backend=FakeBackend(fail=True))
    ) as client:
        response = client.post(
            "/v1/inference/reply",
            json=_reply_payload(),
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "inference_failed", "message": "人格推理失败"}}
    assert "/private/model" not in response.text


def test_runtime_rejects_request_type_mismatching_endpoint() -> None:
    payload = _reply_payload()
    payload["request_type"] = "cognition"
    with TestClient(create_persona_runtime_app(_settings(), backend=FakeBackend())) as client:
        response = client.post(
            "/v1/inference/reply",
            json=payload,
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 422


def test_runtime_accepts_fused_reply_only_on_dedicated_endpoint() -> None:
    backend = FakeBackend()
    payload = _reply_payload()
    payload["request_type"] = "fused_reply"
    with TestClient(create_persona_runtime_app(_settings(), backend=backend)) as client:
        response = client.post(
            "/v1/inference/fused-reply",
            json=payload,
            headers={"Authorization": "Bearer test-token"},
        )
        mismatched = client.post(
            "/v1/inference/reply",
            json=payload,
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    assert backend.requests[0].request_type == "fused_reply"
    assert backend.requests[0].priority == 0
    assert mismatched.status_code == 422
