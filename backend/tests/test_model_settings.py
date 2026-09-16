import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
from moonlightbox.config import Settings
from moonlightbox.model_settings import create_model_settings_router
from moonlightbox.runtime_v1.cloud_models import RuntimeCloudClient


def test_save_updates_runtime_without_exposing_key(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, cognition_model="old", cognition_api_key="old-secret")
    runtime_client = RuntimeCloudClient.from_settings(settings)
    app = FastAPI()
    app.include_router(create_model_settings_router(settings))
    with TestClient(app) as client:
        assert "old-secret" not in client.get('/api/settings/model').text
        result = client.put(
            "/api/settings/model",
            json={
                "model": "MiniMax-M3",
                "endpoint": "https://api.minimaxi.com/v1/chat/completions",
                "api_key": "new-secret",
            },
        )
        assert result.status_code == 200
        assert 'new-secret' not in result.text
        assert settings.cognition_model == 'MiniMax-M3'
        assert settings.node_analysis_model == 'MiniMax-M3'
        assert settings.cognition_api_key.get_secret_value() == 'new-secret'
        runtime_client._refresh_runtime_settings()
        assert runtime_client.model_name == "MiniMax-M3"
        assert runtime_client._api_key == "new-secret"
        path = tmp_path / 'agent-model-settings.json'
        assert path.stat().st_mode & 0o777 == 0o600
        client.put(
            "/api/settings/model",
            json={
                "model": "MiniMax-M3",
                "endpoint": "https://api.minimaxi.com/v1/chat/completions",
                "api_key": None,
            },
        )
        assert json.loads(path.read_text())['cognition_api_key'] == 'new-secret'
        monkeypatch.setenv('MOONLIGHTBOX_DATA_DIR', str(tmp_path))
        monkeypatch.setenv('MOONLIGHTBOX_COGNITION_MODEL', 'env-old')
        restarted = Settings(_env_file=None)
        assert restarted.cognition_model == 'MiniMax-M3'
        assert restarted.node_analysis_model == 'MiniMax-M3'
        assert restarted.cognition_api_key.get_secret_value() == 'new-secret'
        assert Settings(data_dir=tmp_path, cognition_model='explicit').cognition_model == 'explicit'
    runtime_client.close()


def test_save_all_reloads_sidecar_and_preserves_blank_keys(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, cognition_model="old", cognition_api_key="old-key")
    calls = []

    class Response:
        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "moonlightbox.model_settings.httpx.put",
        lambda *args, **kwargs: (calls.append((args, kwargs)) or Response()),
    )
    app = FastAPI()
    app.include_router(create_model_settings_router(settings))
    payload = {
        "agent": {
            "model": "agent-new",
            "endpoint": "https://agent.test/v1/chat/completions",
            "api_key": "agent-key",
        },
        "lightrag": {
            "llm_model": "graph-new",
            "llm_endpoint": "https://graph.test/v1",
            "llm_api_key": "graph-key",
            "embedding_model": "embed-new",
            "embedding_endpoint": "https://embed.test/v1",
            "embedding_dimension": 1024,
            "embedding_api_key": "embed-key",
        },
    }
    with TestClient(app) as client:
        result = client.put("/api/settings", json=payload)
        assert result.status_code == 200
        assert "agent-key" not in result.text
        assert "graph-key" not in result.text
        assert "embed-key" not in result.text
        assert client.get("/api/settings").json()["lightrag"]["llm_key_configured"]
    assert settings.cognition_model == "agent-new"
    assert calls[0][1]["json"]["llm_model"] == "graph-new"
    assert (tmp_path / "lightrag-model-settings.json").stat().st_mode & 0o777 == 0o600


def test_invalid_endpoint_does_not_save(tmp_path):
    app = FastAPI()
    app.include_router(create_model_settings_router(Settings(data_dir=tmp_path)))
    with TestClient(app) as client:
        response = client.put('/api/settings/model', json={"model": "m", "endpoint": "file:///etc/passwd"})
        assert response.status_code == 422
        assert not (tmp_path / 'agent-model-settings.json').exists()
