"""可运行时更新的模型服务配置；密钥永不返回浏览器。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from moonlightbox.config import Settings

_AGENT_FIELDS = {
    f"{prefix}_{suffix}"
    for prefix in ("cognition", "node_analysis")
    for suffix in ("model", "endpoint", "api_key")
} | {"cognition_backend"}


def _valid_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    url = urlsplit(normalized)
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("填写有效的 HTTP API 地址，不能包含密钥或查询参数")
    return normalized


class AgentSettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    model: str = Field(min_length=1, max_length=200)
    endpoint: str = Field(min_length=1, max_length=2000)
    api_key: SecretStr | None = None

    @field_validator("model")
    @classmethod
    def trim_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("模型名称不能为空")
        return value.strip()

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _valid_url(value)


class LightRAGSettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    llm_model: str = Field(min_length=1, max_length=200)
    llm_endpoint: str = Field(min_length=1, max_length=2000)
    llm_api_key: SecretStr | None = None
    embedding_model: str = Field(min_length=1, max_length=200)
    embedding_endpoint: str = Field(min_length=1, max_length=2000)
    embedding_dimension: int = Field(ge=1, le=16384)
    embedding_api_key: SecretStr | None = None

    @field_validator("llm_model", "embedding_model")
    @classmethod
    def trim_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("模型名称不能为空")
        return value.strip()

    @field_validator("llm_endpoint", "embedding_endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _valid_url(value)


class AllModelSettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    agent: AgentSettingsInput
    lightrag: LightRAGSettingsInput


class ConnectionTestInput(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    service: str
    model: str = Field(min_length=1, max_length=200)
    endpoint: str = Field(min_length=1, max_length=2000)
    api_key: SecretStr | None = None
    embedding_dimension: int | None = Field(default=None, ge=1, le=16384)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        return _valid_url(value)


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_write(path: Path, value: dict[str, object], *, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=prefix, dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def apply_saved_agent_settings(settings: Settings) -> None:
    """把共享文件中的 Agent 配置更新到现有 Settings 对象。"""

    value = {
        key: item
        for key, item in _read_json(settings.data_dir / "agent-model-settings.json").items()
        if key in _AGENT_FIELDS
    }
    if not value:
        return
    validated = Settings(**{**settings.model_dump(), **value})
    for key in _AGENT_FIELDS:
        if key in value:
            setattr(settings, key, getattr(validated, key))


def _secret(value: object) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value().strip()
    return str(value or "").strip()


def create_model_settings_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/settings", tags=["settings"])
    agent_path = settings.data_dir / "agent-model-settings.json"
    lightrag_path = settings.data_dir / "lightrag-model-settings.json"

    def sidecar_headers() -> dict[str, str]:
        return {
            "Authorization": f"Bearer {settings.lightrag_sidecar_token.get_secret_value()}"
        }

    def active_graph_config() -> dict[str, object]:
        try:
            result = httpx.get(
                f"{settings.lightrag_sidecar_url.rstrip('/')}/v1/config",
                headers=sidecar_headers(),
                timeout=3,
            )
            result.raise_for_status()
            value = result.json()
        except (httpx.HTTPError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def response() -> dict[str, object]:
        agent = _read_json(agent_path)
        graph = _read_json(lightrag_path)
        active_graph = active_graph_config()
        active_agent_key = settings.resolved_cognition_api_key()
        return {
            "agent": {
                "model": agent.get("cognition_model", settings.cognition_model),
                "endpoint": agent.get("cognition_endpoint", settings.cognition_endpoint),
                "key_configured": bool(agent.get("cognition_api_key") or active_agent_key),
            },
            "lightrag": {
                "llm_model": graph.get("llm_model", active_graph.get("llm_model", "")),
                "llm_endpoint": graph.get(
                    "llm_endpoint", active_graph.get("llm_base_url", "")
                ),
                "llm_key_configured": bool(
                    graph.get("llm_api_key")
                    or active_graph.get("llm_key_configured")
                    or os.environ.get("LIGHTRAG_SIDECAR_LLM_API_KEY")
                ),
                "embedding_model": graph.get(
                    "embedding_model", active_graph.get("embedding_model", "")
                ),
                "embedding_endpoint": graph.get(
                    "embedding_endpoint", active_graph.get("embedding_base_url", "")
                ),
                "embedding_dimension": graph.get(
                    "embedding_dimension", active_graph.get("embedding_dimension", 1024)
                ),
                "embedding_key_configured": bool(
                    graph.get("embedding_api_key")
                    or active_graph.get("embedding_key_configured")
                    or os.environ.get("LIGHTRAG_SIDECAR_EMBEDDING_API_KEY")
                ),
            },
        }

    @router.get("")
    def read_all() -> dict[str, object]:
        return response()

    @router.get("/model")
    def read_model() -> dict[str, object]:
        return dict(response()["agent"])  # type: ignore[arg-type]

    @router.get("/lightrag")
    def read_lightrag() -> dict[str, object]:
        return dict(response()["lightrag"])  # type: ignore[arg-type]

    def build_agent_value(payload: AgentSettingsInput) -> dict[str, object]:
        value = _read_json(agent_path)
        key = _secret(payload.api_key)
        for prefix in ("cognition", "node_analysis"):
            value[f"{prefix}_model"] = payload.model
            value[f"{prefix}_endpoint"] = payload.endpoint
            if key:
                value[f"{prefix}_api_key"] = key
        value["cognition_backend"] = "deepseek"
        return value

    def build_lightrag_value(payload: LightRAGSettingsInput) -> dict[str, object]:
        previous = _read_json(lightrag_path)
        value: dict[str, object] = {
            "llm_model": payload.llm_model,
            "llm_endpoint": payload.llm_endpoint,
            "embedding_model": payload.embedding_model,
            "embedding_endpoint": payload.embedding_endpoint,
            "embedding_dimension": payload.embedding_dimension,
        }
        for field in ("llm_api_key", "embedding_api_key"):
            key = _secret(getattr(payload, field))
            if key:
                value[field] = key
            elif previous.get(field):
                value[field] = previous[field]
        return value

    def reload_sidecar(value: dict[str, object]) -> None:
        try:
            result = httpx.put(
                f"{settings.lightrag_sidecar_url.rstrip('/')}/v1/config",
                headers=sidecar_headers(),
                json={
                    "llm_model": value["llm_model"],
                    "llm_base_url": value["llm_endpoint"],
                    "embedding_model": value["embedding_model"],
                    "embedding_base_url": value["embedding_endpoint"],
                    "embedding_dimension": value["embedding_dimension"],
                    "llm_api_key": value.get("llm_api_key"),
                    "embedding_api_key": value.get("embedding_api_key"),
                },
                timeout=15,
            )
            result.raise_for_status()
        except httpx.HTTPStatusError as error:
            status_code = 422 if error.response.status_code < 500 else 503
            message = (
                "图谱模型配置无效，请检查地址、模型、维度和 API Key。"
                if status_code == 422
                else "图谱服务暂时无法应用配置，请稍后重试。"
            )
            raise HTTPException(status_code, message) from error
        except httpx.HTTPError as error:
            raise HTTPException(
                503, "图谱服务当前不可达，配置未保存；请确认服务运行后重试。"
            ) from error

    def save_all(payload: AllModelSettingsInput) -> dict[str, object]:
        agent = build_agent_value(payload.agent)
        graph = build_lightrag_value(payload.lightrag)
        try:
            Settings(**{**settings.model_dump(), **agent})
        except ValidationError:
            raise HTTPException(422, "Agent 模型配置无效") from None
        reload_sidecar(graph)
        _atomic_write(agent_path, agent, prefix=".agent-model-")
        _atomic_write(lightrag_path, graph, prefix=".lightrag-model-")
        apply_saved_agent_settings(settings)
        return response()

    @router.put("")
    def update_all(payload: AllModelSettingsInput) -> dict[str, object]:
        return save_all(payload)

    @router.put("/model")
    def save_model(payload: AgentSettingsInput) -> dict[str, object]:
        agent = build_agent_value(payload)
        try:
            Settings(**{**settings.model_dump(), **agent})
        except ValidationError:
            raise HTTPException(422, "Agent 模型配置无效") from None
        _atomic_write(agent_path, agent, prefix=".agent-model-")
        apply_saved_agent_settings(settings)
        return dict(response()["agent"])  # type: ignore[arg-type]

    @router.put("/lightrag")
    def save_lightrag(payload: LightRAGSettingsInput) -> dict[str, object]:
        graph = build_lightrag_value(payload)
        reload_sidecar(graph)
        _atomic_write(lightrag_path, graph, prefix=".lightrag-model-")
        return dict(response()["lightrag"])  # type: ignore[arg-type]

    @router.post("/test")
    def test_connection(payload: ConnectionTestInput) -> dict[str, object]:
        if payload.service not in {"agent", "lightrag", "embedding"}:
            raise HTTPException(422, "未知的服务类型")
        saved_agent = _read_json(agent_path)
        supplied = _secret(payload.api_key)
        if payload.service != "agent":
            try:
                result = httpx.post(
                    f"{settings.lightrag_sidecar_url.rstrip('/')}/v1/config:test",
                    headers=sidecar_headers(),
                    json={
                        "service": payload.service,
                        "model": payload.model,
                        "base_url": payload.endpoint,
                        "api_key": supplied or None,
                        "embedding_dimension": payload.embedding_dimension,
                    },
                    timeout=35,
                )
            except httpx.HTTPError as error:
                raise HTTPException(502, "无法连接图谱服务，请检查服务是否运行") from error
            if result.status_code >= 400:
                raise HTTPException(
                    422 if result.status_code < 500 else 502,
                    f"服务连接失败（HTTP {result.status_code}）",
                )
            return {"ok": True, "message": "连接成功"}

        key = supplied or _secret(saved_agent.get("cognition_api_key")) or _secret(
            settings.resolved_cognition_api_key()
        )
        if not key:
            raise HTTPException(422, "请先填写或保存 API Key")
        endpoint = payload.endpoint.rstrip("/")
        if payload.service == "embedding":
            endpoint = endpoint if endpoint.endswith("/embeddings") else f"{endpoint}/embeddings"
            body: dict[str, object] = {"model": payload.model, "input": ["连接测试"]}
            if payload.embedding_dimension:
                body["dimensions"] = payload.embedding_dimension
        else:
            endpoint = (
                endpoint
                if endpoint.endswith("/chat/completions")
                else f"{endpoint}/chat/completions"
            )
            body = {
                "model": payload.model,
                "messages": [{"role": "user", "content": "仅回复 OK"}],
                "max_tokens": 8,
                "stream": False,
            }
        try:
            result = httpx.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body,
                timeout=30,
            )
        except httpx.HTTPError as error:
            raise HTTPException(502, "无法连接该服务，请检查地址和网络") from error
        if result.status_code >= 400:
            raise HTTPException(422, f"服务连接失败（HTTP {result.status_code}）")
        return {"ok": True, "message": "连接成功"}

    return router
