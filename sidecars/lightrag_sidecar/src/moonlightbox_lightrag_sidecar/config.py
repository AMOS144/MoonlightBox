from pathlib import Path

import httpx
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SidecarSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LIGHTRAG_SIDECAR_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    storage_dir: Path = Path("data/lightrag")
    api_token: SecretStr = SecretStr("development-only-token")
    host: str = "127.0.0.1"
    port: int = Field(default=9621, ge=1, le=65535)
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4.1-mini"
    llm_api_key: SecretStr | None = None
    llm_reasoning_split: bool = False
    llm_timeout_seconds: int = Field(default=3600, ge=1, le=3600)
    # Reasoning models can consume their entire default completion budget
    # before emitting the required extraction/summary text. Keep an explicit
    # budget matching the project's output contract.
    llm_max_tokens: int = Field(default=32768, gt=0, le=65536)
    # Indexing keeps every extracted description verbatim. Automatic
    # map/reduce summaries are deferred to the world-profile compiler, where
    # the dedicated fidelity prompt has temporal and speaker constraints.
    summary_context_size: int = Field(default=100000, gt=0, le=200000)
    summary_max_tokens: int = Field(default=100000, gt=0, le=200000)
    force_llm_summary_on_merge: int = Field(default=100000, gt=0, le=200000)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = Field(default=3072, gt=0)
    embedding_max_tokens: int = Field(default=8191, gt=0)
    embedding_api_key: SecretStr | None = None
    chunk_token_size: int = Field(default=1200, gt=0)
    chunk_overlap_token_size: int = Field(default=100, ge=0)
    max_parallel_insert: int = Field(default=3, ge=1, le=10)
    max_async_llm: int = Field(default=4, ge=1, le=32)

    @field_validator("llm_base_url", "embedding_base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        try:
            parsed = httpx.URL(normalized)
        except httpx.InvalidURL:
            raise ValueError("模型服务地址必须是有效的 http(s) URL") from None
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("模型服务地址必须是有效的 http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("模型服务地址不能包含用户凭据")
        return normalized

    @field_validator("llm_model", "embedding_model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("模型名称不能为空")
        return normalized

    def ensure_storage_dir(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def require_model_keys(self) -> tuple[str, str]:
        llm_key = self.llm_api_key.get_secret_value() if self.llm_api_key else ""
        embedding_key = (
            self.embedding_api_key.get_secret_value() if self.embedding_api_key else llm_key
        )
        if not llm_key.strip():
            raise RuntimeError("未配置 LIGHTRAG_SIDECAR_LLM_API_KEY")
        if not embedding_key.strip():
            raise RuntimeError("未配置 LIGHTRAG_SIDECAR_EMBEDDING_API_KEY")
        return llm_key, embedding_key
