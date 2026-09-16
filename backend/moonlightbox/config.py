from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        def saved_model_settings():
            # 用户配置跟随共享数据目录，不修改仓库 .env；显式构造参数仍优先。
            import json
            sources = [init_settings(), env_settings(), dotenv_settings()]
            data_dir = next((v["data_dir"] for v in sources if "data_dir" in v), "data")
            path = Path(data_dir) / "agent-model-settings.json"
            if not path.exists():
                return {}
            value = json.loads(path.read_text(encoding="utf-8"))
            allowed = {f"{prefix}_{suffix}" for prefix in ("cognition", "node_analysis") for suffix in ("model", "endpoint", "api_key")}
            allowed.add("cognition_backend")
            return {k: v for k, v in value.items() if k in allowed}
        return init_settings, saved_model_settings, env_settings, dotenv_settings, file_secret_settings

    model_config = SettingsConfigDict(
        env_prefix="MOONLIGHTBOX_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    data_dir: Path = Path("data")
    # 体验环境可只读复用原项目的媒体库。写入始终留在 data_dir，绝不落到此目录。
    media_read_only_source_dir: Path | None = None
    # Phoenix 是所有 Agent 的统一可观测性后端。默认关闭，避免在未显式启动本地
    # Collector 时影响普通开发；启用后 API 与 Worker 都会向同一 OTLP/HTTP 端点发送 Trace。
    phoenix_enabled: bool = False
    phoenix_collector_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    phoenix_project_name: str = "moonlightbox"
    # 本项目是单用户本机体验环境，默认将 Agent 的输入、输出、工具返回和压缩快照导入
    # 自托管 Phoenix，以便完整回放。密钥仍会被强制脱敏；若需关闭正文导出可显式设 false。
    phoenix_capture_content: bool = True
    phoenix_trace_max_characters: int = Field(default=64_000, ge=512, le=100_000)
    # 根因摘要以 Phoenix CODE Annotation 回写；失败仅影响诊断标签，不影响 Agent 结果。
    phoenix_annotations_enabled: bool = True
    database_url: str = "sqlite:///data/moonlightbox.db"
    chroma_dir: Path = Path("data/chroma")
    model_dir: Path = Path("models")
    auto_create_schema: bool = False
    cloud_evaluation_enabled: bool = True
    cloud_evaluation_endpoint: str = "https://api.openai.com/v1/chat/completions"
    cloud_evaluation_model: str = "gpt-4.1-mini"
    cloud_evaluation_api_key: str | None = None
    node_analysis_enabled: bool = False
    node_analysis_endpoint: str = "https://api.openai.com/v1/chat/completions"
    node_analysis_model: str = "gpt-4.1-mini"
    node_analysis_api_key: SecretStr | None = None
    node_analysis_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    node_analysis_max_retries: int = Field(default=3, ge=0, le=10)
    node_analysis_backoff_seconds: float = Field(default=0.5, gt=0, le=60)
    node_analysis_max_backoff_seconds: float = Field(default=60.0, gt=0, le=300)
    node_analysis_max_retry_after_seconds: float = Field(
        default=3600.0,
        gt=0,
        le=86400,
    )
    node_analysis_response_format: Literal["json_schema", "json_object"] = "json_schema"
    # 节点分析、PersonWorld 等都是多步调查 Agent，默认保留模型思考；只有明确的
    # 低延迟批处理任务才可以通过环境变量关闭。
    node_analysis_thinking_mode: Literal["default", "disabled"] = "default"
    node_analysis_max_output_tokens: int = Field(default=8192, gt=0, le=65536)
    lightrag_enabled: bool = False
    lightrag_sidecar_url: str = "http://127.0.0.1:9621"
    lightrag_sidecar_token: SecretStr = SecretStr("development-only-token")
    lightrag_index_revision: str = Field(
        default="person-world-v1",
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    lightrag_timeout_seconds: float = Field(default=1800.0, gt=0, le=7200)
    lightrag_bundle_gap_hours: float = Field(default=6.0, gt=0, le=48)
    lightrag_bundle_max_characters: int = Field(default=12000, ge=1000, le=100000)
    lightrag_carry_in_turns: int = Field(default=3, ge=0, le=10)
    lightrag_query_top_k: int = Field(default=30, ge=1, le=100)
    lightrag_query_chunk_top_k: int = Field(default=12, ge=1, le=100)
    lightrag_query_max_total_tokens: int = Field(default=16000, ge=1000, le=100000)
    # Send 扇出并发度受限于 7；限流由统一请求退避处理，不自动修改此并发配置。
    person_world_section_concurrency: int = Field(default=3, ge=1, le=7)
    # PersonWorld 是持续调查循环：不按 Prompt 的“轮数”结束。以下只是一致的系统
    # 熔断器，分别处理总成本、重复调用循环与单栏目的墙钟时间。
    person_world_section_max_tool_calls: int = Field(default=48, ge=12, le=200)
    person_world_section_deadline_seconds: float = Field(default=900.0, ge=60, le=7200)
    person_world_section_max_stalled_cycles: int = Field(default=2, ge=1, le=8)
    person_world_section_model_timeout_seconds: float = Field(default=300.0, ge=10, le=1800)
    person_world_section_tool_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)
    world_compiler_timeout_seconds: float = Field(default=300.0, gt=0, le=7200)
    spatial_analysis_enabled: bool = False
    spatial_analysis_backend: Literal["auto", "local", "cloud", "structured_only"] = "cloud"
    spatial_amap_web_key: SecretStr | None = None
    spatial_amap_base_url: str = "https://restapi.amap.com"
    spatial_map_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    spatial_provider_persistence_allowed: bool = False
    spatial_heatmap_default_privacy: Literal["exact", "blurred", "hidden"] = "blurred"
    cognition_backend: Literal["local", "deepseek"] = "local"
    cognition_endpoint: str = "https://api.deepseek.com/chat/completions"
    cognition_model: str = "deepseek-v4-flash"
    cognition_api_key: SecretStr | None = None
    cognition_timeout_seconds: float = Field(default=300.0, gt=0, le=1800)
    cognition_max_retries: int = Field(default=2, ge=0, le=10)
    cognition_response_format: Literal["json_schema", "json_object"] = "json_object"
    # Director / DayPlanAgent / PersonaActor 都需要基于上下文规划与调用工具；默认
    # 开启 MiniMax M3 的思考。只有延迟优先的纯文本回复才应显式设为 disabled。
    cognition_thinking_mode: Literal["default", "disabled"] = "default"
    # M3 的 completion 预算同时包含思考与最终正文。Runtime Agent 不在各领域调用
    # 里另设小上限；统一保留足够预算，以免工具规划尚未输出就被截断。
    cognition_max_output_tokens: int = Field(default=24576, gt=0, le=65536)
    reply_review_enabled: bool = False
    # 中国法定工作日历按年度在线刷新并保存在共享 data 目录。URL 固定在实现中，
    # 不允许通过环境变量把 Runtime 指向任意站点。
    work_calendar_enabled: bool = True
    work_calendar_cache_dir: Path | None = None
    work_calendar_refresh_hours: float = Field(default=24.0, gt=0, le=168)
    work_calendar_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    # Linux 默认使用可被 Transformers 直接加载的模型；旧 MLX 路径文件不可用。
    # 固定到已验证的 HF commit，训练不能跟随可变的 main 标签。
    training_base_model: str = "Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    training_iterations: int = Field(default=600, gt=0)
    training_batch_size: int = Field(default=1, gt=0)
    training_learning_rate: float = Field(default=1e-5, gt=0)
    training_gradient_accumulation_steps: int = Field(default=4, gt=0)
    # Qwen3-1.7B 在 4GB GTX 1650 上做 QLoRA 的保守序列长度。
    training_max_seq_length: int = Field(default=512, ge=128, le=4096)
    training_maximum_event_ratio: float = Field(default=0.25, ge=0, le=1)
    training_style_transfer_ratio: float = Field(default=0.5, ge=0, le=1)
    training_recency_window_days: int = Field(default=10, ge=0)
    training_recency_multiplier: int = Field(default=2, ge=1)
    runtime_style_transfer_enabled: bool = False
    # 生活推进是正式运行时的一部分；这里只配置概率文件，不提供启停开关。
    runtime_life_policy_path: str | None = None
    training_context_turns: int = Field(default=12, gt=0)
    training_protocol_version: str = (
        "persona-plain-text-private-chat-v19-runtime-style-transfer-aligned"
    )
    checkpoint_selection_version: str = "person-identity-held-out-v11-role-separated"
    reply_protocol_version: str = "persona-text-v1"
    memory_protocol_version: str = "evidence-layered-temporal-v3-one-pass-preference"
    memory_preference_training_enabled: bool = True
    memory_preference_iterations: int = Field(default=40, gt=0)
    human_blind_required: bool = True
    human_blind_minimum_ratings: int = Field(default=20, ge=20)
    human_blind_minimum_preference: float = Field(default=0.5, ge=0, le=1)
    behavioral_timezone_offset_minutes: int = Field(default=480, ge=-720, le=840)
    persona_inference_url: str = "http://127.0.0.1:8765"
    # 仅用于本地开发，部署时必须通过环境变量覆盖。
    persona_inference_token: SecretStr = SecretStr("development-only-token")
    persona_inference_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    # auto 优先选择 CUDA；CPU 回退仅适用于开发与排障。
    persona_device: Literal["auto", "cuda", "cpu"] = "auto"
    # 在 4GB NVIDIA 显存上必须使用 4-bit 权重；8B 基座依然不适合本机。
    persona_load_in_4bit: bool = True

    @field_validator("node_analysis_endpoint")
    @classmethod
    def validate_node_analysis_endpoint(cls, value: str) -> str:
        normalized = value.strip()
        try:
            parsed = httpx.URL(normalized)
        except httpx.InvalidURL:
            raise ValueError("节点分析 endpoint 必须是有效的 http(s) URL") from None
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("节点分析 endpoint 必须是有效的 http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("节点分析 endpoint 不允许包含用户凭据")
        return normalized

    @field_validator("phoenix_collector_endpoint")
    @classmethod
    def validate_phoenix_collector_endpoint(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        try:
            parsed = httpx.URL(normalized)
        except httpx.InvalidURL:
            raise ValueError("Phoenix Collector 地址必须是有效的 http(s) URL") from None
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("Phoenix Collector 地址必须是有效的 http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("Phoenix Collector 地址不能包含用户凭据")
        if not parsed.path.endswith("/v1/traces"):
            raise ValueError("Phoenix Collector 地址必须以 /v1/traces 结尾")
        return normalized

    @field_validator("lightrag_sidecar_url")
    @classmethod
    def validate_lightrag_sidecar_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        try:
            parsed = httpx.URL(normalized)
        except httpx.InvalidURL:
            raise ValueError("LightRAG Sidecar 地址必须是有效的 http(s) URL") from None
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("LightRAG Sidecar 地址必须是有效的 http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("LightRAG Sidecar 地址不能包含用户凭据")
        return normalized

    @field_validator("node_analysis_model")
    @classmethod
    def validate_node_analysis_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("节点分析 model 不能为空")
        return normalized

    @field_validator("cognition_endpoint")
    @classmethod
    def validate_cognition_endpoint(cls, value: str) -> str:
        normalized = value.strip()
        try:
            parsed = httpx.URL(normalized)
        except httpx.InvalidURL:
            raise ValueError("认知 endpoint 必须是有效的 http(s) URL") from None
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("认知 endpoint 必须是有效的 http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("认知 endpoint 不允许包含用户凭据")
        return normalized

    @field_validator("cognition_model")
    @classmethod
    def validate_cognition_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("认知 model 不能为空")
        return normalized

    def resolved_cognition_api_key(self) -> SecretStr | None:
        """Allow cognition to reuse the separately protected DeepSeek key."""

        return self.cognition_api_key or self.node_analysis_api_key

    @field_validator("training_base_model")
    @classmethod
    def validate_training_base_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("训练基础模型不能为空")
        return normalized

    def training_config_snapshot(self) -> dict[str, object]:
        return {
            "base_model": self.training_base_model,
            "iterations": self.training_iterations,
            "batch_size": self.training_batch_size,
            "learning_rate": self.training_learning_rate,
            "gradient_accumulation_steps": (self.training_gradient_accumulation_steps),
            "max_seq_length": self.training_max_seq_length,
            "maximum_event_ratio": self.training_maximum_event_ratio,
            "style_transfer_ratio": self.training_style_transfer_ratio,
            "recency_window_days": self.training_recency_window_days,
            "recency_multiplier": self.training_recency_multiplier,
            "runtime_style_transfer_enabled": self.runtime_style_transfer_enabled,
            "context_turns": self.training_context_turns,
            "training_protocol_version": self.training_protocol_version,
            "checkpoint_selection_version": self.checkpoint_selection_version,
            "reply_protocol_version": self.reply_protocol_version,
            "memory_protocol_version": self.memory_protocol_version,
            "memory_preference_training_enabled": (self.memory_preference_training_enabled),
            "memory_preference_iterations": self.memory_preference_iterations,
            "human_blind_required": self.human_blind_required,
            "human_blind_minimum_ratings": self.human_blind_minimum_ratings,
            "human_blind_minimum_preference": self.human_blind_minimum_preference,
            "behavioral_timezone_offset_minutes": (self.behavioral_timezone_offset_minutes),
            "persona_device": self.persona_device,
            "persona_load_in_4bit": self.persona_load_in_4bit,
        }

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.chroma_dir, self.model_dir):
            directory.mkdir(parents=True, exist_ok=True)
