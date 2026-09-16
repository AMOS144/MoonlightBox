from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class RuntimeConfigUpdate(BaseModel):
    """运行时模型配置；密钥只进不出。"""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    llm_model: str = Field(min_length=1, max_length=200)
    llm_base_url: str = Field(min_length=1, max_length=2000)
    llm_api_key: SecretStr | None = None
    embedding_model: str = Field(min_length=1, max_length=200)
    embedding_base_url: str = Field(min_length=1, max_length=2000)
    embedding_dimension: int = Field(gt=0, le=16384)
    embedding_api_key: SecretStr | None = None


class RuntimeConfigResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    llm_model: str
    llm_base_url: str
    llm_key_configured: bool
    embedding_model: str
    embedding_base_url: str
    embedding_dimension: int
    embedding_key_configured: bool


class RuntimeConnectionTest(BaseModel):
    """使用候选参数测试服务；密钥为空时复用 Sidecar 当前密钥。"""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    service: Literal["lightrag", "embedding"]
    model: str = Field(min_length=1, max_length=200)
    base_url: str = Field(min_length=1, max_length=2000)
    api_key: SecretStr | None = None
    embedding_dimension: int | None = Field(default=None, gt=0, le=16384)


class MessageSpan(BaseModel):
    """冻结文档中的消息位置；字符按 Python Unicode 码点计数。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    message_id: str = Field(min_length=1)
    source_ordinal: int = Field(ge=0)
    sent_at: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class TemporalScope(BaseModel):
    """由调用方后端绑定，不属于模型可填写的工具参数。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    source_version: str = Field(min_length=1)
    included_count: int = Field(ge=0)
    period: Literal["before", "after", "all"]
    access: Literal["compiler", "runtime"] = "compiler"
    policy: Literal["mixed_to_before_v1"] = "mixed_to_before_v1"


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    source: str = Field(min_length=1, max_length=180, pattern=r"^[A-Za-z0-9_.-]+$")
    text: str = Field(min_length=1, max_length=500_000)
    source_version: str | None = None
    message_spans: list[MessageSpan] | None = None

    @model_validator(mode="after")
    def validate_spans(self):
        if (self.source_version is None) != (self.message_spans is None):
            raise ValueError("source_version 和 message_spans 必须同时提供")
        if self.message_spans is not None:
            cursor = 0
            seen = set()
            for span in self.message_spans:
                if (
                    span.start < cursor
                    or span.end <= span.start
                    or span.end > len(self.text)
                    or span.message_id in seen
                    or self.text[cursor : span.start].strip()
                ):
                    raise ValueError("消息位置重复、乱序、越界或存在未归属文本")
                # 旧 Bundle 的同秒文本顺序可能不同于起点清单；位置与时间序号分开保存。
                cursor = span.end
                seen.add(span.message_id)
            if not seen or self.text[cursor:].strip():
                raise ValueError("消息位置没有覆盖文档")
        return self


class DocumentBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    documents: list[DocumentInput] = Field(min_length=1, max_length=100)

    @field_validator("documents")
    @classmethod
    def validate_unique_documents(cls, value: list[DocumentInput]) -> list[DocumentInput]:
        ids = [item.id for item in value]
        sources = [item.source for item in value]
        if len(ids) != len(set(ids)) or len(sources) != len(set(sources)):
            raise ValueError("同一批次中的文档 ID 和 source 必须唯一")
        return value


class DocumentStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    document_ids: list[
        Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
    ] = Field(min_length=1, max_length=100)


class SidecarMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lightrag_version: str
    embedding_model: str
    embedding_dimension: int
    extraction_model: str
    chunking_strategy: str
    chunk_token_size: int
    chunk_overlap_token_size: int
    entity_prompt_version: str


class DocumentBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    indexed_document_ids: list[str]
    track_id: str | None = None
    metadata: SidecarMetadata


class WorkspaceCloneRequest(BaseModel):
    """复制已冻结图谱到独立候选 workspace；不触发新的 LLM 抽取。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    source_workspace: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class WorkspaceCloneResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_workspace: str
    workspace: str
    metadata: SidecarMetadata


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=2, max_length=4000)
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int = Field(default=30, ge=1, le=100)
    chunk_top_k: int = Field(default=12, ge=1, le=100)
    max_total_tokens: int = Field(default=16_000, ge=1000, le=100_000)
    temporal: TemporalScope | None = None

    @model_validator(mode="after")
    def validate_temporal_mode(self):
        if self.temporal is not None:
            if self.mode != "mix":
                raise ValueError("时间检索当前仅支持完整 mix 链路")
            if self.temporal.access == "runtime" and self.temporal.period != "before":
                raise ValueError("历史 Runtime 只能查询 before")
        return self


class QueryReference(BaseModel):
    model_config = ConfigDict(extra="allow")

    file_path: str
    reference_id: str | None = None
    content: str | None = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    context: str
    references: list[QueryReference]
    metadata: SidecarMetadata
    temporal_diagnostics: dict | None = None
    global_graph_clues: dict | None = None


class EntityRead(BaseModel):
    model_config = ConfigDict(extra="allow")

    entity_name: str
    graph_data: dict[str, object] | None = None


class EntityListResponse(BaseModel):
    entities: list[EntityRead]


class GraphNodeRead(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, object] = Field(default_factory=dict)


class GraphEdgeRead(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    target: str
    properties: dict[str, object] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    nodes: list[GraphNodeRead]
    edges: list[GraphEdgeRead]
    is_truncated: bool = False


class EntityMergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_entities: list[str] = Field(min_length=1, max_length=20)
    target_entity: str = Field(min_length=1, max_length=500)
    merge_strategy: dict[str, str] = Field(default_factory=dict)
    # 旧版人工别名审核仍可不传；受治理的 Graph ChangeSet 必须传入。
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=160)


class EntityMergeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    result: dict[str, object]


class EntityMutationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    entity_name: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=20_000)
    entity_type: str = Field(default="UNKNOWN", min_length=1, max_length=120)
    allow_rename: bool = False
    new_entity_name: str | None = Field(default=None, min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=1, max_length=160)


class RelationMutationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_entity: str = Field(min_length=1, max_length=500)
    target_entity: str = Field(min_length=1, max_length=500)
    description: str = Field(default="", max_length=20_000)
    keywords: str = Field(default="", max_length=4000)
    weight: float = Field(default=1.0, ge=0, le=1000)
    idempotency_key: str = Field(min_length=1, max_length=160)


class EntityDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    entity_name: str = Field(min_length=1, max_length=500)
    cascade: bool = False
    idempotency_key: str = Field(min_length=1, max_length=160)


class RelationDeleteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_entity: str = Field(min_length=1, max_length=500)
    target_entity: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=1, max_length=160)


class GraphMutationResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    result: dict[str, object]
