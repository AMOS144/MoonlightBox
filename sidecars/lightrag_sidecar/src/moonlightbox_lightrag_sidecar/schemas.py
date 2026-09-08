from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    source: str = Field(min_length=1, max_length=180, pattern=r"^[A-Za-z0-9_.-]+$")
    text: str = Field(min_length=1, max_length=500_000)


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


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=2, max_length=4000)
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int = Field(default=30, ge=1, le=100)
    chunk_top_k: int = Field(default=12, ge=1, le=100)
    max_total_tokens: int = Field(default=16_000, ge=1000, le=100_000)


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


class EntityRead(BaseModel):
    model_config = ConfigDict(extra="allow")

    entity_name: str
    graph_data: dict[str, object] | None = None


class EntityListResponse(BaseModel):
    entities: list[EntityRead]


class EntityMergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_entities: list[str] = Field(min_length=1, max_length=20)
    target_entity: str = Field(min_length=1, max_length=500)
    merge_strategy: dict[str, str] = Field(default_factory=dict)


class EntityMergeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    result: dict[str, object]
