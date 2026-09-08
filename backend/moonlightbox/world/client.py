"""LightRAG Sidecar 客户端：统一处理索引、检索、实体读取和实体归并。"""

from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError


class LightRAGSidecarError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.safe_message = message
        super().__init__(message)


@dataclass(frozen=True)
class LightRAGDocument:
    id: str
    source: str
    text: str


class LightRAGMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lightrag_version: str
    embedding_model: str
    embedding_dimension: int
    extraction_model: str
    chunking_strategy: str
    chunk_token_size: int
    chunk_overlap_token_size: int
    entity_prompt_version: str


class _IndexResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indexed_document_ids: list[str]
    track_id: str | None = None
    metadata: LightRAGMetadata


class LightRAGReference(BaseModel):
    model_config = ConfigDict(extra="allow")

    file_path: str
    reference_id: str | None = None
    content: str | None = None


class LightRAGRetrieval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    context: str
    references: list[LightRAGReference]
    metadata: LightRAGMetadata


class LightRAGEntity(BaseModel):
    model_config = ConfigDict(extra="allow")

    entity_name: str
    graph_data: dict[str, object] | None = None


class LightRAGSidecarClient:
    """通过 Bearer Token 调用 Sidecar，并统一转换网络/API 错误。"""
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_seconds: float = 1800,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._timeout = timeout_seconds

    def index_documents(
        self,
        workspace: str,
        documents: list[LightRAGDocument],
    ) -> LightRAGMetadata:
        if not documents:
            raise ValueError("documents 不能为空")
        result = self._post(
            f"/v1/workspaces/{quote(workspace, safe='')}/documents:batch",
            {
                "documents": [
                    {"id": item.id, "source": item.source, "text": item.text} for item in documents
                ]
            },
        )
        try:
            parsed = _IndexResponse.model_validate(result)
        except ValidationError as error:
            raise LightRAGSidecarError(
                "lightrag_invalid_response", "LightRAG Sidecar 返回了无效的索引结果"
            ) from error
        expected = {item.id for item in documents}
        if set(parsed.indexed_document_ids) != expected:
            raise LightRAGSidecarError(
                "lightrag_incomplete_index", "LightRAG Sidecar 没有确认全部文档已完成索引"
            )
        return parsed.metadata

    def query(
        self,
        workspace: str,
        query: str,
        *,
        mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix",
        top_k: int = 30,
        chunk_top_k: int = 12,
        max_total_tokens: int = 16000,
    ) -> LightRAGRetrieval:
        result = self._post(
            f"/v1/workspaces/{quote(workspace, safe='')}/query",
            {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "chunk_top_k": chunk_top_k,
                "max_total_tokens": max_total_tokens,
            },
        )
        try:
            return LightRAGRetrieval.model_validate(result)
        except ValidationError as error:
            raise LightRAGSidecarError(
                "lightrag_invalid_response", "LightRAG Sidecar 返回了无效的检索结果"
            ) from error

    def list_entities(self, workspace: str) -> list[LightRAGEntity]:
        result = self._get(f"/v1/workspaces/{quote(workspace, safe='')}/entities")
        try:
            return [LightRAGEntity.model_validate(item) for item in result.get("entities", [])]
        except (AttributeError, TypeError, ValidationError) as error:
            raise LightRAGSidecarError("lightrag_invalid_response", "LightRAG 实体列表无效") from error

    def merge_entities(
        self,
        workspace: str,
        *,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] | None = None,
    ) -> dict[str, object]:
        result = self._post(
            f"/v1/workspaces/{quote(workspace, safe='')}/entities:merge",
            {
                "source_entities": source_entities,
                "target_entity": target_entity,
                "merge_strategy": merge_strategy or {},
            },
        )
        return result if isinstance(result, dict) else {"result": result}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _post(self, path: str, payload: dict[str, object]) -> object:
        try:
            response = self._client.post(
                f"{self._base_url}{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                json=payload,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as error:
            raise LightRAGSidecarError("lightrag_timeout", "LightRAG Sidecar 处理超时") from error
        except httpx.RequestError as error:
            raise LightRAGSidecarError(
                "lightrag_unavailable", "无法连接 LightRAG Sidecar"
            ) from error
        if response.status_code == 401:
            raise LightRAGSidecarError("lightrag_authentication", "LightRAG Sidecar 认证失败")
        if response.status_code >= 400:
            raise LightRAGSidecarError(
                "lightrag_request_failed",
                f"LightRAG Sidecar 请求失败（HTTP {response.status_code}）",
            )
        try:
            return response.json()
        except ValueError as error:
            raise LightRAGSidecarError(
                "lightrag_invalid_response", "LightRAG Sidecar 返回了非 JSON 响应"
            ) from error

    def _get(self, path: str) -> object:
        try:
            response = self._client.get(
                f"{self._base_url}{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as error:
            raise LightRAGSidecarError("lightrag_timeout", "LightRAG Sidecar 处理超时") from error
        except httpx.RequestError as error:
            raise LightRAGSidecarError("lightrag_unavailable", "无法连接 LightRAG Sidecar") from error
        if response.status_code == 401:
            raise LightRAGSidecarError("lightrag_authentication", "LightRAG Sidecar 认证失败")
        if response.status_code >= 400:
            raise LightRAGSidecarError("lightrag_request_failed", f"LightRAG Sidecar 请求失败（HTTP {response.status_code}）")
        try:
            return response.json()
        except ValueError as error:
            raise LightRAGSidecarError("lightrag_invalid_response", "LightRAG Sidecar 返回了非 JSON 响应") from error
