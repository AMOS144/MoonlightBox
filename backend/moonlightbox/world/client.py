"""LightRAG Sidecar 客户端：统一处理索引、检索、实体读取和实体归并。"""

from dataclasses import dataclass
from time import monotonic
from typing import Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from moonlightbox.agent_runtime.cancellation import cancellation_requested
from moonlightbox.agent_runtime.deadlines import bounded_timeout
from moonlightbox.agent_runtime.resilience import (
    ResiliencePolicy,
    check_interruption,
    execution_scope,
    managed,
    request_timeout,
    run_operation,
)
from moonlightbox.observability import add_context_snapshot, record_span_output, retriever_span


class LightRAGSidecarError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int | None = None,
                 details: dict[str, object] | None = None) -> None:
        self.code = code
        self.safe_message = message
        self.status_code = status_code
        self.details = details
        super().__init__(message)


def _index_failure(detail, *, status_code=None):
    """只接受已知故障契约；不把供应商堆栈或任意响应正文展示给用户。"""
    messages = {
        'lightrag_content_rejected': '模型服务拒绝处理部分聊天内容，资料已保留。请检查受阻片段后再恢复。',
        'lightrag_provider_authentication': '图谱模型服务认证失败，请检查抽取服务的凭据后再恢复。',
        'lightrag_provider_request_rejected': '图谱模型服务拒绝了请求，请检查模型配置及受阻资料后再恢复。',
        'lightrag_index_error': '资料整理遇到内部错误，已有进度已保留，需要检查后恢复。',
        'lightrag_timeout': '图谱模型请求超时，将稍后恢复。',
        'lightrag_rate_limited': '图谱模型服务暂时限流，将稍后恢复。',
        'lightrag_unavailable': '图谱模型服务暂时无法连接，将稍后恢复。',
        'lightrag_configuration_missing': 'LightRAG Sidecar 缺少模型服务配置，请补充凭据后重试。',
    }
    if (
        isinstance(detail, dict)
        and isinstance(detail.get('code'), str)
        and detail['code'] in messages
    ):
        return LightRAGSidecarError(
            detail['code'], messages[detail['code']],
            status_code=status_code,
            details={k: detail.get(k) for k in ('code', 'document_id', 'chunk_id')},
        )
    return None


def _raise_permanent_index_failure(status):
    permanent = {
        "lightrag_content_rejected", "lightrag_provider_authentication",
        "lightrag_provider_request_rejected", "lightrag_index_error",
        "lightrag_configuration_missing",
    }
    for detail in status.get("failures", []):
        failure = _index_failure(detail)
        if failure and failure.code in permanent:
            raise failure


@dataclass(frozen=True)
class LightRAGDocument:
    id: str
    source: str
    text: str
    source_version: str | None = None
    message_spans: list[dict[str, object]] | None = None


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


class _WorkspaceCloneResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_workspace: str
    workspace: str
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
    temporal_diagnostics: dict | None = None
    global_graph_clues: dict | None = None


class LightRAGEntity(BaseModel):
    model_config = ConfigDict(extra="allow")

    entity_name: str
    graph_data: dict[str, object] | None = None


class LightRAGGraphNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    labels: list[str] = []
    properties: dict[str, object] = {}


class LightRAGGraphEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    target: str
    properties: dict[str, object] = {}


class LightRAGSidecarGraph(BaseModel):
    model_config = ConfigDict(extra="allow")

    nodes: list[LightRAGGraphNode]
    edges: list[LightRAGGraphEdge]
    is_truncated: bool = False


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
        status = self.index_status(workspace, [item.id for item in documents])
        # Sidecar 已经持久化的永久失败不能再次送入模型。临时故障仍交给
        # LightRAG 的失败文档恢复路径处理。
        _raise_permanent_index_failure(status)
        if not status["pipeline_active"] and all(value == "processed" for value in status["documents"].values()):
            return LightRAGMetadata.model_validate(status["metadata"])
        if status["pipeline_active"]:
            raise LightRAGSidecarError(
                "lightrag_index_running", "Sidecar 仍在构建，稍后自动确认进度"
            )
        try:
            return self._insert_documents(workspace, documents)
        except LightRAGSidecarError as error:
            if error.code not in {
                "lightrag_timeout",
                "lightrag_unavailable",
                "lightrag_rate_limited",
                "lightrag_index_running",
            }:
                raise
            # 写请求结果不确定时只读核实，绝不在传输层直接重放写入。
            status = self.index_status(workspace, [item.id for item in documents])
            _raise_permanent_index_failure(status)
            if not status["pipeline_active"] and all(value == "processed" for value in status["documents"].values()):
                return LightRAGMetadata.model_validate(status["metadata"])
            if status["pipeline_active"]:
                raise LightRAGSidecarError(
                    "lightrag_index_running", "连接已恢复，Sidecar 仍在构建，稍后自动确认进度"
                ) from error
            raise

    def index_status(self, workspace: str, document_ids: list[str]) -> dict:
        def read():
            if cancellation_requested():
                raise LightRAGSidecarError("cancelled", "构建已请求取消")
            value = self._post(
                f"/v1/workspaces/{quote(workspace, safe='')}/documents:status",
                {"document_ids": document_ids},
            )
            if (
                not isinstance(value, dict)
                or not isinstance(value.get("documents"), dict)
                or type(value.get("pipeline_active")) is not bool
            ):
                raise LightRAGSidecarError(
                    "lightrag_invalid_response", "构建状态响应不完整，不能安全重放索引"
                )
            if "failures" in value and not isinstance(value["failures"], list):
                raise LightRAGSidecarError(
                    "lightrag_invalid_response", "构建失败详情格式无效，不能安全恢复索引"
                )
            allowed = {
                "missing",
                "pending",
                "parsing",
                "analyzing",
                "processing",
                "processed",
                "failed",
            }
            if set(value["documents"]) != set(document_ids) or any(
                s not in allowed for s in value["documents"].values()
            ):
                raise LightRAGSidecarError("lightrag_invalid_response", "构建状态未覆盖请求的文档")
            try:
                LightRAGMetadata.model_validate(value.get("metadata"))
            except ValidationError as error:
                raise LightRAGSidecarError(
                    "lightrag_invalid_response", "构建状态缺少有效元数据"
                ) from error
            return value

        if managed():
            return run_operation(read, kind="tool", timeout_seconds=10)
        deadline = monotonic() + 40
        with execution_scope(
            ResiliencePolicy(request_timeout_seconds=10, max_retries=2),
            lambda: deadline - monotonic(),
            lambda: "cancelled" if cancellation_requested() else None,
        ):
            return run_operation(read, kind="tool", timeout_seconds=10)

    def _insert_documents(
        self, workspace: str, documents: list[LightRAGDocument]
    ) -> LightRAGMetadata:
        if cancellation_requested():
            raise LightRAGSidecarError("cancelled", "构建已请求取消")
        result = self._post(
            f"/v1/workspaces/{quote(workspace, safe='')}/documents:batch",
            {
                "documents": [
                    {
                        "id": item.id,
                        "source": item.source,
                        "text": item.text,
                        **(
                            {
                                "source_version": item.source_version,
                                "message_spans": item.message_spans,
                            }
                            if item.message_spans is not None
                            else {}
                        ),
                    }
                    for item in documents
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

    def clone_workspace(self, *, source_workspace: str, workspace: str) -> LightRAGMetadata:
        """由 Sidecar 复制已冻结图谱；调用不会把原始聊天重新送去抽取。"""

        result = self._post(
            f"/v1/workspaces/{quote(workspace, safe='')}:clone",
            {"source_workspace": source_workspace},
        )
        try:
            parsed = _WorkspaceCloneResponse.model_validate(result)
        except ValidationError as error:
            raise LightRAGSidecarError(
                "lightrag_invalid_response", "LightRAG Sidecar 返回了无效的复制结果"
            ) from error
        if parsed.source_workspace != source_workspace or parsed.workspace != workspace:
            raise LightRAGSidecarError(
                "lightrag_clone_mismatch", "LightRAG Sidecar 返回了不匹配的候选 workspace"
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
        temporal: dict[str, object] | None = None,
    ) -> LightRAGRetrieval:
        # ``workspace`` 已经是 URL path 的一部分。Sidecar 的 QueryRequest 使用
        # ``extra=forbid``，因此绝不能为了埋点而把它混进实际 POST body；否则请求会
        # 在 FastAPI 参数校验阶段以 422 失败，根本到不了 LightRAG。
        payload: dict[str, object] = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "max_total_tokens": max_total_tokens,
        }
        if temporal is not None:
            payload["temporal"] = temporal
        # Trace 需要记录检索所属 workspace，但这是观测输入，不是 Sidecar 请求契约。
        trace_input = {"workspace": workspace, **payload}
        # 与 Harness 工具 Span 分开：工具说明“谁调用了检索”，此 RETRIEVER Span 说明
        # LightRAG 实际接受了什么查询、返回了多少可追溯资料和耗时。
        with retriever_span(
            "moonlightbox.retrieval.lightrag.query",
            input_value=trace_input,
            attributes={
                "moonlightbox.retrieval.backend": "lightrag",
                "moonlightbox.retrieval.workspace": workspace,
                "moonlightbox.retrieval.mode": mode,
                "moonlightbox.retrieval.top_k": top_k,
                "moonlightbox.retrieval.chunk_top_k": chunk_top_k,
            },
        ) as span:
            result = self._post(
                f"/v1/workspaces/{quote(workspace, safe='')}/query",
                payload,
            )
            try:
                parsed = LightRAGRetrieval.model_validate(result)
            except ValidationError as error:
                raise LightRAGSidecarError(
                    "lightrag_invalid_response", "LightRAG Sidecar 返回了无效的检索结果"
                ) from error
            record_span_output(span, parsed)
            add_context_snapshot(span, event_name="retrieval_result", value=parsed)
            if span is not None:
                span.set_attribute("moonlightbox.retrieval.reference_count", len(parsed.references))
                span.set_attribute("moonlightbox.retrieval.context_characters", len(parsed.context))
            return parsed

    def list_entities(self, workspace: str) -> list[LightRAGEntity]:
        with retriever_span(
            "moonlightbox.retrieval.lightrag.list_entities",
            input_value={"workspace": workspace},
            attributes={
                "moonlightbox.retrieval.backend": "lightrag",
                "moonlightbox.retrieval.workspace": workspace,
            },
        ) as span:
            result = self._get(f"/v1/workspaces/{quote(workspace, safe='')}/entities")
            try:
                if not isinstance(result, dict):
                    raise TypeError("LightRAG 实体响应不是对象")
                entities = result.get("entities", [])
                if not isinstance(entities, list):
                    raise TypeError("LightRAG entities 不是列表")
                parsed = [LightRAGEntity.model_validate(item) for item in entities]
            except (TypeError, ValidationError) as error:
                raise LightRAGSidecarError(
                    "lightrag_invalid_response", "LightRAG 实体列表无效"
                ) from error
            record_span_output(span, parsed)
            add_context_snapshot(span, event_name="retrieval_result", value=parsed)
            return parsed

    def get_graph(self, workspace: str) -> LightRAGSidecarGraph:
        with retriever_span(
            "moonlightbox.retrieval.lightrag.get_graph",
            input_value={"workspace": workspace},
            attributes={
                "moonlightbox.retrieval.backend": "lightrag",
                "moonlightbox.retrieval.workspace": workspace,
            },
        ) as span:
            result = self._get(f"/v1/workspaces/{quote(workspace, safe='')}/graph")
            try:
                if not isinstance(result, dict):
                    raise TypeError("LightRAG 图谱快照响应不是对象")
                parsed = LightRAGSidecarGraph.model_validate(result)
            except (TypeError, ValidationError) as error:
                raise LightRAGSidecarError(
                    "lightrag_invalid_response", "LightRAG 图谱快照无效"
                ) from error
            record_span_output(span, parsed)
            add_context_snapshot(span, event_name="retrieval_result", value=parsed)
            if span is not None:
                span.set_attribute("moonlightbox.retrieval.node_count", len(parsed.nodes))
                span.set_attribute("moonlightbox.retrieval.edge_count", len(parsed.edges))
            return parsed

    def get_entity(self, workspace: str, entity_name: str) -> dict[str, object] | None:
        with retriever_span(
            "moonlightbox.retrieval.lightrag.get_entity",
            input_value={"workspace": workspace, "entity_name": entity_name},
            attributes={
                "moonlightbox.retrieval.backend": "lightrag",
                "moonlightbox.retrieval.workspace": workspace,
            },
        ) as span:
            result = self._get(
                f"/v1/workspaces/{quote(workspace, safe='')}/entities/{quote(entity_name, safe='')}"
            )
            try:
                parsed = LightRAGEntity.model_validate(result)
            except ValidationError as error:
                raise LightRAGSidecarError(
                    "lightrag_invalid_response", "LightRAG 实体详情无效"
                ) from error
            graph_data = dict(parsed.graph_data or {})
            record_span_output(span, graph_data)
            add_context_snapshot(span, event_name="retrieval_result", value=graph_data)
            return graph_data

    def get_relation(
        self, workspace: str, source_entity: str, target_entity: str
    ) -> dict[str, object]:
        with retriever_span(
            "moonlightbox.retrieval.lightrag.get_relation",
            input_value={
                "workspace": workspace,
                "source_entity": source_entity,
                "target_entity": target_entity,
            },
            attributes={
                "moonlightbox.retrieval.backend": "lightrag",
                "moonlightbox.retrieval.workspace": workspace,
            },
        ) as span:
            result = self._get(
                f"/v1/workspaces/{quote(workspace, safe='')}/relations/"
                f"{quote(source_entity, safe='')}/{quote(target_entity, safe='')}"
            )
            parsed = _mutation_result(result)
            record_span_output(span, parsed)
            add_context_snapshot(span, event_name="retrieval_result", value=parsed)
            return parsed

    def merge_entities(
        self,
        workspace: str,
        *,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, object]:
        result = self._post(
            f"/v1/workspaces/{quote(workspace, safe='')}/entities:merge",
            {
                "source_entities": source_entities,
                "target_entity": target_entity,
                "merge_strategy": merge_strategy or {},
                "idempotency_key": idempotency_key,
            },
        )
        return result if isinstance(result, dict) else {"result": result}

    def create_entity(
        self,
        workspace: str,
        *,
        entity_name: str,
        description: str,
        entity_type: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        return _mutation_result(
            self._post(
                f"/v1/workspaces/{quote(workspace, safe='')}/entities",
                {
                    "entity_name": entity_name,
                    "description": description,
                    "entity_type": entity_type,
                    "allow_rename": False,
                    "new_entity_name": None,
                    "idempotency_key": idempotency_key,
                },
            )
        )

    def update_entity(
        self,
        workspace: str,
        *,
        entity_name: str,
        description: str,
        entity_type: str,
        idempotency_key: str,
        new_entity_name: str | None = None,
    ) -> dict[str, object]:
        return _mutation_result(
            self._request(
                "PATCH",
                f"/v1/workspaces/{quote(workspace, safe='')}/entities",
                {
                    "entity_name": entity_name,
                    "description": description,
                    "entity_type": entity_type,
                    "allow_rename": new_entity_name is not None,
                    "new_entity_name": new_entity_name,
                    "idempotency_key": idempotency_key,
                },
            )
        )

    def delete_entity(
        self,
        workspace: str,
        *,
        entity_name: str,
        cascade: bool,
        idempotency_key: str,
    ) -> dict[str, object]:
        return _mutation_result(
            self._post(
                f"/v1/workspaces/{quote(workspace, safe='')}/entities:delete",
                {
                    "entity_name": entity_name,
                    "cascade": cascade,
                    "idempotency_key": idempotency_key,
                },
            )
        )

    def create_relation(
        self,
        workspace: str,
        *,
        source_entity: str,
        target_entity: str,
        description: str,
        keywords: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        return _mutation_result(
            self._post(
                f"/v1/workspaces/{quote(workspace, safe='')}/relations",
                {
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "description": description,
                    "keywords": keywords,
                    "weight": 1.0,
                    "idempotency_key": idempotency_key,
                },
            )
        )

    def update_relation(
        self,
        workspace: str,
        *,
        source_entity: str,
        target_entity: str,
        description: str,
        keywords: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        return _mutation_result(
            self._request(
                "PATCH",
                f"/v1/workspaces/{quote(workspace, safe='')}/relations",
                {
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "description": description,
                    "keywords": keywords,
                    "weight": 1.0,
                    "idempotency_key": idempotency_key,
                },
            )
        )

    def delete_relation(
        self,
        workspace: str,
        *,
        source_entity: str,
        target_entity: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        return _mutation_result(
            self._post(
                f"/v1/workspaces/{quote(workspace, safe='')}/relations:delete",
                {
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "idempotency_key": idempotency_key,
                },
            )
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _post(self, path: str, payload: dict[str, object]) -> object:
        return self._request("POST", path, payload)

    def _request(self, method: str, path: str, payload: dict[str, object]) -> object:
        try:
            response = self._client.request(
                method,
                f"{self._base_url}{path}",
                headers={"Authorization": f"Bearer {self._token}"},
                json=payload,
                timeout=bounded_timeout(request_timeout(self._timeout, model_request=False)),
            )
        except httpx.TimeoutException as error:
            raise LightRAGSidecarError("lightrag_timeout", "LightRAG Sidecar 处理超时") from error
        except httpx.RequestError as error:
            raise LightRAGSidecarError(
                "lightrag_unavailable", "无法连接 LightRAG Sidecar"
            ) from error
        check_interruption()
        if response.status_code in {401, 403}:
            raise LightRAGSidecarError(
                "lightrag_authentication",
                "LightRAG Sidecar 认证失败",
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            try:
                body = response.json()
            except ValueError:
                body = None
            structured = _index_failure(
                body.get('detail') if isinstance(body, dict) else None,
                status_code=response.status_code,
            )
            if structured:
                raise structured
            if response.status_code == 409:
                try:
                    detail = response.json().get("detail", {})
                except (ValueError, AttributeError):
                    detail = {}
                if isinstance(detail, dict) and str(detail.get("code", "")).startswith("temporal_"):
                    raise LightRAGSidecarError(
                        str(detail["code"]),
                        "时间来源索引与查询契约不匹配；需要修复索引，不能将失败当作无相关资料。",
                        status_code=409,
                    )
                if isinstance(detail, dict) and detail.get("code") == "lightrag_index_running":
                    raise LightRAGSidecarError("lightrag_index_running", "Sidecar 正在处理已有构建")
            raise LightRAGSidecarError(
                "lightrag_rate_limited"
                if response.status_code == 429
                else "lightrag_unavailable"
                if response.status_code >= 500
                else "lightrag_request_failed",
                f"LightRAG Sidecar 请求失败（HTTP {response.status_code}）",
                status_code=response.status_code,
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
                timeout=bounded_timeout(request_timeout(self._timeout, model_request=False)),
            )
        except httpx.TimeoutException as error:
            raise LightRAGSidecarError("lightrag_timeout", "LightRAG Sidecar 处理超时") from error
        except httpx.RequestError as error:
            raise LightRAGSidecarError(
                "lightrag_unavailable", "无法连接 LightRAG Sidecar"
            ) from error
        check_interruption()
        if response.status_code in {401, 403}:
            raise LightRAGSidecarError(
                "lightrag_authentication",
                "LightRAG Sidecar 认证失败",
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise LightRAGSidecarError(
                "lightrag_request_failed",
                f"LightRAG Sidecar 请求失败（HTTP {response.status_code}）",
                status_code=response.status_code,
            )
        try:
            return response.json()
        except ValueError as error:
            raise LightRAGSidecarError(
                "lightrag_invalid_response", "LightRAG Sidecar 返回了非 JSON 响应"
            ) from error


def _mutation_result(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {"result": {"value": value}}
    result = value.get("result")
    if isinstance(result, dict):
        return result
    return value
