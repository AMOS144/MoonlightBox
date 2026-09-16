import asyncio
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

from moonlightbox_lightrag_sidecar.config import SidecarSettings
from moonlightbox_lightrag_sidecar.lease import IndexLeaseBusy, WorkspaceIndexLease
from moonlightbox_lightrag_sidecar.schemas import (
    DocumentInput,
    EntityDeleteRequest,
    EntityListResponse,
    EntityMergeRequest,
    EntityMergeResponse,
    EntityMutationRequest,
    EntityRead,
    GraphEdgeRead,
    GraphMutationResponse,
    GraphNodeRead,
    GraphResponse,
    QueryReference,
    QueryRequest,
    QueryResponse,
    RelationDeleteRequest,
    RelationMutationRequest,
    RuntimeConfigResponse,
    RuntimeConfigUpdate,
    RuntimeConnectionTest,
    SidecarMetadata,
)
from moonlightbox_lightrag_sidecar.source_index import (
    SourceIndex,
    TemporalIndexError,
    normalize_document,
    source_aware_chunking,
)

ENTITY_PROMPT_VERSION = "moonlightbox-person-world-v1"
# LightRAG reserves priority 8 for entity/relation description summaries.
# MiniMax's reasoning mode is useful for extraction, but it can spend a very
# long time on the much smaller merge-summary request.  Keep the existing
# extraction setting and disable reasoning only for this bounded maintenance
# call so a summary cannot block the indexing worker.
SUMMARY_LLM_PRIORITY = 8
ENTITY_TYPE_GUIDANCE = """
- Person: 真实人物、称呼、别名以及聊天双方。
- PersonalPlace: 家、公司、学校、常去店铺、城市等与个人生活有关的地点。
- Organization: 公司、学校、团队、社群和其他组织。
- RoleOrOccupation: 工作、职位、学业身份和家庭角色。
- Activity: 工作、通勤、吃饭、运动、娱乐、照料等具体活动。
- Relationship: 家人、朋友、同事、伴侣等稳定社会关系或关系阶段。
- Preference: 喜好、厌恶、选择倾向和长期兴趣。
- Routine: 工作日、周末、睡眠、通勤、回复等重复生活规律。
- LifePhase: 入职、离职、搬家、升学、恋爱等持续一段时间的生活阶段。
- Event: 对人物生活、关系或身份有影响的具体经历。
- Topic: 跨多次对话反复出现的长期话题。

只抽取对理解人物生活世界有帮助的实体。关系描述应保留原意；常见语义包括
knows、family_of、friend_of、colleague_of、partner_of、works_at、studies_at、
lives_at、often_visits、participates_in、likes、dislikes、usually_does、before、
after、related_to 和 changed_by。不要把消息编号、时间戳或文档编号当作实体。
""".strip()


class _OfflineTokenizer:
    """不依赖外网下载词表的稳定分词器。

    LightRAG 默认会在首次创建新 workspace 时下载 tiktoken 的词表；
    WSL/受限网络下这会让整个建图直接失败。这里按 Unicode 码点计数，
    只用于切块和预算，不参与 LLM 语义生成。
    """

    def encode(self, content: str, **_: Any) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def __deepcopy__(self, memo: dict[int, Any]) -> "_OfflineTokenizer":
        return self


@dataclass
class _WorkspaceHandle:
    rag: Any
    lock: asyncio.Lock
    generation: int = 0
    metadata: SidecarMetadata | None = None


class IndexInProgress(RuntimeError):
    """已有请求仍在执行；调用方应只查询状态，不重放写入。"""


class CoreLightRAGRegistry:
    """Own one initialized LightRAG Core instance per immutable graph workspace."""

    def __init__(self, settings: SidecarSettings) -> None:
        self._settings = settings
        self._handles: dict[str, _WorkspaceHandle] = {}
        self._retired_handles: list[_WorkspaceHandle] = []
        self._generation = 0
        self._registry_lock = asyncio.Lock()

    @property
    def metadata(self) -> SidecarMetadata:
        return self._metadata_for(self._settings)

    @property
    def runtime_config(self) -> RuntimeConfigResponse:
        llm_key, embedding_key = self._configured_keys(self._settings)
        return RuntimeConfigResponse(
            llm_model=self._settings.llm_model,
            llm_base_url=self._settings.llm_base_url,
            llm_key_configured=bool(llm_key),
            embedding_model=self._settings.embedding_model,
            embedding_base_url=self._settings.embedding_base_url,
            embedding_dimension=self._settings.embedding_dimension,
            embedding_key_configured=bool(embedding_key),
        )

    @staticmethod
    def _configured_keys(settings: SidecarSettings) -> tuple[str, str]:
        llm_key = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key else ""
        embedding_key = (
            settings.embedding_api_key.get_secret_value().strip()
            if settings.embedding_api_key
            else llm_key
        )
        return llm_key, embedding_key

    @staticmethod
    def _metadata_for(settings: SidecarSettings) -> SidecarMetadata:
        try:
            installed_version = version("lightrag-hku")
        except PackageNotFoundError:
            installed_version = "unknown"
        return SidecarMetadata(
            lightrag_version=installed_version,
            embedding_model=settings.embedding_model,
            embedding_dimension=settings.embedding_dimension,
            extraction_model=settings.llm_model,
            chunking_strategy="fixed_token",
            chunk_token_size=settings.chunk_token_size,
            chunk_overlap_token_size=settings.chunk_overlap_token_size,
            entity_prompt_version=ENTITY_PROMPT_VERSION,
        )

    async def reconfigure(self, update: RuntimeConfigUpdate) -> RuntimeConfigResponse:
        values = update.model_dump(exclude_none=True)
        validated = SidecarSettings(**{**self._settings.model_dump(), **values})
        validated.require_model_keys()
        async with self._registry_lock:
            self._settings = validated
            self._generation += 1
        return self.runtime_config

    async def test_connection(self, request: RuntimeConnectionTest) -> None:
        import httpx

        llm_key, embedding_key = self._configured_keys(self._settings)
        supplied = request.api_key.get_secret_value().strip() if request.api_key else ""
        key = supplied or (embedding_key if request.service == "embedding" else llm_key)
        if not key:
            raise ValueError("请先填写或保存 API Key")
        base_url = request.base_url.strip().rstrip("/")
        if request.service == "embedding":
            endpoint = base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
            body: dict[str, object] = {"model": request.model, "input": ["连接测试"]}
            if request.embedding_dimension is not None:
                body["dimensions"] = request.embedding_dimension
        else:
            endpoint = (
                base_url
                if base_url.endswith("/chat/completions")
                else f"{base_url}/chat/completions"
            )
            body = {
                "model": request.model,
                "messages": [{"role": "user", "content": "仅回复 OK"}],
                "max_tokens": 8,
                "stream": False,
            }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                result = await client.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {key}"},
                    json=body,
                )
        except httpx.HTTPError as error:
            raise RuntimeError("无法连接该服务，请检查地址和网络") from error
        if result.status_code >= 400:
            raise ValueError(f"服务连接失败（HTTP {result.status_code}）")

    async def index(self, workspace: str, documents: list[DocumentInput]) -> str | None:
        if self._lease_active(workspace):
            raise IndexInProgress("LightRAG workspace 正在由其他进程构建")
        lease = WorkspaceIndexLease(
            self._settings.storage_dir, workspace, self._settings.index_lease_ttl_seconds
        )
        try:
            await lease.__aenter__()
        except IndexLeaseBusy as error:
            raise IndexInProgress(str(error)) from error
        try:
            return await self._index_impl(workspace, documents)
        finally:
            await lease.__aexit__(None, None, None)

    async def _index_impl(self, workspace: str, documents: list[DocumentInput]) -> str | None:
        handle = await self._get_handle(workspace)
        if handle.lock.locked() or await self._pipeline_active(handle.rag) or handle.lock.locked():
            raise IndexInProgress("LightRAG 流水线仍在执行，请稍后查询状态")
        async with handle.lock:
            # 先冻结来源契约；即使抽取中断，重试仍使用同一份消息位置。
            source_documents = [d for d in documents if d.message_spans is not None]
            if source_documents:
                existing_docs = await handle.rag.full_docs.get_by_ids(
                    [d.id for d in source_documents]
                )
                for document, existing in zip(source_documents, existing_docs, strict=True):
                    if existing and existing.get("content") != normalize_document(document).text:
                        raise TemporalIndexError("temporal_frozen_document_mismatch")
            SourceIndex(self._settings.storage_dir / workspace).publish(
                documents,
                embedding=(
                    (handle.metadata or self.metadata).embedding_model,
                    (handle.metadata or self.metadata).embedding_dimension,
                ),
            )
            document_ids = [item.id for item in documents]
            statuses = await handle.rag.aget_docs_by_ids(document_ids)
            from .index_errors import document_failures

            for failure in document_failures(statuses):
                if failure.code in {
                    "lightrag_content_rejected",
                    "lightrag_provider_authentication",
                    "lightrag_provider_request_rejected",
                }:
                    raise failure
            failed_ids = {
                document_id
                for document_id, record in statuses.items()
                if self._status_value(record) == "failed"
            }
            interrupted_ids = {
                document_id
                for document_id, record in statuses.items()
                if self._status_value(record) in {"pending", "parsing", "analyzing", "processing"}
            }
            result: str | None = None
            if failed_ids:
                result = await self._recover_failed_documents(handle.rag)
                statuses = await handle.rag.aget_docs_by_ids(document_ids)
            elif interrupted_ids:
                # LightRAG automatically resets interrupted pipeline states to
                # PENDING when its pipeline is run again. This is separate from
                # the explicit FAILED-document retry flow above.
                await handle.rag.apipeline_process_enqueue_documents()
                statuses = await handle.rag.aget_docs_by_ids(document_ids)

            missing = [item for item in documents if item.id not in statuses]
            if missing:
                inserted = await handle.rag.ainsert(
                    [item.text for item in missing],
                    ids=[item.id for item in missing],
                    file_paths=[item.source for item in missing],
                )
                result = inserted if isinstance(inserted, str) else result
                statuses = await handle.rag.aget_docs_by_ids(document_ids)
        from .index_errors import document_failures
        failures = document_failures(statuses)
        if failures:
            # 永久拒绝优先，不能被另一片段的临时超时掩盖。
            raise next((f for f in failures if f.status_code == 422), failures[0])
        counts: Counter[str] = Counter()
        for item in documents:
            record = statuses.get(item.id)
            counts[self._status_value(record)] += 1
        if counts != {"processed": len(documents)}:
            summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            raise RuntimeError(f"LightRAG 索引未完整成功（{summary}）")
        return result

    async def _pipeline_active(self, rag: Any) -> bool:
        from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

        if not getattr(rag, "workspace", None):
            return False
        state = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        async with get_namespace_lock("pipeline_status", workspace=rag.workspace):
            return bool(state.get("busy", False))

    async def index_status(self, workspace: str, document_ids: list[str]) -> dict:
        """不等待索引写锁；processing 文档与活跃流水线分开报告。"""
        handle = await self._get_handle(workspace)
        statuses = await handle.rag.aget_docs_by_ids(document_ids)
        from .index_errors import document_failures
        return {
            "failures": [f.detail for f in document_failures(statuses)],
            "documents": {key: self._status_value(statuses.get(key)) for key in document_ids},
            "pipeline_active": (
                handle.lock.locked()
                or self._lease_active(workspace)
                or await self._pipeline_active(handle.rag)
            ),
            "metadata": self.metadata.model_dump(mode="json"),
        }

    def _lease_active(self, workspace: str) -> bool:
        path = self._settings.storage_dir / workspace / ".moonlightbox-index-lease"
        if not path.exists():
            return False
        probe = WorkspaceIndexLease(
            self._settings.storage_dir, workspace, self._settings.index_lease_ttl_seconds
        )
        return not probe._is_stale()

    async def clone_workspace(self, source_workspace: str, workspace: str) -> None:
        """复制已完成的 workspace，给候选 Profile/Graph 提供独立可写副本。

        这是存储层复制，绝不重新发送历史聊天给抽取模型。复制前会刷新来源实例，
        复制后清掉内存 handle，避免旧文件句柄继续写入来源或候选目录。
        """

        source_path = self._settings.storage_dir / source_workspace
        destination_path = self._settings.storage_dir / workspace
        if not source_path.is_dir():
            raise RuntimeError("来源 LightRAG workspace 不存在")
        if destination_path.exists():
            raise RuntimeError("候选 LightRAG workspace 已存在，不能覆盖")
        if self._lease_active(source_workspace):
            raise IndexInProgress("来源 LightRAG workspace 正在构建，不能复制")

        source_handle = await self._get_handle(source_workspace)
        async with source_handle.lock:
            await source_handle.rag.finalize_storages()
            async with self._registry_lock:
                # finalize 后移除缓存；后续请求会从各自目录重新初始化。
                self._handles.pop(source_workspace, None)
                if workspace in self._handles:
                    raise RuntimeError("候选 LightRAG workspace 已被占用")
            try:
                await asyncio.to_thread(shutil.copytree, source_path, destination_path)
            except FileExistsError as error:
                raise RuntimeError("候选 LightRAG workspace 已存在，不能覆盖") from error
            except OSError as error:
                raise RuntimeError("复制候选 LightRAG workspace 失败") from error

    @staticmethod
    def _status_value(record: Any) -> str:
        status = (
            record.get("status") if isinstance(record, dict) else getattr(record, "status", None)
        )
        normalized = getattr(status, "value", status)
        return str(normalized) if normalized is not None else "missing"

    async def _recover_failed_documents(self, rag: Any) -> str:
        from lightrag.kg.shared_storage import (
            commit_manual_retry_request,
            get_namespace_data,
            get_namespace_lock,
            get_pipeline_ingress,
        )

        request_id = uuid4().hex
        pipeline_status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        pipeline_status_lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
        ingress = await get_pipeline_ingress(rag.workspace)
        state: dict[str, Any] = {}
        refusal = await commit_manual_retry_request(
            pipeline_status,
            pipeline_status_lock,
            ingress,
            request_id,
            state,
        )
        if refusal is not None:
            raise RuntimeError(f"LightRAG 失败文档恢复被拒绝：{refusal}")
        await rag.apipeline_process_enqueue_documents()
        return request_id

    async def query(self, workspace: str, request: QueryRequest) -> QueryResponse:
        from lightrag import QueryParam

        handle = await self._get_handle(workspace)
        param = QueryParam(
            mode=request.mode,
            only_need_context=True,
            stream=False,
            top_k=request.top_k,
            chunk_top_k=request.chunk_top_k,
            max_total_tokens=request.max_total_tokens,
            enable_rerank=False,
            include_references=True,
        )
        async with handle.lock:
            if request.temporal is not None:
                from .temporal_query import query_temporal

                return await query_temporal(
                    handle.rag,
                    request,
                    param,
                    self._settings.storage_dir / workspace,
                    self.metadata,
                )
            result = await handle.rag.aquery_llm(request.query, param=param)
        if not isinstance(result, dict) or result.get("status") == "failure":
            # LightRAG's complete query API reports retrieval/model failures in
            # a successful Python return value. Do not turn that into a false
            # 200 response with an empty context; the caller needs a retryable
            # Sidecar failure instead.
            raise RuntimeError("LightRAG 查询失败")
        llm_response = result.get("llm_response", {}) if isinstance(result, dict) else {}
        data = result.get("data", {}) if isinstance(result, dict) else {}
        content = llm_response.get("content", "")
        references = data.get("references", [])
        chunks = data.get("chunks", [])
        content_by_reference: dict[str, list[str]] = {}
        for chunk in chunks if isinstance(chunks, list) else []:
            if not isinstance(chunk, dict):
                continue
            reference_id = chunk.get("reference_id")
            chunk_content = chunk.get("content")
            if isinstance(reference_id, str) and isinstance(chunk_content, str):
                content_by_reference.setdefault(reference_id, []).append(chunk_content)
        normalized: list[QueryReference] = []
        for reference in references if isinstance(references, list) else []:
            if not isinstance(reference, dict):
                continue
            file_path = reference.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue
            reference_id = reference.get("reference_id")
            normalized.append(
                QueryReference(
                    file_path=file_path,
                    reference_id=reference_id if isinstance(reference_id, str) else None,
                    content=(
                        "\n\n".join(content_by_reference.get(reference_id, []))
                        if isinstance(reference_id, str)
                        else None
                    )
                    or None,
                )
            )
        return QueryResponse(
            context=content if isinstance(content, str) else "",
            references=normalized,
            metadata=self.metadata,
        )

    async def entities(self, workspace: str) -> EntityListResponse:
        handle = await self._get_handle(workspace)
        async with handle.lock:
            labels = await handle.rag.get_graph_labels()
            valid_labels = (
                [label for label in labels if isinstance(label, str)]
                if isinstance(labels, list)
                else []
            )

            async def read_entity(label: str) -> EntityRead:
                info = await handle.rag.get_entity_info(label)
                graph_data = info.get("graph_data") if isinstance(info, dict) else None
                return EntityRead(
                    entity_name=label,
                    graph_data=graph_data if isinstance(graph_data, dict) else None,
                )

            # get_graph_labels 返回的节点可能很多；并发读取详情避免审核页
            # 因逐个等待数百次存储访问而长时间无响应。
            entities = list(await asyncio.gather(*(read_entity(label) for label in valid_labels)))
        return EntityListResponse(entities=entities)

    async def graph(self, workspace: str) -> GraphResponse:
        """全量导出知识图谱。get_knowledge_graph 的 max_nodes 会被 global_config
        钳制(默认 1000),前端可视化要求看到所有节点,因此直接读存储层。"""
        handle = await self._get_handle(workspace)
        async with handle.lock:
            storage = handle.rag.chunk_entity_relation_graph
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        return GraphResponse(
            nodes=[
                GraphNodeRead(
                    id=str(node.get("id", "")),
                    labels=[],
                    properties={key: value for key, value in node.items() if key != "id"},
                )
                for node in all_nodes
            ],
            edges=[
                GraphEdgeRead(
                    source=str(edge.get("source", "")),
                    target=str(edge.get("target", "")),
                    properties={
                        key: value
                        for key, value in edge.items()
                        if key not in {"source", "target"}
                    },
                )
                for edge in all_edges
            ],
            is_truncated=False,
        )

    async def merge_entities(
        self, workspace: str, request: EntityMergeRequest
    ) -> EntityMergeResponse:
        handle = await self._get_handle(workspace)
        async with handle.lock:

            async def mutate() -> object:
                return await handle.rag.amerge_entities(
                    request.source_entities,
                    request.target_entity,
                    request.merge_strategy or None,
                )

            result = await self._run_mutation(
                workspace,
                idempotency_key=request.idempotency_key,
                request_payload=request.model_dump(mode="json"),
                mutate=mutate,
            )
        return EntityMergeResponse(result=result)

    async def entity(self, workspace: str, entity_name: str) -> EntityRead:
        handle = await self._get_handle(workspace)
        async with handle.lock:
            info = await handle.rag.get_entity_info(entity_name)
        graph_data = info.get("graph_data") if isinstance(info, dict) else None
        return EntityRead(
            entity_name=entity_name,
            graph_data=graph_data if isinstance(graph_data, dict) else None,
        )

    async def relation(
        self, workspace: str, source_entity: str, target_entity: str
    ) -> GraphMutationResponse:
        handle = await self._get_handle(workspace)
        async with handle.lock:
            info = await handle.rag.get_relation_info(source_entity, target_entity)
        return GraphMutationResponse(result=info if isinstance(info, dict) else {})

    async def create_entity(
        self, workspace: str, request: EntityMutationRequest
    ) -> GraphMutationResponse:
        handle = await self._get_handle(workspace)
        async with handle.lock:

            async def mutate() -> object:
                return await handle.rag.acreate_entity(
                    request.entity_name,
                    {"description": request.description, "entity_type": request.entity_type},
                )

            result = await self._run_mutation(
                workspace,
                idempotency_key=request.idempotency_key,
                request_payload=request.model_dump(mode="json"),
                mutate=mutate,
            )
        return GraphMutationResponse(result=result)

    async def update_entity(
        self, workspace: str, request: EntityMutationRequest
    ) -> GraphMutationResponse:
        updated: dict[str, str] = {
            "description": request.description,
            "entity_type": request.entity_type,
        }
        if request.new_entity_name is not None:
            updated["entity_name"] = request.new_entity_name
        handle = await self._get_handle(workspace)
        async with handle.lock:

            async def mutate() -> object:
                return await handle.rag.aedit_entity(
                    request.entity_name,
                    updated,
                    allow_rename=request.allow_rename,
                    allow_merge=False,
                )

            result = await self._run_mutation(
                workspace,
                idempotency_key=request.idempotency_key,
                request_payload=request.model_dump(mode="json"),
                mutate=mutate,
            )
        return GraphMutationResponse(result=result)

    async def delete_entity(
        self, workspace: str, request: EntityDeleteRequest
    ) -> GraphMutationResponse:
        if not request.cascade:
            raise RuntimeError("删除实体会级联删除关系，必须显式批准 cascade=true")
        handle = await self._get_handle(workspace)
        async with handle.lock:

            async def mutate() -> object:
                return await handle.rag.adelete_by_entity(request.entity_name)

            result = await self._run_mutation(
                workspace,
                idempotency_key=request.idempotency_key,
                request_payload=request.model_dump(mode="json"),
                mutate=mutate,
            )
        return GraphMutationResponse(result=result)

    async def create_relation(
        self, workspace: str, request: RelationMutationRequest
    ) -> GraphMutationResponse:
        handle = await self._get_handle(workspace)
        async with handle.lock:

            async def mutate() -> object:
                return await handle.rag.acreate_relation(
                    request.source_entity,
                    request.target_entity,
                    {
                        "description": request.description,
                        "keywords": request.keywords,
                        "weight": request.weight,
                    },
                )

            result = await self._run_mutation(
                workspace,
                idempotency_key=request.idempotency_key,
                request_payload=request.model_dump(mode="json"),
                mutate=mutate,
            )
        return GraphMutationResponse(result=result)

    async def update_relation(
        self, workspace: str, request: RelationMutationRequest
    ) -> GraphMutationResponse:
        handle = await self._get_handle(workspace)
        async with handle.lock:

            async def mutate() -> object:
                return await handle.rag.aedit_relation(
                    request.source_entity,
                    request.target_entity,
                    {
                        "description": request.description,
                        "keywords": request.keywords,
                        "weight": request.weight,
                    },
                )

            result = await self._run_mutation(
                workspace,
                idempotency_key=request.idempotency_key,
                request_payload=request.model_dump(mode="json"),
                mutate=mutate,
            )
        return GraphMutationResponse(result=result)

    async def delete_relation(
        self, workspace: str, request: RelationDeleteRequest
    ) -> GraphMutationResponse:
        handle = await self._get_handle(workspace)
        async with handle.lock:

            async def mutate() -> object:
                return await handle.rag.adelete_by_relation(
                    request.source_entity, request.target_entity
                )

            result = await self._run_mutation(
                workspace,
                idempotency_key=request.idempotency_key,
                request_payload=request.model_dump(mode="json"),
                mutate=mutate,
            )
        return GraphMutationResponse(result=result)

    async def _run_mutation(
        self,
        workspace: str,
        *,
        idempotency_key: str | None,
        request_payload: dict[str, object],
        mutate: Callable[[], Awaitable[object]],
    ) -> dict[str, object]:
        """执行并持久化幂等结果；调用方必须已持有 workspace 锁。"""

        if idempotency_key is None:
            return _mutation_result(await mutate())
        request_hash = hashlib.sha256(
            json.dumps(
                request_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        path = self._settings.storage_dir / ".moonlightbox-mutations" / f"{workspace}.json"
        records = _read_mutation_records(path)
        previous = records.get(idempotency_key)
        if isinstance(previous, dict):
            if previous.get("request_hash") != request_hash:
                raise RuntimeError("同一 idempotency_key 对应了不同的图谱请求")
            cached = previous.get("result")
            if previous.get("status") == "succeeded" and isinstance(cached, dict):
                return cached
            # 上一个进程可能已经完成了图操作但还没来得及保存结果。候选图
            # 可以废弃，因此这里拒绝猜测和重复写入。
            raise RuntimeError("图谱操作上次执行结果不明确，必须废弃候选图后重建")
        records[idempotency_key] = {"request_hash": request_hash, "status": "pending"}
        _write_mutation_records(path, records)
        try:
            result = _mutation_result(await mutate())
        except Exception:
            records[idempotency_key] = {"request_hash": request_hash, "status": "unknown"}
            _write_mutation_records(path, records)
            raise
        records[idempotency_key] = {
            "request_hash": request_hash,
            "status": "succeeded",
            "result": result,
        }
        _write_mutation_records(path, records)
        return result

    async def close(self) -> None:
        async with self._registry_lock:
            handles = [*self._handles.values(), *self._retired_handles]
            self._handles.clear()
            self._retired_handles.clear()
        for handle in handles:
            await handle.rag.finalize_storages()

    async def _get_handle(self, workspace: str) -> _WorkspaceHandle:
        existing = self._handles.get(workspace)
        if existing is not None and (
            existing.generation == self._generation or existing.lock.locked()
        ):
            return existing
        async with self._registry_lock:
            existing = self._handles.get(workspace)
            if existing is not None and (
                existing.generation == self._generation or existing.lock.locked()
            ):
                return existing
            if existing is not None:
                # 不在这里 finalize：可能仍有只读查询持有旧 handle。旧实例仅在
                # Sidecar 关闭时回收；后续请求立即使用新配置创建的新实例。
                self._retired_handles.append(existing)
            settings = self._settings
            handle = _WorkspaceHandle(
                rag=await self._initialize_rag(workspace, settings),
                lock=asyncio.Lock(),
                generation=self._generation,
                metadata=self._metadata_for(settings),
            )
            self._handles[workspace] = handle
            return handle

    async def _initialize_rag(self, workspace: str, settings: SidecarSettings | None = None) -> Any:
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import Tokenizer, wrap_embedding_func_with_attrs

        current = settings or self._settings
        llm_key, embedding_key = current.require_model_keys()

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, str]] | None = None,
            **kwargs: Any,
        ) -> str:
            kwargs = self._llm_request_kwargs(kwargs, settings=current)
            return await openai_complete_if_cache(
                current.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=llm_key,
                base_url=current.llm_base_url,
                **kwargs,
            )

        @wrap_embedding_func_with_attrs(
            embedding_dim=current.embedding_dimension,
            max_token_size=current.embedding_max_tokens,
            model_name=current.embedding_model,
        )
        async def embedding_func(texts: list[str]) -> Any:
            return await openai_embed.func(
                texts,
                model=current.embedding_model,
                api_key=embedding_key,
                base_url=current.embedding_base_url,
            )

        rag = LightRAG(
            working_dir=str(current.storage_dir),
            workspace=workspace,
            llm_model_func=llm_model_func,
            llm_model_name=current.llm_model,
            tokenizer=Tokenizer(current.llm_model, _OfflineTokenizer()),
            chunking_func=source_aware_chunking,
            embedding_func=embedding_func,
            chunk_token_size=current.chunk_token_size,
            chunk_overlap_token_size=current.chunk_overlap_token_size,
            max_parallel_insert=current.max_parallel_insert,
            llm_model_max_async=current.max_async_llm,
            default_llm_timeout=current.llm_timeout_seconds,
            llm_model_kwargs={"max_tokens": current.llm_max_tokens},
            summary_context_size=current.summary_context_size,
            summary_max_tokens=current.summary_max_tokens,
            force_llm_summary_on_merge=current.force_llm_summary_on_merge,
            addon_params={
                "language": "Chinese",
                "entity_types_guidance": ENTITY_TYPE_GUIDANCE,
            },
        )
        await rag.initialize_storages()
        return rag

    def _llm_request_kwargs(
        self, kwargs: dict[str, Any], *, settings: SidecarSettings | None = None
    ) -> dict[str, Any]:
        if not (settings or self._settings).llm_reasoning_split:
            return kwargs
        request_kwargs = dict(kwargs)
        extra_body = request_kwargs.get("extra_body")
        merged_extra_body = dict(extra_body) if isinstance(extra_body, dict) else {}
        # LightRAG uses priority 8 exclusively for entity/relation merge
        # summaries.  These calls must remain short and deterministic; using
        # MiniMax's reasoning channel here was observed to hold a worker until
        # its 600-second execution timeout.  Extraction calls retain the
        # configured reasoning mode.
        priority = request_kwargs.get("_priority")
        merged_extra_body.setdefault("reasoning_split", priority != SUMMARY_LLM_PRIORITY)
        request_kwargs["extra_body"] = merged_extra_body
        return request_kwargs


def _mutation_result(value: Any) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {"value": str(dumped)}
    return {"value": str(value)}


def _read_mutation_records(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # 损坏的幂等账本不能被静默覆盖，否则可能重复执行破坏性操作。
        raise RuntimeError("图谱幂等账本无法读取，需要人工检查") from None
    if not isinstance(value, dict):
        raise RuntimeError("图谱幂等账本格式无效，需要人工检查")
    return value


def _write_mutation_records(path: Path, records: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(records, ensure_ascii=False, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
