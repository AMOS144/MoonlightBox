import asyncio
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import uuid4

from moonlightbox_lightrag_sidecar.config import SidecarSettings
from moonlightbox_lightrag_sidecar.schemas import (
    DocumentInput,
    QueryReference,
    QueryRequest,
    QueryResponse,
    SidecarMetadata,
    EntityListResponse,
    EntityMergeRequest,
    EntityMergeResponse,
    EntityRead,
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


class CoreLightRAGRegistry:
    """Own one initialized LightRAG Core instance per immutable graph workspace."""

    def __init__(self, settings: SidecarSettings) -> None:
        self._settings = settings
        self._handles: dict[str, _WorkspaceHandle] = {}
        self._registry_lock = asyncio.Lock()

    @property
    def metadata(self) -> SidecarMetadata:
        try:
            installed_version = version("lightrag-hku")
        except PackageNotFoundError:
            installed_version = "unknown"
        return SidecarMetadata(
            lightrag_version=installed_version,
            embedding_model=self._settings.embedding_model,
            embedding_dimension=self._settings.embedding_dimension,
            extraction_model=self._settings.llm_model,
            chunking_strategy="fixed_token",
            chunk_token_size=self._settings.chunk_token_size,
            chunk_overlap_token_size=self._settings.chunk_overlap_token_size,
            entity_prompt_version=ENTITY_PROMPT_VERSION,
        )

    async def index(self, workspace: str, documents: list[DocumentInput]) -> str | None:
        handle = await self._get_handle(workspace)
        async with handle.lock:
            document_ids = [item.id for item in documents]
            statuses = await handle.rag.aget_docs_by_ids(document_ids)
            failed_ids = {
                document_id
                for document_id, record in statuses.items()
                if self._status_value(record) == "failed"
            }
            interrupted_ids = {
                document_id
                for document_id, record in statuses.items()
                if self._status_value(record)
                in {"pending", "parsing", "analyzing", "processing"}
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
        counts: Counter[str] = Counter()
        for item in documents:
            record = statuses.get(item.id)
            counts[self._status_value(record)] += 1
        if counts != {"processed": len(documents)}:
            summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            raise RuntimeError(f"LightRAG 索引未完整成功（{summary}）")
        return result

    @staticmethod
    def _status_value(record: Any) -> str:
        status = (
            record.get("status")
            if isinstance(record, dict)
            else getattr(record, "status", None)
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
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status_lock = get_namespace_lock(
            "pipeline_status", workspace=rag.workspace
        )
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
            valid_labels = [label for label in labels if isinstance(label, str)] if isinstance(labels, list) else []

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

    async def merge_entities(self, workspace: str, request: EntityMergeRequest) -> EntityMergeResponse:
        handle = await self._get_handle(workspace)
        async with handle.lock:
            result = await handle.rag.amerge_entities(
                request.source_entities,
                request.target_entity,
                request.merge_strategy or None,
            )
        return EntityMergeResponse(result=result if isinstance(result, dict) else {"value": result})

    async def close(self) -> None:
        async with self._registry_lock:
            handles = list(self._handles.values())
            self._handles.clear()
        for handle in handles:
            await handle.rag.finalize_storages()

    async def _get_handle(self, workspace: str) -> _WorkspaceHandle:
        existing = self._handles.get(workspace)
        if existing is not None:
            return existing
        async with self._registry_lock:
            existing = self._handles.get(workspace)
            if existing is not None:
                return existing
            handle = _WorkspaceHandle(
                rag=await self._initialize_rag(workspace),
                lock=asyncio.Lock(),
            )
            self._handles[workspace] = handle
            return handle

    async def _initialize_rag(self, workspace: str) -> Any:
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import Tokenizer
        from lightrag.utils import wrap_embedding_func_with_attrs

        llm_key, embedding_key = self._settings.require_model_keys()

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, str]] | None = None,
            **kwargs: Any,
        ) -> str:
            kwargs = self._llm_request_kwargs(kwargs)
            return await openai_complete_if_cache(
                self._settings.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=llm_key,
                base_url=self._settings.llm_base_url,
                **kwargs,
            )

        @wrap_embedding_func_with_attrs(
            embedding_dim=self._settings.embedding_dimension,
            max_token_size=self._settings.embedding_max_tokens,
            model_name=self._settings.embedding_model,
        )
        async def embedding_func(texts: list[str]) -> Any:
            return await openai_embed.func(
                texts,
                model=self._settings.embedding_model,
                api_key=embedding_key,
                base_url=self._settings.embedding_base_url,
            )

        rag = LightRAG(
            working_dir=str(self._settings.storage_dir),
            workspace=workspace,
            llm_model_func=llm_model_func,
            llm_model_name=self._settings.llm_model,
            tokenizer=Tokenizer(self._settings.llm_model, _OfflineTokenizer()),
            embedding_func=embedding_func,
            chunk_token_size=self._settings.chunk_token_size,
            chunk_overlap_token_size=self._settings.chunk_overlap_token_size,
            max_parallel_insert=self._settings.max_parallel_insert,
            llm_model_max_async=self._settings.max_async_llm,
            default_llm_timeout=self._settings.llm_timeout_seconds,
            llm_model_kwargs={"max_tokens": self._settings.llm_max_tokens},
            summary_context_size=self._settings.summary_context_size,
            summary_max_tokens=self._settings.summary_max_tokens,
            force_llm_summary_on_merge=self._settings.force_llm_summary_on_merge,
            addon_params={
                "language": "Chinese",
                "entity_types_guidance": ENTITY_TYPE_GUIDANCE,
            },
        )
        await rag.initialize_storages()
        return rag

    def _llm_request_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self._settings.llm_reasoning_split:
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
        merged_extra_body.setdefault(
            "reasoning_split", priority != SUMMARY_LLM_PRIORITY
        )
        request_kwargs["extra_body"] = merged_extra_body
        return request_kwargs
