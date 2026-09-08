import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from lightrag import LightRAG, QueryParam
from pydantic import SecretStr

from moonlightbox_lightrag_sidecar.config import SidecarSettings
from moonlightbox_lightrag_sidecar.core import CoreLightRAGRegistry, _WorkspaceHandle
from moonlightbox_lightrag_sidecar.schemas import DocumentInput, QueryRequest


def test_reasoning_split_preserves_existing_extra_body_options() -> None:
    registry = CoreLightRAGRegistry(
        SidecarSettings.model_construct(llm_reasoning_split=True)
    )

    assert registry._llm_request_kwargs({}) == {
        "extra_body": {"reasoning_split": True}
    }
    assert registry._llm_request_kwargs(
        {"extra_body": {"service_tier": "standard"}}
    ) == {
        "extra_body": {
            "service_tier": "standard",
            "reasoning_split": True,
        }
    }


def test_reasoning_split_is_disabled_for_lightrag_summary_priority() -> None:
    registry = CoreLightRAGRegistry(
        SidecarSettings.model_construct(llm_reasoning_split=True)
    )

    assert registry._llm_request_kwargs({"_priority": 8}) == {
        "_priority": 8,
        "extra_body": {"reasoning_split": False},
    }
    assert registry._llm_request_kwargs({"_priority": 10}) == {
        "_priority": 10,
        "extra_body": {"reasoning_split": True},
    }


def test_pinned_lightrag_core_supports_required_workspace_contract(tmp_path) -> None:
    assert "workspace" in inspect.signature(LightRAG).parameters
    assert "ids" in inspect.signature(LightRAG.ainsert).parameters
    assert "file_paths" in inspect.signature(LightRAG.ainsert).parameters
    assert "include_references" in inspect.signature(QueryParam).parameters

    async def initialize() -> None:
        settings = SidecarSettings.model_construct(
            storage_dir=tmp_path,
            llm_base_url="http://127.0.0.1:1/v1",
            llm_model="fake",
            llm_api_key=SecretStr("fake"),
            embedding_base_url="http://127.0.0.1:1/v1",
            embedding_model="fake-embedding",
            embedding_dimension=3,
            embedding_max_tokens=1024,
            embedding_api_key=SecretStr("fake"),
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            max_parallel_insert=3,
            max_async_llm=4,
            llm_timeout_seconds=3600,
        )
        registry = CoreLightRAGRegistry(settings)
        handle = await registry._get_handle("world_contract")
        assert handle.rag.workspace == "world_contract"
        assert handle.rag.default_llm_timeout == 3600
        await registry.close()

    asyncio.run(initialize())


def test_core_query_surfaces_lightrag_failure_status() -> None:
    class FailedQueryRag:
        async def aquery_llm(self, *_args, **_kwargs):
            return {
                "status": "failure",
                "message": "upstream details must not be returned as context",
                "data": {},
                "llm_response": {"content": None},
            }

    async def query() -> None:
        registry = CoreLightRAGRegistry(SidecarSettings.model_construct())
        registry._handles["world_failure"] = _WorkspaceHandle(
            rag=FailedQueryRag(),
            lock=asyncio.Lock(),
        )
        with pytest.raises(RuntimeError, match="LightRAG 查询失败"):
            await registry.query(
                "world_failure",
                QueryRequest(query="目标人物在哪里工作？"),
            )

    asyncio.run(query())


def test_core_index_rejects_partially_failed_batch() -> None:
    class PartiallyFailedRag:
        status_reads = 0

        async def ainsert(self, *_args, **_kwargs):
            return "track-1"

        async def aget_docs_by_ids(self, _ids):
            self.status_reads += 1
            if self.status_reads == 1:
                return {}
            return {
                "doc-1": SimpleNamespace(status="processed"),
                "doc-2": SimpleNamespace(status="failed"),
            }

    async def index() -> None:
        registry = CoreLightRAGRegistry(SidecarSettings.model_construct())
        registry._handles["world_failure"] = _WorkspaceHandle(
            rag=PartiallyFailedRag(),
            lock=asyncio.Lock(),
        )
        with pytest.raises(
            RuntimeError,
            match=r"LightRAG 索引未完整成功（failed=1, processed=1）",
        ):
            await registry.index(
                "world_failure",
                [
                    DocumentInput(id="doc-1", source="doc-1.txt", text="hello"),
                    DocumentInput(id="doc-2", source="doc-2.txt", text="world"),
                ],
            )

    asyncio.run(index())


def test_core_index_recovers_existing_failed_documents() -> None:
    class RecoverableRag:
        workspace = "world_retry"
        recovered = False

        async def aget_docs_by_ids(self, _ids):
            status = "processed" if self.recovered else "failed"
            return {"doc-1": {"status": status}}

        async def ainsert(self, *_args, **_kwargs):
            raise AssertionError("existing failed documents must not be reinserted")

    async def index() -> None:
        rag = RecoverableRag()
        registry = CoreLightRAGRegistry(SidecarSettings.model_construct())
        registry._handles["world_retry"] = _WorkspaceHandle(
            rag=rag,
            lock=asyncio.Lock(),
        )

        async def recover(_rag):
            rag.recovered = True
            return "retry-1"

        registry._recover_failed_documents = AsyncMock(side_effect=recover)
        result = await registry.index(
            "world_retry",
            [DocumentInput(id="doc-1", source="doc-1.txt", text="hello")],
        )

        assert result == "retry-1"
        registry._recover_failed_documents.assert_awaited_once_with(rag)

    asyncio.run(index())


@pytest.mark.parametrize("interrupted_status", ["pending", "parsing", "analyzing", "processing"])
def test_core_index_resumes_interrupted_documents(interrupted_status: str) -> None:
    class InterruptedRag:
        resumed = False

        async def aget_docs_by_ids(self, _ids):
            status = "processed" if self.resumed else interrupted_status
            return {"doc-1": {"status": status}}

        async def apipeline_process_enqueue_documents(self):
            self.resumed = True

        async def ainsert(self, *_args, **_kwargs):
            raise AssertionError("interrupted documents must be resumed, not reinserted")

    async def index() -> None:
        rag = InterruptedRag()
        registry = CoreLightRAGRegistry(SidecarSettings.model_construct())
        registry._handles["world_resume"] = _WorkspaceHandle(
            rag=rag,
            lock=asyncio.Lock(),
        )

        result = await registry.index(
            "world_resume",
            [DocumentInput(id="doc-1", source="doc-1.txt", text="hello")],
        )

        assert result is None
        assert rag.resumed is True

    asyncio.run(index())
