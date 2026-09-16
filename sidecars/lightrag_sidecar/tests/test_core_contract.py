import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from lightrag import LightRAG, QueryParam
from pydantic import SecretStr

from moonlightbox_lightrag_sidecar.config import SidecarSettings
from moonlightbox_lightrag_sidecar.core import CoreLightRAGRegistry, _WorkspaceHandle
from moonlightbox_lightrag_sidecar.lease import IndexLeaseBusy, WorkspaceIndexLease
from moonlightbox_lightrag_sidecar.schemas import (
    DocumentInput,
    EntityMutationRequest,
    QueryRequest,
    RuntimeConfigUpdate,
)


def test_status_does_not_wait_for_busy_index_lock():
    from moonlightbox_lightrag_sidecar.core import IndexInProgress

    async def check():
        registry = CoreLightRAGRegistry(SidecarSettings.model_construct(
            embedding_model="test", embedding_dimension=3, llm_model="test",
            chunk_token_size=1200, chunk_overlap_token_size=100,
        ))
        lock = asyncio.Lock()
        rag = SimpleNamespace(
            aget_docs_by_ids=AsyncMock(return_value={"doc-1": {"status": "processing"}})
        )
        registry._handles["workspace"] = _WorkspaceHandle(rag=rag, lock=lock)
        async with lock:
            status = await asyncio.wait_for(
                registry.index_status("workspace", ["doc-1"]), timeout=1
            )
            assert status["pipeline_active"] is True
            assert status["documents"] == {"doc-1": "processing"}
            with pytest.raises(IndexInProgress):
                await registry.index(
                    "workspace",
                    [DocumentInput(id="doc-1", source="doc.txt", text="hello")],
                )
        status = await registry.index_status("workspace", ["doc-1"])
        assert status["pipeline_active"] is False  # 同样的 processing 文档，现在可以走中断恢复。

    asyncio.run(check())


def test_workspace_index_lease_is_exclusive_across_registry_instances(tmp_path):
    async def check() -> None:
        first = WorkspaceIndexLease(tmp_path, "world", ttl_seconds=30)
        second = WorkspaceIndexLease(tmp_path, "world", ttl_seconds=30)
        async with first:
            with pytest.raises(IndexLeaseBusy):
                await second.__aenter__()
        async with second:
            assert (tmp_path / "world" / ".moonlightbox-index-lease").exists()

    asyncio.run(check())


def test_stale_workspace_index_lease_can_be_reclaimed(tmp_path):
    async def check() -> None:
        lease = WorkspaceIndexLease(tmp_path, "world", ttl_seconds=30)
        lease.path.mkdir(parents=True)
        (lease.path / "owner.json").write_text(
            '{"owner":"dead","updated_at":0}', encoding="utf-8"
        )
        async with WorkspaceIndexLease(tmp_path, "world", ttl_seconds=30):
            assert (tmp_path / "world" / ".moonlightbox-index-lease" / "owner.json").exists()

    asyncio.run(check())


def test_reasoning_split_preserves_existing_extra_body_options() -> None:
    registry = CoreLightRAGRegistry(SidecarSettings.model_construct(llm_reasoning_split=True))

    assert registry._llm_request_kwargs({}) == {"extra_body": {"reasoning_split": True}}
    assert registry._llm_request_kwargs({"extra_body": {"service_tier": "standard"}}) == {
        "extra_body": {
            "service_tier": "standard",
            "reasoning_split": True,
        }
    }


def test_runtime_reconfigure_preserves_blank_keys_and_rotates_handles(tmp_path) -> None:
    async def check() -> None:
        settings = SidecarSettings(
            storage_dir=tmp_path,
            llm_model="old-llm",
            llm_base_url="https://old-llm.test/v1",
            llm_api_key=SecretStr("old-llm-key"),
            embedding_model="old-embedding",
            embedding_base_url="https://old-embedding.test/v1",
            embedding_dimension=3,
            embedding_api_key=SecretStr("old-embedding-key"),
        )
        registry = CoreLightRAGRegistry(settings)
        current = await registry.reconfigure(
            RuntimeConfigUpdate(
                llm_model="new-llm",
                llm_base_url="https://new-llm.test/v1",
                llm_api_key=None,
                embedding_model="new-embedding",
                embedding_base_url="https://new-embedding.test/v1",
                embedding_dimension=1024,
                embedding_api_key=None,
            )
        )

        assert current.llm_model == "new-llm"
        assert current.embedding_dimension == 1024
        assert current.llm_key_configured is True
        assert current.embedding_key_configured is True
        assert registry._generation == 1
        assert registry._settings.llm_api_key.get_secret_value() == "old-llm-key"
        assert (
            registry._settings.embedding_api_key.get_secret_value()
            == "old-embedding-key"
        )

    asyncio.run(check())


def test_reasoning_split_is_disabled_for_lightrag_summary_priority() -> None:
    registry = CoreLightRAGRegistry(SidecarSettings.model_construct(llm_reasoning_split=True))

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
        from moonlightbox_lightrag_sidecar.index_errors import IndexFailure
        with pytest.raises(IndexFailure) as caught:
            await registry.index(
                "world_failure",
                [
                    DocumentInput(id="doc-1", source="doc-1.txt", text="hello"),
                    DocumentInput(id="doc-2", source="doc-2.txt", text="world"),
                ],
            )
        assert caught.value.detail["code"] == "lightrag_index_error"

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
        registry._pipeline_active = AsyncMock(return_value=False)
        result = await registry.index(
            "world_retry",
            [DocumentInput(id="doc-1", source="doc-1.txt", text="hello")],
        )

        assert result == "retry-1"
        registry._recover_failed_documents.assert_awaited_once_with(rag)

    asyncio.run(index())


def test_graph_mutation_idempotency_survives_registry_restart(tmp_path) -> None:
    class MutationRag:
        def __init__(self) -> None:
            self.calls = 0

        async def acreate_entity(self, name, payload):
            self.calls += 1
            return {"entity_name": name, **payload}

    async def mutate() -> None:
        settings = SidecarSettings.model_construct(storage_dir=tmp_path)
        request = EntityMutationRequest(
            entity_name="洪欣羽",
            description="目标人物",
            entity_type="Person",
            idempotency_key="change-set-1:operation-1",
        )
        first_rag = MutationRag()
        first = CoreLightRAGRegistry(settings)
        first._handles["world_candidate"] = _WorkspaceHandle(
            rag=first_rag,
            lock=asyncio.Lock(),
        )
        one = await first.create_entity("world_candidate", request)
        two = await first.create_entity("world_candidate", request)
        assert one == two
        assert first_rag.calls == 1

        restarted_rag = MutationRag()
        restarted = CoreLightRAGRegistry(settings)
        restarted._handles["world_candidate"] = _WorkspaceHandle(
            rag=restarted_rag,
            lock=asyncio.Lock(),
        )
        three = await restarted.create_entity("world_candidate", request)
        assert three == one
        assert restarted_rag.calls == 0

        changed = request.model_copy(update={"description": "不同请求"})
        with pytest.raises(RuntimeError, match="不同的图谱请求"):
            await restarted.create_entity("world_candidate", changed)

    asyncio.run(mutate())


def test_graph_mutation_with_unknown_result_is_not_repeated(tmp_path) -> None:
    class UncertainMutationRag:
        def __init__(self) -> None:
            self.calls = 0

        async def acreate_entity(self, name, payload):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection lost after mutation")
            return {"entity_name": name, **payload}

    async def mutate() -> None:
        settings = SidecarSettings.model_construct(storage_dir=tmp_path)
        request = EntityMutationRequest(
            entity_name="洪欣羽",
            description="目标人物",
            entity_type="Person",
            idempotency_key="change-set-2:operation-1",
        )
        rag = UncertainMutationRag()
        registry = CoreLightRAGRegistry(settings)
        registry._handles["world_candidate"] = _WorkspaceHandle(
            rag=rag,
            lock=asyncio.Lock(),
        )

        with pytest.raises(RuntimeError, match="connection lost"):
            await registry.create_entity("world_candidate", request)
        with pytest.raises(RuntimeError, match="执行结果不明确"):
            await registry.create_entity("world_candidate", request)
        assert rag.calls == 1

    asyncio.run(mutate())


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
