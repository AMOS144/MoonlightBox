from fastapi.testclient import TestClient
from pydantic import SecretStr

from moonlightbox_lightrag_sidecar.app import create_app
from moonlightbox_lightrag_sidecar.config import SidecarSettings
from moonlightbox_lightrag_sidecar.schemas import (
    DocumentInput,
    EntityMutationRequest,
    GraphEdgeRead,
    GraphMutationResponse,
    GraphNodeRead,
    GraphResponse,
    QueryReference,
    QueryRequest,
    QueryResponse,
    RuntimeConfigResponse,
    SidecarMetadata,
)


class FakeRegistry:
    def __init__(self) -> None:
        self.index_calls: list[tuple[str, list[DocumentInput]]] = []
        self.clone_calls: list[tuple[str, str]] = []
        self.query_calls: list[tuple[str, QueryRequest]] = []
        self.entity_mutation_calls: list[tuple[str, EntityMutationRequest]] = []
        self.closed = False
        self.metadata = SidecarMetadata(
            lightrag_version="test",
            embedding_model="embedding-test",
            embedding_dimension=3,
            extraction_model="extract-test",
            chunking_strategy="fixed_token",
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            entity_prompt_version="prompt-v1",
        )
        self.runtime_config = RuntimeConfigResponse(
            llm_model="extract-test",
            llm_base_url="https://llm.test/v1",
            llm_key_configured=True,
            embedding_model="embedding-test",
            embedding_base_url="https://embedding.test/v1",
            embedding_dimension=3,
            embedding_key_configured=True,
        )
        self.connection_tests = []

    async def index(self, workspace: str, documents: list[DocumentInput]) -> str:
        self.index_calls.append((workspace, documents))
        return "track-1"

    async def index_status(self, workspace, document_ids):
        return {"documents": {key: "processing" for key in document_ids},
                "pipeline_active": True, "metadata": self.metadata.model_dump()}
    async def clone_workspace(self, source_workspace: str, workspace: str) -> None:
        self.clone_calls.append((source_workspace, workspace))

    async def query(self, workspace: str, request: QueryRequest) -> QueryResponse:
        self.query_calls.append((workspace, request))
        return QueryResponse(
            context="目标人物在新公司工作。",
            references=[QueryReference(file_path="doc-1.txt")],
            metadata=self.metadata,
        )

    async def graph(self, workspace: str) -> GraphResponse:
        return GraphResponse(
            nodes=[
                GraphNodeRead(
                    id="小月",
                    labels=["小月"],
                    properties={"entity_type": "person", "description": "在新公司工作"},
                )
            ],
            edges=[
                GraphEdgeRead(source="小月", target="新公司", properties={"keywords": "工作"})
            ],
            is_truncated=False,
        )

    async def create_entity(
        self, workspace: str, request: EntityMutationRequest
    ) -> GraphMutationResponse:
        self.entity_mutation_calls.append((workspace, request))
        return GraphMutationResponse(
            result={"entity_name": request.entity_name, "description": request.description}
        )

    async def close(self) -> None:
        self.closed = True

    async def reconfigure(self, update):
        self.metadata = self.metadata.model_copy(update={
            "extraction_model": update.llm_model,
            "embedding_model": update.embedding_model,
            "embedding_dimension": update.embedding_dimension,
        })
        self.runtime_config = self.runtime_config.model_copy(update={
            "llm_model": update.llm_model,
            "llm_base_url": update.llm_base_url,
            "embedding_model": update.embedding_model,
            "embedding_base_url": update.embedding_base_url,
            "embedding_dimension": update.embedding_dimension,
        })
        return self.runtime_config

    async def test_connection(self, request):
        self.connection_tests.append(request)


def settings(tmp_path) -> SidecarSettings:
    return SidecarSettings.model_construct(
        storage_dir=tmp_path,
        api_token=SecretStr("test-token"),
    )


def test_index_failure_returns_stable_content_rejection_contract(tmp_path):
    from unittest.mock import AsyncMock

    registry = FakeRegistry()
    registry.index = AsyncMock(side_effect=RuntimeError(
        "doc-1-chunk-004: UnprocessableEntityError: input new_sensitive (1026)"
    ))
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        response = client.post(
            "/v1/workspaces/project_1/documents:batch",
            headers={"Authorization": "Bearer test-token"},
            json={"documents": [{"id": "doc-1", "source": "doc.txt", "text": "private"}]},
        )
    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "lightrag_content_rejected",
        "message": "模型服务拒绝处理部分聊天内容，资料已保留。请检查受阻片段后再恢复。",
            "document_id": "doc-1",
        "chunk_id": "doc-1-chunk-004",
    }


def test_requires_bearer_token(tmp_path) -> None:
    registry = FakeRegistry()
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        response = client.post(
            "/v1/workspaces/project_1/documents:batch",
            json={"documents": [{"id": "doc-1", "source": "doc-1.txt", "text": "hello"}]},
        )
    assert response.status_code == 401


def test_runtime_config_update_is_authenticated_and_never_returns_keys(tmp_path) -> None:
    registry = FakeRegistry()
    payload = {
        "llm_model": "llm-new", "llm_base_url": "https://llm.test/v1",
        "llm_api_key": "llm-secret", "embedding_model": "embed-new",
        "embedding_base_url": "https://embed.test/v1", "embedding_dimension": 1024,
        "embedding_api_key": "embed-secret",
    }
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        assert client.get("/v1/config").status_code == 401
        current = client.get(
            "/v1/config", headers={"Authorization": "Bearer test-token"}
        )
        assert client.put("/v1/config", json=payload).status_code == 401
        response = client.put(
            "/v1/config", headers={"Authorization": "Bearer test-token"}, json=payload
        )
        tested = client.post(
            "/v1/config:test",
            headers={"Authorization": "Bearer test-token"},
            json={
                "service": "embedding",
                "model": "embed-new",
                "base_url": "https://embed.test/v1",
                "embedding_dimension": 1024,
            },
        )
    assert current.status_code == 200
    assert current.json()["llm_base_url"] == "https://llm.test/v1"
    assert response.status_code == 200
    assert response.json()["llm_model"] == "llm-new"
    assert response.json()["embedding_model"] == "embed-new"
    assert "llm-secret" not in response.text
    assert "embed-secret" not in response.text
    assert tested.json() == {"ok": True, "message": "连接成功"}
    assert registry.connection_tests[0].service == "embedding"


def test_status_is_authenticated_and_busy_is_not_a_server_failure(tmp_path):
    from unittest.mock import AsyncMock

    from moonlightbox_lightrag_sidecar.core import IndexInProgress

    registry = FakeRegistry()
    registry.index = AsyncMock(side_effect=IndexInProgress("busy"))
    headers = {"Authorization": "Bearer test-token"}
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        path = "/v1/workspaces/project_1/documents:status"
        assert client.post(path, json={"document_ids": ["doc-1"]}).status_code == 401
        response = client.post(path, headers=headers, json={"document_ids": ["doc-1"]})
        assert response.json()["pipeline_active"] is True
        invalid = client.post(path, headers=headers, json={"document_ids": ["../bad"]})
        assert invalid.status_code == 422
        response = client.post(
            "/v1/workspaces/project_1/documents:batch",
            headers=headers,
            json={"documents": [{"id": "doc-1", "source": "doc.txt", "text": "hello"}]},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "lightrag_index_running"


def test_indexes_and_queries_an_isolated_workspace(tmp_path) -> None:
    registry = FakeRegistry()
    headers = {"Authorization": "Bearer test-token"}
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        indexed = client.post(
            "/v1/workspaces/project_1/documents:batch",
            headers=headers,
            json={"documents": [{"id": "doc-1", "source": "doc-1.txt", "text": "hello"}]},
        )
        queried = client.post(
            "/v1/workspaces/project_1/query",
            headers=headers,
            json={"query": "她在哪里工作？", "mode": "mix"},
        )

    assert indexed.status_code == 200
    assert indexed.json()["indexed_document_ids"] == ["doc-1"]
    assert queried.status_code == 200
    assert queried.json()["context"] == "目标人物在新公司工作。"
    assert registry.index_calls[0][0] == "project_1"
    assert registry.query_calls[0][1].mode == "mix"
    assert registry.closed is True


def test_clones_a_frozen_workspace_without_indexing_documents(tmp_path) -> None:
    registry = FakeRegistry()
    headers = {"Authorization": "Bearer test-token"}
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        response = client.post(
            "/v1/workspaces/world_candidate:clone",
            headers=headers,
            json={"source_workspace": "world_baseline"},
        )

    assert response.status_code == 200
    assert response.json()["source_workspace"] == "world_baseline"
    assert response.json()["workspace"] == "world_candidate"
    assert registry.clone_calls == [("world_baseline", "world_candidate")]
    assert registry.index_calls == []


def test_rejects_workspace_path_traversal(tmp_path) -> None:
    registry = FakeRegistry()
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        response = client.post(
            "/v1/workspaces/..%2Fother/query",
            headers={"Authorization": "Bearer test-token"},
            json={"query": "test query"},
        )
    assert response.status_code in {404, 422}


def test_creates_entity_with_explicit_idempotency_key(tmp_path) -> None:
    registry = FakeRegistry()
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        response = client.post(
            "/v1/workspaces/world_candidate/entities",
            headers={"Authorization": "Bearer test-token"},
            json={
                "entity_name": "洪欣羽",
                "description": "目标人物",
                "entity_type": "Person",
                "allow_rename": False,
                "new_entity_name": None,
                "idempotency_key": "change-set-1:operation-1",
            },
        )

    assert response.status_code == 200
    assert response.json()["result"]["entity_name"] == "洪欣羽"
    assert registry.entity_mutation_calls[0][0] == "world_candidate"
    assert registry.entity_mutation_calls[0][1].idempotency_key == "change-set-1:operation-1"


def test_graph_endpoint_returns_nodes_and_edges(tmp_path) -> None:
    registry = FakeRegistry()
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        assert client.get("/v1/workspaces/world_project/graph").status_code == 401
        response = client.get(
            "/v1/workspaces/world_project/graph",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "nodes": [
            {
                "id": "小月",
                "labels": ["小月"],
                "properties": {"entity_type": "person", "description": "在新公司工作"},
            }
        ],
        "edges": [{"source": "小月", "target": "新公司", "properties": {"keywords": "工作"}}],
        "is_truncated": False,
    }
