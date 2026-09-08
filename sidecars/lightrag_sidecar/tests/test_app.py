from fastapi.testclient import TestClient
from pydantic import SecretStr

from moonlightbox_lightrag_sidecar.app import create_app
from moonlightbox_lightrag_sidecar.config import SidecarSettings
from moonlightbox_lightrag_sidecar.schemas import (
    DocumentInput,
    QueryReference,
    QueryRequest,
    QueryResponse,
    SidecarMetadata,
)


class FakeRegistry:
    def __init__(self) -> None:
        self.index_calls: list[tuple[str, list[DocumentInput]]] = []
        self.query_calls: list[tuple[str, QueryRequest]] = []
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

    async def index(self, workspace: str, documents: list[DocumentInput]) -> str:
        self.index_calls.append((workspace, documents))
        return "track-1"

    async def query(self, workspace: str, request: QueryRequest) -> QueryResponse:
        self.query_calls.append((workspace, request))
        return QueryResponse(
            context="目标人物在新公司工作。",
            references=[QueryReference(file_path="doc-1.txt")],
            metadata=self.metadata,
        )

    async def close(self) -> None:
        self.closed = True


def settings(tmp_path) -> SidecarSettings:
    return SidecarSettings.model_construct(
        storage_dir=tmp_path,
        api_token=SecretStr("test-token"),
    )


def test_requires_bearer_token(tmp_path) -> None:
    registry = FakeRegistry()
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        response = client.post(
            "/v1/workspaces/project_1/documents:batch",
            json={"documents": [{"id": "doc-1", "source": "doc-1.txt", "text": "hello"}]},
        )
    assert response.status_code == 401


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


def test_rejects_workspace_path_traversal(tmp_path) -> None:
    registry = FakeRegistry()
    with TestClient(create_app(settings(tmp_path), registry)) as client:
        response = client.post(
            "/v1/workspaces/..%2Fother/query",
            headers={"Authorization": "Bearer test-token"},
            json={"query": "test query"},
        )
    assert response.status_code in {404, 422}
