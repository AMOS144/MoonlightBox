import httpx
from moonlightbox.world.client import LightRAGDocument, LightRAGSidecarClient


def test_client_indexes_and_requests_context_only_query() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("documents:batch"):
            return httpx.Response(
                200,
                json={
                    "indexed_document_ids": ["doc-1"],
                    "track_id": "track-1",
                    "metadata": metadata(),
                },
            )
        return httpx.Response(
            200,
            json={
                "context": "context",
                "references": [{"file_path": "doc-1.txt"}],
                "metadata": metadata(),
            },
        )

    transport = httpx.MockTransport(handler)
    client = LightRAGSidecarClient(
        "http://sidecar:9621",
        "secret",
        client=httpx.Client(transport=transport),
    )
    result = client.index_documents(
        "workspace_1",
        [LightRAGDocument(id="doc-1", source="doc-1.txt", text="自然对话")],
    )
    retrieval = client.query("workspace_1", "她在哪里工作？")

    assert result.embedding_model == "embedding"
    assert retrieval.context == "context"
    assert requests[0].headers["authorization"] == "Bearer secret"
    assert b'"mode":"mix"' in requests[1].content


def metadata() -> dict[str, object]:
    return {
        "lightrag_version": "1.5.6",
        "embedding_model": "embedding",
        "embedding_dimension": 3,
        "extraction_model": "extractor",
        "chunking_strategy": "fixed_token",
        "chunk_token_size": 1200,
        "chunk_overlap_token_size": 100,
        "entity_prompt_version": "prompt-v1",
    }
