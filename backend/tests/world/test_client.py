import json

import httpx
import pytest
from moonlightbox.world.client import LightRAGDocument, LightRAGSidecarClient
from moonlightbox.world.client import LightRAGSidecarError


def test_persisted_content_rejection_is_not_submitted_again():
    writes = 0

    def handler(request):
        nonlocal writes
        if request.url.path.endswith("documents:status"):
            return httpx.Response(200, json={
                "documents": {"doc-1": "failed"}, "pipeline_active": False,
                "metadata": metadata(),
                "failures": [{"code": "lightrag_content_rejected", "document_id": "doc-1", "chunk_id": "doc-1-chunk-004"}],
            })
        writes += 1
        return httpx.Response(200, json={})

    client = LightRAGSidecarClient("http://sidecar:9621", "secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(LightRAGSidecarError) as caught:
        client.index_documents("workspace", [LightRAGDocument(id="doc-1", source="doc.txt", text="private")])
    assert caught.value.code == "lightrag_content_rejected"
    assert caught.value.details["chunk_id"] == "doc-1-chunk-004"
    assert writes == 0


@pytest.mark.parametrize("status_code,expected_reads", [(503, 2), (401, 1)])
def test_status_read_retries_only_transient_failures(status_code, expected_reads):
    reads = 0

    def handler(request):
        nonlocal reads
        assert request.url.path.endswith("documents:status")
        reads += 1
        if reads == 1:
            return httpx.Response(status_code, json={})
        return httpx.Response(200, json={"documents": {"doc-1": "processed"},
                                         "pipeline_active": False, "metadata": metadata()})

    client = LightRAGSidecarClient("http://sidecar:9621", "secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    if status_code == 401:
        with pytest.raises(LightRAGSidecarError):
            client.index_status("workspace", ["doc-1"])
    else:
        assert client.index_status("workspace", ["doc-1"])["documents"]["doc-1"] == "processed"
    assert reads == expected_reads


@pytest.mark.parametrize("still_running", [False, True])
def test_uncertain_write_is_checked_not_replayed(still_running):
    writes = 0

    def handler(request):
        nonlocal writes
        if request.url.path.endswith("documents:status"):
            return httpx.Response(200, json={
                "documents": {"doc-1": "processed" if writes else "missing"},
                "pipeline_active": bool(writes and still_running), "metadata": metadata(),
            })
        writes += 1
        raise httpx.ReadTimeout("lost reply", request=request)

    client = LightRAGSidecarClient("http://sidecar:9621", "secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    documents = [LightRAGDocument(id="doc-1", source="doc-1.txt", text="hello")]
    if still_running:
        for _ in range(2):
            with pytest.raises(LightRAGSidecarError) as caught:
                client.index_documents("workspace", documents)
            assert caught.value.code == "lightrag_index_running"
    else:
        assert client.index_documents("workspace", documents).embedding_model == "embedding"
        client.index_documents("workspace", documents)
    assert writes == 1


def test_client_indexes_and_requests_context_only_query() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("documents:status"):
            return httpx.Response(200, json={"documents": {"doc-1": "missing"}, "pipeline_active": False, "metadata": metadata()})
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
    assert json.loads(requests[-1].content) == {
        "query": "她在哪里工作？",
        "mode": "mix",
        "top_k": 30,
        "chunk_top_k": 12,
        "max_total_tokens": 16000,
    }


def test_client_clones_workspace_without_sending_documents() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "source_workspace": "world_baseline",
                "workspace": "world_candidate",
                "metadata": metadata(),
            },
        )

    client = LightRAGSidecarClient(
        "http://sidecar:9621",
        "secret",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = client.clone_workspace(
        source_workspace="world_baseline",
        workspace="world_candidate",
    )

    assert result.extraction_model == "extractor"
    assert requests[0].url.path == "/v1/workspaces/world_candidate:clone"
    assert b'"source_workspace":"world_baseline"' in requests[0].content


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
