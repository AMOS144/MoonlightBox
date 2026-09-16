"""时间视图冒烟测试：混合边界、排序前过滤、来源失败与共享图隔离。"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from nano_vectordb import NanoVectorDB
from pydantic import ValidationError

from moonlightbox_lightrag_sidecar.schemas import DocumentInput, TemporalScope
from moonlightbox_lightrag_sidecar.source_index import (
    SourceIndex,
    TemporalIndexError,
    classify_chunks,
    resolve_chunk,
    source_aware_chunking,
)
from moonlightbox_lightrag_sidecar.temporal_query import ScopedGraph, ScopedVectors


def document():
    return DocumentInput(
        id="doc",
        source="doc.txt",
        text="上午\n下午",
        source_version="v1",
        message_spans=[
            {
                "message_id": "a",
                "source_ordinal": 0,
                "sent_at": "2026-01-01T12:00:00",
                "start": 0,
                "end": 2,
            },
            {
                "message_id": "b",
                "source_ordinal": 1,
                "sent_at": "2026-01-01T12:00:00",
                "start": 3,
                "end": 5,
            },
        ],
    )


def test_mixed_whole_before_same_timestamp_and_runtime():
    doc = document()
    chunks = {
        key: {"content": text, "full_doc_id": "doc"}
        for key, text in {
            "mixed": doc.text,
            "before": "上午",
            "after": "下午",
            "unknown": "不存在",
        }.items()
    }
    scope = TemporalScope(source_version="v1", included_count=1, period="before")
    rows, meta, unknown = classify_chunks(chunks, {"doc": doc}, scope)
    assert set(rows) == {"mixed", "before"}
    assert rows["mixed"]["content"] == doc.text
    assert meta["mixed"]["boundary_relation"] == "mixed"
    assert [m["period"] for m in meta["mixed"]["messages"]] == ["before", "after"]
    assert unknown == ["unknown"]
    assert set(
        classify_chunks(chunks, {"doc": doc}, scope.model_copy(update={"period": "after"}))[0]
    ) == {"after"}
    assert set(
        classify_chunks(chunks, {"doc": doc}, scope.model_copy(update={"access": "runtime"}))[0]
    ) == {"before"}


def test_source_index_immutable_and_versioned(tmp_path):
    store = SourceIndex(tmp_path)
    with pytest.raises(TemporalIndexError, match="missing"):
        store.read()
    doc = document()
    store.publish([doc])
    store.publish([doc])
    assert store.read()["doc"] == doc
    with pytest.raises(TemporalIndexError, match="conflict|version_mismatch"):
        store.publish([doc.model_copy(update={"source_version": "v2"})])
    with pytest.raises(TemporalIndexError, match="version_mismatch"):
        classify_chunks(
            {}, store.read(), TemporalScope(source_version="v2", included_count=1, period="all")
        )
    with pytest.raises(ValidationError):
        DocumentInput(
            id="bad", source="bad.txt", text="遗漏", source_version="v1", message_spans=[]
        )


def test_real_chunker_retains_repeated_text_offsets():
    from lightrag.utils import Tokenizer

    from moonlightbox_lightrag_sidecar.core import _OfflineTokenizer

    rows = source_aware_chunking(
        Tokenizer("test", _OfflineTokenizer()),
        "重复重复重复",
        chunk_token_size=2,
        chunk_overlap_token_size=0,
    )
    assert [r["moonlight_source_span"] for r in rows] == [
        {"start": 0, "end": 2},
        {"start": 2, "end": 4},
        {"start": 4, "end": 6},
    ]
    doc = DocumentInput(id="doc", source="doc.txt", text="重复重复重复")
    assert resolve_chunk({"content": "重复"}, doc) is None
    assert resolve_chunk(rows[1], doc) == (2, 4)
    from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result

    persisted = build_chunks_dict_from_chunking_result(rows, doc_id="doc", file_path="doc.txt")
    assert persisted["doc-chunk-001"]["moonlight_source_span"] == {"start": 2, "end": 4}


def test_index_follows_lightrag_sanitization_once(tmp_path):
    from moonlightbox_lightrag_sidecar.source_index import normalize_document

    doc = DocumentInput(
        id="doc",
        source="doc.txt",
        text="甲\x05\n&amp;lt;",
        source_version="v1",
        message_spans=[
            {"message_id": "a", "source_ordinal": 0, "sent_at": "2026-01-01", "start": 0, "end": 2},
            {
                "message_id": "b",
                "source_ordinal": 1,
                "sent_at": "2026-01-01",
                "start": 3,
                "end": 11,
            },
        ],
    )
    normalized = normalize_document(doc)
    assert normalized.text == "甲\n&lt;"
    assert normalized.message_spans[1].start == 2
    store = SourceIndex(tmp_path)
    store.publish([doc])
    store.publish([doc])
    assert store.read()["doc"] == normalized


def test_real_nano_prefilter_recovers_below_global_top_k(tmp_path):
    client = NanoVectorDB(2, storage_file=str(tmp_path / "vdb.json"))
    client.upsert(
        [
            {"__id__": "future", "__vector__": np.array([1.0, 0.0])},
            {"__id__": "past", "__vector__": np.array([0.8, 0.2])},
        ]
    )

    async def get_client():
        return client

    storage = SimpleNamespace(_get_client=get_client, cosine_better_than_threshold=-1)
    wrapper = ScopedVectors(storage, lambda row: row["__id__"] == "past")
    result = asyncio.run(wrapper.query("query", 1, np.array([1.0, 0.0])))
    assert result[0]["id"] == "past"
    assert client.query(np.array([1.0, 0.0]), top_k=1)[0]["__id__"] == "future"


def test_graph_expansion_and_degree_use_same_scope():
    nodes = [
        {"id": "a", "source_id": "past"},
        {"id": "b", "source_id": "past"},
        {"id": "c", "source_id": "future"},
    ]
    edges = [
        {"source": "a", "target": "b", "source_id": "past"},
        {"source": "a", "target": "c", "source_id": "future"},
    ]
    view = ScopedGraph(nodes, edges, {"past"})
    assert asyncio.run(view.node_degrees_batch(["a"])) == {"a": 1}
    assert asyncio.run(view.get_nodes_edges_batch(["a"])) == {"a": [("a", "b")]}
    assert nodes[2]["source_id"] == "future"
