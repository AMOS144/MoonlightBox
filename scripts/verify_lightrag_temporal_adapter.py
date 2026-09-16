"""真实只读资产回放生产时间适配器，使用已存向量，不冒充在线语义质量测试。"""

import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
from lightrag import QueryParam
from lightrag.utils import Tokenizer
from moonlightbox_lightrag_sidecar.core import _OfflineTokenizer
from moonlightbox_lightrag_sidecar.schemas import QueryRequest, SidecarMetadata, TemporalScope
from moonlightbox_lightrag_sidecar.source_index import SourceIndex
from moonlightbox_lightrag_sidecar.temporal_query import query_temporal
from nano_vectordb import NanoVectorDB


class ReadOnlyVectors:
    def __init__(self, path, embedding):
        self.raw = json.loads(path.read_text())
        self.client = NanoVectorDB(self.raw["embedding_dim"], storage_file=str(path))
        self.embedding_func = embedding
        self.cosine_better_than_threshold = -1
        self.matrix = np.frombuffer(base64.b64decode(self.raw["matrix"]), dtype=np.float32).reshape(
            -1, self.raw["embedding_dim"]
        )

    @property
    async def client_storage(self):
        return self.raw

    async def _get_client(self):
        return self.client

    async def get_vectors_by_ids(self, ids):
        wanted = set(ids)
        return {
            r["__id__"]: self.matrix[i]
            for i, r in enumerate(self.raw["data"])
            if r["__id__"] in wanted
        }


async def verify(workspace, index_dir):
    files = [p for p in workspace.iterdir() if p.suffix in {".json", ".graphml"}]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    docs = SourceIndex(index_dir).read()
    source_version = next(iter(docs.values())).source_version
    ordinals = {s.source_ordinal for d in docs.values() for s in d.message_spans}
    included = sorted(ordinals)[len(ordinals) // 2]
    raw = json.loads((workspace / "vdb_chunks.json").read_text())
    seed = np.frombuffer(base64.b64decode(raw["matrix"]), dtype=np.float32).reshape(
        -1, raw["embedding_dim"]
    )[0]

    async def embedding(texts, **kwargs):
        return np.array([seed for _ in texts])

    graph = nx.read_graphml(workspace / "graph_chunk_entity_relation.graphml")

    async def nodes():
        return [{**data, "id": key} for key, data in graph.nodes(data=True)]

    async def edges():
        return [{**data, "source": a, "target": b} for a, b, data in graph.edges(data=True)]

    chunks = json.loads((workspace / "kv_store_text_chunks.json").read_text())

    async def get_by_ids(ids):
        return [chunks.get(key) for key in ids]

    config = {
        "tokenizer": Tokenizer("offline", _OfflineTokenizer()),
        "kg_chunk_pick_method": "WEIGHT",
        "related_chunk_number": 5,
        "rerank_model_func": None,
    }
    rag = SimpleNamespace(
        vector_storage="NanoVectorDBStorage",
        chunk_entity_relation_graph=SimpleNamespace(get_all_nodes=nodes, get_all_edges=edges),
        text_chunks=SimpleNamespace(
            get_by_ids=get_by_ids, embedding_func=embedding, global_config=config
        ),
        chunks_vdb=ReadOnlyVectors(workspace / "vdb_chunks.json", embedding),
        entities_vdb=ReadOnlyVectors(workspace / "vdb_entities.json", embedding),
        relationships_vdb=ReadOnlyVectors(workspace / "vdb_relationships.json", embedding),
        _build_global_config=lambda: config,
    )
    metadata = SidecarMetadata(
        lightrag_version="1.5.6",
        embedding_model=SourceIndex(index_dir).embedding_signature()[0],
        embedding_dimension=raw["embedding_dim"],
        extraction_model="not-called",
        chunking_strategy="fixed_token",
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        entity_prompt_version="replay",
    )
    results = []
    for method in ["WEIGHT", "VECTOR"]:
        config["kg_chunk_pick_method"] = method
        for access, period in [
            ("compiler", "before"),
            ("compiler", "after"),
            ("compiler", "all"),
            ("runtime", "before"),
        ]:
            request = QueryRequest(
                query="人物生活与工作",
                temporal=TemporalScope(
                    source_version=source_version,
                    included_count=included,
                    period=period,
                    access=access,
                ),
            )
            param = QueryParam(
                mode="mix",
                top_k=8,
                chunk_top_k=12,
                max_total_tokens=100000,
                max_entity_tokens=30000,
                max_relation_tokens=30000,
                enable_rerank=False,
                hl_keywords=["生活"],
                ll_keywords=["人物"],
            )
            response = await query_temporal(rag, request, param, index_dir, metadata)
            assert response.references, (method, access, period)
            assert response.temporal_diagnostics["mapping_complete"]
            for ref in response.references:
                assert ref.content == chunks[ref.chunk_id]["content"]
                if access == "runtime":
                    assert ref.boundary_relation == "before"
                if period == "after":
                    assert ref.boundary_relation == "after"
            assert (response.global_graph_clues is None) == (access == "runtime")
            results.append(
                {
                    "pick_method": method,
                    **response.temporal_diagnostics,
                    "returned_chunks": len(response.references),
                }
            )
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return {"online_models_called": False, "original_assets_unchanged": True, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(verify(args.workspace, args.index_dir)), ensure_ascii=False, indent=2
        )
    )
