"""LightRAG 1.5.6 / NanoVectorDB 的请求局部时间视图。

候选先按来源筛选再做相似度排名；不修改共享 storage、不复制查询流程。
私有 API 仅集中在本模块，升级 LightRAG 时必须跑真实混合召回契约测试。
"""

import json
from importlib.metadata import version
from types import SimpleNamespace

from .schemas import QueryReference, QueryResponse
from .source_index import SourceIndex, TemporalIndexError, classify_chunks


class ScopedVectors:
    """把允许集合交给 Nano 的 filter_lambda，而非 top-k 后过滤。"""

    def __init__(self, storage, allows):
        self.storage, self.allows = storage, allows
        self.cosine_better_than_threshold = storage.cosine_better_than_threshold

    async def query(self, query, top_k, query_embedding=None):
        embedding = query_embedding
        if embedding is None:
            embedding = (await self.storage.embedding_func([query], context="query", _priority=5))[
                0
            ]
        client = await self.storage._get_client()
        rows = client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.storage.cosine_better_than_threshold,
            filter_lambda=self.allows,
        )
        return [
            {
                **{k: v for k, v in row.items() if k != "vector"},
                "id": row["__id__"],
                "distance": row["__metrics__"],
            }
            for row in rows
        ]

    async def get_vectors_by_ids(self, ids):
        # VECTOR 二次挑选也必须受限，不能只限制最初向量查询。
        stored = await self.storage.client_storage
        allowed = {row["__id__"] for row in stored["data"] if self.allows(row)}
        return await self.storage.get_vectors_by_ids([key for key in ids if key in allowed])


class ScopedGraph:
    def __init__(self, nodes, edges, allowed_chunks):
        import networkx as nx
        from lightrag.constants import GRAPH_FIELD_SEP

        self.g = nx.Graph()
        self.excluded_endpoint_edges = 0

        def scoped(row):
            data = dict(row)
            sources = [
                s for s in data.get("source_id", "").split(GRAPH_FIELD_SEP) if s in allowed_chunks
            ]
            data["source_id"] = GRAPH_FIELD_SEP.join(sources)
            return data if sources else None

        for row in nodes:
            data = scoped(row)
            if data:
                self.g.add_node(row["id"], **data)
        for row in edges:
            data = scoped(row)
            if data:
                if row["source"] in self.g and row["target"] in self.g:
                    self.g.add_edge(row["source"], row["target"], **data)
                else:
                    self.excluded_endpoint_edges += 1

    async def get_nodes_batch(self, ids):
        return {key: dict(self.g.nodes[key]) for key in ids if key in self.g}

    async def node_degrees_batch(self, ids):
        return {key: self.g.degree(key) for key in ids if key in self.g}

    async def get_nodes_edges_batch(self, ids):
        return {key: list(self.g.edges(key)) for key in ids if key in self.g}

    async def get_edges_batch(self, pairs):
        return {
            (p["src"], p["tgt"]): dict(self.g.edges[p["src"], p["tgt"]])
            for p in pairs
            if self.g.has_edge(p["src"], p["tgt"])
        }

    async def edge_degrees_batch(self, pairs):
        return {
            (a, b): self.g.degree(a) + self.g.degree(b) for a, b in pairs if self.g.has_edge(a, b)
        }


async def query_temporal(rag, request, param, workspace_path, metadata):
    """调用真实关键词、embedding 与 mix 链路；不复用无时间作用域的结果缓存。"""
    from lightrag.constants import GRAPH_FIELD_SEP
    from lightrag.operate import _build_query_context, get_keywords_from_query

    if version("lightrag-hku") != "1.5.6" or rag.vector_storage != "NanoVectorDBStorage":
        raise TemporalIndexError("temporal_adapter_incompatible")
    scope = request.temporal
    source_index = SourceIndex(workspace_path)
    documents = source_index.read()
    if source_index.embedding_signature() != (
        metadata.embedding_model,
        metadata.embedding_dimension,
    ):
        raise TemporalIndexError("temporal_embedding_mismatch")
    ordinals = {m.source_ordinal for doc in documents.values() for m in doc.message_spans or []}
    if not ordinals or scope.included_count > max(ordinals) + 1:
        raise TemporalIndexError("temporal_boundary_out_of_range")
    nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
    edges = await rag.chunk_entity_relation_graph.get_all_edges()
    vector_rows = (await rag.chunks_vdb.client_storage)["data"]
    chunk_ids = {r["__id__"] for r in vector_rows}
    for row in [*nodes, *edges]:
        chunk_ids.update(s for s in row.get("source_id", "").split(GRAPH_FIELD_SEP) if s)
    ordered_ids = sorted(chunk_ids)
    records = await rag.text_chunks.get_by_ids(ordered_ids)
    chunks = {key: row for key, row in zip(ordered_ids, records, strict=True) if row}
    chunks = source_index.attach_chunk_spans(chunks)
    selected, source_metadata, unknown = classify_chunks(chunks, documents, scope)
    excluded_mixed_runtime = 0
    if scope.access == "runtime":
        _, compiler_metadata, _ = classify_chunks(
            chunks, documents, scope.model_copy(update={"access": "compiler"})
        )
        excluded_mixed_runtime = sum(
            m["boundary_relation"] == "mixed" for m in compiler_metadata.values()
        )
    unknown.extend(sorted(chunk_ids - chunks.keys()))
    if not selected and unknown:
        raise TemporalIndexError("temporal_mapping_unavailable")
    graph = ScopedGraph(nodes, edges, set(selected))

    async def get_by_ids(ids):
        return [selected.get(key) for key in ids]

    kv = SimpleNamespace(
        embedding_func=rag.text_chunks.embedding_func,
        global_config=rag.text_chunks.global_config,
        get_by_ids=get_by_ids,
    )
    data = {}
    if selected:
        # 关键词只依赖当前问题；仍不接全图查询结果缓存，避免时间作用域串用。
        high, low = await get_keywords_from_query(
            request.query, param, rag._build_global_config(), None
        )
        result = await _build_query_context(
            request.query,
            ", ".join(low),
            ", ".join(high),
            graph,
            ScopedVectors(rag.entities_vdb, lambda row: row.get("entity_name") in graph.g),
            ScopedVectors(
                rag.relationships_vdb,
                lambda row: graph.g.has_edge(row.get("src_id"), row.get("tgt_id")),
            ),
            kv,
            param,
            ScopedVectors(rag.chunks_vdb, lambda row: row["__id__"] in selected),
        )
        data = result.raw_data.get("data", {}) if result else {}
    references = []
    for row in data.get("chunks", []):
        key = row.get("chunk_id")
        if key not in selected:
            raise TemporalIndexError("temporal_scope_violation")
        references.append(
            QueryReference(
                file_path=selected[key]["file_path"],
                reference_id=row.get("reference_id"),
                content=selected[key]["content"],
                chunk_id=key,
                **source_metadata[key],
            )
        )
    diagnostics = {
        "policy": scope.policy,
        "period": scope.period,
        "access": scope.access,
        "source_version": scope.source_version,
        "included_count": scope.included_count,
        "candidate_chunks": len(selected),
        "candidate_entities": len(graph.g),
        "candidate_relationships": graph.g.number_of_edges(),
        "mixed_chunks": sum(m["boundary_relation"] == "mixed" for m in source_metadata.values()),
        "unknown_chunk_ids": unknown,
        "mapping_complete": not unknown,
        "excluded_endpoint_edges": graph.excluded_endpoint_edges,
        "historical_vectors": False,
        "excluded_mixed_runtime": excluded_mixed_runtime,
        # 该指标只描述 Runtime 的纯前段原文覆盖；compiler before 可合法包含 mixed。
        "historical_raw_coverage_complete": (
            not unknown and not excluded_mixed_runtime if scope.access == "runtime" else None
        ),
        "result_cache": "disabled",
    }
    # Context 仅包含带来源分类的片段。全图归纳不是当时事实，单独交给编译器。
    return QueryResponse(
        context=json.dumps([r.model_dump() for r in references], ensure_ascii=False),
        references=references,
        metadata=metadata,
        temporal_diagnostics=diagnostics,
        global_graph_clues={
            "scope": "global_retrospective",
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }
        if scope.access == "compiler"
        else None,
    )
