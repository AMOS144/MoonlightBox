"""真实资产只读审计：消息映射与 LightRAG mix 上下文构建回放。

使用已存向量作为查询向量，不联网，不声称测量自然语言检索质量。
SQLite 以 mode=ro 打开；图和向量只载入内存，不初始化可写 LightRAG 实例。
"""

import asyncio
import base64
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
from lightrag import QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import _build_query_context
from lightrag.utils import Tokenizer as RuntimeTokenizer

ROOT = Path(__file__).resolve().parents[1] / "../.runtime-data/data"
PROJECT = "e089f2fe-b7de-4988-a399-019e10b22b78"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
# 只读使用项目后端的 SQLAlchemy；不安装或改动 sidecar 依赖环境。
sys.path.extend(
    str(p)
    for p in (Path(__file__).resolve().parents[1] / ".venv/lib").glob("python*/site-packages")
)
from moonlightbox.imports.quoted_sender_preprocess import (
    build_clean_world_messages,
    preprocess_quoted_senders,
)
from moonlightbox.world.bundles import WorldMessage, render_world_message


class Tokenizer:
    def encode(self, text):
        return list(text.encode())

    def decode(self, tokens):
        return bytes(tokens).decode(errors="replace")


class Vectors:
    cosine_better_than_threshold = -1

    def __init__(self, path, allowed, query_vector):
        raw = json.loads(path.read_text())
        self.rows = raw["data"]
        self.matrix = np.frombuffer(base64.b64decode(raw["matrix"]), dtype=np.float32).reshape(
            -1, raw["embedding_dim"]
        )
        self.allowed, self.query_vector = allowed, query_vector
        self.calls = []
        self.vector_reads = 0

    async def get_vectors_by_ids(self, ids):
        self.vector_reads += 1
        wanted = set(ids)
        return {
            r["__id__"]: self.matrix[i]
            for i, r in enumerate(self.rows)
            if r["__id__"] in wanted and self.allowed(r)
        }

    async def query(self, query, top_k, query_embedding=None):
        ix = [i for i, row in enumerate(self.rows) if self.allowed(row)]
        self.calls.append(len(ix))
        if not ix:
            return []
        q = np.asarray(query_embedding if query_embedding is not None else self.query_vector)
        q = q / np.linalg.norm(q)
        mat = self.matrix[ix]
        scores = (mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)) @ q
        chosen = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.rows[ix[i]], "id": self.rows[ix[i]]["__id__"], "distance": float(scores[i])}
            for i in chosen
        ]


class Graph:
    def __init__(self, graph):
        self.g = graph

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
        return {(a, b): self.g.degree(a) + self.g.degree(b) for a, b in pairs}


def audit(connection, graph_id, chunks):
    bundles = {
        r["document_id"]: r
        for r in connection.execute(
            "SELECT * FROM conversation_bundles WHERE graph_version_id=?", (graph_id,)
        )
    }
    raw = connection.execute(
        "SELECT DISTINCT m.*,p.name,p.role FROM messages m JOIN participants p ON p.id=m.participant_id "
        "JOIN conversation_bundle_messages bm ON bm.message_id=m.id "
        "JOIN conversation_bundles b ON b.id=bm.bundle_id WHERE b.graph_version_id=? "
        "ORDER BY m.timestamp,m.id",
        (graph_id,),
    ).fetchall()
    originals = [
        WorldMessage(
            id=r["id"],
            import_id=r["import_id"],
            participant_id=r["participant_id"],
            participant_name=r["name"],
            participant_role=r["role"],
            timestamp=datetime.fromisoformat(r["timestamp"]),
            kind=r["kind"],
            content=r["content"],
        )
        for r in raw
    ]
    cleaned = {
        m.id: render_world_message(m)
        for m in build_clean_world_messages(originals, preprocess_quoted_senders(originals))
    }
    spans, stats, details = {}, Counter(), []
    for key, chunk in chunks.items():
        bundle = bundles.get(chunk["full_doc_id"])
        if bundle is None:
            stats["missing_bundle"] += 1
            continue
        rows = connection.execute(
            "SELECT m.id,m.timestamp,m.content,p.name FROM conversation_bundle_messages bm "
            "JOIN messages m ON m.id=bm.message_id JOIN participants p ON p.id=m.participant_id "
            "WHERE bm.bundle_id=? ORDER BY bm.ordinal",
            (bundle["id"],),
        ).fetchall()
        rendered = [
            f"{datetime.fromisoformat(r['timestamp']):%Y-%m-%d %H:%M} {r['name']}：{' '.join(r['content'].split())}"
            for r in rows
        ]
        stats["old_mapping_zero_full_lines"] += int(
            not any(line in chunk["content"] for line in rendered)
        )
        if "\n".join(rendered) != bundle["content"]:
            stats["raw_render_mismatch_chunks"] += 1
        raw_rendered = rendered
        rendered = [cleaned[r["id"]] for r in rows]
        # 重放真实建图清洗；只有文本完全一致才使用字符位置映射。
        if "\n".join(rendered) != bundle["content"]:
            stats["bundle_render_mismatch"] += 1
            continue
        content = chunk["content"]
        start = bundle["content"].find(content)
        if start < 0 or bundle["content"].find(content, start + 1) >= 0:
            stats["unresolved_or_ambiguous_chunk"] += 1
            continue
        end, offset, matches, partial = start + len(content), 0, [], False
        old_matches = 0
        for row, line, raw_line in zip(rows, rendered, raw_rendered):
            if raw_line in content:
                old_matches += 1
            if offset < end and offset + len(line) > start:
                matches.append((row["id"], row["timestamp"]))
                partial |= offset < start or offset + len(line) > end
            offset += len(line) + 1
        spans[key] = matches
        stats["exact_offset_chunks"] += 1
        stats["chunks_with_partial_message"] += int(partial)
        stats["old_full_line_mapping_misses_messages"] += int(len(matches) > old_matches)
        if partial and len(details) < 5:
            details.append(
                {"chunk_id": key, "message_count": len(matches), "old_full_line_count": old_matches}
            )
    return spans, dict(stats), details


async def main():
    with sqlite3.connect(
        f"file:{(ROOT / 'moonlightbox.db').resolve()}?mode=ro", uri=True
    ) as connection:
        connection.row_factory = sqlite3.Row
        graph_row = connection.execute(
            "SELECT g.* FROM world_publications p JOIN world_graph_versions g ON g.id=p.graph_version_id "
            "WHERE p.project_id=? AND p.status='active'",
            (PROJECT,),
        ).fetchone()
        workspace = ROOT / "lightrag" / graph_row["workspace_key"]
        asset_files = [
            workspace / name
            for name in [
                "kv_store_text_chunks.json",
                "graph_chunk_entity_relation.graphml",
                "vdb_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
            ]
        ]
        fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in asset_files}
        chunks = json.loads((workspace / "kv_store_text_chunks.json").read_text())
        spans, stats, details = audit(connection, graph_row["id"], chunks)
    graph = nx.read_graphml(workspace / "graph_chunk_entity_relation.graphml")
    times = sorted({time for values in spans.values() for _, time in values})
    assert times, stats
    cutoff = times[len(times) // 2]
    periods = {
        "before": {key for key, rows in spans.items() if rows and all(t < cutoff for _, t in rows)},
        "after": {key for key, rows in spans.items() if rows and all(t >= cutoff for _, t in rows)},
    }
    seed = Vectors(workspace / "vdb_chunks.json", lambda r: True, None)
    results = []
    # 选择三个不同的真实向量作为回放输入，所有检索路径使用同一个向量以检查接线。
    for seed_index, pick_method in [
        (i, method)
        for i in [0, len(seed.rows) // 2, len(seed.rows) - 1]
        for method in ["WEIGHT", "VECTOR"]
    ]:
        query_vector = seed.matrix[seed_index]
        for period, allowed in [("all", set(chunks)), *periods.items()]:
            scoped = graph.copy()
            for _, data in scoped.nodes(data=True):
                data["source_id"] = GRAPH_FIELD_SEP.join(
                    s for s in data.get("source_id", "").split(GRAPH_FIELD_SEP) if s in allowed
                )
            for a, b, data in list(scoped.edges(data=True)):
                data["source_id"] = GRAPH_FIELD_SEP.join(
                    s for s in data.get("source_id", "").split(GRAPH_FIELD_SEP) if s in allowed
                )
                if not data["source_id"]:
                    scoped.remove_edge(a, b)
            scoped.remove_nodes_from(
                [n for n, d in scoped.nodes(data=True) if not d.get("source_id")]
            )
            entity = Vectors(
                workspace / "vdb_entities.json",
                lambda r: r.get("entity_name") in scoped,
                query_vector,
            )
            relation = Vectors(
                workspace / "vdb_relationships.json",
                lambda r: scoped.has_edge(r.get("src_id"), r.get("tgt_id")),
                query_vector,
            )
            vector = Vectors(
                workspace / "vdb_chunks.json", lambda r: r["__id__"] in allowed, query_vector
            )

            async def embedding(texts, **kwargs):
                return np.array([query_vector for _ in texts])

            async def get_by_ids(ids):
                return [dict(chunks[key]) if key in allowed else None for key in ids]

            kv = SimpleNamespace(
                embedding_func=embedding,
                get_by_ids=get_by_ids,
                global_config={
                    "tokenizer": RuntimeTokenizer("probe-bytes", Tokenizer()),
                    "kg_chunk_pick_method": pick_method,
                    "related_chunk_number": 5,
                    "rerank_model_func": None,
                },
            )
            response = await _build_query_context(
                "stored-vector replay",
                "entity replay",
                "relation replay",
                Graph(scoped),
                entity,
                relation,
                kv,
                QueryParam(
                    mode="mix",
                    top_k=8,
                    chunk_top_k=12,
                    max_entity_tokens=100000,
                    max_relation_tokens=100000,
                    max_total_tokens=300000,
                    enable_rerank=False,
                ),
                vector,
            )
            assert response is not None
            data = response.raw_data["data"]
            returned = data.get("chunks", [])
            ids = [c.get("chunk_id", c.get("id")) for c in returned]
            assert ids and all(key in allowed for key in ids), ids
            assert entity.calls and relation.calls and vector.calls
            # 原始节点描述是全局合并的；即使来源已裁剪，仍记录混合时期的风险数量。
            mixed_descriptions = sum(
                bool(set(graph.nodes[n].get("source_id", "").split(GRAPH_FIELD_SEP)) - allowed)
                for n in scoped.nodes
            )
            results.append(
                {
                    "seed_index": seed_index,
                    "period": period,
                    "pick_method": pick_method,
                    "vector_reads": vector.vector_reads,
                    "eligible_nodes_with_out_of_scope_original_sources": mixed_descriptions,
                    "candidate_counts": [entity.calls, relation.calls, vector.calls],
                    "returned_chunks": len(ids),
                    "outside_range": 0,
                    "entities": len(data.get("entities", [])),
                    "relationships": len(data.get("relationships", [])),
                }
            )
    assert fingerprints == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in asset_files}
    print(
        json.dumps(
            {
                "graph_id": graph_row["id"],
                "chunks": len(chunks),
                "mapping": stats,
                "asset_sha256_unchanged": fingerprints,
                "partial_examples": details,
                "cutoff": cutoff,
                "eligible": {k: len(v) for k, v in periods.items()},
                "mix_replays": results,
                "limitations": [
                    "没有调用自然语言embedding或关键词模型，使用真实已存向量回放",
                    "保留图的全局描述以观察现有上下文构建，不保证描述无未来信息",
                    "跨界chunk排除，尚未实现裁剪重嵌入",
                    "未测试aquery缓存和HTTP接线",
                ],
                "real_asset_writes": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
