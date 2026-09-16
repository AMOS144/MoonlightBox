"""只读旧资产，向新目录生成可审核的来源索引；绝不覆盖正式 workspace。

用 sidecar Python 运行，无后端 ORM 依赖。全部冻结行和实际切块重放通过才发布。
部署时将产物复制到对应不可变 workspace；再次执行不能覆盖已有产物。
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from lightrag.utils import Tokenizer
from moonlightbox_lightrag_sidecar.core import _OfflineTokenizer
from moonlightbox_lightrag_sidecar.schemas import DocumentInput
from moonlightbox_lightrag_sidecar.source_index import (
    SourceIndex,
    normalize_document,
    source_aware_chunking,
)


def build(db_path: Path, workspace: Path, graph_id: str, output: Path):
    if output.resolve() == workspace.resolve():
        raise ValueError("请输出到独立审核目录，不允许直接写入真实 workspace")
    if (output / "moonlight_source_index.sqlite3").exists():
        raise ValueError("输出来源索引已存在，拒绝覆盖")
    assets = [p for p in workspace.iterdir() if p.suffix in {"json", ".json", ".graphml"}]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        graph = db.execute("SELECT * FROM world_graph_versions WHERE id=?", (graph_id,)).fetchone()
        if graph is None or graph["workspace_key"] != workspace.name:
            raise ValueError("数据库图版本与 workspace 不一致")
        imports = json.loads(graph["source_import_ids"])
        rows = db.execute(
            "SELECT m.id,m.timestamp,m.import_id,m.source_id FROM messages m WHERE m.import_id IN ("
            + ",".join("?" for _ in imports)
            + ") ORDER BY m.timestamp,m.id",
            imports,
        ).fetchall()
        rows.sort(
            key=lambda r: (
                r["timestamp"],
                r["import_id"],
                (0, int(r["source_id"])) if r["source_id"].isdecimal() else (1, r["source_id"]),
                r["id"],
            )
        )
        order = {r["id"]: i for i, r in enumerate(rows)}
        documents = []
        for bundle in db.execute(
            "SELECT * FROM conversation_bundles WHERE graph_version_id=? ORDER BY ordinal",
            (graph_id,),
        ):
            messages = db.execute(
                "SELECT m.id,m.timestamp,p.name FROM conversation_bundle_messages bm "
                "JOIN messages m ON m.id=bm.message_id "
                "JOIN participants p ON p.id=m.participant_id "
                "WHERE bm.bundle_id=? ORDER BY bm.ordinal",
                (bundle["id"],),
            ).fetchall()
            lines = bundle["content"].split("\n")
            if len(lines) != len(messages):
                raise ValueError("冻结行数与持久化消息映射不一致")
            spans, cursor = [], 0
            for line, message in zip(lines, messages, strict=True):
                prefix = message["timestamp"][:16].replace("T", " ") + " " + message["name"] + "："
                if not line.startswith(prefix):
                    raise ValueError("冻结行的时间/发送者与持久化消息映射不一致")
                spans.append(
                    {
                        "message_id": message["id"],
                        "source_ordinal": order[message["id"]],
                        "sent_at": message["timestamp"],
                        "start": cursor,
                        "end": cursor + len(line),
                    }
                )
                cursor += len(line) + 1
            documents.append(
                DocumentInput(
                    id=bundle["document_id"],
                    source=bundle["source_name"],
                    text=bundle["content"],
                    source_version=graph["source_fingerprint"],
                    message_spans=spans,
                )
            )
    full_docs = json.loads((workspace / "kv_store_full_docs.json").read_text())
    chunks = json.loads((workspace / "kv_store_text_chunks.json").read_text())
    generated = {}
    for original in documents:
        doc = normalize_document(original)
        if full_docs.get(doc.id, {}).get("content") != doc.text:
            raise ValueError("编码清理后仍与 LightRAG 冻结文档不一致")
        generated[doc.id] = source_aware_chunking(
            Tokenizer("offline", _OfflineTokenizer()),
            doc.text,
            chunk_token_size=graph["chunk_token_size"],
            chunk_overlap_token_size=graph["chunk_overlap_token_size"],
        )
    spans = {}
    for key, chunk in chunks.items():
        candidates = generated.get(chunk["full_doc_id"], [])
        index = chunk["chunk_order_index"]
        if (
            index < 0
            or index >= len(candidates)
            or candidates[index]["content"] != chunk["content"]
        ):
            raise ValueError("实际 chunk 无法由冻结文档与原切块配置精确重放")
        spans[key] = {
            "document_id": chunk["full_doc_id"],
            "content_hash": hashlib.sha256(chunk["content"].encode()).hexdigest(),
            "span": candidates[index]["moonlight_source_span"],
        }
    if hashes != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}:
        raise ValueError("验证期间真实资产发生变化，请在冻结版本上重试")
    store = SourceIndex(output)
    store.publish(documents, embedding=(graph["embedding_model"], graph["embedding_dimension"]))
    store.publish_chunk_spans(spans)
    return {
        "graph_id": graph_id,
        "source_version": graph["source_fingerprint"],
        "documents": len(documents),
        "chunks": len(spans),
        "source_messages": len(order),
        "original_assets_unchanged": True,
        "index": str(store.path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.db, args.workspace, args.graph_id, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
