#!/usr/bin/env python3
"""把同一份聊天记录生成的 LightRAG 图谱迁移到另一个项目。

迁移保留源图的实体、关系、chunk ID 和向量，只重绑来源：

* 使用 source_id、时间、发送者和原始消息 JSON 验证两边消息一一对应；
* 将来源索引中的 message_id、source_ordinal、sent_at、source_version 改为目标值；
* 将 LightRAG 的 file_path 映射到目标项目的 bundle；
* 为目标 bundle 写入 processed 状态，使后端重试时复用图谱而不再调用抽取模型。

默认仅检查并输出报告。--apply 会创建一个全新的 workspace，并在最后一个
SQLite 事务中把目标 graph 指向它；源 workspace 和失败 workspace 均不覆盖。
迁移后目标 graph 仍保持 failed，用户点击“重试构建”后会跳过索引、继续人物编译。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


METADATA_FIELDS = (
    "lightrag_version",
    "embedding_model",
    "embedding_dimension",
    "extraction_model",
    "chunking_strategy",
    "chunk_token_size",
    "chunk_overlap_token_size",
    "entity_prompt_version",
)


def _json(value: str | None) -> Any:
    return json.loads(value) if value else None


def _canonical_json(value: str | None) -> str:
    try:
        return json.dumps(_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return value or ""


def _source_sort_key(row: sqlite3.Row) -> tuple[Any, ...]:
    source_id = row["source_id"] or ""
    source_part: tuple[int, Any] = (
        (0, int(source_id)) if source_id.isdecimal() else (1, source_id)
    )
    return row["timestamp"], row["import_id"], source_part, row["id"]


def _read_graph(db: sqlite3.Connection, graph_id: str) -> sqlite3.Row:
    row = db.execute("SELECT * FROM world_graph_versions WHERE id=?", (graph_id,)).fetchone()
    if row is None:
        raise ValueError(f"找不到 graph：{graph_id}")
    return row


def _read_messages(db: sqlite3.Connection, project_id: str) -> list[sqlite3.Row]:
    rows = db.execute(
        "SELECT m.*,p.name AS participant_name,p.role AS participant_role "
        "FROM messages m JOIN participants p ON p.id=m.participant_id "
        "WHERE m.project_id=?",
        (project_id,),
    ).fetchall()
    return sorted(rows, key=_source_sort_key)


def _message_identity(row: sqlite3.Row) -> tuple[str, str, str, str]:
    # content/kind 可以因预处理版本变化；raw 是同一次聊天导入的稳定事实来源。
    return (
        row["source_id"],
        row["timestamp"],
        row["participant_name"],
        _canonical_json(row["raw"]),
    )


def _verify_message_mapping(
    source: list[sqlite3.Row], target: list[sqlite3.Row]
) -> tuple[dict[str, sqlite3.Row], int]:
    if len(source) != len(target):
        raise ValueError(f"消息数量不一致：源 {len(source)}，目标 {len(target)}")
    mapping: dict[str, sqlite3.Row] = {}
    changed_content = 0
    mismatches: list[dict[str, Any]] = []
    for ordinal, (old, new) in enumerate(zip(source, target, strict=True)):
        if _message_identity(old) != _message_identity(new):
            mismatches.append(
                {
                    "ordinal": ordinal,
                    "source_id": old["source_id"],
                    "target_source_id": new["source_id"],
                    "source_time": old["timestamp"],
                    "target_time": new["timestamp"],
                    "source_sender": old["participant_name"],
                    "target_sender": new["participant_name"],
                }
            )
            if len(mismatches) >= 20:
                break
        if (old["kind"], old["content"]) != (new["kind"], new["content"]):
            changed_content += 1
        mapping[old["id"]] = new
    if mismatches:
        raise ValueError(
            "聊天记录不能安全地一一映射，前 20 个差异：\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )
    return mapping, changed_content


def _read_bundles(
    db: sqlite3.Connection, graph_id: str
) -> tuple[dict[str, sqlite3.Row], dict[str, str]]:
    bundles = {
        row["document_id"]: row
        for row in db.execute(
            "SELECT * FROM conversation_bundles WHERE graph_version_id=? ORDER BY ordinal",
            (graph_id,),
        )
    }
    primary: dict[str, str] = {}
    for row in db.execute(
        "SELECT bm.message_id,b.document_id,bm.is_carry_in "
        "FROM conversation_bundle_messages bm "
        "JOIN conversation_bundles b ON b.id=bm.bundle_id "
        "WHERE b.graph_version_id=? ORDER BY b.ordinal,bm.ordinal",
        (graph_id,),
    ):
        if not row["is_carry_in"]:
            if row["message_id"] in primary:
                raise ValueError(f"目标消息重复出现在 primary bundle：{row['message_id']}")
            primary[row["message_id"]] = row["document_id"]
    return bundles, primary


def _read_source_index(workspace: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    path = workspace / "moonlight_source_index.sqlite3"
    if not path.exists():
        raise ValueError(f"源图没有来源索引：{path}")
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"documents", "metadata", "chunk_spans"}.issubset(tables):
            raise ValueError("源图来源索引不完整，缺少 documents/metadata/chunk_spans")
        contract = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if contract is None or contract[0] != "frozen_source_spans_v1/import_source_order_v1":
            raise ValueError("源图来源索引 contract 不受支持")
        documents = {key: json.loads(payload) for key, payload in db.execute(
            "SELECT id,payload FROM documents"
        )}
        chunks = {key: json.loads(payload) for key, payload in db.execute(
            "SELECT id,payload FROM chunk_spans"
        )}
    if not documents or not chunks:
        raise ValueError("源图来源索引为空")
    return documents, chunks


def _dominant_bundle(ids: list[str]) -> str:
    if not ids:
        raise ValueError("来源位置没有对应到任何目标 bundle")
    counts = Counter(ids)
    # 相同票数时按最先出现决定，保证稳定且更贴近 chunk 开头证据。
    return max(counts, key=lambda item: (counts[item], -ids.index(item)))


def _build_plan(
    documents: dict[str, dict],
    chunk_spans: dict[str, dict],
    message_map: dict[str, sqlite3.Row],
    target_order: dict[str, int],
    target_primary_bundle: dict[str, str],
    target_bundles: dict[str, sqlite3.Row],
    target_source_version: str,
) -> tuple[dict[str, dict], dict[str, str], dict[str, str], dict[str, Any]]:
    migrated_docs: dict[str, dict] = {}
    doc_paths: dict[str, str] = {}
    doc_bundle_sets: dict[str, set[str]] = {}
    for document_id, document in documents.items():
        spans = document.get("message_spans") or []
        if not spans:
            raise ValueError(f"文档缺少 message_spans：{document_id}")
        mapped_spans = []
        bundle_ids = []
        for span in spans:
            new = message_map.get(span["message_id"])
            if new is None:
                raise ValueError(f"来源消息无法映射：{span['message_id']}")
            target_bundle = target_primary_bundle.get(new["id"])
            if target_bundle is None:
                raise ValueError(f"目标消息没有 primary bundle：{new['id']}")
            bundle_ids.append(target_bundle)
            mapped_spans.append(
                {
                    **span,
                    "message_id": new["id"],
                    "source_ordinal": target_order[new["id"]],
                    "sent_at": new["timestamp"],
                }
            )
        dominant = _dominant_bundle(bundle_ids)
        source_name = target_bundles[dominant]["source_name"]
        migrated_docs[document_id] = {
            **document,
            "source": source_name,
            "source_version": target_source_version,
            "message_spans": mapped_spans,
        }
        doc_paths[document_id] = source_name
        doc_bundle_sets[document_id] = set(bundle_ids)

    chunk_paths: dict[str, str] = {}
    cross_chunk_count = 0
    for chunk_id, item in chunk_spans.items():
        document = migrated_docs.get(item.get("document_id"))
        if document is None:
            raise ValueError(f"chunk 指向未知文档：{chunk_id}")
        start, end = item["span"]["start"], item["span"]["end"]
        messages = [
            span for span in document["message_spans"]
            if span["start"] < end and span["end"] > start
        ]
        bundle_ids = [target_primary_bundle[span["message_id"]] for span in messages]
        if not bundle_ids:
            raise ValueError(f"chunk 无法落到目标消息：{chunk_id}")
        if len(set(bundle_ids)) > 1:
            cross_chunk_count += 1
        dominant = _dominant_bundle(bundle_ids)
        chunk_paths[chunk_id] = target_bundles[dominant]["source_name"]

    report = {
        "source_documents": len(documents),
        "source_chunks": len(chunk_spans),
        "cross_bundle_documents": sum(len(v) > 1 for v in doc_bundle_sets.values()),
        "cross_bundle_chunks": cross_chunk_count,
        "target_bundles_referenced": len(
            {Path(path).stem for path in [*doc_paths.values(), *chunk_paths.values()]}
        ),
    }
    return migrated_docs, doc_paths, chunk_paths, report


def _rewrite_source_index(path: Path, documents: dict[str, dict]) -> None:
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        for document_id, payload in documents.items():
            db.execute(
                "UPDATE documents SET payload=? WHERE id=?",
                (json.dumps(payload, ensure_ascii=False, separators=(",", ":")), document_id),
            )
        db.commit()


def _patch_paths(value: Any, doc_paths: dict[str, str], chunk_paths: dict[str, str]) -> None:
    if isinstance(value, dict):
        chunk_id = value.get("_id") or value.get("__id__")
        if isinstance(chunk_id, str) and chunk_id in chunk_paths and "file_path" in value:
            value["file_path"] = chunk_paths[chunk_id]
        elif isinstance(value.get("full_doc_id"), str) and "file_path" in value:
            doc_id = value["full_doc_id"]
            if doc_id in doc_paths:
                value["file_path"] = doc_paths[doc_id]
        elif isinstance(value.get("file_path"), str):
            parts = value["file_path"].split("<SEP>")
            value["file_path"] = "<SEP>".join(
                dict.fromkeys(doc_paths.get(Path(part).stem, part) for part in parts)
            )
        for child in value.values():
            _patch_paths(child, doc_paths, chunk_paths)
    elif isinstance(value, list):
        for child in value:
            _patch_paths(child, doc_paths, chunk_paths)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def _rewrite_assets(
    workspace: Path,
    doc_paths: dict[str, str],
    chunk_paths: dict[str, str],
    target_bundles: dict[str, sqlite3.Row],
) -> None:
    for path in workspace.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        _patch_paths(payload, doc_paths, chunk_paths)
        _write_json(path, payload)

    full_path = workspace / "kv_store_full_docs.json"
    status_path = workspace / "kv_store_doc_status.json"
    full_docs = json.loads(full_path.read_text(encoding="utf-8"))
    statuses = json.loads(status_path.read_text(encoding="utf-8"))
    text_chunks = json.loads(
        (workspace / "kv_store_text_chunks.json").read_text(encoding="utf-8")
    )
    reverse: dict[str, list[str]] = defaultdict(list)
    for old_doc, path in doc_paths.items():
        reverse[Path(path).stem].append(old_doc)
    # 新版 bundle 比旧版更大/边界不同。某个目标 bundle 可能只覆盖旧文档中的
    # 一部分 chunk，而不是任何旧文档的主归属；此时从 chunk 精确反查模板文档。
    for chunk_id, path in chunk_paths.items():
        chunk = text_chunks.get(chunk_id)
        if chunk and chunk.get("full_doc_id") in full_docs:
            reverse[Path(path).stem].append(chunk["full_doc_id"])
    for target_id, bundle in target_bundles.items():
        candidates = reverse.get(target_id)
        if not candidates:
            raise ValueError(f"目标 bundle 没有任何源文档覆盖：{target_id}")
        template_id = candidates[0]
        template = dict(full_docs[template_id])
        template.update(
            {
                "_id": target_id,
                "content": bundle["content"],
                "content_hash": bundle["content_hash"],
                "file_path": bundle["source_name"],
            }
        )
        full_docs[target_id] = template
        state = dict(statuses[template_id])
        metadata = dict(state.get("metadata") or {})
        metadata["moonlight_migrated_graph"] = True
        state.update(
            {
                "status": "processed",
                "content_summary": bundle["content"][:240],
                "content_length": len(bundle["content"]),
                "file_path": bundle["source_name"],
                "content_hash": bundle["content_hash"],
                "metadata": metadata,
                "chunks_count": 0,
                "chunks_list": [],
            }
        )
        statuses[target_id] = state
    _write_json(full_path, full_docs)
    _write_json(status_path, statuses)

    graphml = workspace / "graph_chunk_entity_relation.graphml"
    if graphml.exists():
        text = graphml.read_text(encoding="utf-8")
        for old_doc, new_path in doc_paths.items():
            text = text.replace(f"{old_doc}.txt", new_path)
        graphml.write_text(text, encoding="utf-8")


def _workspace_digest(path: Path) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def migrate(args: argparse.Namespace) -> dict[str, Any]:
    db_path = args.database.resolve()
    workspace_root = args.workspace_root.resolve()
    if not db_path.is_file():
        raise ValueError(f"数据库不存在：{db_path}")
    if not workspace_root.is_dir():
        raise ValueError(f"workspace 根目录不存在：{workspace_root}")

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        source_graph = _read_graph(db, args.source_graph)
        target_graph = _read_graph(db, args.target_graph)
        if source_graph["project_id"] == target_graph["project_id"]:
            raise ValueError("源图和目标图属于同一个项目，无需跨项目迁移")
        if source_graph["status"] not in {"ready", "awaiting_profile_review", "superseded"}:
            raise ValueError(f"源图状态不可迁移：{source_graph['status']}")
        source_workspace = workspace_root / source_graph["workspace_key"]
        if not source_workspace.is_dir():
            raise ValueError(f"源 workspace 不存在：{source_workspace}")
        source_messages = _read_messages(db, source_graph["project_id"])
        target_messages = _read_messages(db, target_graph["project_id"])
        message_map, changed_content = _verify_message_mapping(source_messages, target_messages)
        target_order = {row["id"]: index for index, row in enumerate(target_messages)}
        target_bundles, target_primary = _read_bundles(db, target_graph["id"])
        if len(target_primary) != len(target_messages):
            raise ValueError(
                f"目标 bundle 仅覆盖 {len(target_primary)}/{len(target_messages)} 条 primary 消息"
            )

    documents, chunk_spans = _read_source_index(source_workspace)
    migrated_docs, doc_paths, chunk_paths, plan = _build_plan(
        documents,
        chunk_spans,
        message_map,
        target_order,
        target_primary,
        target_bundles,
        target_graph["source_fingerprint"],
    )
    original_digest = _workspace_digest(source_workspace)
    workspace_key = args.workspace_key or (
        f"{target_graph['workspace_key']}_migrated_{source_graph['id'].replace('-', '')[:8]}"
    )
    output = workspace_root / workspace_key
    report: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry-run",
        "source_graph_id": source_graph["id"],
        "source_workspace": source_graph["workspace_key"],
        "target_graph_id": target_graph["id"],
        "target_previous_workspace": target_graph["workspace_key"],
        "target_migrated_workspace": workspace_key,
        "source_version": target_graph["source_fingerprint"],
        "messages": len(target_messages),
        "messages_with_preprocessing_changes": changed_content,
        "target_bundles": len(target_bundles),
        **plan,
    }
    if not args.apply:
        return report
    if output.exists():
        raise ValueError(f"目标迁移 workspace 已存在，拒绝覆盖：{output}")

    temporary = workspace_root / f".{workspace_key}.tmp"
    if temporary.exists():
        raise ValueError(f"临时目录已存在，请先人工检查：{temporary}")
    try:
        shutil.copytree(source_workspace, temporary)
        _rewrite_source_index(temporary / "moonlight_source_index.sqlite3", migrated_docs)
        _rewrite_assets(temporary, doc_paths, chunk_paths, target_bundles)
        if _workspace_digest(source_workspace) != original_digest:
            raise ValueError("迁移期间源 workspace 发生变化，已中止")
        manifest = {**report, "created_at": datetime.now(UTC).isoformat()}
        (temporary / "moonlight_migration_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.rename(output)
        with sqlite3.connect(db_path) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            current = _read_graph(db, target_graph["id"])
            if current["workspace_key"] != target_graph["workspace_key"]:
                raise ValueError("目标 graph 已被其他操作修改，拒绝提交迁移")
            assignments = ["workspace_key=?"] + [f"{field}=?" for field in METADATA_FIELDS]
            values = [workspace_key] + [source_graph[field] for field in METADATA_FIELDS]
            db.execute(
                f"UPDATE world_graph_versions SET {','.join(assignments)} WHERE id=?",
                (*values, target_graph["id"]),
            )
            db.execute(
                "UPDATE conversation_bundles SET indexed=1 WHERE graph_version_id=?",
                (target_graph["id"],),
            )
            db.commit()
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        # output 已 rename、但 DB 未提交时也保持资产供人工检查，不做危险删除。
        raise
    report["manifest"] = str(output / "moonlight_migration_manifest.json")
    report["next_step"] = "在页面点击“重试构建”；后端将复用迁移图谱并继续后续编译。"
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True, help="moonlightbox.db 路径")
    parser.add_argument("--workspace-root", type=Path, required=True, help="LightRAG workspace 根目录")
    parser.add_argument("--source-graph", required=True, help="已有完整图谱的 graph id")
    parser.add_argument("--target-graph", required=True, help="需要接收图谱的 graph id")
    parser.add_argument("--workspace-key", help="新 workspace 名称；默认从目标和源 graph id 生成")
    parser.add_argument("--apply", action="store_true", help="通过全部校验后写入并绑定目标 graph")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        print(json.dumps(migrate(parse_args()), ensure_ascii=False, indent=2))
    except Exception as exc:  # 运维脚本需要给出具体失败原因。
        print(f"迁移失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
