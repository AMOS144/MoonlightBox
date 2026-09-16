"""独立来源索引：保存冻结文本的位置，不修改 LightRAG 图或向量。

新文档由后端提供消息位置；旧文档没有可靠位置时保持 unknown。
SQLite 的单事务用于发布完整批次，workspace 复制时随目录一起复制。
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from .schemas import DocumentInput, TemporalScope

INDEX_CONTRACT = "frozen_source_spans_v1/import_source_order_v1"


class TemporalIndexError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def normalize_document(document: DocumentInput) -> DocumentInput:
    """同步 LightRAG 入库编码清理，并重新计算清理后的字符坐标。

    真实旧资产含 U+0005/U+0001；它们在 full_docs 中已被删除。
    不能把 Bundle 的旧坐标直接套在已清理的 chunk 文本上。
    """
    from lightrag.utils import sanitize_text_for_encoding

    if document.message_spans is None:
        return document
    pieces, spans, cursor = [], [], 0
    for span in document.message_spans:
        text = sanitize_text_for_encoding(document.text[span.start : span.end])
        if not text:
            raise TemporalIndexError("temporal_empty_rendered_message")
        spans.append(span.model_copy(update={"start": cursor, "end": cursor + len(text)}))
        pieces.append(text)
        cursor += len(text) + 1
    text = "\n".join(pieces)
    if text != sanitize_text_for_encoding(document.text):
        raise TemporalIndexError("temporal_normalization_mismatch")
    return DocumentInput.model_validate(
        {**document.model_dump(), "text": text, "message_spans": [s.model_dump() for s in spans]}
    )


def source_aware_chunking(*args, **kwargs):
    """保留上游切块内容、ID 和向量，仅将原始位置存入 chunk KV。

    上游会剥离私有 _source_span，因此复制到项目命名空间字段。
    使用上游真实切块位置，重复文字也不需要通过 find 猜测。
    """
    from lightrag.chunker import chunking_by_token_size

    kwargs["_emit_source_span"] = True
    rows = chunking_by_token_size(*args, **kwargs)
    for row in rows:
        if "_source_span" in row:
            row["moonlight_source_span"] = dict(row["_source_span"])
    return rows


class SourceIndex:
    def __init__(self, workspace_path: Path):
        self.path = workspace_path / "moonlight_source_index.sqlite3"

    def publish(
        self, documents: list[DocumentInput], *, embedding: tuple[str, int] | None = None
    ) -> None:
        documents = [normalize_document(d) for d in documents if d.message_spans is not None]
        if not documents:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            current = [
                DocumentInput.model_validate_json(row[0])
                for row in db.execute("SELECT payload FROM documents")
            ]
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            contract = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
            if (current and contract is None) or (contract and contract[0] != INDEX_CONTRACT):
                raise TemporalIndexError("temporal_index_contract_mismatch")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract', ?)", (INDEX_CONTRACT,))
            if embedding is not None:
                expected = json.dumps(list(embedding))
                stored = db.execute("SELECT value FROM metadata WHERE key='embedding'").fetchone()
                if stored and stored[0] != expected:
                    raise TemporalIndexError("temporal_embedding_mismatch")
                db.execute("INSERT OR IGNORE INTO metadata VALUES ('embedding', ?)", (expected,))
            versions, messages, ordinals = set(), {}, {}
            for document in [*current, *documents]:
                versions.add(document.source_version)
                for span in document.message_spans or []:
                    identity = (span.source_ordinal, span.sent_at)
                    if (span.message_id in messages and messages[span.message_id] != identity) or (
                        span.source_ordinal in ordinals
                        and ordinals[span.source_ordinal] != span.message_id
                    ):
                        raise TemporalIndexError("temporal_message_order_conflict")
                    messages[span.message_id] = identity
                    ordinals[span.source_ordinal] = span.message_id
            if len(versions) != 1:
                raise TemporalIndexError("temporal_source_version_mismatch")
            for document in documents:
                payload = document.model_dump_json()
                old = db.execute(
                    "SELECT payload FROM documents WHERE id=?", (document.id,)
                ).fetchone()
                if old and json.loads(old[0]) != json.loads(payload):
                    raise TemporalIndexError("temporal_source_conflict")
                db.execute("INSERT OR IGNORE INTO documents VALUES (?, ?)", (document.id, payload))

    def read(self) -> dict[str, DocumentInput]:
        if not self.path.exists():
            raise TemporalIndexError("temporal_index_missing")
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='metadata'").fetchone():
                raise TemporalIndexError("temporal_index_contract_mismatch")
            contract = db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
            if contract is None or contract[0] != INDEX_CONTRACT:
                raise TemporalIndexError("temporal_index_contract_mismatch")
            return {
                key: DocumentInput.model_validate_json(value)
                for key, value in db.execute("SELECT id,payload FROM documents")
            }

    def publish_chunk_spans(self, rows: dict[str, dict]) -> None:
        """回填独立来源表，不给旧图谱或向量补写字段。"""
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS chunk_spans "
                "(id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            for key, value in rows.items():
                payload = json.dumps(value, sort_keys=True)
                old = db.execute("SELECT payload FROM chunk_spans WHERE id=?", (key,)).fetchone()
                if old and old[0] != payload:
                    raise TemporalIndexError("temporal_chunk_conflict")
                db.execute("INSERT OR IGNORE INTO chunk_spans VALUES (?, ?)", (key, payload))

    def embedding_signature(self) -> tuple[str, int]:
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='embedding'").fetchone()
        if row is None:
            raise TemporalIndexError("temporal_embedding_binding_missing")
        name, dimension = json.loads(row[0])
        return name, dimension

    def attach_chunk_spans(self, chunks: dict[str, dict]) -> dict[str, dict]:
        result = {key: dict(row) for key, row in chunks.items()}
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='chunk_spans'").fetchone():
                return result
            for key, payload in db.execute("SELECT id,payload FROM chunk_spans"):
                if key not in result:
                    continue
                row = json.loads(payload)
                chunk = result[key]
                if row["content_hash"] != hashlib.sha256(
                    chunk["content"].encode()
                ).hexdigest() or row["document_id"] != chunk.get("full_doc_id"):
                    raise TemporalIndexError("temporal_chunk_version_mismatch")
                chunk["moonlight_source_span"] = row["span"]
        return result


def resolve_chunk(chunk: dict, document: DocumentInput) -> tuple[int, int] | None:
    """验证实际文本，旧资产仅接受唯一精确匹配；不猜模糊或重复位置。"""
    content = chunk.get("content", "")
    if not content:
        return None
    span = chunk.get("moonlight_source_span")
    if span is not None:
        start, end = span.get("start"), span.get("end")
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(document.text)
        ):
            # 上游范围可能包含被 strip 移除的两端空白。
            raw = document.text[start:end]
            if raw.strip() == content:
                left = start + len(raw) - len(raw.lstrip())
                return left, left + len(content)
        return None
    start = document.text.find(content)
    if start < 0 or document.text.find(content, start + 1) >= 0:
        return None
    return start, start + len(content)


def classify_chunks(
    chunks: dict[str, dict], documents: dict[str, DocumentInput], scope: TemporalScope
):
    """按消息序号分组。mixed 完整归 before；Runtime 禁止借此读未来原话。"""
    selected, metadata, unknown = {}, {}, []
    if any(d.source_version != scope.source_version for d in documents.values()):
        raise TemporalIndexError("temporal_source_version_mismatch")
    for chunk_id, chunk in chunks.items():
        doc = documents.get(chunk.get("full_doc_id"))
        span = resolve_chunk(chunk, doc) if doc else None
        messages = (
            [m for m in (doc.message_spans or []) if span and m.start < span[1] and m.end > span[0]]
            if doc
            else []
        )
        if not messages:
            unknown.append(chunk_id)
            continue
        before = [m.source_ordinal < scope.included_count for m in messages]
        relation = "before" if all(before) else "mixed" if any(before) else "after"
        assigned = "before" if any(before) else "after"
        # 历史 Runtime 暂时排除 mixed；不能把编译器权限误当角色知识权限。
        if scope.access == "runtime" and relation != "before":
            continue
        if scope.period != "all" and assigned != scope.period:
            continue
        selected[chunk_id] = dict(chunk)
        metadata[chunk_id] = {
            "assigned_period": assigned,
            "boundary_relation": relation,
            "content_hash": hashlib.sha256(chunk["content"].encode()).hexdigest(),
            "messages": [
                {**m.model_dump(), "period": "before" if b else "after"}
                for m, b in zip(messages, before, strict=True)
            ],
        }
    return selected, metadata, unknown
