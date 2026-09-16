"""分支语义索引：SQLite 保存可重建向量，不将虚拟经历写入原始 LightRAG。"""

import json
from functools import lru_cache
from math import sqrt
from pathlib import Path

from sqlalchemy import JSON, ForeignKey, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from moonlightbox.config import Settings
from moonlightbox.db import Base

from .branch_models import BranchMessage
from .db_models import RuntimeLifeEventRow, RuntimeMemoryRow


class ConversationVector(Base):
    __tablename__ = "runtime_conversation_vectors"
    source_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    branch_id: Mapped[str] = mapped_column(
        ForeignKey("branches.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    vector: Mapped[list] = mapped_column(JSON)
    model: Mapped[str] = mapped_column(String(100))


MODEL = "BAAI/bge-small-zh-v1.5"


@lru_cache(maxsize=1)
def _load(root):
    from moonlightbox.embeddings import LocalChineseEmbedder

    return LocalChineseEmbedder(Path(root))


def embedder():
    root = Settings().model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
    if not root.is_dir():
        raise RuntimeError("conversation_embedding_unavailable")
    return _load(str(root.resolve()))


def index_branch(session, branch_id, encoder=None):
    """后台维护入口；每批持久化，可在任务重试时跳过已完成来源。"""
    encoder = encoder or embedder()
    existing = {
        row.source_id: row
        for row in session.scalars(
            select(ConversationVector).where(ConversationVector.branch_id == branch_id)
        )
    }
    messages = list(
        session.scalars(
            select(BranchMessage)
            .where(
                BranchMessage.branch_id == branch_id,
                BranchMessage.generation_status != "failed",
            )
            .order_by(BranchMessage.sequence)
        )
    )
    docs = []
    for i, row in enumerate(messages):
        text = "\n".join(f"{item.role}: {item.content}" for item in messages[max(0, i - 2) : i + 1])
        docs.append((row.id, "message", text))
    docs.extend(
        (row.id, "memory", f"{row.subject}: {row.summary}")
        for row in session.scalars(
            select(RuntimeMemoryRow).where(
                RuntimeMemoryRow.branch_id == branch_id,
                RuntimeMemoryRow.status.in_(["asserted", "confirmed"]),
            )
        )
    )
    # 与分支索引一同异步维护；聊天检索只嵌入查询，不同步重建全部使用片段。
    from .db_models import RuntimeSnapshotRow
    from .style_history import usage_documents

    snapshot = session.scalar(
        select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
    )
    docs.extend(
        (ref, "style", text) for ref, _, text in usage_documents(session, branch_id, snapshot)
    )
    pending = [
        (ref, kind, text)
        for ref, kind, text in docs
        if ref not in existing or existing[ref].text != text or existing[ref].model != MODEL
    ]
    for event in session.scalars(
        select(RuntimeLifeEventRow).where(
            RuntimeLifeEventRow.branch_id == branch_id,
            RuntimeLifeEventRow.event_type == "simulated_life",
        )
    ):
        if event.id not in existing:
            pending.append((event.id, "simulation", json.dumps(event.payload, ensure_ascii=False)))
    for offset in range(0, len(pending), 32):
        batch = pending[offset : offset + 32]
        vectors = encoder.embed([item[2] for item in batch])
        if len(vectors) != len(batch):
            raise ValueError("embedding_count_mismatch")
        for (ref, kind, text), vector in zip(batch, vectors, strict=True):
            row = existing.get(ref) or ConversationVector(source_id=ref, branch_id=branch_id)
            row.kind, row.text, row.vector, row.model = kind, text, vector, MODEL
            session.add(row)
        session.commit()


def search_branch(session, branch_id, query, now, limit=8, encoder=None, kind=None):
    """仅对有效、可见原始来源返回命中；未建立索引不冒充无历史。"""
    from sqlalchemy import func

    sources = {
        row.id: row
        for row in session.scalars(
            select(BranchMessage).where(
                BranchMessage.branch_id == branch_id,
                BranchMessage.generation_status != "failed",
                func.coalesce(BranchMessage.observed_at, BranchMessage.created_at) <= now,
            )
        )
    }
    if kind != "message":
        sources.update(
            {
                row.id: row
                for row in session.scalars(
                    select(RuntimeMemoryRow).where(
                        RuntimeMemoryRow.branch_id == branch_id,
                        RuntimeMemoryRow.status.in_(["asserted", "confirmed"]),
                        (
                            RuntimeMemoryRow.valid_from.is_(None)
                            | (RuntimeMemoryRow.valid_from <= now)
                        ),
                        (RuntimeMemoryRow.valid_to.is_(None) | (RuntimeMemoryRow.valid_to >= now)),
                    )
                )
            }
        )
        sources.update(
            {
                row.id: row
                for row in session.scalars(
                    select(RuntimeLifeEventRow).where(
                        RuntimeLifeEventRow.branch_id == branch_id,
                        RuntimeLifeEventRow.event_type == "simulated_life",
                        RuntimeLifeEventRow.occurred_at <= now,
                    )
                )
            }
        )
    rows = [
        row
        for row in session.scalars(
            select(ConversationVector).where(
                ConversationVector.branch_id == branch_id, ConversationVector.model == MODEL
            )
        )
        if row.source_id in sources and (kind is None or row.kind == kind)
    ]
    coverage = {"visible_sources": len(sources), "indexed_sources": len(rows)}
    if not rows:
        return {"status": "index_pending" if sources else "empty", "coverage": coverage, "data": []}
    try:
        vector = (encoder or embedder()).embed([query])[0]
    except Exception:
        return {"status": "embedding_unavailable", "coverage": coverage, "data": []}

    def score(row):
        if len(vector) != len(row.vector):
            raise ValueError("embedding_dimension_mismatch")
        norm = sqrt(sum(x * x for x in vector) * sum(x * x for x in row.vector))
        return sum(a * b for a, b in zip(vector, row.vector, strict=True)) / norm if norm else 0

    hits = sorted(rows, key=score, reverse=True)[:limit]
    ordered_messages = sorted(
        (row for row in sources.values() if isinstance(row, BranchMessage)),
        key=lambda row: row.sequence,
    )
    positions = {row.id: i for i, row in enumerate(ordered_messages)}

    def current_text(ref):
        # 返回重新读取的可见原文，不能返回旧索引文本或包含未来消息的邻域。
        source = sources[ref]
        if isinstance(source, RuntimeLifeEventRow):
            return "simulation: " + json.dumps(source.payload, ensure_ascii=False)
        if isinstance(source, BranchMessage):
            i = positions[ref]
            return "\n".join(
                f"{item.role}: {item.content}" for item in ordered_messages[max(0, i - 2) : i + 1]
            )
        return f"{source.subject} ({source.predicate}): {source.summary}"

    return {
        "status": "ready" if len(rows) == len(sources) else "partial",
        "coverage": coverage,
        "selection": "semantic_similarity",
        "data": [
            {
                "source_ref": row.source_id,
                "source_ids": [
                    item.id
                    for item in ordered_messages[
                        max(0, positions[row.source_id] - 2) : positions[row.source_id] + 1
                    ]
                ]
                if row.kind == "message"
                else (
                    [row.source_id]
                    if row.kind == "simulation"
                    else sources[row.source_id].source_ids
                ),
                "kind": row.kind,
                "text": current_text(row.source_id),
                "similarity": score(row),
            }
            for row in hits
        ],
    }
