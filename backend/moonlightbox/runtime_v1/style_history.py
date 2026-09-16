"""表情包按历史使用上下文检索；不读取像素，不推断固定图片含义。"""

from math import sqrt
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import select

from moonlightbox.config import Settings
from moonlightbox.imports.models import Message, Participant
from moonlightbox.media.models import MediaAsset
from moonlightbox.media.service import MediaStore

from .snapshot_sources import ordered_frozen_message_ids


def source_rows(session, snapshot):
    if snapshot is None:
        return []
    order = {key: index for index, key in enumerate(ordered_frozen_message_ids(session, snapshot))}
    rows = list(
        session.execute(
            select(Message, Participant.role)
            .join(Participant, Participant.id == Message.participant_id)
            .where(
                Message.id.in_(order),
                Participant.role.in_(["self", "target"]),
            )
            .order_by(Message.timestamp, Message.id)
        )
    )
    return sorted(rows, key=lambda row: order[row[0].id])


def usage_documents(session, branch_id, snapshot):
    """索引附近文本，不把表情占位符或资产 ID 当作语义文本。"""
    rows = source_rows(session, snapshot)
    result = []
    for i, (message, role) in enumerate(rows):
        if role != "target" or message.kind != "sticker" or not message.media_asset_id:
            continue
        text = "\n".join(
            f"{speaker} {item.timestamp.isoformat()}: {item.content}"
            for item, speaker in rows[max(0, i - 4) : i + 5]
            if item.kind == "text" and item.content.strip()
        )
        if text:
            ref = str(uuid5(NAMESPACE_URL, f"style:{branch_id}:{snapshot.id}:{message.id}"))
            result.append((ref, message.id, text))
    return result


def search_usage(session, branch_id, snapshot, query, limit=6, encoder=None):
    from .conversation_index import MODEL, ConversationVector, embedder

    docs = {
        ref: (source, text) for ref, source, text in usage_documents(session, branch_id, snapshot)
    }
    rows = [
        row
        for row in session.scalars(
            select(ConversationVector).where(
                ConversationVector.branch_id == branch_id,
                ConversationVector.kind == "style",
                ConversationVector.model == MODEL,
            )
        )
        if row.source_id in docs and row.text == docs[row.source_id][1]
    ]
    coverage = {"eligible_usages": len(docs), "indexed_usages": len(rows)}
    if not rows:
        return {"status": "index_pending" if docs else "empty", "source_ids": [], **coverage}
    try:
        vector = (encoder or embedder()).embed([query])[0]
    except Exception:
        return {"status": "embedding_unavailable", "source_ids": [], **coverage}

    def score(row):
        norm = sqrt(sum(x * x for x in vector) * sum(x * x for x in row.vector))
        return sum(a * b for a, b in zip(vector, row.vector, strict=True)) / norm if norm else 0

    return {
        "status": "ready" if len(rows) == len(docs) else "partial",
        "source_ids": [
            docs[row.source_id][0] for row in sorted(rows, key=score, reverse=True)[:limit]
        ],
        **coverage,
    }


def validate_asset(session, branch, snapshot, asset_ref, settings=None):
    """发送资格由真实目标人物用法、快照范围、资产类型及文件共同决定。"""
    asset = session.get(MediaAsset, asset_ref)
    if asset is None or asset.project_id != branch.project_id or asset.kind != "sticker":
        raise ValueError("sticker_asset_out_of_scope")
    if not any(
        m.media_asset_id == asset_ref and role == "target" and m.kind == "sticker"
        for m, role in source_rows(session, snapshot)
    ):
        raise ValueError("sticker_not_used_by_target")
    settings = settings or Settings()
    path = MediaStore(
        settings.data_dir, read_only_source_dir=settings.media_read_only_source_dir
    ).path_for(asset)
    try:
        with path.open("rb") as stream:
            if not stream.read(1):
                raise ValueError("sticker_asset_empty")
    except OSError as error:
        raise ValueError("sticker_asset_unavailable") from error
    return asset
