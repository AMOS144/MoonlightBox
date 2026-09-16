"""连续对话的只读投影：分支新增原文与绑定快照的导入前缀共用来源边界。"""

from sqlalchemy import select

from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.wechat_rendering import normalize_wechat_display

from .snapshot_sources import frozen_message_ids


def imported_prefix(session, snapshot, limit=30, before=None):
    ids = frozen_message_ids(session, snapshot)
    if not ids:
        return []
    query = (
        select(Message, Participant.role)
        .join(Participant, Participant.id == Message.participant_id)
        .where(
            Message.id.in_(ids),
            Message.timestamp <= snapshot.cutoff_at,
            Participant.role.in_(["self", "target"]),
        )
    )
    if before is not None:
        # 时间与 ID 共同排序，防止同秒多条消息翻页丢失。
        from sqlalchemy import and_, or_

        anchor = session.get(Message, before)
        if anchor is None or anchor.id not in ids:
            raise ValueError("history_cursor_out_of_scope")
        query = query.where(
            or_(
                Message.timestamp < anchor.timestamp,
                and_(Message.timestamp == anchor.timestamp, Message.id < anchor.id),
            )
        )
    rows = list(
        session.execute(query.order_by(Message.timestamp.desc(), Message.id.desc()).limit(limit))
    )
    result = []
    for message, role in reversed(rows):
        kind, content = normalize_wechat_display(message.kind, message.content)
        result.append(
            {
                "source_id": message.id,
                "role": "user" if role == "self" else "assistant",
                "origin": "import",
                "content": content,
                "type": kind,
                "asset_ref": message.media_asset_id,
                "usage_refs": [message.id] if kind == "sticker" else [],
                "occurred_at": message.timestamp.isoformat(),
                "sequence": None,
            }
        )
    return result
