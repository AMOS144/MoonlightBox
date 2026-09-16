"""时间分支起点之前的只读聊天历史。

旧版分支把这段历史放进 Baseline 清单，再由预处理任务负责提供。Runtime v1
不需要那套持久化副本：节点本身已经记录了结束消息，界面只需从原始导入记录中
按同一边界只读分页即可。这个模块刻意不参与 Director / PersonaActor 的上下文，
它只是聊天界面的历史投影。
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.wechat_rendering import normalize_wechat_display
from moonlightbox.world.models import ConversationBundle, ConversationBundleMessage

from .branch_models import Branch
from .db_models import RuntimeSnapshotRow


class SourceHistoryError(ValueError):
    """历史请求的游标或分页参数不合法。"""


@dataclass(frozen=True)
class SourceHistoryMessage:
    id: str
    source_id: str
    role: str
    content: str
    type: str
    media_asset_id: str | None
    timestamp: datetime


@dataclass(frozen=True)
class SourceHistoryPage:
    items: list[SourceHistoryMessage]
    next_cursor: str | None
    has_more: bool
    # 为兼容既有前端契约保留字段名；v1 没有 Baseline manifest。
    manifest_id: str


class SourceHistoryService:
    """从导入的原始记录构造分支起点以前的微信式聊天时间线。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def page(
        self,
        project_id: str,
        branch_id: str,
        *,
        before: str | None,
        limit: int,
    ) -> SourceHistoryPage:
        if limit < 20 or limit > 100:
            raise SourceHistoryError("limit 必须在 20 到 100 之间")
        branch = self._branch(project_id, branch_id)
        snapshot = self._session.scalar(
            select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch.id)
        )
        if snapshot is not None and snapshot.snapshot_mode != "latest_profile":
            return self._historical_page(branch, snapshot, before=before, limit=limit)
        boundary = self._boundary_message(branch)
        scope_id = _scope_id(branch)
        if boundary is None:
            # 极少数旧数据可能只有事件摘要、没有可追溯的原始消息。此时不要把别的
            # 导入批次误展示为该节点之前的对话。
            return SourceHistoryPage([], None, False, scope_id)

        conditions: list[ColumnElement[bool]] = [
            Message.project_id == project_id,
            Message.import_id == boundary.import_id,
            Participant.role.in_(("self", "target")),
            _at_or_before(Message, boundary.timestamp, boundary.source_id, boundary.id),
        ]
        if before is not None:
            cursor = _decode_cursor(before, branch.id, scope_id)
            conditions.append(
                _before(
                    Message,
                    datetime.fromisoformat(cursor["timestamp"]),
                    cursor["source_id"],
                    cursor["id"],
                )
            )

        rows = list(
            self._session.execute(
                select(Message, Participant.role)
                .join(Participant, Participant.id == Message.participant_id)
                .where(*conditions)
                .order_by(Message.timestamp.desc(), Message.source_id.desc(), Message.id.desc())
                .limit(limit + 1)
            )
        )
        has_more = len(rows) > limit
        selected = rows[:limit]
        selected.reverse()
        items = [_history_message(message, role) for message, role in selected]
        next_cursor = _encode_cursor(branch.id, scope_id, selected[0][0]) if has_more else None
        return SourceHistoryPage(items, next_cursor, has_more, scope_id)

    def _historical_page(self, branch, snapshot, *, before, limit):
        """历史分支按冻结消息清单分页，不能回退旧 EventNode 或同秒时间比较。"""
        from moonlightbox.agent_runtime.tool_errors import ToolServiceError
        from moonlightbox.world.models import WorldGraphVersion

        from .snapshot_sources import temporal_retrieval_scope

        graph = self._session.get(WorldGraphVersion, snapshot.graph_version_id)
        if graph is None or graph.project_id != branch.project_id:
            raise SourceHistoryError("历史分支绑定图谱不存在")
        try:
            temporal_retrieval_scope(self._session, snapshot, graph)
        except ToolServiceError as error:
            raise SourceHistoryError(str(error)) from error
        ids = list(snapshot.source_message_ids or [])
        scope_id = f"snapshot:{snapshot.id}"
        if before is not None:
            cursor = _decode_cursor(before, branch.id, scope_id)
            if cursor["id"] not in ids:
                raise SourceHistoryError("历史游标不属于冻结消息清单")
            ids = ids[: ids.index(cursor["id"])]
        # 顺序来自已批准清单，既支持多批导入，也不把数值 source_id 按字符串排序。
        rows = self._session.execute(
            select(Message, Participant.role)
            .join(Participant, Participant.id == Message.participant_id)
            .where(
                Message.project_id == branch.project_id,
                Message.id.in_(ids),
                Participant.role.in_(("self", "target")),
            )
        ).all()
        by_id = {message.id: (message, role) for message, role in rows}
        ordered = [by_id[key] for key in ids if key in by_id]
        has_more = len(ordered) > limit
        selected = ordered[-limit:]
        cursor = _encode_cursor(branch.id, scope_id, selected[0][0]) if has_more else None
        return SourceHistoryPage(
            [_history_message(message, role) for message, role in selected],
            cursor,
            has_more,
            scope_id,
        )

    def _branch(self, project_id: str, branch_id: str) -> Branch:
        branch = self._session.get(Branch, branch_id)
        if branch is None or branch.project_id != project_id:
            raise LookupError("时间分支不存在")
        return branch

    def _boundary_message(self, branch: Branch) -> Message | None:
        """定位节点的结束消息，并同时确定本次历史所属的导入批次。

        EventNode 存的是导入方的 ``source_id``，而不是 messages 表的主键。优先用
        节点结束时间消除重复导入时同一 source_id 的歧义；老数据缺少结束时间时才
        退化为该项目内最新的一条同 source_id 记录。
        """

        snapshot = self._session.scalar(
            select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch.id)
        )
        if (
            snapshot is not None
            and snapshot.snapshot_mode == "latest_profile"
            and snapshot.graph_version_id
        ):
            # 当前体验按完整图谱末端起步，界面也必须使用同一边界；旧事件节点
            # 可能早于导入末端，不能因此把已有聊天藏掉。只读绑定图的原文。
            return self._session.scalar(
                select(Message)
                .join(ConversationBundleMessage, ConversationBundleMessage.message_id == Message.id)
                .join(
                    ConversationBundle, ConversationBundle.id == ConversationBundleMessage.bundle_id
                )
                .where(
                    ConversationBundle.graph_version_id == snapshot.graph_version_id,
                    ConversationBundle.project_id == branch.project_id,
                    Message.project_id == branch.project_id,
                    Message.timestamp <= snapshot.cutoff_at,
                )
                .order_by(Message.timestamp.desc(), Message.source_id.desc(), Message.id.desc())
                .limit(1)
            )
        event = self._session.get(EventNode, branch.origin_event_id)
        if event is None or event.project_id != branch.project_id:
            return None
        statement = select(Message).where(
            Message.project_id == branch.project_id,
            Message.source_id == event.end_message_id,
        )
        if event.ended_at is not None:
            exact = self._session.scalar(statement.where(Message.timestamp == event.ended_at))
            if exact is not None:
                return exact
        return self._session.scalar(
            statement.order_by(Message.timestamp.desc(), Message.id.desc()).limit(1)
        )


def _history_message(message: Message, role: str) -> SourceHistoryMessage:
    kind, content = normalize_wechat_display(message.kind, message.content)
    return SourceHistoryMessage(
        id=message.id,
        source_id=message.source_id,
        role=role,
        content=content,
        type=kind,
        media_asset_id=message.media_asset_id,
        timestamp=message.timestamp,
    )


def _before(model: type[Message], at: datetime, source_id: str, row_id: str) -> ColumnElement[bool]:
    return or_(
        model.timestamp < at,
        and_(model.timestamp == at, model.source_id < source_id),
        and_(model.timestamp == at, model.source_id == source_id, model.id < row_id),
    )


def _at_or_before(
    model: type[Message], at: datetime, source_id: str, row_id: str
) -> ColumnElement[bool]:
    return or_(_before(model, at, source_id, row_id), model.id == row_id)


def _scope_id(branch: Branch) -> str:
    """让游标只能用于创建它的分支，避免跨节点翻页。"""

    return f"runtime-source-history-v1:{branch.id}"


def _encode_cursor(branch_id: str, scope_id: str, message: Message) -> str:
    payload = {
        "branch_id": branch_id,
        "scope_id": scope_id,
        "timestamp": message.timestamp.isoformat(),
        "source_id": message.source_id,
        "id": message.id,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def _decode_cursor(encoded: str, branch_id: str, scope_id: str) -> dict[str, str]:
    try:
        value = json.loads(base64.urlsafe_b64decode(encoded.encode()))
        if not isinstance(value, dict):
            raise ValueError
        cursor = {str(key): str(item) for key, item in value.items()}
        required = {"branch_id", "scope_id", "timestamp", "source_id", "id"}
        if set(cursor) != required:
            raise ValueError
        datetime.fromisoformat(cursor["timestamp"])
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise SourceHistoryError("历史游标无效") from error
    if cursor["branch_id"] != branch_id or cursor["scope_id"] != scope_id:
        raise SourceHistoryError("历史游标不属于当前分支")
    return cursor
