import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from moonlightbox.branches.baseline_models import BranchBaselineManifest
from moonlightbox.branches.models import Branch
from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.wechat_rendering import normalize_wechat_display


class BranchHistoryCursorError(ValueError):
    pass


@dataclass(frozen=True)
class BranchHistoryMessage:
    id: str
    source_id: str
    role: str
    content: str
    type: str
    media_asset_id: str | None
    timestamp: datetime


@dataclass(frozen=True)
class BranchHistoryPage:
    items: list[BranchHistoryMessage]
    next_cursor: str | None
    has_more: bool
    manifest_id: str


class BranchHistoryService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def page(
        self,
        project_id: str,
        branch_id: str,
        *,
        before: str | None,
        limit: int,
    ) -> BranchHistoryPage:
        if limit < 20 or limit > 100:
            raise BranchHistoryCursorError("limit 必须在 20 到 100 之间")
        branch, manifest = self._ready_manifest(project_id, branch_id)
        conditions = [
            Message.project_id == project_id,
            Message.import_id == manifest.import_id,
            Participant.role.in_(("self", "target")),
            (
                _at_or_before(
                    Message,
                    manifest.boundary_timestamp,
                    manifest.boundary_source_id,
                    manifest.boundary_message_id,
                )
                if manifest.boundary_inclusive
                else _before(
                    Message,
                    manifest.boundary_timestamp,
                    manifest.boundary_source_id,
                    manifest.boundary_message_id,
                )
            ),
        ]
        if before is not None:
            cursor = _decode_cursor(before, branch.id, manifest.id)
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
                .order_by(
                    Message.timestamp.desc(),
                    Message.source_id.desc(),
                    Message.id.desc(),
                )
                .limit(limit + 1)
            )
        )
        has_more = len(rows) > limit
        selected = rows[:limit]
        selected.reverse()
        items = [_history_message(message, role) for message, role in selected]
        next_cursor = None
        if has_more and selected:
            oldest = selected[0][0]
            next_cursor = _encode_cursor(branch.id, manifest.id, oldest)
        return BranchHistoryPage(
            items=items,
            next_cursor=next_cursor,
            has_more=has_more,
            manifest_id=manifest.id,
        )

    def recent_rows(
        self,
        branch: Branch,
        *,
        message_limit: int = 80,
    ) -> list[tuple[Message, str]]:
        manifest = (
            self._session.get(BranchBaselineManifest, branch.baseline_manifest_id)
            if branch.baseline_manifest_id is not None
            else None
        )
        if manifest is None:
            return []
        rows = cast(
            list[tuple[Message, str]],
            self._session.execute(
                select(Message, Participant.role)
                .join(Participant, Participant.id == Message.participant_id)
                .where(
                    Message.id.in_(manifest.recent_tail_message_ids),
                    Message.import_id == manifest.import_id,
                )
                .order_by(
                    Message.timestamp.desc(),
                    Message.source_id.desc(),
                    Message.id.desc(),
                )
                .limit(message_limit)
            ).all(),
        )
        return list(reversed(rows))

    def authentic_example_rows(
        self,
        branch: Branch,
        *,
        message_limit: int = 500,
    ) -> list[tuple[Message, str]]:
        """Read a wider, still cutoff-safe window for persona example retrieval."""

        manifest = (
            self._session.get(BranchBaselineManifest, branch.baseline_manifest_id)
            if branch.baseline_manifest_id is not None
            else None
        )
        if manifest is None or message_limit <= 0:
            return []
        boundary = (
            _at_or_before(
                Message,
                manifest.boundary_timestamp,
                manifest.boundary_source_id,
                manifest.boundary_message_id,
            )
            if manifest.boundary_inclusive
            else _before(
                Message,
                manifest.boundary_timestamp,
                manifest.boundary_source_id,
                manifest.boundary_message_id,
            )
        )
        rows = cast(
            list[tuple[Message, str]],
            self._session.execute(
                select(Message, Participant.role)
                .join(Participant, Participant.id == Message.participant_id)
                .where(
                    Message.project_id == branch.project_id,
                    Message.import_id == manifest.import_id,
                    Participant.role.in_(("self", "target")),
                    boundary,
                )
                .order_by(
                    Message.timestamp.desc(),
                    Message.source_id.desc(),
                    Message.id.desc(),
                )
                .limit(message_limit)
            ).all(),
        )
        return list(reversed(rows))

    def _ready_manifest(
        self,
        project_id: str,
        branch_id: str,
    ) -> tuple[Branch, BranchBaselineManifest]:
        branch = self._session.get(Branch, branch_id)
        if (
            branch is None
            or branch.project_id != project_id
            or branch.baseline_status != "ready"
            or branch.baseline_manifest_id is None
        ):
            raise LookupError("分支基础历史尚未准备完成")
        manifest = self._session.get(BranchBaselineManifest, branch.baseline_manifest_id)
        if manifest is None:
            raise LookupError("分支基础历史清单不存在")
        return branch, manifest


def _history_message(message: Message, role: str) -> BranchHistoryMessage:
    kind, content = normalize_wechat_display(message.kind, message.content)
    return BranchHistoryMessage(
        id=message.id,
        source_id=message.source_id,
        role=role,
        content=content,
        type=kind,
        media_asset_id=message.media_asset_id,
        timestamp=message.timestamp,
    )


def _before(
    model: type[Message],
    at: datetime,
    source_id: str,
    row_id: str,
) -> ColumnElement[bool]:
    return or_(
        model.timestamp < at,
        and_(model.timestamp == at, model.source_id < source_id),
        and_(
            model.timestamp == at,
            model.source_id == source_id,
            model.id < row_id,
        ),
    )


def _at_or_before(
    model: type[Message],
    at: datetime,
    source_id: str,
    row_id: str,
) -> ColumnElement[bool]:
    return or_(
        _before(model, at, source_id, row_id),
        model.id == row_id,
    )


def _encode_cursor(branch_id: str, manifest_id: str, message: Message) -> str:
    payload = {
        "branch_id": branch_id,
        "manifest_id": manifest_id,
        "timestamp": message.timestamp.isoformat(),
        "source_id": message.source_id,
        "id": message.id,
    }
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


def _decode_cursor(
    encoded: str,
    branch_id: str,
    manifest_id: str,
) -> dict[str, str]:
    try:
        value = json.loads(base64.urlsafe_b64decode(encoded.encode()))
        if not isinstance(value, dict):
            raise ValueError
        cursor = {str(key): str(item) for key, item in value.items()}
        required = {"branch_id", "manifest_id", "timestamp", "source_id", "id"}
        if set(cursor) != required:
            raise ValueError
        datetime.fromisoformat(cursor["timestamp"])
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise BranchHistoryCursorError("历史游标无效") from error
    if cursor["branch_id"] != branch_id or cursor["manifest_id"] != manifest_id:
        raise BranchHistoryCursorError("历史游标不属于当前分支")
    return cursor
