"""Runtime v1 统一记忆账本与只读 search_memory 工具。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message

from .branch_models import BranchMessage
from .db_models import (
    RuntimeLifeEventRow,
    RuntimeMemoryIndexRow,
    RuntimeMemoryRow,
    RuntimeSnapshotRow,
)
from .schemas import MemoryEvidence, MemoryRecord
from .tools.memory_search import SearchMemoryArgs


class MemoryService:
    """所有事实写入和读取都经由此服务，避免旧记忆表之间互相竞争。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, record: MemoryRecord) -> MemoryRecord:
        if record.scope == "branch" and not record.branch_id:
            raise ValueError("branch 记忆必须提供 branch_id")
        row = RuntimeMemoryRow(
            id=record.id,
            scope=record.scope,
            branch_id=record.branch_id,
            snapshot_id=record.snapshot_id,
            subject=record.subject,
            predicate=record.predicate,
            object=record.object,
            summary=record.summary,
            status=record.status,
            source_ids=record.source_ids,
            confidence=record.confidence,
            valid_from=record.valid_from,
            valid_to=record.valid_to,
            supersedes_id=record.supersedes_id,
        )
        self.session.add(row)
        return record

    def current_index(self, branch_id: str) -> RuntimeMemoryIndexRow | None:
        return self.session.scalar(
            select(RuntimeMemoryIndexRow).where(
                RuntimeMemoryIndexRow.branch_id == branch_id,
                RuntimeMemoryIndexRow.is_current.is_(True),
            )
        )

    def rebuild_index(self, branch_id: str, *, reason: str) -> RuntimeMemoryIndexRow:
        """为当前有效的 branch 记忆建立稳定读取边界。"""

        now = datetime.now(UTC)
        active_ids = list(
            self.session.scalars(
                select(RuntimeMemoryRow.id)
                .where(
                    RuntimeMemoryRow.scope == "branch",
                    RuntimeMemoryRow.branch_id == branch_id,
                    RuntimeMemoryRow.status.in_(["asserted", "confirmed"]),
                    or_(
                        RuntimeMemoryRow.valid_to.is_(None),
                        RuntimeMemoryRow.valid_to >= now,
                    ),
                )
                .order_by(RuntimeMemoryRow.created_at, RuntimeMemoryRow.id)
            )
        )
        current = self.current_index(branch_id)
        if current is not None and list(current.active_record_ids or []) == active_ids:
            return current
        version = (current.version + 1) if current is not None else 1
        if current is not None:
            current.is_current = False
        index = RuntimeMemoryIndexRow(
            branch_id=branch_id,
            version=version,
            active_record_ids=active_ids,
            previous_version_id=current.id if current is not None else None,
            reason=reason[:120],
            is_current=True,
        )
        self.session.add(index)
        self.session.flush()
        return index

    def search(
        self,
        *,
        branch_id: str,
        snapshot_id: str | None,
        args: SearchMemoryArgs,
        excluded_source_ids: set[str] | None = None,
        as_of: datetime | None = None,
        branch_record_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        """按 scope 和关键词读取当前有效记录，并保留来源边界。"""
        scopes = {args.scope} if args.scope != "both" else {"branch", "world"}
        if snapshot_id is None:
            scopes.discard("world")
        query = select(RuntimeMemoryRow).where(
            RuntimeMemoryRow.status.in_(["asserted", "confirmed"]),
            RuntimeMemoryRow.scope.in_(scopes),
        )
        if "branch" in scopes:
            query = query.where(
                or_(RuntimeMemoryRow.scope == "world", RuntimeMemoryRow.branch_id == branch_id)
            )
        if snapshot_id is not None:
            query = query.where(
                or_(RuntimeMemoryRow.scope == "branch", RuntimeMemoryRow.snapshot_id == snapshot_id)
            )
        current = as_of or datetime.now(UTC)
        rows = [
            row
            for row in self.session.scalars(query.order_by(RuntimeMemoryRow.created_at.desc()))
            if (row.valid_to is None or _at_or_after(row.valid_to, current))
            and (row.valid_from is None or _at_or_after(current, row.valid_from))
            and (row.scope != "branch" or branch_record_ids is None or row.id in branch_record_ids)
        ]
        excluded = excluded_source_ids or set()
        snapshot_source_ids = self._snapshot_source_ids(snapshot_id)
        branch_source_ids = self._branch_source_ids(branch_id)

        def score(row: RuntimeMemoryRow) -> tuple[int, int, float, datetime]:
            # SQL 部分是分支/画像的近期上下文候选，不再用字面词频冒充语义相关性。
            matches = 0
            priority = 2 if row.scope == "branch" and row.status == "confirmed" else 1
            return matches, priority, float(row.confidence), _aware_or_min(row.created_at)

        results: list[MemoryEvidence] = []
        for row in sorted(rows, key=score, reverse=True):
            # 旧 Profile 编译产物中曾混入“相关对话”“日期文本”等展示标签。记忆
            # 摘要仍可作为弱背景阅读，但这些标签绝不能随 source_ids 传给 Agent，
            # 否则它们可能被误写为 DayPlan 的可审计证据。
            allowed_sources = (
                snapshot_source_ids
                if row.scope == "world"
                else branch_source_ids | snapshot_source_ids
            )
            source_ids = [
                str(item) for item in (row.source_ids or []) if str(item) in allowed_sources
            ]
            if source_ids and set(source_ids).issubset(excluded):
                continue
            original = self._original_excerpt(source_ids) if args.include_original else None
            results.append(
                MemoryEvidence(
                    basis=row.predicate,
                    record_id=row.id,
                    scope=row.scope,  # type: ignore[arg-type]
                    summary=row.summary,
                    source_ids=source_ids,
                    occurred_at=row.valid_from,
                    valid_until=row.valid_to,
                    certainty=(
                        "inferred"
                        if row.predicate == "inferred"
                        or any(source.startswith("profile:") for source in source_ids)
                        else "approved"
                        if row.status == "confirmed"
                        else "observed"
                    ),
                    original=original,
                )
            )
            if len(results) >= args.limit:
                break
        return {
            "tool_name": "search_memory",
            "scope": args.scope,
            "as_of": current.isoformat(),
            "source_ids": [source for item in results for source in item.source_ids],
            "truncated": len(results) >= args.limit and len(rows) > len(results),
            "data": [item.model_dump(mode="json") for item in results],
            "selection": "recent_context_not_semantic_matches",
        }

    def _snapshot_source_ids(self, snapshot_id: str | None) -> set[str]:
        if snapshot_id is None:
            return set()
        snapshot = self.session.get(RuntimeSnapshotRow, snapshot_id)
        if snapshot is None:
            return set()
        from .snapshot_sources import frozen_source_ids

        return frozen_source_ids(self.session, snapshot)

    def _branch_source_ids(self, branch_id: str) -> set[str]:
        """分支记忆只能引向本分支真实消息或已提交生活事件。"""

        source_ids = set(
            self.session.scalars(
                select(BranchMessage.id).where(BranchMessage.branch_id == branch_id)
            )
        )
        source_ids.update(
            self.session.scalars(
                select(RuntimeLifeEventRow.id).where(RuntimeLifeEventRow.branch_id == branch_id)
            )
        )
        return source_ids

    def _original_excerpt(self, source_ids: list[str]) -> str | None:
        """按需返回少量原文；未找到时绝不拿摘要冒充原文。"""

        if not source_ids:
            return None
        branch_message = self.session.scalar(
            select(BranchMessage.content).where(BranchMessage.id.in_(source_ids)).limit(1)
        )
        if isinstance(branch_message, str):
            return branch_message[:1200]
        imported = self.session.scalar(
            select(Message.content).where(Message.id.in_(source_ids)).limit(1)
        )
        if isinstance(imported, str):
            return imported[:1200]
        life_event = self.session.scalar(
            select(RuntimeLifeEventRow.payload)
            .where(RuntimeLifeEventRow.id.in_(source_ids))
            .limit(1)
        )
        if isinstance(life_event, dict):
            return str(life_event)[:1200]
        return None


def _at_or_after(value: datetime, current: datetime) -> bool:
    """兼容 SQLite 读出的 naive 时间，统一按 UTC 比较。"""

    left = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    right = current if current.tzinfo is not None else current.replace(tzinfo=UTC)
    return left >= right


def _aware_or_min(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
