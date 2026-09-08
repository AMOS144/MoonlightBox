"""Runtime v1 统一记忆账本与只读 search_memory 工具。"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, cast

from langchain_core.tools import StructuredTool
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_models import BranchMemoryEpisode, BranchMemoryItem
from moonlightbox.branches.models import BranchMessage
from moonlightbox.imports.models import Message

from .config import TOOL_DESCRIPTIONS, TOOL_SCHEMAS
from .db_models import RuntimeLifeEventRow, RuntimeMemoryIndexRow, RuntimeMemoryRow
from .schemas import MemoryEvidence, MemoryRecord, SearchMemoryArgs


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

    def migrate_legacy_branch_items(self, branch_id: str) -> int:
        """一次性投影旧的已审核记忆，Runtime v1 绝不读取 pending 项。

        历史 ``BranchMemoryItem`` 仍可能被旧功能使用，因此这里采用可重复执行的
        复制迁移，而不是删除或改写旧表。旧 episode ID 保留为 source_ids，查询时
        可以继续追溯到原始双边对话。
        """

        existing_ids = set(
            self.session.scalars(
                select(RuntimeMemoryRow.id).where(
                    RuntimeMemoryRow.scope == "branch",
                    RuntimeMemoryRow.branch_id == branch_id,
                )
            )
        )
        legacy_rows = self.session.scalars(
            select(BranchMemoryItem).where(
                BranchMemoryItem.branch_id == branch_id,
                BranchMemoryItem.review_status == "approved",
                BranchMemoryItem.invalidated_at.is_(None),
            )
        )
        migrated = 0
        for item in legacy_rows:
            if item.id in existing_ids:
                continue
            self.session.add(
                RuntimeMemoryRow(
                    id=item.id,
                    scope="branch",
                    branch_id=branch_id,
                    subject=item.subject,
                    predicate=item.predicate,
                    object=item.object[:500],
                    summary=item.content[:2000],
                    status="confirmed",
                    source_ids=list(item.source_episode_ids or []),
                    confidence=float(item.confidence),
                    valid_from=item.valid_from,
                    valid_to=item.valid_to,
                    supersedes_id=item.supersedes_id,
                    created_at=item.created_at,
                )
            )
            migrated += 1
        if migrated:
            self.session.flush()
        return migrated

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
            and (
                row.scope != "branch"
                or branch_record_ids is None
                or row.id in branch_record_ids
            )
        ]
        terms = _query_terms(args.query)
        excluded = excluded_source_ids or set()

        def score(row: RuntimeMemoryRow) -> tuple[int, int, float, datetime]:
            haystack = f"{row.subject} {row.predicate} {row.object} {row.summary}".lower()
            matches = sum(term in haystack for term in terms)
            priority = 2 if row.scope == "branch" and row.status == "confirmed" else 1
            return matches, priority, float(row.confidence), _aware_or_min(row.created_at)

        results: list[MemoryEvidence] = []
        for row in sorted(rows, key=score, reverse=True):
            source_ids = [str(item) for item in (row.source_ids or [])]
            if source_ids and set(source_ids).issubset(excluded):
                continue
            original = self._original_excerpt(source_ids) if args.include_original else None
            results.append(
                MemoryEvidence(
                    record_id=row.id,
                    scope=row.scope,  # type: ignore[arg-type]
                    summary=row.summary,
                    source_ids=source_ids,
                    occurred_at=row.valid_from,
                    valid_until=row.valid_to,
                    certainty="approved" if row.status == "confirmed" else "observed",
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
        }

    def tool(
        self,
        *,
        branch_id: str,
        snapshot_id: str | None,
        excluded_source_ids: set[str] | None = None,
        as_of: datetime | None = None,
    ) -> StructuredTool:
        """绑定当前 Cycle 身份，模型只能传 scope/query/limit 等参数。"""

        index = self.current_index(branch_id)
        active_ids = set(index.active_record_ids or []) if index is not None else set()

        def invoke(**kwargs: Any) -> dict[str, Any]:
            args = SearchMemoryArgs.model_validate(kwargs)
            return self.search(
                branch_id=branch_id,
                snapshot_id=snapshot_id,
                args=args,
                excluded_source_ids=excluded_source_ids,
                as_of=as_of,
                branch_record_ids=active_ids,
            )

        return StructuredTool.from_function(
            name="search_memory",
            description=TOOL_DESCRIPTIONS["search_memory"],
            func=invoke,
            args_schema=cast(Any, TOOL_SCHEMAS["search_memory"]),
        )

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
        episode = self.session.scalar(
            select(BranchMemoryEpisode.user_content)
            .where(BranchMemoryEpisode.id.in_(source_ids))
            .limit(1)
        )
        if isinstance(episode, str):
            return episode[:1200]
        return None


def _query_terms(query: str) -> list[str]:
    """英文按词、中文按双字切分，使没有空格的查询也能排序。"""

    normalized = query.lower().strip()
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized)
    terms: list[str] = []
    for word in words:
        if len(word) <= 2 or not re.fullmatch(r"[\u4e00-\u9fff]+", word):
            terms.append(word)
        else:
            terms.extend(word[index : index + 2] for index in range(len(word) - 1))
    return list(dict.fromkeys(term for term in terms if term))


def _at_or_after(value: datetime, current: datetime) -> bool:
    """兼容 SQLite 读出的 naive 时间，统一按 UTC 比较。"""

    left = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    right = current if current.tzinfo is not None else current.replace(tzinfo=UTC)
    return left >= right


def _aware_or_min(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
