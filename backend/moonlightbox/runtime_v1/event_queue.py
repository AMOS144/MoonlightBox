"""RuntimeEvent 持久化队列的轻量访问层。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .db_models import RuntimeEventRow


class RuntimeEventQueue:
    """持久化队列的窄访问层；合并与 Cycle 编排仍由 RuntimeService 负责。"""

    def __init__(self, session: Session) -> None:
        self.session = session

    def enqueue(
        self,
        *,
        project_id: str,
        branch_id: str,
        event_type: str,
        payload: dict[str, object],
        idempotency_key: str,
        occurred_at: datetime | None = None,
        priority: int = 0,
    ) -> RuntimeEventRow:
        existing = self.session.scalar(
            select(RuntimeEventRow).where(
                RuntimeEventRow.branch_id == branch_id,
                RuntimeEventRow.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return existing
        row = RuntimeEventRow(
            project_id=project_id,
            branch_id=branch_id,
            event_type=event_type,
            payload=payload,
            idempotency_key=idempotency_key,
            occurred_at=occurred_at or datetime.now(UTC),
            priority=priority,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def claim(self, branch_id: str, *, limit: int = 16) -> list[RuntimeEventRow]:
        """CAS 领取简单批次，避免备用维护调用重复执行同一事件。"""

        rows = list(
            self.session.scalars(
                select(RuntimeEventRow)
                .where(RuntimeEventRow.branch_id == branch_id, RuntimeEventRow.status == "queued")
                .order_by(RuntimeEventRow.priority.desc(), RuntimeEventRow.occurred_at)
                .limit(limit)
            )
        )
        now = datetime.now(UTC)
        claimed_rows: list[RuntimeEventRow] = []
        for row in rows:
            result = cast(
                Any,
                self.session.execute(
                    update(RuntimeEventRow)
                    .where(RuntimeEventRow.id == row.id, RuntimeEventRow.status == "queued")
                    .values(status="claimed", claimed_at=now)
                ),
            )
            if result.rowcount == 1:
                claimed_rows.append(row)
        self.session.flush()
        return claimed_rows

    def complete(self, rows: list[RuntimeEventRow]) -> None:
        now = datetime.now(UTC)
        for row in rows:
            row.status = "completed"
            row.completed_at = now
        self.session.flush()
