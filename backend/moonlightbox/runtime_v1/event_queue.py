"""RuntimeEvent 持久化队列的轻量访问层。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
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
            occurred_at=(
                occurred_at.replace(tzinfo=UTC)
                if occurred_at and occurred_at.tzinfo is None
                else occurred_at or datetime.now(UTC)
            ).astimezone(UTC),
            priority=priority,
        )
        self.session.add(row)
        self.session.flush()
        return row
