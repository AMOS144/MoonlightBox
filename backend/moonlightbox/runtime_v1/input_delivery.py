"""统一输入交付边界：到期唤醒入事件队列，提交时只消费实际读取集合。"""

from datetime import UTC, datetime

from sqlalchemy import select, update

from .db_models import RuntimeEventRow, RuntimeWakeupRow
from .event_queue import RuntimeEventQueue


def deliver_wakeups(session, project_id, branch_id, clock):
    if clock.status != "running":
        return
    due = list(
        session.scalars(
            select(RuntimeWakeupRow)
            .where(
                RuntimeWakeupRow.branch_id == branch_id,
                RuntimeWakeupRow.status == "scheduled",
                RuntimeWakeupRow.trigger_type != "life_followup",
                RuntimeWakeupRow.wake_at <= clock.now(),
            )
            .order_by(RuntimeWakeupRow.wake_at, RuntimeWakeupRow.id)
        )
    )
    for wakeup in due:
        changed = session.execute(
            update(RuntimeWakeupRow)
            .where(
                RuntimeWakeupRow.id == wakeup.id,
                RuntimeWakeupRow.status == "scheduled",
            )
            .values(status="delivered")
        )
        if changed.rowcount != 1:
            continue
        RuntimeEventQueue(session).enqueue(
            project_id=project_id,
            branch_id=branch_id,
            event_type=wakeup.trigger_type,
            occurred_at=wakeup.wake_at,
            idempotency_key=f"wakeup-delivery:{wakeup.id}",
            payload={"wakeup_id": wakeup.id, "reason": wakeup.reason},
        )


def consume_inputs(session, branch_id, ids):
    """调用者持有业务提交事务；未读输入永远不在这个集合内。"""
    rows = list(
        session.scalars(
            select(RuntimeEventRow).where(
                RuntimeEventRow.branch_id == branch_id,
                RuntimeEventRow.id.in_(ids),
            )
        )
    )
    wakeup_ids = [row.payload["wakeup_id"] for row in rows if row.payload.get("wakeup_id")]
    session.execute(
        update(RuntimeEventRow)
        .where(
            RuntimeEventRow.branch_id == branch_id,
            RuntimeEventRow.id.in_(ids),
            RuntimeEventRow.status.in_(["queued", "claimed"]),
        )
        .values(status="completed", completed_at=datetime.now(UTC))
    )
    session.execute(
        update(RuntimeWakeupRow)
        .where(
            RuntimeWakeupRow.branch_id == branch_id,
            RuntimeWakeupRow.id.in_([*ids, *wakeup_ids]),
            RuntimeWakeupRow.status.in_(["delivered", "executing"]),
        )
        .values(status="completed", executing_at=None)
    )
