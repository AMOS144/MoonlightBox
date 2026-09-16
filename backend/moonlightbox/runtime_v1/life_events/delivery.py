"""公共消息承载协商，文字回复续接原工作，不再要求另一份心理批准。"""

from langchain_core.messages import messages_to_dict
from sqlalchemy import select

from ..branch_models import Branch
from ..collaboration.messages import shared_message
from ..collaboration.transport import enqueue_peer_message
from ..db_models import RuntimeEventRow
from .models import LifeOpportunityRow


def request_advice(session, row, state, now):
    """优先复用 Agent 已发问题；否则通过同一消息传输提交具体问题。"""
    task_id = f"life:{row.id}:{row.step}"
    existing = [
        e
        for e in session.scalars(
            select(RuntimeEventRow)
            .where(
                RuntimeEventRow.branch_id == row.branch_id,
                RuntimeEventRow.event_type == "agent_message",
            )
            .order_by(RuntimeEventRow.received_at)
        )
        if any(
            m.get("task_id") == task_id and m.get("sender") == "day_planner"
            for m in e.payload.get("discussion", [])
        )
    ]
    candidate = state.get("life", {}).get("candidate", {})
    event = (
        existing[-1]
        if existing
        else enqueue_peer_message(
            session,
            branch=session.get(Branch, row.branch_id),
            sender="day_planner",
            recipient="director",
            content=candidate.get("question") or candidate.get("reason") or "需要讨论当前生活安排",
            task_id=task_id,
            occurred_at=now,
            payload={"candidate": candidate, "status": "not_occurred"},
        )
    )
    row.work = {**row.work, "state": state, "awaiting_director_event_id": event.id}


def route_inbox_to_life(session, branch_id, rows):
    """只有一个活动生活任务；来信交回它理解，不推断来信等于批准。"""
    active = session.scalar(
        select(LifeOpportunityRow).where(
            LifeOpportunityRow.branch_id == branch_id, LifeOpportunityRow.status == "active"
        )
    )
    if not active:
        return False
    # 未让出时留在 inbox，避免修改正在执行的工作；下一拍再交付。
    if not active.work.get("awaiting_director_event_id"):
        return True
    messages = [
        shared_message(
            m["sender"],
            "day_planner",
            "answer",
            m["content"],
            f"life:{active.id}",
            m.get("payload"),
        )
        for r in rows
        for m in r.payload.get("discussion", [])
    ]
    active.work = {
        **active.work,
        "awaiting_director_event_id": None,
        "peer_messages": [*active.work.get("peer_messages", []), *messages_to_dict(messages)],
        "reply_event_ids": [*active.work.get("reply_event_ids", []), *[r.id for r in rows]],
    }
    for row in rows:
        row.status = "assigned"  # 不属于 Director 的临时 claim，不能被其租约回收器重置。
    return True
