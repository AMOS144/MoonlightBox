"""平级通信复用 RuntimeEvent/Job：入队不等于对方已读，更不等于计划提交。"""

import hashlib
import json

from sqlalchemy import select, update

from moonlightbox.jobs.models import Job

from ..branch_models import Branch
from ..db_models import RuntimeEventRow
from ..event_queue import RuntimeEventQueue
from .messages import PeerMessage, shared_message


def enqueue_peer_message(
    session, *, branch, sender, recipient, content, task_id, occurred_at, key=None, payload=None
):
    """调用者控制事务；Executor 可将成功通知与计划写入原子提交。"""
    digest = hashlib.sha256(
        json.dumps([task_id, sender, recipient, content], ensure_ascii=False).encode()
    ).hexdigest()
    envelope = PeerMessage(
        sender=sender,
        recipient=recipient,
        kind="notification",
        content=content,
        task_id=task_id,
        payload=payload or {},
    )
    return RuntimeEventQueue(session).enqueue(
        project_id=branch.project_id,
        branch_id=branch.id,
        event_type="planner_inbox" if recipient == "day_planner" else "agent_message",
        payload={"discussion": [envelope.model_dump(mode="json")]},
        idempotency_key=key or f"peer:{digest}",
        occurred_at=occurred_at,
    )


def schedule_planner_inbox(session, branch_id, now):
    """每拍合并所有未分配消息；运行/恢复中的 Planner 不被新输入打断。"""
    from .planner_tasks import enqueue_dispatch

    # 与消息写入使用同一分支短事务锁，两个扫描器不能同时拆分同一批消息。
    session.execute(
        update(Branch)
        .where(Branch.id == branch_id)
        .values(runtime_input_revision=Branch.runtime_input_revision)
    )
    jobs = list(
        session.scalars(
            select(Job).where(
                Job.kind == "runtime-v1-cycle", Job.payload["branch_id"].as_string() == branch_id
            )
        )
    )
    jobs = [
        j for j in jobs if j.payload.get("branch_id") == branch_id and j.payload.get("planner_task")
    ]
    if any(j.status in {"queued", "running", "cancelling"} for j in jobs):
        return False
    # 已分配任务由 Job 负责恢复，包括失败任务，不能绕开恢复预算重建任务。
    assigned = {ref for j in jobs for ref in j.payload["planner_task"].get("input_ids", [])}
    rows = [
        row
        for row in session.scalars(
            select(RuntimeEventRow)
            .where(
                RuntimeEventRow.branch_id == branch_id,
                RuntimeEventRow.event_type == "planner_inbox",
                RuntimeEventRow.status == "queued",
                RuntimeEventRow.occurred_at <= now,
            )
            .order_by(RuntimeEventRow.received_at, RuntimeEventRow.id)
        )
        if row.id not in assigned
    ]
    if not rows:
        return False
    from ..life_events.delivery import route_inbox_to_life
    if route_inbox_to_life(session, branch_id, rows):
        return False
    messages = [
        shared_message(
            **{
                "sender": item["sender"],
                "recipient": item["recipient"],
                "kind": item["kind"],
                "content": item["content"],
                "task_id": item["task_id"],
                "payload": item.get("payload"),
            }
        )
        for row in rows
        for item in row.payload.get("discussion", [])
    ]
    dispatch = {
        "input_ids": [row.id for row in rows],
        "request": {"reason": "处理本批协作消息；需要调整则提交计划，否则完成处理即可。"},
    }
    waiting = [job for job in jobs if (job.checkpoint or {}).get("awaiting_answer")]
    if len(waiting) == 1:
        # 单个悬而未决的调查直接续接，不要求模型填写内部任务 ID。
        parent = waiting[0]
        dispatch.update(resume=parent.checkpoint["planner_state"], parent_job_id=parent.id)
        dispatch.pop("request")
    enqueue_dispatch(
        session,
        branch_id,
        f"inbox:{rows[0].id}",
        dispatch,
        messages,
    )
    return True


def peer_history(session, branch_id, *, recipient, current=()):
    """公开发言延续到后续任务；未读来信由各自输入边界交付，不能偷读。"""
    result = []
    for row in session.scalars(
        select(RuntimeEventRow)
        .where(
            RuntimeEventRow.branch_id == branch_id,
        )
        .order_by(RuntimeEventRow.received_at, RuntimeEventRow.id)
    ):
        for envelope in row.payload.get("discussion", []):
            if row.status == "completed" or envelope.get("sender") == recipient:
                result.append(envelope)
    # 同一发言在队列和图检查点中可能各有一份，按完整语义合并。
    unique = {}
    for item in [*result, *current]:
        unique.setdefault(json.dumps(item, sort_keys=True, ensure_ascii=False), item)
    return list(unique.values())
