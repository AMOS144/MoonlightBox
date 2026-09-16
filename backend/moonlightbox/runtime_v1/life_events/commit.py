"""经历、跨块计划与通知原子提交；不规定事件额度或要求心理批准。"""

import json
from copy import deepcopy
from datetime import UTC, datetime

from sqlalchemy import select, update

from ..branch_models import Branch
from ..db_models import RuntimeDayPlanRow, RuntimeLifeEventRow, RuntimeWakeupRow
from ..event_queue import RuntimeEventQueue
from .contracts import LifeAdvanceDecision
from .models import DayPlanVersionRow, LifeOpportunityRow
from .scheduling import utc


def archive_plan(session, plan):
    key = f"{plan.id}:{plan.version}"
    if not session.get(DayPlanVersionRow, key):
        session.add(
            DayPlanVersionRow(
                id=key,
                branch_id=plan.branch_id,
                plan_date=plan.plan_date,
                version=plan.version,
                blocks=deepcopy(plan.blocks),
                generation_metadata=deepcopy(plan.generation_metadata),
            )
        )
        session.flush()


def minute_projection(blocks):
    """比较真实语义前缀；拆块不能偷偷改写过去。"""
    result = {}
    for block in blocks:
        start, end = (
            sum(int(n) * m for n, m in zip(t.split(":"), (60, 1), strict=True))
            for t in (block["start"], block["end"])
        )
        if start < 0 or end > 1440 or end <= start:
            raise ValueError("计划时间必须正向且在当天内")
        semantic = tuple(
            block.get(k)
            for k in (
                "activity",
                "location_role",
                "default_availability",
                "basis",
                "confidence",
                "assumption",
            )
        )
        semantic += (json.dumps(sorted(block.get("evidence_ids", []))),)
        for minute in range(start, end):
            if minute in result:
                raise ValueError("计划块重叠")
            result[minute] = semantic
    if set(result) != set(range(1440)):
        raise ValueError("计划必须完整覆盖当天")
    return result


def validate_plan_change(old_blocks, new_blocks, now, target_date):
    new = minute_projection(new_blocks)
    if not old_blocks:
        return
    old = minute_projection(old_blocks)
    changed = {m for m in old if old[m] != new[m]}
    minute = now.hour * 60 + now.minute + bool(now.second or now.microsecond)
    boundary = ((minute + 14) // 15) * 15 if target_date == now.date() else 0
    protected = set()
    for b in old_blocks:
        if b.get("basis") == "branch_commitment":
            start, end = (int(t[:2]) * 60 + int(t[3:]) for t in (b["start"], b["end"]))
            protected.update(range(start, end))
    if any(m < boundary or m in protected for m in changed):
        raise ValueError("不能改写已发生前缀或未经相应流程更新的已确认约定")


def validate_life_result(value, *, now, plans, requests):
    """提交工具先反馈可修复参数；事务中再用最新状态检查。"""
    if value.follow_up_at is not None:
        if value.follow_up_at.tzinfo is None or utc(value.follow_up_at) <= utc(now):
            raise ValueError("follow_up_at 必须是带时区且晚于当前虚拟时间的时间点")
        if value.outcome == "no_event" and not any(r.get("parent_event_id") for r in requests):
            raise ValueError("no_event 的后续提醒必须关联本次正在跟进的已有事件")
    for proposal in value.plan_proposals:
        from ..tools.plan_validation import validate_plan_structure

        validate_plan_structure(proposal)
        if proposal.plan_date < now.date():
            raise ValueError("只能修改当前或未来日期，不能回写过去日程")
        old = plans.get(proposal.plan_date.isoformat(), {})
        validate_plan_change(
            old.get("blocks", []),
            [b.model_dump(mode="json") for b in proposal.blocks],
            now,
            proposal.plan_date,
        )
    if value.event:
        at = value.event.occurrence.occurred_at
        if at is not None:
            if at.tzinfo is None or utc(at) > utc(now):
                raise ValueError("occurred_at 必须带时区，不能将未来写为已发生")
            ranges = [r["recovery"] for r in requests if r.get("recovery")]
            if not any(
                utc(datetime.fromisoformat(r["from"]))
                <= utc(at)
                <= utc(datetime.fromisoformat(r["to"]))
                for r in ranges
            ):
                raise ValueError("仅离线恢复允许指定恢复区间内的 occurred_at；普通事件省略时间")
        if value.event.handling.actual_outcome and not any(
            r.get("parent_event_id") or r.get("recovery") for r in requests
        ):
            raise ValueError("新事件不能提前填写 actual_outcome；后续推进时再确认实际结果")


def batch_requests(session, row):
    rows = [session.get(LifeOpportunityRow, key) for key in row.work.get("batch_ids", [row.id])]
    return [
        {
            "received_at": utc(r.cursor_at).isoformat(),
            **({"opportunity": r.brief} if r.kind == "opportunity" else {}),
            **({"recovery": r.work["recovery"]} if r.kind == "recovery" else {}),
            **({"parent_event_id": r.work["parent_event_id"]} if r.kind == "followup" else {}),
        }
        for r in rows
        if r
    ]


def commit_simulated_event(executor, row, *, decision, input_revision, now):
    session = executor.session
    key = f"simulated-life:{row.id}"
    existing = session.scalar(
        select(RuntimeLifeEventRow).where(RuntimeLifeEventRow.idempotency_key == key)
    )
    if existing:
        return existing
    if row.status != "active":
        raise ValueError("生活任务已结束")
    locked = session.execute(
        update(Branch)
        .where(Branch.id == row.branch_id, Branch.runtime_input_revision == input_revision)
        .values(runtime_input_revision=input_revision)
    )
    if locked.rowcount != 1:
        raise ValueError("stale_input_revision")
    value = LifeAdvanceDecision.model_validate_json(json.dumps(decision))
    if value.outcome != "submit_event" or value.event is None:
        raise ValueError("只能提交完整事件")
    plans = {
        p.plan_date: p
        for p in session.scalars(
            select(RuntimeDayPlanRow).where(RuntimeDayPlanRow.branch_id == row.branch_id)
        )
    }
    requests = batch_requests(session, row)
    validate_life_result(
        value, now=now, plans={k: {"blocks": p.blocks} for k, p in plans.items()}, requests=requests
    )
    versions = row.work.get("plan_versions", {})
    for proposal in value.plan_proposals:
        date_key = proposal.plan_date.isoformat()
        old = plans.get(date_key)
        version = old.version if old else 0
        if version != versions.get(date_key, 0):
            raise ValueError("计划版本已改变，请结合新计划调整，不要覆盖")
        plans[date_key] = executor.commit_day_plan_proposal(
            branch_id=row.branch_id,
            proposal=proposal,
            virtual_now=now,
            mode="life_event",
            expected_version=version,
            idempotency_key=f"{key}:{date_key}",
        )
    occurred_at = value.event.occurrence.occurred_at or now
    event = RuntimeLifeEventRow(
        branch_id=row.branch_id,
        event_type="simulated_life",
        occurred_at=utc(occurred_at),
        idempotency_key=key,
        payload={
            "origin": "simulation",
            "event": value.event.model_dump(mode="json"),
            "opportunity_id": row.id,
            "recorded_at": datetime.now(UTC).isoformat(),
            "generated_during_recovery": any(r.get("recovery") for r in requests),
            "parent_event_ids": [
                r["parent_event_id"] for r in requests if r.get("parent_event_id")
            ],
            "plan_versions": {
                p.plan_date.isoformat(): plans[p.plan_date.isoformat()].version
                for p in value.plan_proposals
            },
        },
    )
    session.add(event)
    session.flush()
    branch = session.get(Branch, row.branch_id)
    RuntimeEventQueue(session).enqueue(
        project_id=branch.project_id,
        branch_id=row.branch_id,
        event_type="system",
        payload={
            "reason": "simulated_life_committed",
            "origin": "simulation",
            "source_event_id": event.id,
            "event": event.payload,
            "status": "committed",
        },
        idempotency_key=f"life-notify:{row.id}",
        occurred_at=utc(now),
    )
    schedule_followup(session, row, value.follow_up_at, event.id)
    row.status, row.event_id = "committed", event.id
    # 与经历同事务完成整个输入批次；提交后进程中断也不能遗留永远 batched 的机会。
    for member_id in row.work.get("batch_ids", []):
        member = session.get(LifeOpportunityRow, member_id)
        if member and member.branch_id == row.branch_id and member.status == "batched":
            member.status, member.event_id = "committed", event.id
    session.flush()
    return event


def schedule_followup(session, row, at, event_id):
    if at is None:
        return
    key = f"life-followup:{row.id}"
    if not session.scalar(
        select(RuntimeWakeupRow.id).where(
            RuntimeWakeupRow.branch_id == row.branch_id, RuntimeWakeupRow.idempotency_key == key
        )
    ):
        session.add(
            RuntimeWakeupRow(
                branch_id=row.branch_id,
                wake_at=utc(at),
                reason=event_id,
                trigger_type="life_followup",
                idempotency_key=key,
            )
        )
