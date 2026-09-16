"""虚拟时间节拍、批次与恢复；只写队列，不在 Scheduler 中调用模型。"""

import logging
from datetime import UTC, timedelta
from uuid import uuid4

from sqlalchemy import select, update

from moonlightbox.config import Settings
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService

from ..branch_models import Branch
from ..clock import create_clock
from ..collaboration.plans import local_time
from ..db_models import RuntimeClockRow, RuntimeDayPlanRow, RuntimeWakeupRow
from .models import LifeOpportunityRow, LifeScheduleCursor
from .policy import load_policy

logger = logging.getLogger(__name__)
# 检测 Scheduler 心跳中断，不是事件有效期或生活冷却。
HEARTBEAT_GAP_SECONDS = 60


def utc(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def branch_now(row, wall_now):
    clock = create_clock(
        row.branch_id, row.virtual_anchor, wall_anchor=row.wall_anchor, timezone=row.timezone
    ).model_copy(update={"time_scale": row.time_scale, "status": row.status})
    return local_time(clock.now(wall_now), row.timezone)


def current_plan(session, branch_id, now):
    plan = session.scalar(
        select(RuntimeDayPlanRow).where(
            RuntimeDayPlanRow.branch_id == branch_id,
            RuntimeDayPlanRow.plan_date == now.date().isoformat(),
        )
    )
    if not plan or (plan.generation_metadata or {}).get("status") != "agent":
        return None, None
    block = next((b for b in plan.blocks if b["start"] <= now.strftime("%H:%M") < b["end"]), None)
    return plan, block


def midnight_time(now, hhmm):
    hour, minute = map(int, hhmm.split(":"))
    return now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        hours=hour, minutes=minute
    )


def scan_life_opportunities(session, wall_now, limit=100):
    policy = load_policy(Settings().runtime_life_policy_path or None)
    ids = list(
        session.scalars(
            select(Branch.id).where(Branch.lifecycle_status == "active").order_by(Branch.id)
        )
    )
    count = 0
    for branch_id in ids:
        # 不获取跨模型请求的文件锁，否则 Director 忙时概率时钟也会停住。
        session.execute(
            update(Branch)
            .where(Branch.id == branch_id)
            .values(runtime_input_revision=Branch.runtime_input_revision)
        )
        count += scan_branch(
            session, session.get(Branch, branch_id), wall_now, policy=policy, dispatch=count < limit
        )
        session.commit()
    return count


def _new(session, branch_id, key, at, kind, policy, *, brief=None, work=None):
    existing = session.scalar(
        select(LifeOpportunityRow).where(
            LifeOpportunityRow.branch_id == branch_id, LifeOpportunityRow.check_key == key
        )
    )
    if existing:
        return existing
    row = LifeOpportunityRow(
        branch_id=branch_id,
        check_key=key,
        kind=kind,
        status="queued",
        cursor_at=utc(at),
        policy=policy.model_dump(),
        brief=brief or {},
        work=work or {},
    )
    session.add(row)
    session.flush()
    return row


def scan_branch(session, branch, wall_now, *, policy=None, dispatch=True):
    policy = policy or load_policy(Settings().runtime_life_policy_path or None)
    clock = session.get(RuntimeClockRow, branch.id)
    if not clock:
        return 0
    now = branch_now(clock, wall_now)
    cursor = session.get(LifeScheduleCursor, branch.id)
    interval = timedelta(minutes=policy.check_interval_minutes)
    if cursor is None:
        cursor = LifeScheduleCursor(
            branch_id=branch.id,
            sequence=0,
            last_check_at=utc(now),
            next_check_at=utc(now) + interval,
            last_seen_wall=utc(wall_now),
            last_seen_virtual=utc(now),
            interval_minutes=policy.check_interval_minutes,
            clock_status=clock.status,
        )
        session.add(cursor)
        session.flush()
    if clock.status != "running":
        cursor.last_seen_wall, cursor.last_seen_virtual = utc(wall_now), utc(now)
        cursor.clock_status = clock.status
        return 0
    if cursor.clock_status != "running":
        # 暂停期间虚拟时间未流逝；恢复后重建未来节拍，不补造暂停期间的机会。
        cursor.next_check_at = utc(now) + interval
    elif (
        utc(wall_now) - utc(cursor.last_seen_wall)
    ).total_seconds() > HEARTBEAT_GAP_SECONDS and utc(now) >= utc(cursor.next_check_at):
        _new(
            session,
            branch.id,
            f"recovery:{cursor.sequence}:{utc(cursor.last_seen_virtual).isoformat()}",
            now,
            "recovery",
            policy,
            work={
                "recovery": {
                    "from": utc(cursor.last_seen_virtual).isoformat(),
                    "to": utc(now).isoformat(),
                    "recorded_at": utc(wall_now).isoformat(),
                }
            },
        )
        cursor.last_check_at, cursor.next_check_at = utc(now), utc(now) + interval
    if cursor.interval_minutes != policy.check_interval_minutes:
        cursor.next_check_at = max(utc(cursor.last_check_at) + interval, utc(now) + interval)
        cursor.interval_minutes = policy.check_interval_minutes
    while utc(cursor.next_check_at) <= utc(now):
        at = utc(cursor.next_check_at)
        cursor.sequence += 1
        seed = str(uuid4())
        probability, brief = policy.sample(seed, local_time(at, clock.timezone))
        diagnostic = {
            "sequence": cursor.sequence,
            "at": at.isoformat(),
            "seed": seed,
            "probability": probability,
            "hit": brief is not None,
            "policy_hash": policy.fingerprint,
            "algorithm": "smoothstep-beta-v2",
        }
        cursor.last_check = diagnostic
        if brief is not None:
            row = _new(
                session,
                branch.id,
                f"check:{cursor.sequence}",
                at,
                "opportunity",
                policy,
                brief=brief.model_dump(),
                work={"sampling": diagnostic},
            )
            row.seed = seed
        logger.debug("life_check branch=%s result=%s", branch.id, diagnostic)
        cursor.last_check_at, cursor.next_check_at = at, at + interval
    cursor.last_seen_wall, cursor.last_seen_virtual = utc(wall_now), utc(now)
    cursor.clock_status = clock.status
    # 后续推进不重新抽概率，也不会覆盖正常检查节拍。
    for wake in session.scalars(
        select(RuntimeWakeupRow).where(
            RuntimeWakeupRow.branch_id == branch.id,
            RuntimeWakeupRow.trigger_type == "life_followup",
            RuntimeWakeupRow.status == "scheduled",
            RuntimeWakeupRow.wake_at <= utc(now),
        )
    ):
        _new(
            session,
            branch.id,
            f"followup:{wake.id}",
            wake.wake_at,
            "followup",
            policy,
            work={"parent_event_id": wake.reason},
        )
        wake.status = "completed"
    if not dispatch:
        return 0
    plan, block = current_plan(session, branch.id, now)
    if not block:
        return 0  # 机会保留，但不在没有准备好资产时强行运行模型。
    active = session.scalar(
        select(LifeOpportunityRow).where(
            LifeOpportunityRow.branch_id == branch.id, LifeOpportunityRow.status == "active"
        )
    )
    if active:
        job = session.get(Job, active.job_id) if active.job_id else None
        if job and job.status in {"queued", "running", "cancelling", "failed", "interrupted"}:
            if job.status in {"failed", "interrupted"}:
                from moonlightbox.jobs.runtime_recovery import recovery

                if recovery(job).get("status") == "terminal":
                    active.status, active.error_code = "blocked", job.error_code
            return 0  # 原 Job 统一恢复，绝不换身份绕预算。
        if job and job.status == "cancelled":
            active.status = "cancelled"
            for key in active.work.get("batch_ids", []):
                member = session.get(LifeOpportunityRow, key)
                if member and member.status == "batched":
                    member.status = "cancelled"
            return 0
        if active.work.get("awaiting_director_event_id"):
            return 0
        leader = active
    else:
        # 与日常规划使用同一执行权；忙时只累计，不再生成并行 Planner。
        jobs = session.scalars(
            select(Job).where(
                Job.kind == "runtime-v1-cycle",
                Job.payload["branch_id"].as_string() == branch.id,
                Job.status.in_(["queued", "running", "cancelling"]),
            )
        )
        if any(
            j.payload.get("life_opportunity_id")
            or j.payload.get("planner_task")
            or j.payload.get("plan_date")
            or j.payload.get("bootstrap")
            for j in jobs
        ):
            return 0
        rows = list(
            session.scalars(
                select(LifeOpportunityRow)
                .where(
                    LifeOpportunityRow.branch_id == branch.id, LifeOpportunityRow.status == "queued"
                )
                .order_by(LifeOpportunityRow.cursor_at, LifeOpportunityRow.id)
            )
        )
        if not rows:
            return 0
        leader = rows[0]
        leader.status = "active"
        leader.work = {**leader.work, "batch_ids": [r.id for r in rows]}
        for row in rows[1:]:
            row.status = "batched"
            row.work = {**row.work, "leader_id": leader.id}
    job = JobService(session).enqueue_unique(
        "runtime-v1-cycle",
        {"project_id": branch.project_id, "branch_id": branch.id, "life_opportunity_id": leader.id},
        dedupe_key=f"life:{leader.id}:step:{leader.step}",
        commit=False,
    )
    leader.job_id = job.id
    return 1
