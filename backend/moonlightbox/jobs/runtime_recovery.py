"""按任务类型声明的统一恢复策略；检查点内容与累计 Agent 用量始终保留。

failed + recovery.status=waiting 表示退避，terminal 表示停止自动恢复。
不增加数据库状态枚举，现有任务 API 仍能展示 failed 与具体错误。
"""

from datetime import UTC, datetime, timedelta
from random import uniform
from types import SimpleNamespace

from sqlalchemy import select, update

from moonlightbox.agent_runtime.resilience import classify_failure

from .models import Job

RUNTIME_KIND = "runtime-v1-cycle"
MAX_RETRIES = 3
WORLD_KIND = "lightrag_world_build_v1"
RECOVERY_POLICIES = {RUNTIME_KIND: MAX_RETRIES, WORLD_KIND: 5}


def recovery_keys(job):
    # Runtime 历史检查点仍可读；新任务使用通用命名，不迁移已有数据。
    return (
        ("runtime_recovery", "runtime_retry_attempt")
        if job.kind == RUNTIME_KIND
        else ("job_recovery", "job_retry_attempt")
    )


def utc(value):
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def recovery(job):
    return (job.checkpoint or {}).get(recovery_keys(job)[0], {})


def settle_failure(session, job, now):
    """调用者已经 CAS 获得终态写权；迁移旧 failed 行也用同一判定。"""
    if recovery(job):
        return
    state_key, count_key = recovery_keys(job)
    count = int((job.checkpoint or {}).get(count_key, 0))
    max_retries = RECOVERY_POLICIES[job.kind]
    opportunity = None
    if job.payload.get("life_opportunity_id"):
        from moonlightbox.runtime_v1.life_events.models import LifeOpportunityRow

        opportunity = session.get(LifeOpportunityRow, job.payload["life_opportunity_id"])
    code = job.error_code or "worker_interrupted"
    failure = classify_failure(SimpleNamespace(code=code))
    transient = failure.retryable or code in {"worker_interrupted", "branch_busy"}
    waiting_for_sidecar = job.kind == WORLD_KIND and code == "lightrag_index_running"
    transient = transient or waiting_for_sidecar
    terminal = not transient or (count >= max_retries and not waiting_for_sidecar)
    delay = 30 * 2 ** min(count, max_retries)
    if job.kind == WORLD_KIND:
        delay = 30 if waiting_for_sidecar else uniform(delay / 2, delay)
    job.checkpoint = {
        **(job.checkpoint or {}),
        state_key: {
            "status": "terminal" if terminal else "waiting",
            "retries": count,
            "max_retries": max_retries,
            "reason": "retry_exhausted" if terminal and transient and count >= max_retries else code,
            "error_code": code,
            "retry_at": None if terminal else (utc(now) + timedelta(seconds=delay)).isoformat(),
        },
    }
    if job.kind == WORLD_KIND:
        return
    if not terminal and job.payload.get("plan_date"):
        from moonlightbox.runtime_v1.db_models import RuntimeDayPlanRow
        from moonlightbox.runtime_v1.plan_status import preparation

        plan = session.scalar(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == job.payload["branch_id"],
                RuntimeDayPlanRow.plan_date == job.payload["plan_date"],
            )
        )
        if plan:
            plan.generation_metadata = {
                **(plan.generation_metadata or {}),
                "preparation": {
                    **preparation(plan),
                    "status": "retryable_failed",
                    "retry_at": recovery(job)["retry_at"],
                    "error_code": code,
                    "job_id": job.id,
                },
            }
    if terminal:
        release_failed_inputs(session, job)
        if opportunity:
            opportunity.error_code = code
            if opportunity.status == "active":
                opportunity.status = "blocked"
                for key in opportunity.work.get("batch_ids", []):
                    member = session.get(type(opportunity), key)
                    if member and member.status == "batched":
                        member.status, member.error_code = "blocked", code
                from moonlightbox.runtime_v1.db_models import RuntimeEventRow

                for key in opportunity.work.get("reply_event_ids", []):
                    reply = session.get(RuntimeEventRow, key)
                    if (
                        reply
                        and reply.branch_id == opportunity.branch_id
                        and reply.status == "assigned"
                    ):
                        reply.status = "failed"


def release_failed_inputs(session, job):
    """只关闭此任务已经接管的输入；后到消息不丢弃、不标成已回复。

    原消息、原检查点保留。failed 事件不会再被扫描成一个全新任务绕过预算。
    """
    from moonlightbox.runtime_v1.branch_models import BranchMessage
    from moonlightbox.runtime_v1.db_models import RuntimeEventRow, RuntimeWakeupRow

    cycle = (job.checkpoint or {}).get("runtime_cycle", {})
    ids = set(cycle.get("trigger_ids", [])) | set(cycle.get("input_event_ids", []))
    # 首次还没建立 cycle 就失败时，幂等键仍记录了最初唤起输入。
    if not ids and job.payload.get("input_tick") and job.dedupe_key:
        ids.add(job.dedupe_key.rsplit(":", 1)[-1])
    if ids:
        wakeup_ids = set(ids)
        events = session.scalars(
            select(RuntimeEventRow).where(
                RuntimeEventRow.id.in_(ids),
                RuntimeEventRow.branch_id == job.payload.get("branch_id"),
                RuntimeEventRow.status.in_(["queued", "claimed"]),
            )
        )
        for event in events:
            if event.payload.get("wakeup_id"):
                wakeup_ids.add(event.payload["wakeup_id"])
            event.status, event.claimed_at = "failed", None
            message_id = event.payload.get("branch_message_id")
            message = session.get(BranchMessage, message_id) if message_id else None
            if message:
                message.generation_metadata = {
                    **(message.generation_metadata or {}),
                    "input_status": "failed",
                    "error_code": job.error_code,
                    "failed_job_id": job.id,
                }
        session.execute(
            update(RuntimeWakeupRow)
            .where(
                RuntimeWakeupRow.id.in_(wakeup_ids),
                RuntimeWakeupRow.branch_id == job.payload.get("branch_id"),
                RuntimeWakeupRow.status.in_(["scheduled", "executing"]),
            )
            .values(status="failed", executing_at=None)
        )
    if job.payload.get("plan_date"):
        from moonlightbox.runtime_v1.plan_status import set_preparation

        set_preparation(
            session,
            job.payload["branch_id"],
            datetime.fromisoformat(job.payload["plan_date"]).date(),
            "blocked",
            error_code=job.error_code,
            job_id=job.id,
        )


def resume_job(session, job, now, *, manual=False):
    """原 Job 原检查点恢复。显式恢复也不清零总次数；manual 仅跳过退避等待。"""
    # LightRAG 自动恢复耗尽后会将任务收口为 cancelled；这不是用户主动取消，
    # 应允许用户修复配置/供应商问题后再次手动重试。
    resumable_cancelled_world = manual and job.kind == WORLD_KIND and job.status == "cancelled"
    if job.status not in {"failed", "interrupted"} and not resumable_cancelled_world:
        return False
    locked = session.execute(
        update(Job)
        .where(
            Job.id == job.id,
            Job.status == job.status,
            Job.updated_at == job.updated_at,
        )
        .values(updated_at=job.updated_at)
    )
    if locked.rowcount != 1:
        return False
    session.refresh(job)
    settle_failure(session, job, now)
    state = recovery(job)
    # terminal 只阻止自动恢复；用户修复配额/密钥等外部原因后显式重试应当放行，
    # 累计次数保留，再次失败仍按既有策略收口。
    if state["status"] == "terminal" and not manual:
        return False
    if not manual and datetime.fromisoformat(state["retry_at"]) > utc(now):
        return False
    checkpoint = dict(job.checkpoint or {})
    state_key, count_key = recovery_keys(job)
    checkpoint.pop(state_key, None)
    # 确认 Sidecar 仍在执行只是轮询，不消耗故障恢复次数。
    checkpoint[count_key] = state["retries"] + (state.get("error_code") != "lightrag_index_running")
    session.flush()
    result = session.execute(
        update(Job)
        .where(
            Job.id == job.id,
            Job.status == job.status,
            Job.updated_at == job.updated_at,
        )
        .values(
            status="queued",
            checkpoint=checkpoint,
            error_code=None,
            error_message=None,
            worker_token=None,
            lease_expires_at=None,
            updated_at=now,
        )
    )
    return result.rowcount == 1


def scan_recovery(session, now):
    """先结清终态再扫描新输入；bootstrap 仍需用户显式恢复。"""
    resumed = 0
    for job in list(
        session.scalars(
            select(Job).where(
                Job.kind.in_(RECOVERY_POLICIES),
                Job.status.in_(["failed", "interrupted"]),
                Job.checkpoint["runtime_recovery"]["status"]
                .as_string()
                .is_distinct_from("terminal"),
            )
        )
    ):
        if recovery(job).get("status") == "terminal":
            continue
        # 对旧失败行的迁移也先 CAS 取写权，防止与人工恢复/其他扫描器交错覆盖。
        locked = session.execute(
            update(Job)
            .where(
                Job.id == job.id,
                Job.status == job.status,
                Job.updated_at == job.updated_at,
            )
            .values(updated_at=job.updated_at)
        )
        if locked.rowcount != 1:
            continue
        session.refresh(job)
        settle_failure(session, job, job.updated_at)
        if job.kind == WORLD_KIND:
            resumed += int(resume_job(session, job, now))
            continue
        if job.payload.get("bootstrap"):
            continue
        from moonlightbox.runtime_v1.db_models import RuntimeClockRow

        clock = session.get(RuntimeClockRow, job.payload.get("branch_id"))
        if clock and clock.status == "running":
            resumed += int(resume_job(session, job, now))
    session.flush()
    return resumed


def recover_leases(session, now, *, job_id=None, token=None):
    """进程退出也要经过整任务预算，不能用直接 queued 绕过策略。"""
    statement = select(Job).where(
        Job.kind.in_(RECOVERY_POLICIES), Job.status == "running", Job.lease_expires_at <= now
    )
    if job_id:
        statement = statement.where(Job.id == job_id, Job.worker_token == token)
    count = 0
    for job in list(session.scalars(statement)):
        result = session.execute(
            update(Job)
            .where(
                Job.id == job.id,
                Job.status == "running",
                Job.worker_token == job.worker_token,
                Job.lease_expires_at <= now,
            )
            .values(
                status="interrupted",
                worker_token=None,
                lease_expires_at=None,
                error_code="worker_interrupted",
                error_message="Worker 租约已过期，等待恢复",
                updated_at=now,
            )
            .execution_options(synchronize_session="fetch")
        )
        if result.rowcount:
            settle_failure(session, job, now)
            count += 1
    session.flush()
    return count
