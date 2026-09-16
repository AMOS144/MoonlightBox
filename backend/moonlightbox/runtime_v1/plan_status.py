"""计划准备状态独立于 Job 和已提交日程；修订失败不能抹掉仍有效的旧计划。"""

from datetime import UTC, datetime

from sqlalchemy import select

from .db_models import RuntimeDayPlanRow


def preparation(plan):
    metadata = (plan.generation_metadata or {}) if plan is not None else {}
    value = metadata.get("preparation")
    if isinstance(value, dict):
        return dict(value)
    legacy = metadata.get("status", "pending")
    return {
        "status": "ready"
        if legacy == "agent"
        else "retryable_failed"
        if legacy == "unavailable"
        else "pending",
        "attempt": 0,
        "retry_at": None,
        "error_code": None,
    }


def set_preparation(
    session,
    branch_id,
    target_date,
    status,
    *,
    error_code=None,
    job_id=None,
    owner_key=None,
    now=None,
):
    """短事务调用者控制提交；这里只维护准备元数据，不改变正式 blocks/version。"""
    now = now or datetime.now(UTC)
    row = session.scalar(
        select(RuntimeDayPlanRow).where(
            RuntimeDayPlanRow.branch_id == branch_id,
            RuntimeDayPlanRow.plan_date == target_date.isoformat(),
        )
    )
    if row is None:
        row = RuntimeDayPlanRow(
            branch_id=branch_id,
            plan_date=target_date.isoformat(),
            blocks=[],
            generation_metadata={"status": "pending"},
            version=0,
        )
        session.add(row)
    current = preparation(row)
    continuing = (
        current["status"] == "running"
        and current.get("owner_key") == owner_key
        and owner_key is not None
    )
    attempt = (0 if current["status"] == "ready" else current.get("attempt", 0)) + (
        1 if status == "running" and not continuing else 0
    )
    if status == "retryable_failed":
        from types import SimpleNamespace

        from moonlightbox.agent_runtime.resilience import classify_failure

        # 这是准备状态的展示映射；次数和是否再次运行由 Job 决定。
        if not classify_failure(SimpleNamespace(code=error_code)).retryable and error_code not in {
            "worker_interrupted",
            "branch_busy",
        }:
            status = "blocked"
    state = {
        **current,
        "status": status,
        "attempt": attempt,
        "error_code": error_code,
        "updated_at": now.isoformat(),
        "job_id": job_id or current.get("job_id"),
        "owner_key": owner_key or current.get("owner_key"),
        # 这里只呈现状态；Job 记录失败后回填唯一的退避截止时间。
        "retry_at": None,
    }
    row.generation_metadata = {**(row.generation_metadata or {}), "preparation": state}
    session.flush()
    return row


def can_schedule(plan, now):
    state = preparation(plan)
    if state["status"] not in {"pending", "retryable_failed"}:
        return False
    retry_at = state.get("retry_at")
    return not retry_at or datetime.fromisoformat(retry_at) <= now
