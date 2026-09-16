"""将 Runtime EventQueue 接入现有通用 Worker。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.world.models import PersonWorldProfile, WorldGraphVersion

from .branch_models import Branch
from .clock import create_clock
from .cloud_models import RuntimeCloudClient, create_cloud_models
from .collaboration.plans import local_time
from .db_models import RuntimeClockRow, RuntimeDayPlanRow, RuntimeSnapshotRow, RuntimeWakeupRow
from .plan_status import can_schedule, preparation, set_preparation
from .service import RuntimeModelExecutionError, RuntimeService
from .work_calendar import ChinaWorkCalendarService

RUNTIME_CYCLE_JOB_KIND = "runtime-v1-cycle"


def enqueue_input_tick(session, branch_id, first_input_id):
    """短事务先锁定已有分支行，再检查/投递；不能仅靠先查后写防并发。"""
    locked = session.execute(
        update(Branch)
        .where(Branch.id == branch_id, Branch.lifecycle_status == "active")
        .values(runtime_input_revision=Branch.runtime_input_revision)
    )
    if locked.rowcount != 1:
        return False
    existing = session.scalars(
        select(Job).where(
            Job.kind == RUNTIME_CYCLE_JOB_KIND,
            Job.status.in_(["queued", "running", "cancelling", "failed", "interrupted"]),
        )
    )
    if any(
        job.payload.get("branch_id") == branch_id
        and (job.checkpoint or {}).get("runtime_recovery", {}).get("status") != "terminal"
        and not job.payload.get("conversation_maintenance")
        and not job.payload.get("plan_date")
        and not job.payload.get("life_opportunity_id")
        and not job.payload.get("planner_task")
        for job in existing
    ):
        return False
    key = f"{RUNTIME_CYCLE_JOB_KIND}:input:{branch_id}:{first_input_id}"
    if session.scalar(select(Job.id).where(Job.dedupe_key == key)):
        return False
    JobService(session).enqueue_unique(
        RUNTIME_CYCLE_JOB_KIND,
        {"project_id": session.get(Branch, branch_id).project_id,
         "branch_id": branch_id, "input_tick": True},
        dedupe_key=key,
        commit=False,
    )
    return True


def enqueue_due_runtime_cycles(
    database: Database,
    *,
    now: datetime | None = None,
    limit: int = 100,
) -> int:
    """按每条分支的 VirtualClock 扫描到期 Wakeup，并启动已就绪分支。

    ``wake_at`` 是虚拟时间，不能直接拿墙上时间比较；暂停中的分支因而不会被
    重复无效唤醒，未来改变 time_scale 时也无需修改调度器语义。
    """

    wall_now = now if now is not None else datetime.now(UTC)
    count = 0
    with Session(database.engine) as session:
        from moonlightbox.jobs.runtime_recovery import scan_recovery

        scan_recovery(session, wall_now)
        session.commit()
        rows = list(
            session.scalars(
                select(RuntimeWakeupRow)
                .where(
                    RuntimeWakeupRow.status == "scheduled",
                    RuntimeWakeupRow.trigger_type != "life_followup",
                )
                .order_by(RuntimeWakeupRow.wake_at, RuntimeWakeupRow.id)
            )
        )
        for row in rows:
            if count >= limit:
                break
            clock_row = session.get(RuntimeClockRow, row.branch_id)
            if clock_row is None:
                continue
            clock = create_clock(
                row.branch_id,
                clock_row.virtual_anchor,
                wall_anchor=clock_row.wall_anchor,
                timezone=clock_row.timezone,
            ).model_copy(update={"time_scale": clock_row.time_scale, "status": clock_row.status})
            virtual_now = clock.now(wall_now)
            if clock.status != "running":
                continue
            if _as_utc(row.wake_at) > _as_utc(virtual_now):
                continue
            count += int(enqueue_input_tick(session, row.branch_id, row.id))
        # 分支与人物世界都准备好后自动建立 Snapshot/DayPlan/首个 Wakeup。这样不必
        # 依赖前端额外调用 /runtime/bootstrap，分支可以真正持续运行。
        candidates = session.scalars(
            select(Branch)
            .outerjoin(RuntimeSnapshotRow, RuntimeSnapshotRow.branch_id == Branch.id)
            .where(
                Branch.lifecycle_status.in_(["active", "preparing"]),
                RuntimeSnapshotRow.id.is_(None),
            )
            .limit(limit)
        )
        for branch in candidates:
            if count >= limit:
                break
            profile_id = session.scalar(
                select(PersonWorldProfile.id)
                .join(
                    WorldGraphVersion,
                    PersonWorldProfile.graph_version_id == WorldGraphVersion.id,
                )
                .where(
                    PersonWorldProfile.project_id == branch.project_id,
                    PersonWorldProfile.node_boundary_hash.is_(None),
                    WorldGraphVersion.status == "ready",
                )
                .limit(1)
            )
            if profile_id is None:
                continue
            JobService(session).enqueue_unique(
                RUNTIME_CYCLE_JOB_KIND,
                {"project_id": branch.project_id, "branch_id": branch.id, "bootstrap": True},
                dedupe_key=f"{RUNTIME_CYCLE_JOB_KIND}:bootstrap:{branch.id}",
                commit=False,
            )
            count += 1
        # 已经有 Snapshot 的旧分支同样需要迁移到 DayPlanAgent。旧计划行没有
        # generation_metadata 时视为 pending；这里入队一个纯规划 Cycle，不会生成
        # 用户可见消息。否则代码升级后只能等用户下一次发言才会重新规划当天。
        active_branches = list(
            session.scalars(
                select(Branch)
                .join(RuntimeSnapshotRow, RuntimeSnapshotRow.branch_id == Branch.id)
                .join(RuntimeClockRow, RuntimeClockRow.branch_id == Branch.id)
                .where(Branch.lifecycle_status == "active")
                .limit(limit)
            )
        )
        for branch in active_branches:
            if count >= limit:
                break
            clock_row = session.get(RuntimeClockRow, branch.id)
            if clock_row is None:
                continue
            clock = create_clock(
                branch.id,
                clock_row.virtual_anchor,
                wall_anchor=clock_row.wall_anchor,
                timezone=clock_row.timezone,
            ).model_copy(update={"time_scale": clock_row.time_scale, "status": clock_row.status})
            if clock.status != "running":
                continue
            plan = session.scalar(
                select(RuntimeDayPlanRow).where(
                    RuntimeDayPlanRow.branch_id == branch.id,
                    RuntimeDayPlanRow.plan_date
                    == local_time(clock.now(wall_now), clock.timezone).date().isoformat(),
                )
            )
            target_date = local_time(clock.now(wall_now), clock.timezone).date().isoformat()
            metadata = plan.generation_metadata if plan is not None else None
            if (
                plan is not None
                and isinstance(metadata, dict)
                and preparation(plan)["status"] == "ready"
            ):
                # 今天已有正式计划时提前维护明天，不经过 Director 审批。
                target_date = (
                    local_time(clock.now(wall_now), clock.timezone).date() + timedelta(days=1)
                ).isoformat()
                plan = session.scalar(
                    select(RuntimeDayPlanRow).where(
                        RuntimeDayPlanRow.branch_id == branch.id,
                        RuntimeDayPlanRow.plan_date == target_date,
                    )
                )
                if plan is not None and preparation(plan)["status"] == "ready":
                    continue
            # 准备失败不是“空闲”；重试退避、人工阻塞与正式版本相互独立。
            if plan is not None and preparation(plan)["status"] == "running":
                owner = (
                    session.get(Job, preparation(plan).get("job_id"))
                    if preparation(plan).get("job_id")
                    else None
                )
                if owner is not None and owner.status not in {"running", "queued"}:
                    set_preparation(
                        session,
                        branch.id,
                        datetime.fromisoformat(target_date).date(),
                        "retryable_failed",
                        error_code=owner.error_code or "worker_interrupted",
                        now=wall_now,
                    )
            if not can_schedule(plan, wall_now):
                continue
            owner_id = preparation(plan).get("job_id")
            owner = session.get(Job, owner_id) if owner_id else None
            if (
                owner is not None
                and owner.status in {"queued", "running", "failed", "interrupted"}
                and owner.payload.get("input_revision", branch.runtime_input_revision)
                == branch.runtime_input_revision
            ):
                # 同一准备任务继续使用原 Job，不能因占位行/版本变化并行排出第二份调查。
                continue
            if plan is None:
                plan = set_preparation(
                    session, branch.id, datetime.fromisoformat(target_date).date(), "pending"
                )
            JobService(session).enqueue_unique(
                RUNTIME_CYCLE_JOB_KIND,
                {
                    "project_id": branch.project_id,
                    "branch_id": branch.id,
                    "day_plan_id": plan.id if plan is not None else None,
                    "plan_date": target_date,
                    "input_revision": branch.runtime_input_revision,
                },
                dedupe_key=(
                    f"{RUNTIME_CYCLE_JOB_KIND}:plan:{plan.id}:{plan.version}:input:{branch.runtime_input_revision}"
                    if plan is not None
                    else f"{RUNTIME_CYCLE_JOB_KIND}:plan:{branch.id}:{target_date}"
                ),
                commit=False,
            )
            count += 1
        session.commit()
        # 已提交事件的 outbox 即使关闭新机会也继续投递，不丢已发生经历。
        from .db_models import RuntimeEventRow

        notifications = session.scalars(
            select(RuntimeEventRow)
            .join(Branch, Branch.id == RuntimeEventRow.branch_id)
            .where(RuntimeEventRow.status == "queued", Branch.lifecycle_status == "active")
            .order_by(RuntimeEventRow.received_at, RuntimeEventRow.id)
        )
        seen_branches = set()
        for notice in notifications:
            if count >= limit:
                break
            recipient = "day_planner" if notice.event_type == "planner_inbox" else "director"
            recipient_key = (notice.branch_id, recipient)
            if recipient_key in seen_branches:
                continue
            clock = RuntimeService(session).get_clock(
                session.get(Branch, notice.branch_id).project_id, notice.branch_id
            )
            if clock.status == "running" and _as_utc(notice.occurred_at) <= clock.now(wall_now):
                seen_branches.add(recipient_key)
                if recipient == "day_planner":
                    from .collaboration.transport import schedule_planner_inbox

                    count += int(
                        schedule_planner_inbox(session, notice.branch_id, clock.now(wall_now))
                    )
                else:
                    count += int(enqueue_input_tick(session, notice.branch_id, notice.id))
        session.commit()
        from .life_events.scheduling import scan_life_opportunities

        count += scan_life_opportunities(session, wall_now, limit)
    return count


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def create_runtime_cycle_handler(
    *,
    settings: Settings,
) -> Callable[[JobService, Job], None]:
    """Worker 领取独立工作；Director 的实际表达由 speaking skill 完成。"""

    from .life_events.policy import load_policy

    # 生活推进是必需运行能力；策略错误在启动时暴露，不能静默停掉后台推进。
    load_policy(settings.runtime_life_policy_path)
    cloud_client = RuntimeCloudClient.from_settings(settings)
    work_calendar = ChinaWorkCalendarService.from_settings(settings)

    def handle(service: JobService, job: Job) -> None:
        project_id = job.payload.get("project_id")
        branch_id = job.payload.get("branch_id")
        if not isinstance(branch_id, str):
            raise ValueError("Runtime job 缺少 project_id/branch_id")
        if not isinstance(project_id, str):
            branch = service.session.get(Branch, branch_id)
            project_id = branch.project_id if branch is not None else None
        if not isinstance(project_id, str):
            raise ValueError("Runtime job 缺少 project_id/branch_id")
        director, actor = create_cloud_models(cloud_client)
        if job.payload.get("planner_task"):
            from .collaboration.planner_tasks import run_planner_task

            try:
                run_planner_task(
                    RuntimeService(
                        service.session, director_model=director, work_calendar=work_calendar
                    ),
                    job,
                )
            except RuntimeModelExecutionError as error:
                raise JobHandlerError(error.code, error.safe_message) from error
            return
        if job.payload.get("conversation_maintenance"):
            from .collaboration.persistence import branch_writer
            from .conversation_index import index_branch
            from .conversation_maintenance import summarize

            # 独立派生任务；禁止调度器线程里加载模型。维护之间也防止重复覆盖区间。
            with branch_writer(
                service.session, f"{branch_id}:conversation-maintenance"
            ) as acquired:
                if not acquired:
                    raise JobHandlerError("branch_busy", "历史维护等待分支写锁")
                more = summarize(service.session, branch_id, project_id, director)
                if more:
                    from .conversation_maintenance import enqueue_maintenance

                    enqueue_maintenance(service.session, branch_id, f"continue:{job.id}")
                    service.session.commit()
                index_branch(service.session, branch_id)
            return
        if job.payload.get("life_opportunity_id"):
            from .life_events.service import process_life_step

            process_life_step(
                RuntimeService(
                    service.session,
                    director_model=director,
                    actor_model=actor,
                    work_calendar=work_calendar,
                ),
                job,
            )
            return
        try:
            RuntimeService(
                service.session,
                director_model=director,
                work_calendar=work_calendar,
            ).process_next(
                project_id=project_id,
                branch_id=branch_id,
                job_id=job.id,
                prepare_branch=bool(job.payload.get("bootstrap")),
                plan_date=job.payload.get("plan_date"),
            )
        except RuntimeModelExecutionError as error:
            target_date = job.payload.get("plan_date")
            if target_date:
                # 在进入 Planner 前失败（如分支写锁忙）也必须有退避，不能每秒重排。
                plan = service.session.scalar(
                    select(RuntimeDayPlanRow).where(
                        RuntimeDayPlanRow.branch_id == branch_id,
                        RuntimeDayPlanRow.plan_date == target_date,
                    )
                )
                if preparation(plan)["status"] == "pending":
                    day = datetime.fromisoformat(target_date).date()
                    set_preparation(
                        service.session, branch_id, day, "running", job_id=job.id, owner_key=job.id
                    )
                    set_preparation(
                        service.session, branch_id, day, "retryable_failed", error_code=error.code
                    )
                    service.session.commit()
            # Worker 只保存安全码与通用说明；具体失败上下文在 Cycle Trace 中。
            raise JobHandlerError(error.code, error.safe_message) from error

    return handle
