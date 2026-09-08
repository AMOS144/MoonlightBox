"""将 Runtime EventQueue 接入现有通用 Worker。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService
from moonlightbox.world.models import PersonWorldProfile, WorldGraphVersion

from .branch_models import Branch
from .clock import create_clock
from .db_models import RuntimeClockRow, RuntimeSnapshotRow, RuntimeWakeupRow
from .inference_client import RuntimeInferenceClient
from .remote_models import create_remote_models
from .service import RuntimeService

RUNTIME_CYCLE_JOB_KIND = "runtime-v1-cycle"


def enqueue_due_runtime_cycles(
    database: Database, *, now: datetime | None = None, limit: int = 100
) -> int:
    """按每条分支的 VirtualClock 扫描到期 Wakeup，并启动已就绪分支。

    ``wake_at`` 是虚拟时间，不能直接拿墙上时间比较；暂停中的分支因而不会被
    重复无效唤醒，未来改变 time_scale 时也无需修改调度器语义。
    """

    wall_now = now if now is not None else datetime.now(UTC)
    count = 0
    with Session(database.engine) as session:
        rows = list(
            session.scalars(
                select(RuntimeWakeupRow)
                .where(RuntimeWakeupRow.status == "scheduled")
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
            if _as_utc(row.wake_at) > _as_utc(virtual_now):
                continue
            JobService(session).enqueue_unique(
                RUNTIME_CYCLE_JOB_KIND,
                {"branch_id": row.branch_id},
                dedupe_key=f"{RUNTIME_CYCLE_JOB_KIND}:wakeup:{row.id}",
                commit=False,
            )
            count += 1
        # 分支与人物世界都准备好后自动建立 Snapshot/DayPlan/首个 Wakeup。这样不必
        # 依赖前端额外调用 /runtime/bootstrap，分支可以真正持续运行。
        candidates = session.scalars(
            select(Branch)
            .outerjoin(RuntimeSnapshotRow, RuntimeSnapshotRow.branch_id == Branch.id)
            .where(
                Branch.lifecycle_status == "active",
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
        session.commit()
    return count


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def create_runtime_cycle_handler(
    *,
    persona_client: RuntimeInferenceClient,
) -> Callable[[JobService, Job], None]:
    """Worker 每次只处理一个分支 Cycle，并经 HTTP 请求 Linux 人格推理服务。"""

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
        director, actor = create_remote_models(service.session, branch_id, client=persona_client)
        RuntimeService(
            service.session,
            director_model=director,
            actor_model=actor,
        ).process_next(project_id=project_id, branch_id=branch_id)

    return handle
