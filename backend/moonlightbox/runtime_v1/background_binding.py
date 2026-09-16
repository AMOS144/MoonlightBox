"""显式更新体验分支的已发布背景绑定，不重做历史切片，不读取待审核草稿。"""

from datetime import UTC, datetime

from sqlalchemy import select, update

from moonlightbox.world.models import PersonWorldProfile, WorldGraphVersion, WorldPublication

from .branch_models import Branch
from .db_models import (
    RuntimeClockRow,
    RuntimeDayPlanRow,
    RuntimeLifeEventRow,
    RuntimeMemoryRow,
    RuntimeSnapshotRow,
)


def refresh_published_background(session, project_id, branch_id):
    from .executor import (
        _latest_profile_cutoff,
        _profile_payload,
        _runtime_routine_profile,
        _seed_world_memory,
    )
    from .memory import MemoryService

    branch = session.get(Branch, branch_id)
    snapshot = session.scalar(
        select(RuntimeSnapshotRow).where(RuntimeSnapshotRow.branch_id == branch_id)
    )
    if branch is None or branch.project_id != project_id or snapshot is None:
        raise LookupError("分支或已绑定背景不存在")
    if snapshot.snapshot_mode != "latest_profile":
        raise ValueError("历史冻结分支不能隐式升级；仅支持 latest_profile 体验约定")
    publication = session.scalar(
        select(WorldPublication)
        .where(
            WorldPublication.project_id == project_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash.is_(None),
        )
        .order_by(WorldPublication.published_at.desc())
    )
    if publication is None:
        raise ValueError("没有当前已发布画像，不能用最新草稿替代")
    profile = session.get(PersonWorldProfile, publication.profile_id)
    graph = session.get(WorldGraphVersion, publication.graph_version_id)
    if (
        profile is None
        or profile.profile_schema_version != "v3"
        or graph is None
        or graph.status != "ready"
        or profile.graph_version_id != graph.id
        or profile.project_id != project_id
    ):
        raise ValueError("Publication 必须绑定同项目的完整 ready 图与 v3 画像")
    binding = {
        "publication_id": publication.id,
        "profile_id": profile.id,
        "graph_version_id": graph.id,
        "profile_schema_version": "v3",
    }
    if snapshot.profile.get("_runtime_binding") == binding:
        return binding
    # 修改前完整保留原背景及引用，允许人工审计/恢复，不删除历史计划或聊天。
    previous = {
        "profile_id": snapshot.profile_id,
        "graph_version_id": snapshot.graph_version_id,
        "profile": snapshot.profile,
        "routine_profile": snapshot.routine_profile,
        "source_message_ids": snapshot.source_message_ids,
        "cutoff_at": snapshot.cutoff_at.isoformat(),
        "compiler_version": snapshot.compiler_version,
    }
    session.add(
        RuntimeLifeEventRow(
            branch_id=branch_id,
            event_type="background_rebound",
            occurred_at=datetime.now(UTC),
            payload={"previous": previous, "binding": binding},
            idempotency_key=f"background:{branch_id}:{publication.id}",
        )
    )
    session.execute(
        update(RuntimeMemoryRow)
        .where(
            RuntimeMemoryRow.snapshot_id == snapshot.id,
            RuntimeMemoryRow.scope == "world",
            RuntimeMemoryRow.status.in_(["asserted", "confirmed"]),
        )
        .values(status="superseded")
    )
    snapshot.profile_id, snapshot.graph_version_id = profile.id, graph.id
    snapshot.profile = {**_profile_payload(profile), "_runtime_binding": binding}
    snapshot.routine_profile = _runtime_routine_profile(profile)
    snapshot.source_message_ids = list(profile.source_message_ids)
    snapshot.cutoff_at = _latest_profile_cutoff(session, profile, branch.origin_time)
    snapshot.compiler_version = profile.compiler_version
    session.execute(
        update(Branch)
        .where(Branch.id == branch_id)
        .values(runtime_input_revision=Branch.runtime_input_revision + 1)
    )
    session.flush()
    _seed_world_memory(session, snapshot)
    MemoryService(session).rebuild_index(branch_id, reason="published_background_rebound")
    # 已提交计划仍可读；仅把今天及未来的准备状态置为待修订，不篡改已经发生的块。
    from .clock import create_clock
    from .collaboration.plans import local_time

    clock_row = session.get(RuntimeClockRow, branch_id)
    if clock_row is not None:
        clock = create_clock(
            branch_id,
            clock_row.virtual_anchor,
            wall_anchor=clock_row.wall_anchor,
            timezone=clock_row.timezone,
        ).model_copy(update={"time_scale": clock_row.time_scale, "status": clock_row.status})
        today = local_time(clock.now(), clock.timezone).date().isoformat()
        for plan in session.scalars(
            select(RuntimeDayPlanRow).where(
                RuntimeDayPlanRow.branch_id == branch_id, RuntimeDayPlanRow.plan_date >= today
            )
        ):
            plan.generation_metadata = {
                **(plan.generation_metadata or {}),
                "background_binding": binding,
                "preparation": {
                    "status": "pending",
                    "attempt": 0,
                    "retry_at": None,
                    "error_code": "published_background_changed",
                },
            }
    return binding
