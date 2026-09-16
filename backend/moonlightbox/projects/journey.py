"""用户流程的只读投影；不保存另一套完成状态，也不启动后台任务。

Job 只用于解释暂时失败，不能代替发布、批准或分支准备的业务结果。
所有后续写接口仍须在事务内检查自己的对象与版本。
"""

from sqlalchemy import func, select

from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.runtime_recovery import WORLD_KIND, recovery
from moonlightbox.jobs.scope import project_job_condition
from moonlightbox.node_investigation.models import NodeInvestigation
from moonlightbox.runtime_v1.branch_models import Branch, BranchMessage
from moonlightbox.world.models import (
    PersonWorldRevisionSession,
    WorldGraphVersion,
    WorldPublication,
)


def project_journey(session, project_id):
    build_job = session.scalar(select(Job).where(
        Job.kind == WORLD_KIND, project_job_condition(project_id),
    ).order_by(Job.created_at.desc(), Job.id.desc()).limit(1))
    build_recovery = recovery(build_job) if build_job else {}
    build_waiting = bool(
        build_job
        and build_job.status in {"failed", "interrupted"}
        and build_recovery.get("status") == "waiting"
    )
    imports = list(
        session.scalars(
            select(ImportSource)
            .where(ImportSource.project_id == project_id)
            .order_by(ImportSource.confirmed_at.desc())
        )
    )
    people = list(session.scalars(select(Participant).where(Participant.project_id == project_id)))
    graph = session.scalar(
        select(WorldGraphVersion)
        .where(WorldGraphVersion.project_id == project_id)
        .order_by(WorldGraphVersion.created_at.desc(), WorldGraphVersion.id.desc())
    )
    publication = session.scalar(
        select(WorldPublication)
        .where(
            WorldPublication.project_id == project_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash.is_(None),
        )
        .order_by(WorldPublication.published_at.desc(), WorldPublication.id.desc())
    )
    investigations = list(
        session.scalars(
            select(NodeInvestigation)
            .where(NodeInvestigation.project_id == project_id)
            .order_by(NodeInvestigation.created_at.desc(), NodeInvestigation.id.desc())
        )
    )
    branches = list(
        session.scalars(
            select(Branch).where(Branch.project_id == project_id).order_by(Branch.created_at.desc())
        )
    )
    active_jobs = list(
        session.scalars(
            select(Job).where(
                project_job_condition(project_id),
                Job.status.in_(["queued", "running", "cancelling"]),
            )
        )
    )
    stages, tasks = [], []

    def stage(key, label, state, action, detail, object_id=None):
        value = dict(
            key=key, label=label, state=state, action=action, detail=detail, object_id=object_id
        )
        stages.append(value)
        if state in {"needs_user", "blocked", "paused"}:
            tasks.append({**value, "id": f"{key}:{object_id or project_id}"})
        return value

    stage(
        "import",
        "导入记录",
        "completed" if imports else "not_started",
        "import",
        "原始记录已保存；可追加导入，不会清除已有分支。"
        if imports
        else "选择聊天导出目录，先解析再确认人物。",
    )
    bound = {p.role for p in people} >= {"self", "target"}
    aliases_pending = graph is not None and graph.status == "awaiting_alias_review"
    stage(
        "participants",
        "确认人物与资料",
        "needs_user"
        if aliases_pending or (imports and not bound)
        else "confirmed"
        if bound
        else "not_started",
        "participants",
        "请确认这些名字是否指同一个人。"
        if aliases_pending
        else "人物绑定已保存；可查看头像、记录范围和资料整理状态。"
        if bound
        else "在导入预览中确认我和目标人物。",
        graph.id if graph else None,
    )
    investigation = investigations[0] if investigations else None
    node_state = investigation.state if investigation else {}
    node_status = node_state.get("status")
    state = (
        "processing"
        if node_status in {"queued", "running", "cancelling", "waiting_for_user"}
        else "blocked"
        if node_status in {"failed", "interrupted"}
        else "confirmed"
        if node_state.get("confirmed")
        else "paused"
        if node_status in {"paused", "cancelled"}
        else "needs_user"
        if node_state.get("candidates")
        else "not_started"
    )
    stage(
        "nodes",
        "调查并选择起点",
        state,
        "nodes",
        "Agent 会自行完成调查并给出候选；选择一个起点后才编译该时刻的人物背景。",
        investigation.id if investigation else None,
    )
    for revision in session.scalars(
        select(PersonWorldRevisionSession).where(
            PersonWorldRevisionSession.project_id == project_id,
            PersonWorldRevisionSession.status.not_in(["published", "cancelled", "failed", "stale"]),
        )
    ):
        tasks.append(
            dict(
                id=f"revision:{revision.id}",
                key="revision",
                label="继续人物背景纠正",
                state="processing" if revision.pending_turn_id else "needs_user",
                action="node_profile" if revision.scope.get("node_scope") else "revision",
                object_id=revision.base_profile_id
                if revision.scope.get("node_scope")
                else revision.id,
                detail="仅在逐步批准后应用修改；关闭面板不会取消会话。",
            )
        )
    for branch in branches:
        if branch.lifecycle_status in {"preparing", "prepare_failed"}:
            tasks.append(
                dict(
                    id=f"branch:{branch.id}",
                    key="branch",
                    label=branch.title,
                    state="blocked"
                    if branch.lifecycle_status == "prepare_failed"
                    else "processing",
                    action="branch",
                    object_id=branch.id,
                    detail="查看这条分支自己的准备状态。",
                )
            )
    node_publication = session.scalar(
        select(WorldPublication).where(
            WorldPublication.project_id == project_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash
            == (node_state.get("confirmed") or {}).get("preview_hash"),
            WorldPublication.node_boundary_hash.is_not(None),
        )
    )
    stage(
        "prepare",
        "准备该起点的分支",
        "confirmed"
        if node_publication
        else "needs_user"
        if node_state.get("confirmed")
        else "not_started",
        "selection",
        "节点背景已发布，可以准备分支。"
        if node_publication
        else "先在起点工作台编译、审核此刻的人物背景。",
        investigation.id if investigation else None,
    )
    stage(
        "chat",
        "开始聊天",
        "completed" if any(b.lifecycle_status == "active" for b in branches) else "not_started",
        "branches",
        "已有分支可独立打开，新资料整理不会强制重新准备。"
        if branches
        else "准备完成后在这里继续聊天。",
    )
    # 不以任何 Job 成功或模型数量证明资料可用。
    next_stage = next(
        (s for s in stages if s["state"] not in {"completed", "confirmed"}), stages[-1]
    )
    # 图谱尚未完成时，用户当前能处理的是资料整理/重试，而不是人物背景审阅。
    # 不把后续页面暴露成“下一步”，避免用户跳入必然受阻的页面。
    if build_job is not None and build_job.status != "succeeded":
        participants_stage = next(stage for stage in stages if stage["key"] == "participants")
        next_stage = {
            **participants_stage,
            "label": "处理资料整理",
            "detail": "图谱构建尚未完成，请在人物资料页查看进度并重试。",
        }
    lower, upper = session.execute(
        select(func.min(Message.timestamp), func.max(Message.timestamp)).where(
            Message.project_id == project_id
        )
    ).one()
    # 一次窗口查询取得各分支最后一条已提交消息，不拉取完整历史或逐分支查询。
    ranked = (
        select(
            BranchMessage.branch_id,
            BranchMessage.content,
            BranchMessage.type,
            BranchMessage.role,
            BranchMessage.observed_at,
            func.row_number()
            .over(
                partition_by=BranchMessage.branch_id,
                order_by=BranchMessage.sequence.desc(),
            )
            .label("rank"),
        )
        .join(Branch, Branch.id == BranchMessage.branch_id)
        .where(
            Branch.project_id == project_id,
            BranchMessage.generation_status == "completed",
        )
        .subquery()
    )
    previews = {
        row.branch_id: dict(
            text=row.content[:240]
            if row.type == "text"
            else {
                "image": "[图片]",
                "sticker": "[表情包]",
                "voice": "[语音]",
            }.get(row.type, "[媒体消息]"),
            role=row.role,
            virtual_time=row.observed_at,
        )
        for row in session.execute(select(ranked).where(ranked.c.rank == 1))
    }
    return dict(
        project_id=project_id,
        stages=stages,
        tasks=tasks,
        next_action=next_stage,
        processing=bool(active_jobs) or build_waiting,
        world_build={
            "failure": (build_job.checkpoint or {}).get("index_failure"),
            "id": build_job.id, "status": build_job.status,
            "progress": build_job.progress, "error_message": build_job.error_message,
            "recovery": build_recovery,
            "indexed_bundles": (build_job.checkpoint or {}).get("indexed_bundles", 0),
            "bundle_count": (build_job.checkpoint or {}).get("bundle_count", 0),
        } if build_job else None,
        participants=[
            dict(id=p.id, name=p.name, role=p.role, avatar_asset_id=p.avatar_asset_id)
            for p in people
        ],
        imports=[
            dict(
                id=i.id,
                preview_id=i.preview_id,
                message_count=i.message_count,
                confirmed_at=i.confirmed_at,
            )
            for i in imports
        ],
        time_range=[lower, upper],
        publication=dict(
            id=publication.id,
            profile_id=publication.profile_id,
            graph_version_id=publication.graph_version_id,
        )
        if publication
        else None,
        graph=dict(id=graph.id, status=graph.status) if graph else None,
        historical_branch_creation=dict(
            available=bool(node_publication),
            reason=None if node_publication else "node_profile_approval_required",
        ),
        branches=[
            dict(
                id=b.id,
                title=b.title,
                origin_time=b.origin_time,
                lifecycle_status=b.lifecycle_status,
                latest_message=previews.get(b.id),
            )
            for b in branches
        ],
    )
