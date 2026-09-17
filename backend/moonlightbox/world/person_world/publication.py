"""Graph 与 Profile 的原子发布服务。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService
from moonlightbox.world.jobs import enqueue_node_investigation
from moonlightbox.world.models import (
    PersonWorldProfile,
    PersonWorldProfileDraft,
    PersonWorldRevisionMessage,
    PersonWorldRevisionSession,
    PersonWorldSectionTask,
    WorldCorrection,
    WorldGraphChangeSet,
    WorldGraphVersion,
    WorldPublication,
)


class WorldPublicationError(RuntimeError):
    pass


class WorldPublicationStaleBaseError(WorldPublicationError):
    """候选完成后活动发布对发生竞争，旧批准不能切换当前世界。"""


def active_publication(
    session: Session, project_id: str, node_boundary_hash=None
) -> WorldPublication | None:
    return session.scalar(
        select(WorldPublication)
        .where(
            WorldPublication.project_id == project_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash == node_boundary_hash,
        )
        .order_by(WorldPublication.published_at.desc(), WorldPublication.id.desc())
    )


def approve_initial_profile(
    session: Session,
    *,
    project_id: str,
    graph_version_id: str,
    profile_id: str,
) -> WorldPublication:
    graph = session.get(WorldGraphVersion, graph_version_id)
    profile = session.get(PersonWorldProfile, profile_id)
    if (
        graph is None
        or profile is None
        or graph.project_id != project_id
        or profile.project_id != project_id
        or profile.graph_version_id != graph.id
    ):
        raise WorldPublicationError("待审核的 Graph 与 Profile 不匹配")
    if graph.status != "awaiting_profile_review":
        raise WorldPublicationError("当前图版本不在 Profile 审核阶段")
    if profile.generation_summary.get("node_scope"):
        raise WorldPublicationError("节点画像需要按节点发布，不能通过项目全量画像入口覆盖活动版本")
    if profile.profile_schema_version == "v3" and profile.generation_summary.get("failed_sections"):
        raise WorldPublicationError("v3 仍有运行失败的栏目，请完成重试后发布；不能把失败当作未知")
    if profile.agent_run_id and session.scalar(
        select(PersonWorldSectionTask.id).where(
            PersonWorldSectionTask.agent_run_id == profile.agent_run_id,
            PersonWorldSectionTask.status.in_({"pending", "researching"}),
        )
    ):
        raise WorldPublicationError("仍有栏目正在重试，请等待候选档案稳定后再发布")
    current = active_publication(session, project_id)
    if (
        current is not None
        and graph.parent_version_id is not None
        and current.graph_version_id != graph.parent_version_id
    ):
        retired_base = session.get(WorldGraphVersion, graph.parent_version_id)
        retired_profile = session.scalar(
            select(PersonWorldProfile)
            .where(PersonWorldProfile.node_boundary_hash.is_(None))
            .where(PersonWorldProfile.graph_version_id == graph.parent_version_id)
        )
        # 旧候选升级会多一层只读父版本；仅允许跨过这一层，仍核对原活动基线。
        # 活动版本真的变化时不能因“升级”而绕过并发发布保护。
        if not (
            retired_base is not None
            and retired_base.status == "superseded"
            and retired_base.parent_version_id == current.graph_version_id
            and retired_profile is not None
            and retired_profile.profile_schema_version in {"v1", "v2"}
        ):
            raise WorldPublicationStaleBaseError(
                "活动人物世界版本已经变化，必须基于最新版本重新调查并审核"
            )
    publication = _publish_pair(
        session,
        graph=graph,
        profile=profile,
        correction_head_hash=graph.correction_head_hash,
        published_by="local_user",
    )
    # 旧数据库没有 WorldPublication 时，候选仍可能以 legacy ready 图为基线。
    # 成功批准后也要显式退役该基线，避免“两个 ready 图”被旧读取路径误选。
    if graph.parent_version_id is not None:
        base = session.get(WorldGraphVersion, graph.parent_version_id)
        if base is not None and base.id != graph.id and base.status == "ready":
            base.status = "superseded"
            base.superseded_at = datetime.now(UTC)
    return publication


def publish_change_set(
    session: Session,
    *,
    project_id: str,
    revision_session_id: str,
    change_set_id: str,
    expected_session_revision: int,
) -> WorldPublication:
    from sqlalchemy import update

    from moonlightbox.projects.models import Project

    # 与直接批准使用同一项目写锁，发布基线检查和切换不可交错。
    session.execute(update(Project).where(Project.id == project_id).values(name=Project.name))
    revision = session.get(PersonWorldRevisionSession, revision_session_id)
    change_set = session.get(WorldGraphChangeSet, change_set_id)
    if (
        revision is None
        or change_set is None
        or revision.project_id != project_id
        or change_set.project_id != project_id
        or change_set.revision_session_id != revision.id
    ):
        raise WorldPublicationError("纠正会话或 ChangeSet 不存在")
    if revision.status != "publish_ready" or change_set.status != "publish_ready":
        raise WorldPublicationError("候选图尚未完成执行和验证")
    if revision.session_revision != expected_session_revision:
        raise WorldPublicationError("纠正会话已经更新，请刷新后再发布")
    candidate = (
        session.get(WorldGraphVersion, change_set.candidate_graph_version_id)
        if change_set.candidate_graph_version_id
        else None
    )
    profile_id = (
        change_set.execution_result.get("profile_id")
        if isinstance(change_set.execution_result, dict)
        else None
    )
    profile = session.get(PersonWorldProfile, profile_id) if isinstance(profile_id, str) else None
    if candidate is None or profile is None or profile.graph_version_id != candidate.id:
        raise WorldPublicationError("候选 Graph 与重新编译的 Profile 不匹配")
    if candidate.status != "publish_ready":
        raise WorldPublicationError("候选图状态不允许发布")
    node_scope = revision.scope.get("node_scope")
    if bool(node_scope) != bool(profile.node_boundary_hash):
        raise WorldPublicationError("纠正发布不能切换节点与全量作用域")
    if node_scope:
        from .node_scope import NodeCompilationScope, same_node_scope

        candidate_scope = profile.generation_summary.get("node_scope")
        if (
            not same_node_scope(node_scope, candidate_scope)
            or profile.node_boundary_hash != node_scope.get("preview_hash")
            or candidate.parent_version_id != revision.base_graph_version_id
        ):
            raise WorldPublicationError("纠正产物未继承原节点边界")
        try:
            NodeCompilationScope.restore(session, graph=candidate, envelope=candidate_scope)
        except ValueError as error:
            raise WorldPublicationError(str(error)) from error
    current = active_publication(session, project_id, profile.node_boundary_hash)
    from .review.profile_v3 import revision_base_is_current

    v3_current = bool(revision.scope.get("base_profile_hash")) and revision_base_is_current(
        session, revision, current
    )
    if (revision.scope.get("base_profile_hash") and not v3_current) or (
        current is not None
        and not v3_current
        and (
            current.graph_version_id != change_set.base_graph_version_id
            or current.profile_id != revision.base_profile_id
        )
    ):
        _invalidate_stale_publication_attempt(
            session,
            revision=revision,
            change_set=change_set,
            candidate=candidate,
            current=current,
        )
        raise WorldPublicationStaleBaseError(
            "活动人物世界版本已经变化，必须基于最新版本重新生成并审核 Patch"
        )
    publication = _publish_pair(
        session,
        graph=candidate,
        profile=profile,
        correction_head_hash=change_set.canonical_payload_hash,
        published_by="local_user",
    )
    now = datetime.now(UTC)
    # 首次候选尚未形成 active publication；若用户先纠正再发布，也要把
    # 原候选明确退役，避免它继续显示成另一份等待批准的草稿。
    base_graph = session.get(WorldGraphVersion, change_set.base_graph_version_id)
    if (
        base_graph is not None
        and profile.node_boundary_hash is None
        and base_graph.id != candidate.id
        and base_graph.status != "superseded"
    ):
        base_graph.status = "superseded"
        base_graph.superseded_at = now
    for correction in session.scalars(
        select(WorldCorrection).where(
            WorldCorrection.approved_change_set_id == change_set.id,
            WorldCorrection.status == "approved",
        )
    ):
        correction.status = "active"
    change_set.status = "published"
    revision.status = "published"
    revision.session_revision += 1
    candidate.status = "ready"
    candidate.published_at = now
    session.flush()
    return publication


def enqueue_publication_analysis(
    session: Session,
    *,
    settings: Settings,
    graph: WorldGraphVersion,
) -> Job:
    """发布后进入节点调查工作台；不再启动历史 EventNode 评分。"""

    service = JobService(session)
    return enqueue_node_investigation(
        service,
        project_id=graph.project_id,
    )


def _invalidate_stale_publication_attempt(
    session: Session,
    *,
    revision: PersonWorldRevisionSession,
    change_set: WorldGraphChangeSet,
    candidate: WorldGraphVersion,
    current: WorldPublication | None,
) -> None:
    """发布竞争发生在最后一步时，也要留下可读的过期状态和审计消息。"""

    now = datetime.now(UTC)
    change_set.status = "stale"
    if candidate.status not in {"ready", "superseded"}:
        candidate.status = "rejected"
        candidate.completed_at = now
    for correction in session.scalars(
        select(WorldCorrection).where(
            WorldCorrection.approved_change_set_id == change_set.id,
            WorldCorrection.status.in_(["proposed", "approved"]),
        )
    ):
        correction.status = "revoked"
    revision.scope = {
        **dict(revision.scope or {}),
        "stale_base": {
            "reason": "active_publication_changed_before_publish",
            "base_graph_version_id": revision.base_graph_version_id,
            "base_profile_id": revision.base_profile_id,
            "active_graph_version_id": current.graph_version_id if current else None,
            "active_profile_id": current.profile_id if current else None,
            "invalidated_at": now.isoformat(),
        },
    }
    revision.status = "stale"
    revision.pending_turn_id = None
    revision.understanding_payload = None
    revision.understanding_payload_hash = None
    revision.session_revision += 1
    session.add(
        PersonWorldRevisionMessage(
            session_id=revision.id,
            role="system",
            kind="base_stale",
            session_revision=revision.session_revision,
            content="活动人物世界版本已变化，已完成的候选图不会发布。请基于最新版本重新开始。",
            payload={
                "reason": "active_publication_changed_before_publish",
                "base_graph_version_id": revision.base_graph_version_id,
                "base_profile_id": revision.base_profile_id,
                "active_graph_version_id": current.graph_version_id if current else None,
                "active_profile_id": current.profile_id if current else None,
            },
        )
    )
    session.flush()


def _publish_pair(
    session: Session,
    *,
    graph: WorldGraphVersion,
    profile: PersonWorldProfile,
    correction_head_hash: str,
    published_by: str,
) -> WorldPublication:
    node_hash = profile.node_boundary_hash
    if profile.profile_schema_version == "v3":
        from .context_module_snapshots import ModuleSnapshot, validate_references
        from .contracts.profile_v3 import PersonWorldProfileV3

        if profile.generation_summary.get("failed_sections"):
            raise WorldPublicationError("v3 仍有失败栏目，不能发布不完整候选")
        canonical = PersonWorldProfileV3.model_validate(profile.profile_v3).model_dump(mode="json")
        validate_references(
            canonical,
            ModuleSnapshot.create(graph.project_id, canonical["life_context"]["context_modules"]),
        )
    current = active_publication(session, graph.project_id, node_hash)
    now = datetime.now(UTC)
    if current is not None:
        if current.graph_version_id == graph.id and current.profile_id == profile.id:
            return current
        current.status = "superseded"
        # 先释放作用域内的唯一活动指针，再插入新指针。
        session.flush()
        previous_graph = session.get(WorldGraphVersion, current.graph_version_id)
        if previous_graph is not None and node_hash is None:
            previous_graph.status = "superseded"
            previous_graph.superseded_at = now
    publication = WorldPublication(
        project_id=graph.project_id,
        graph_version_id=graph.id,
        profile_id=profile.id,
        node_boundary_hash=node_hash,
        correction_head_hash=correction_head_hash,
        previous_publication_id=current.id if current is not None else None,
        status="active",
        published_by=published_by,
        published_at=now,
    )
    session.add(publication)
    # 节点发布不改变共享图的全量审核/退役状态；只有它自己的纠正候选需要就绪。
    if node_hash is None or graph.status == "publish_ready":
        graph.status = "ready"
        graph.published_at = now
    draft = (
        session.scalar(
            select(PersonWorldProfileDraft).where(
                PersonWorldProfileDraft.agent_run_id == profile.agent_run_id
            )
        )
        if profile.agent_run_id
        else None
    )
    if draft is not None:
        draft.status = "approved"
    session.flush()
    return publication
