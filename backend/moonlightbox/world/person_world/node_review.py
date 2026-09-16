"""节点草稿到不可变审核版本、节点发布指针；不修改共享全量画像。"""

from copy import deepcopy

from sqlalchemy import select

from moonlightbox.world.models import PersonWorldProfile, PersonWorldProfileDraft, WorldGraphVersion

from .coordinator_v3 import empty_legacy_projection
from .node_scope import NodeCompilationScope
from .publication import WorldPublicationError, _publish_pair, active_publication
from .review.context import canonical_context_hash


def review_node_draft(session, *, project_id, draft_id):
    draft = session.get(PersonWorldProfileDraft, draft_id)
    if draft is None or draft.project_id != project_id:
        raise LookupError("节点草稿不存在")
    graph = session.get(WorldGraphVersion, draft.graph_version_id)
    if graph is None:
        raise ValueError("节点草稿绑定的图版本不存在")
    scope = NodeCompilationScope.restore(
        session, graph=graph, envelope=draft.generation_summary.get("node_scope")
    )
    fingerprint = canonical_context_hash(
        {"profile": draft.profile_v3, "summary": draft.generation_summary}
    )
    profile = session.scalar(
        select(PersonWorldProfile).where(
            PersonWorldProfile.project_id == project_id,
            PersonWorldProfile.node_boundary_hash == scope.boundary["preview_hash"],
            PersonWorldProfile.generation_summary["source_draft_hash"].as_string() == fingerprint,
        )
    )
    if profile is None:
        # 审核版本不可变。栏目重试生成新草稿内容后，旧的审核/纠正基线仍可追溯。
        current = active_publication(session, project_id, scope.boundary["preview_hash"])
        profile = PersonWorldProfile(
            project_id=project_id,
            graph_version_id=graph.id,
            subject_person_id=draft.profile_v3["subject_participant_id"],
            node_boundary_hash=scope.boundary["preview_hash"],
            **empty_legacy_projection().model_dump(mode="json"),
            profile_schema_version="v3",
            profile_v3=deepcopy(draft.profile_v3),
            source_message_ids=[],
            retrieval_manifest=[],
            compiler_version="node-profile-v3",
            agent_run_id=draft.agent_run_id,
            investigation_report=deepcopy(draft.investigation_report),
            generation_summary={
                **deepcopy(draft.generation_summary),
                "source_draft_id": draft.id,
                "source_draft_hash": fingerprint,
                "base_node_publication_id": current.id if current else None,
            },
        )
        session.add(profile)
        session.flush()
    return profile


def node_review_payload(session, profile):
    current = active_publication(session, profile.project_id, profile.node_boundary_hash)
    # 用户完成纠正并发布后，工作台展示该节点当前版本，而非又回到原始草稿。
    displayed = (
        session.get(PersonWorldProfile, current.profile_id)
        if current and profile.generation_summary.get("base_node_publication_id") != current.id
        else profile
    )
    return {
        "profile_id": displayed.id,
        "draft_profile_id": profile.id,
        "agent_run_id": displayed.agent_run_id,
        "source_draft_id": displayed.generation_summary.get("source_draft_id"),
        "profile_v3": displayed.profile_v3,
        "node_scope": displayed.generation_summary["node_scope"],
        "investigation_report": displayed.investigation_report,
        "failed_sections": displayed.generation_summary.get("failed_sections", []),
        "approval_hash": canonical_context_hash(
            {"profile": displayed.profile_v3, "scope": displayed.generation_summary["node_scope"]}
        ),
        "publication_id": current.id if current and displayed.id == current.profile_id else None,
    }


def approve_node_profile(session, *, project_id, profile_id, approval_hash):
    from sqlalchemy import update

    from moonlightbox.projects.models import Project

    # 在读取发布指针前取得项目写锁，使并发批准检查与写入处于同一事务。
    session.execute(update(Project).where(Project.id == project_id).values(name=Project.name))
    profile = session.get(PersonWorldProfile, profile_id)
    if profile is None or profile.project_id != project_id or not profile.node_boundary_hash:
        raise LookupError("节点审核版本不存在")
    graph = session.get(WorldGraphVersion, profile.graph_version_id)
    NodeCompilationScope.restore(
        session, graph=graph, envelope=profile.generation_summary["node_scope"]
    )
    expected = canonical_context_hash(
        {"profile": profile.profile_v3, "scope": profile.generation_summary["node_scope"]}
    )
    if approval_hash != expected:
        raise WorldPublicationError("审核内容已变化，请重新确认")
    current = active_publication(session, project_id, profile.node_boundary_hash)
    if current and current.profile_id == profile.id:
        return current
    if (current.id if current else None) != profile.generation_summary.get(
        "base_node_publication_id"
    ):
        raise WorldPublicationError("该节点已发布另一版本，请刷新审核页")
    draft_id = profile.generation_summary.get("source_draft_id")
    from moonlightbox.world.models import PersonWorldSectionTask

    if session.scalar(
        select(PersonWorldSectionTask.id).where(
            PersonWorldSectionTask.agent_run_id == profile.agent_run_id,
            PersonWorldSectionTask.status.in_(["pending", "researching"]),
        )
    ):
        raise WorldPublicationError("栏目仍在重试，请等待草稿稳定后再批准")
    draft = session.get(PersonWorldProfileDraft, draft_id) if draft_id else None
    if draft and canonical_context_hash(
        {"profile": draft.profile_v3, "summary": draft.generation_summary}
    ) != profile.generation_summary.get("source_draft_hash"):
        raise WorldPublicationError("栏目重试已更新草稿，请重新读取审核版本")
    return _publish_pair(
        session,
        graph=graph,
        profile=profile,
        correction_head_hash=graph.correction_head_hash or "",
        published_by="local_user",
    )


def runtime_node_projection(value):
    """只保留获批理解，不把前后段调查引用、工具材料和审核笔记下发给角色。"""
    if isinstance(value, list):
        return [runtime_node_projection(item) for item in value]
    if isinstance(value, dict):
        return {
            key: runtime_node_projection(item)
            for key, item in value.items()
            if key
            not in {
                "reference_message_ids",
                "source_message_ids",
                "evidence_ids",
                "source_ids",
                "investigation",
                "generation_summary",
                "section_work",
                "retrieval_manifest",
            }
        }
    return value
