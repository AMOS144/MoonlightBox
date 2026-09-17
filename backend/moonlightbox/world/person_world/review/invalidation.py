"""纠正会话的基线失效与下游草稿作废逻辑。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.world.models import (
    AtomicWorldClaim,
    PersonWorldRevisionMessage,
    PersonWorldRevisionSession,
    WorldCorrection,
    WorldGraphChangeSet,
    WorldGraphVersion,
    WorldPublication,
)

from .payloads import _object_refs, _string_list


def ensure_base_is_current(session: Session, revision: PersonWorldRevisionSession) -> None:
    """用发布指针而非展示文字验证本轮纠正的乐观锁基线。

    这项检查只比较固定的 Graph/Profile UUID。它不阅读聊天正文、不会借助正则判断
    用户意图；一旦活动版本改变，旧 ContextSnapshot、Patch 和 Approval 都不得继续
    使用，必须由新会话在新基线上重新调查。
    """

    from .service import RevisionBaseStaleError

    base_graph = session.get(WorldGraphVersion, revision.base_graph_version_id)
    publication = session.scalar(
        select(WorldPublication)
        .where(
            WorldPublication.project_id == revision.project_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash
            == (revision.scope.get("node_scope") or {}).get("preview_hash"),
        )
        .order_by(WorldPublication.published_at.desc(), WorldPublication.id.desc())
    )
    stale_reason: str | None = None
    active_graph_id: str | None = None
    active_profile_id: str | None = None
    if revision.scope.get("base_profile_hash"):
        from .profile_v3 import revision_base_is_current

        if revision_base_is_current(session, revision, publication):
            return
        stale_reason = "v3_profile_or_publication_changed"
    elif base_graph is None:
        stale_reason = "base_graph_missing"
    elif publication is not None:
        active_graph_id = publication.graph_version_id
        active_profile_id = publication.profile_id
        if publication.graph_version_id != revision.base_graph_version_id:
            stale_reason = "active_graph_changed"
        elif publication.profile_id != revision.base_profile_id:
            stale_reason = "active_profile_changed"
    elif base_graph.status == "superseded":
        # 尚未创建 Publication 的旧项目兼容路径：一旦基础 Graph 已退役，同样
        # 不能让纠正沿用它的证据窗口。
        stale_reason = "base_graph_superseded"
    if stale_reason is None:
        return
    if revision.status != "stale":
        invalidate_stale_base(
            session,
            revision,
            reason=stale_reason,
            active_graph_id=active_graph_id,
            active_profile_id=active_profile_id,
        )
    raise RevisionBaseStaleError(
        "人物世界的活动版本已变化；本次纠正及其批准已失效，请基于最新版本重新开始"
    )


def invalidate_stale_base(
    session: Session,
    revision: PersonWorldRevisionSession,
    *,
    reason: str,
    active_graph_id: str | None,
    active_profile_id: str | None,
) -> None:
    """原子作废旧基线的派生产物，不改动已经发布的 Graph/Profile。"""

    now = datetime.now(UTC)
    change_sets = list(
        session.scalars(
            select(WorldGraphChangeSet).where(
                WorldGraphChangeSet.revision_session_id == revision.id,
                WorldGraphChangeSet.status != "published",
            )
        )
    )
    for change_set in change_sets:
        change_set.status = "stale"
        candidate = (
            session.get(WorldGraphVersion, change_set.candidate_graph_version_id)
            if change_set.candidate_graph_version_id
            else None
        )
        if candidate is not None and candidate.status not in {"ready", "superseded"}:
            candidate.status = "rejected"
            candidate.completed_at = now
    for correction in session.scalars(
        select(WorldCorrection).where(
            WorldCorrection.revision_session_id == revision.id,
            WorldCorrection.status.in_(["proposed", "approved"]),
        )
    ):
        correction.status = "revoked"
    revision.scope = {
        **dict(revision.scope or {}),
        "stale_base": {
            "reason": reason,
            "base_graph_version_id": revision.base_graph_version_id,
            "base_profile_id": revision.base_profile_id,
            "active_graph_version_id": active_graph_id,
            "active_profile_id": active_profile_id,
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
            content="活动人物世界版本已变化，原纠正范围、理解与批准已失效。请基于最新版本重新开始。",
            payload={
                "reason": reason,
                "base_graph_version_id": revision.base_graph_version_id,
                "base_profile_id": revision.base_profile_id,
                "active_graph_version_id": active_graph_id,
                "active_profile_id": active_profile_id,
            },
        )
    )
    session.flush()


def invalidate_unapproved_outputs(session: Session, revision: PersonWorldRevisionSession) -> None:
    """用户改变意图或范围时作废下游草稿，不触碰活动图谱。"""

    previous_change_set = (
        session.get(WorldGraphChangeSet, revision.graph_change_set_id)
        if revision.graph_change_set_id
        else None
    )
    if previous_change_set is None:
        return
    previous_change_set.status = "rejected"
    candidate = (
        session.get(WorldGraphVersion, previous_change_set.candidate_graph_version_id)
        if previous_change_set.candidate_graph_version_id
        else None
    )
    if candidate is not None and candidate.status != "ready":
        candidate.status = "rejected"
    for correction in session.scalars(
        select(WorldCorrection).where(
            WorldCorrection.approved_change_set_id == previous_change_set.id,
            WorldCorrection.status.in_(["proposed", "approved"]),
        )
    ):
        correction.status = "revoked"


def correction_scope(session: Session, revision: PersonWorldRevisionSession) -> dict[str, object]:
    """把 Revision Scope 中的稳定 Claim 转成可供重编译筛选的结构边界。

    这里不理解或匹配任何中文陈述。数据库的 ``primary_domain``、``fact_type`` 和
    ``subject_kind`` 是 Claim 创建时已经固化的结构字段；没有 Claim 的自由纠正保持
    ``provisional``，不会伪装成可自动归类的已知人物事实。
    """

    scope = dict(revision.scope or {})
    claim_ids = _string_list(scope.get("selected_claim_ids"))
    claims = (
        list(
            session.scalars(
                select(AtomicWorldClaim).where(
                    AtomicWorldClaim.project_id == revision.project_id,
                    AtomicWorldClaim.graph_version_id == revision.base_graph_version_id,
                    AtomicWorldClaim.id.in_(claim_ids),
                )
            )
        )
        if claim_ids
        else []
    )
    domains = list(
        dict.fromkeys(
            claim.primary_domain
            if claim.primary_domain and claim.primary_domain != "unknown"
            else claim.profile_section.split(".", 1)[0]
            for claim in claims
            if claim.profile_section
        )
    )
    return {
        "claim_ids": [claim.id for claim in claims],
        "node_scope": scope.get("node_scope"),
        "primary_domains": domains,
        "fact_types": list(
            dict.fromkeys(
                claim.fact_type
                if claim.fact_type and claim.fact_type != "unknown"
                else claim.predicate
                for claim in claims
            )
        ),
        "subject_kinds": list(dict.fromkeys(claim.subject_kind for claim in claims)),
        "graph_object_refs": _object_refs(scope.get("included_graph_objects")),
        "provisional": not claims,
    }
