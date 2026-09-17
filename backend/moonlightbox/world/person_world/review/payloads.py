"""纠正流程的模块级纯函数：活动版本查询、Profile 载荷与 Scope/消息辅助。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message, Participant
from moonlightbox.world.models import (
    PersonWorldProfile,
    WorldGraphVersion,
    WorldPublication,
)


def _active_profile_and_graph(
    session: Session, project_id: str
) -> tuple[PersonWorldProfile | None, WorldGraphVersion | None]:
    publication = session.scalar(
        select(WorldPublication)
        .where(
            WorldPublication.project_id == project_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash.is_(None),
        )
        .order_by(WorldPublication.published_at.desc())
    )
    if publication is not None:
        return (
            session.get(PersonWorldProfile, publication.profile_id),
            session.get(WorldGraphVersion, publication.graph_version_id),
        )
    graph = session.scalar(
        select(WorldGraphVersion)
        .where(
            WorldGraphVersion.project_id == project_id,
            WorldGraphVersion.status.in_(["ready", "awaiting_profile_review"]),
        )
        .order_by(WorldGraphVersion.created_at.desc())
    )
    profile = (
        session.scalar(
            select(PersonWorldProfile)
            .where(PersonWorldProfile.node_boundary_hash.is_(None))
            .where(
                PersonWorldProfile.project_id == project_id,
                PersonWorldProfile.graph_version_id == graph.id,
            )
        )
        if graph is not None
        else None
    )
    return profile, graph


def graph_bundles(session: Session, graph_version_id: str) -> list[Any]:
    from moonlightbox.world.models import ConversationBundle

    return list(
        session.scalars(
            select(ConversationBundle).where(
                ConversationBundle.graph_version_id == graph_version_id
            )
        )
    )


def _profile_payload(profile: PersonWorldProfile | None) -> dict[str, object]:
    if profile is None:
        return {}
    if profile.profile_schema_version == "v3":
        return dict(profile.profile_v3 or {})
    if profile.profile_schema_version == "v2" and isinstance(profile.profile_v2, dict):
        # v2 是 Revision 的唯一人物事实读模型。去掉内部 Claim UUID 后再交给模型：具体
        # 修改对象仅由服务器根据 selected_scope_items 的稳定位置绑定，不能让模型复制 ID。
        return _without_claim_ids(profile.profile_v2)
    return {
        "identity": profile.identity,
        "work_and_education": profile.work_and_education,
        "places": profile.places,
        "social_relationships": profile.social_relationships,
        "preferences": profile.preferences,
        "recurring_activities": profile.recurring_activities,
        "routine_summary": profile.routine_summary,
        "life_phases": profile.life_phases,
        "relationship_with_user": profile.relationship_with_user,
        "important_events": profile.important_events,
        "unresolved_candidates": profile.unresolved_candidates,
    }


def _without_claim_ids(value: object) -> dict[str, object]:
    """递归移除模型不应看到的数据库事实身份；不改变任何可读事实内容。"""

    def sanitize(item: object) -> object:
        if isinstance(item, dict):
            return {key: sanitize(value) for key, value in item.items() if key != "claim_id"}
        if isinstance(item, list):
            return [sanitize(child) for child in item]
        return item

    cleaned = sanitize(value)
    return dict(cleaned) if isinstance(cleaned, dict) else {}


def _scope_proposal_candidates(scope: object, related_claim_ids: list[str]) -> list[str]:
    """从 Snapshot 的有序相关 Claim 中移除已选项，得到模型序号的后端绑定。"""

    selected = set(_string_list(scope.get("selected_claim_ids")) if isinstance(scope, dict) else [])
    return [item for item in related_claim_ids if item not in selected]


def _resolve_scope_proposal_items(scope: dict[str, object], item_ids: list[int]) -> list[str]:
    from .service import RevisionStateError

    pending = scope.get("pending_scope_proposal")
    if not isinstance(pending, dict):
        if item_ids:
            raise RevisionStateError("没有可供确认的范围提案")
        return []
    items = pending.get("items")
    if not isinstance(items, list):
        raise RevisionStateError("范围提案已损坏，请重新发起调查")
    requested = set(item_ids)
    return [
        item["claim_id"]
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("proposal_item"), int)
        and item["proposal_item"] in requested
        and isinstance(item.get("claim_id"), str)
    ]


def _source_message_context(
    session: Session, project_id: str, message_ids: list[str]
) -> list[dict[str, object]]:
    if not message_ids:
        return []
    return [
        {
            "message_id": message.id,
            "timestamp": message.timestamp.isoformat(),
            "participant": participant.name,
            "role": participant.role,
            "content": message.content,
        }
        for message, participant in session.execute(
            select(Message, Participant)
            .join(Participant, Participant.id == Message.participant_id)
            .where(Message.project_id == project_id, Message.id.in_(message_ids))
            .order_by(Message.timestamp.asc(), Message.id.asc())
        ).all()
    ]


def _dedupe_message_payloads(items: list[Any]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("message_id"), str):
            continue
        if item["message_id"] in seen:
            continue
        seen.add(item["message_id"])
        result.append(item)
    return result


def _affected_entities(operations: list[dict[str, object]]) -> list[str]:
    values: list[str] = []
    for operation in operations:
        for key in ("entity_name", "source_entity", "target_entity"):
            value = operation.get(key)
            if isinstance(value, str) and value:
                values.append(value)
        sources = operation.get("source_entities")
        if isinstance(sources, list):
            values.extend(item for item in sources if isinstance(item, str) and item)
    return list(dict.fromkeys(values))


def _affected_relations(operations: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"source_entity": source, "target_entity": target}
        for item in operations
        for source, target in [(item.get("source_entity"), item.get("target_entity"))]
        if isinstance(source, str) and isinstance(target, str) and source and target
    ]


def _graph_state_description(value: object) -> str | None:
    """从已签发 precondition 的结构状态提取可读旧描述，不解释消息或图谱语义。"""

    if not isinstance(value, dict):
        return None
    description = value.get("description")
    return description if isinstance(description, str) and description.strip() else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str)))


def _object_refs(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {key: item[key] for key in sorted(item) if isinstance(item[key], str)}
        for item in value
        if isinstance(item, dict) and all(isinstance(key, str) for key in item)
    ]


def _merge_values(existing: list[str], incoming: list[str]) -> list[str]:
    return list(dict.fromkeys([*existing, *(item for item in incoming if isinstance(item, str))]))


def _object_ref_key(value: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(value.items()))


def _merge_object_refs(
    existing: list[dict[str, str]], incoming: list[dict[str, str]]
) -> list[dict[str, str]]:
    result = list(existing)
    seen = {_object_ref_key(item) for item in existing}
    for item in incoming:
        normalized = {key: value for key, value in sorted(item.items()) if isinstance(value, str)}
        key = _object_ref_key(normalized)
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _remove_object_refs(
    existing: list[dict[str, str]], removed: list[dict[str, str]]
) -> list[dict[str, str]]:
    removed_keys = {
        _object_ref_key({key: value for key, value in item.items() if isinstance(value, str)})
        for item in removed
    }
    return [item for item in existing if _object_ref_key(item) not in removed_keys]
