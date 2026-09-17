"""Runtime 快照投影：把已发布 Profile 编译为 Runtime 只读视图与 world 记忆。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.events.models import EventNode
from moonlightbox.world.models import ConversationBundle, PersonWorldProfile

from .branch_models import Branch
from .db_models import RuntimeClockRow, RuntimeMemoryRow, RuntimeSnapshotRow


def _latest_profile_cutoff(
    session: Session, profile: PersonWorldProfile | None, fallback: datetime
) -> datetime:
    if profile is None:
        return fallback
    latest = session.scalar(
        select(func.max(ConversationBundle.ended_at)).where(
            ConversationBundle.graph_version_id == profile.graph_version_id
        )
    )
    return latest or profile.created_at or fallback


def _latest_import_node(session: Session, project_id: str) -> str | None:
    """记录当前导入末端节点，而非分支原节点，匹配 v1 的最新完整图谱约定。"""

    sort_time = func.coalesce(EventNode.ended_at, EventNode.started_at, EventNode.created_at)
    return session.scalar(
        select(EventNode.id)
        .where(EventNode.project_id == project_id)
        .order_by(sort_time.desc(), EventNode.id.desc())
        .limit(1)
    )


def _runtime_time_anchor(session: Session, branch: Branch) -> tuple[datetime, str]:
    """新分支读取批准边界；旧分支保留已存时钟，不重新猜时区。"""
    boundary = branch.origin_boundary
    if boundary:
        origin = datetime.fromisoformat(boundary["cutoff_at"])
        if origin.tzinfo is None:
            raise ValueError("起点边界必须包含明确时区")
        zone = boundary["timezone"]
        return origin.astimezone(ZoneInfo(zone)), zone
    clock = session.get(RuntimeClockRow, branch.id)
    if clock is not None:
        return clock.virtual_anchor, clock.timezone
    # 升级前尚未初始化的分支缺乏可信时区，不再查询 IdentityKernel。
    origin = branch.origin_time
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=UTC)
    return origin.astimezone(UTC), "UTC"


def _profile_payload(profile: PersonWorldProfile | None) -> dict[str, Any]:
    if profile is None:
        return {}
    if profile.profile_schema_version == "v3":
        # 来自分支绑定的已发布 Profile，绝不查询候选草稿或当前全局最新版本。
        return {**profile.profile_v3, "profile_schema_version": "v3"}
    if profile.profile_schema_version == "v2" and isinstance(profile.profile_v2, dict):
        return _runtime_projection_v2(profile.profile_v2)
    return {
        "profile_schema_version": "v1",
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
    }


def _runtime_projection_v2(profile: dict[str, Any]) -> dict[str, Any]:
    """从七栏目规范 Profile 构造 Runtime 所需的最小只读投影。

    这里仅调整数据形状，不归纳或改写任何事实。旧 key 是 Runtime 现有工具的稳定输入名，
    值则完全取自 v2；这样迁移期不会重新依赖旧 Profile 的兼容投影。
    """

    def values(domain: str, *fields: str) -> list[object]:
        section = profile.get(domain, {})
        if not isinstance(section, dict):
            return []
        return [
            item for field in fields for item in section.get(field, []) if isinstance(item, dict)
        ]

    return {
        "profile_schema_version": "v2",
        "identity": profile.get("identity", {}),
        "work_and_education": values("life_context", "work_and_learning"),
        "places": values("life_context", "places_and_environment"),
        "social_relationships": values("social_world", "ties"),
        "preferences": values(
            "agency",
            "preferences",
            "values_and_interpretations",
            "goals_and_commitments",
        ),
        "recurring_activities": values("practices", "recurring_activities"),
        "routine_summary": {
            "workdays": [],
            "weekends": [],
            "other_patterns": values("practices", "temporal_rhythms"),
        },
        "relationship_with_user": profile.get("relationship_with_user", {}),
        "important_events": values("life_course", "episodes"),
        "life_phases": values("life_course", "transitions", "trajectories"),
    }


def _runtime_routine_profile(profile: PersonWorldProfile) -> dict[str, Any]:
    """为既有 DayPlan 工具提供 v2 practices 的兼容运行时视图。"""

    projection = _profile_payload(profile)
    if profile.profile_schema_version == "v3":
        return {
            "practices": projection.get("practices", {}),
            "context_modules": [
                item
                for item in projection.get("life_context", {}).get("context_modules", [])
                if item.get("status") in {"current", "planned", "paused"}
            ],
            "interpretation": (
                "模块是处境和制度背景，practices 是实际规律理解；均不是当天已发生安排。"
            ),
        }
    routine = projection.get("routine_summary", {})
    return dict(routine) if isinstance(routine, dict) else {}


def _seed_world_memory(session: Session, snapshot: RuntimeSnapshotRow) -> None:
    """把人物档案编译为统一 world 记忆，供 search_memory 只读检索。"""
    from .snapshot_sources import profile_entries

    if snapshot.profile.get("profile_schema_version") == "v3":
        for entry in profile_entries(snapshot):
            record_id = str(uuid5(NAMESPACE_URL, entry["source_id"]))
            if session.get(RuntimeMemoryRow, record_id) is not None:
                continue
            session.add(
                RuntimeMemoryRow(
                    id=record_id,
                    scope="world",
                    snapshot_id=snapshot.id,
                    subject="目标人物",
                    predicate=entry["field_path"],
                    object=entry["text"],
                    summary=entry["text"],
                    status="asserted",
                    source_ids=[entry["source_id"]],
                    confidence=0.7,
                )
            )
        session.flush()
        return
    # 只有 v2 规范事实投影经过了栏目 Contract 和本地来源边界校验，才可以成为
    # Runtime 的 confirmed 世界记忆；遗留 Profile 仍可展示，但绝不自动升级。
    if snapshot.profile.get("profile_schema_version") != "v2":
        return
    allowed_source_ids = set(snapshot.source_message_ids or [])
    for field in (
        "identity",
        "work_and_education",
        "places",
        "social_relationships",
        "preferences",
        "recurring_activities",
        "routine_summary",
        "relationship_with_user",
        "important_events",
        "life_phases",
    ):
        for summary, source_ids in _profile_statements(snapshot.profile.get(field)):
            verified_source_ids = [item for item in source_ids if item in allowed_source_ids]
            if not verified_source_ids:
                continue
            session.add(
                RuntimeMemoryRow(
                    id=str(uuid4()),
                    scope="world",
                    snapshot_id=snapshot.id,
                    subject="目标人物",
                    predicate=field,
                    object=str(summary)[:500],
                    summary=str(summary)[:2000],
                    status="confirmed",
                    source_ids=verified_source_ids,
                    confidence=1.0,
                )
            )


def _profile_statements(value: object) -> list[tuple[str, list[str]]]:
    """从编译档案提取最小可检索陈述，保留每条原始 message ID。"""

    if isinstance(value, dict):
        text = value.get("statement") or value.get("text") or value.get("summary")
        if isinstance(text, str) and text.strip():
            sources = value.get("evidence_message_ids", value.get("source_message_ids", []))
            return [
                (
                    text.strip(),
                    [item for item in sources if isinstance(item, str)]
                    if isinstance(sources, list)
                    else [],
                )
            ]
        statements: list[tuple[str, list[str]]] = []
        for nested in value.values():
            statements.extend(_profile_statements(nested))
        return statements
    if isinstance(value, list):
        nested_statements: list[tuple[str, list[str]]] = []
        for nested in value:
            nested_statements.extend(_profile_statements(nested))
        return nested_statements
    if isinstance(value, str) and value.strip():
        return [(value.strip(), [])]
    return []
