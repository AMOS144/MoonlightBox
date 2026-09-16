"""PersonWorld 已发布栏目与人工纠正的只读工具。"""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from moonlightbox.world.models import (
    PersonWorldProfile,
    WorldCorrection,
    WorldPublication,
)


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GetCurrentProfileSectionArgs(_StrictArgs):
    section: Literal[
        "identity",
        "life_context",
        "social_world",
        "agency",
        "practices",
        "life_course",
        "relationship_with_user",
    ] = Field(
        description="栏目键：identity、life_context、social_world、agency、practices、life_course、relationship_with_user；读取当前绑定版本已有结论",
    )


class GetActiveCorrectionsArgs(_StrictArgs):
    """只允许结构化缩小范围，不能把自然语言交给后端猜语义。"""

    primary_domain: str | None = Field(
        default=None,
        max_length=80,
        description="纠正 scope 中的栏目键，精确匹配；未知时省略，不写自然语言主题",
    )
    fact_type: str | None = Field(
        default=None,
        max_length=80,
        description="已有纠正 scope.fact_type 的精确值；省略不按事实类型筛选",
    )
    subject_kind: (
        Literal["target_person", "user", "graph_entity", "target_user_pair", "unknown"] | None
    ) = Field(default=None, description="纠正所针对的事实主体，不等于消息发送者；省略不筛选主体")
    related_fact_id: str | None = Field(
        default=None,
        max_length=80,
        description="已提供的 related_fact_id，精确关联一条已有事实；未知则省略",
    )
    limit: int = Field(
        default=40,
        ge=1,
        le=100,
        description="最多读取的纠正记录数；只读取当前有效及本次允许的已批准纠正，不支持历史时间查询",
    )


_CURRENT_DESCRIPTION = """读取当前绑定版本中一个人物世界栏目及其 Publication 版本。返回的是已有
结论，不是新事实；Agent 必须用原始消息复核后才可保留或修改。"""
_CORRECTIONS_DESCRIPTION = """读取当前项目仍生效的用户纠正。筛选只依据持久化的结构字段，
不解析中文、不卡关键词，也不自行判定纠正与候选事实的语义冲突。"""


def build_current_profile_tools(
    session: Session,
    *,
    project_id: str,
    graph_version_id: str,
    correction_change_set_ids: set[str] | None = None,
    node_scope: dict | None = None,
    profile_snapshot: dict | None = None,
) -> dict[str, StructuredTool]:
    """把项目、图版本和可见纠正封在工具闭包内。"""

    visible_change_sets = correction_change_set_ids or set()

    def get_current_profile_section(**kwargs: Any) -> dict[str, object]:
        args = GetCurrentProfileSectionArgs.model_validate(kwargs)
        if node_scope is not None:
            from copy import deepcopy

            return {
                "section": args.section,
                "profile_id": None,
                "publication_id": None,
                "profile_graph_version_id": graph_version_id,
                "content": deepcopy((profile_snapshot or {}).get(args.section)),
                "scope": "frozen_node",
            }
        publication = session.scalar(
            select(WorldPublication)
            .where(
                WorldPublication.project_id == project_id,
                WorldPublication.status == "active",
                WorldPublication.node_boundary_hash.is_(None),
            )
            .order_by(WorldPublication.published_at.desc(), WorldPublication.id.desc())
        )
        profile = session.scalar(
            select(PersonWorldProfile)
            .where(PersonWorldProfile.node_boundary_hash.is_(None))
            .where(
                PersonWorldProfile.project_id == project_id,
                PersonWorldProfile.graph_version_id == graph_version_id,
            )
        )
        # 候选图重编译时它尚没有自己的 Profile；此时应读取同一 Publication 的
        # 基线 v2 档案，让栏目 Agent 能明确判断“保留、删除还是替换”，而不是把
        # 一份已批准 Patch 重新当成从零抽取任务。
        if profile is None and publication is not None:
            profile = session.get(PersonWorldProfile, publication.profile_id)
        if profile is None:
            return {
                "section": args.section,
                "profile_id": None,
                "publication_id": publication.id if publication is not None else None,
                "profile_graph_version_id": None,
                "content": None,
            }
        payload = _profile_sections(profile)
        return {
            "section": args.section,
            "profile_id": profile.id,
            "publication_id": publication.id if publication is not None else None,
            "profile_graph_version_id": profile.graph_version_id,
            "content": payload.get(args.section),
        }

    def get_active_corrections(**kwargs: Any) -> list[dict[str, object]]:
        args = GetActiveCorrectionsArgs.model_validate(kwargs)
        corrections = list(
            session.scalars(
                select(WorldCorrection)
                .where(
                    WorldCorrection.project_id == project_id,
                    WorldCorrection.corrected_interpretation["scope"]["node_scope"][
                        "preview_hash"
                    ].as_string()
                    == (node_scope or {}).get("preview_hash"),
                    or_(
                        WorldCorrection.status == "active",
                        WorldCorrection.approved_change_set_id.in_(visible_change_sets),
                    ),
                )
                .order_by(WorldCorrection.approved_at.asc(), WorldCorrection.created_at.asc())
                .limit(args.limit)
            )
        )
        # 旧数据并不一定携带 v2 的结构化 scope；没有这些字段时只会被保守返回，
        # 不会通过内容匹配假装知道它属于哪一栏。
        result: list[dict[str, object]] = []
        for correction in corrections:
            corrected = dict(correction.corrected_interpretation or {})
            scope = corrected.get("scope")
            structured_scope = dict(scope) if isinstance(scope, dict) else {}
            from ..node_scope import same_node_scope

            if not same_node_scope(structured_scope.get("node_scope"), node_scope):
                continue
            primary_domains = _string_list(structured_scope.get("primary_domains"))
            primary_domain = structured_scope.get("primary_domain")
            if (
                args.primary_domain
                and args.primary_domain != primary_domain
                and args.primary_domain not in primary_domains
            ):
                continue
            if args.fact_type and structured_scope.get("fact_type") != args.fact_type:
                continue
            if args.subject_kind and structured_scope.get("subject_kind") != args.subject_kind:
                continue
            if (
                args.related_fact_id
                and structured_scope.get("related_fact_id") != args.related_fact_id
            ):
                continue
            result.append(
                {
                    "correction_id": correction.id,
                    "status": correction.status,
                    # Claim UUID 只给服务端 Scope / ContextAssembler 使用，不能作为栏目模型
                    # 的提示词上下文；模型只需要可读的结构范围与用户确认的纠正内容。
                    "scope": {
                        key: value
                        for key, value in structured_scope.items()
                        if key not in {"claim_ids", "graph_object_refs"}
                    },
                    "original_interpretation": dict(correction.original_interpretation or {}),
                    "corrected_interpretation": corrected,
                    "source_message_ids": list(correction.source_message_ids or []),
                    "approved_at": correction.approved_at.isoformat()
                    if correction.approved_at is not None
                    else None,
                }
            )
        return result

    return {
        "get_current_profile_section": StructuredTool.from_function(
            name="get_current_profile_section",
            description=_CURRENT_DESCRIPTION,
            func=get_current_profile_section,
            args_schema=cast(Any, GetCurrentProfileSectionArgs),
        ),
        "get_active_corrections": StructuredTool.from_function(
            name="get_active_corrections",
            description=_CORRECTIONS_DESCRIPTION,
            func=get_active_corrections,
            args_schema=cast(Any, GetActiveCorrectionsArgs),
        ),
    }


def _profile_sections(profile: PersonWorldProfile) -> dict[str, object]:
    """集中维护当前 Profile 的栏目投影，迁移期不让 Agent 反射数据库字段。"""

    if profile.profile_schema_version == "v3":
        return dict(profile.profile_v3 or {})
    # v2 是历史档案的规范人物事实结构。这里只投影已持久化的顶层键。
    # 投影给对应 Section Agent，不解释任何字段正文，也不把审计报告混进事实。
    if profile.profile_schema_version == "v2" and isinstance(profile.profile_v2, dict):
        v2 = profile.profile_v2
        return {
            "identity": _v2_section(v2, "identity"),
            "life_context": _v2_section(v2, "life_context"),
            "social_world": _v2_section(v2, "social_world"),
            "agency": _v2_section(v2, "agency"),
            "practices": _v2_section(v2, "practices"),
            "life_course": _v2_section(v2, "life_course"),
            "relationship_with_user": _v2_section(v2, "relationship_with_user"),
        }

    return {
        "identity": dict(profile.identity or {}),
        "life_context": {
            "work_and_learning": list(profile.work_and_education or []),
            "places_and_environment": list(profile.places or []),
        },
        "social_world": {"ties": list(profile.social_relationships or [])},
        "agency": {"preferences": list(profile.preferences or [])},
        "practices": {
            "recurring_activities": list(profile.recurring_activities or []),
            "temporal_rhythms": dict(profile.routine_summary or {}),
        },
        "life_course": {
            "episodes": list(profile.important_events or []),
            "transitions": list(profile.life_phases or []),
        },
        "relationship_with_user": dict(profile.relationship_with_user or {}),
    }


def _v2_section(payload: dict[str, object], section: str) -> dict[str, object]:
    value = payload.get(section)
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str)))
