"""旧 PersonWorldProfile 到 v2 的候选映射审计。

迁移不解释旧文案，也不把它们直接写入 v2。它仅保存旧字段的稳定位置、原有来源和可选
去向，要求新的七栏目 Agent 重新回到原始消息核验后才可生成新的 AtomicWorldClaim。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from moonlightbox.world.models import PersonWorldProfile


@dataclass(frozen=True, slots=True)
class LegacyStatementAudit:
    """一个旧陈述的审计定位，不复制可能敏感的正文。"""

    legacy_path: str
    legacy_index: int
    candidate_v2_paths: tuple[str, ...]
    source_message_ids: tuple[str, ...]
    state: str

    def as_dict(self) -> dict[str, object]:
        return {
            "legacy_path": self.legacy_path,
            "legacy_index": self.legacy_index,
            "candidate_v2_paths": list(self.candidate_v2_paths),
            "source_message_ids": list(self.source_message_ids),
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class LegacyProfileMigrationAudit:
    """v1 基线的只读迁移审计；它不是人物事实也不是发布许可。"""

    source_profile_id: str
    candidates: tuple[LegacyStatementAudit, ...]
    unresolved_questions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_profile_id": self.source_profile_id,
            "candidate_count": len(self.candidates),
            "candidates": [item.as_dict() for item in self.candidates],
            "unresolved_questions": list(self.unresolved_questions),
        }


_DIRECT_MAPPINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity.names", ("identity.identifiers",)),
    ("identity.aliases", ("identity.identifiers",)),
    ("identity.self_descriptions", ("identity.self_descriptions", "identity.self_narratives")),
    ("work_and_education", ("life_context.work_and_learning",)),
    ("places", ("life_context.places_and_environment",)),
    ("social_relationships", ("social_world.ties",)),
    ("recurring_activities", ("practices.recurring_activities",)),
    ("relationship_with_user.overview", ("relationship_with_user.standing",)),
    ("relationship_with_user.changes_over_time", ("relationship_with_user.history",)),
)

_AMBIGUOUS_MAPPINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "identity.roles",
        ("life_context.work_and_learning", "life_context.home_and_care", "social_world.ties"),
    ),
    (
        "preferences",
        (
            "agency.preferences",
            "agency.values_and_interpretations",
            "agency.goals_and_commitments",
        ),
    ),
    (
        "routine_summary.workdays",
        ("practices.recurring_activities", "practices.temporal_rhythms"),
    ),
    (
        "routine_summary.weekends",
        ("practices.recurring_activities", "practices.temporal_rhythms"),
    ),
    (
        "routine_summary.other_patterns",
        ("practices.recurring_activities", "practices.temporal_rhythms"),
    ),
    (
        "life_phases",
        ("life_course.episodes", "life_course.transitions", "life_course.trajectories"),
    ),
    (
        "important_events",
        ("life_course.episodes", "life_course.transitions", "life_course.trajectories"),
    ),
)


def audit_legacy_profile(profile: PersonWorldProfile) -> LegacyProfileMigrationAudit | None:
    """返回 v1 的候选映射，v2 Profile 则无需迁移。

    映射只遍历 JSON 的既有列表结构，不读取 `text` 字段判断事实含义；同一份旧数据可以
    因此重复得到同一审计结果，不受模型或检索排序影响。
    """

    if profile.profile_schema_version == "v2":
        return None

    candidates: list[LegacyStatementAudit] = []
    unresolved: list[str] = []
    for path, destinations in _DIRECT_MAPPINGS:
        candidates.extend(
            _audits_for_values(profile, path, destinations, state="requires_reverification")
        )
    for path, destinations in _AMBIGUOUS_MAPPINGS:
        rows = _audits_for_values(profile, path, destinations, state="ambiguous_destination")
        candidates.extend(rows)
        if rows:
            unresolved.append(f"旧档案 {path} 有 {len(rows)} 条陈述需要重新判断唯一 v2 栏目。")
    unresolved.extend(
        f"旧档案 unresolved_candidates[{index}] 仅保留为待调查问题，不能发布为人物事实。"
        for index, _ in enumerate(_values_at(profile, "unresolved_candidates"))
    )
    return LegacyProfileMigrationAudit(
        source_profile_id=profile.id,
        candidates=tuple(candidates),
        unresolved_questions=tuple(unresolved),
    )


def _audits_for_values(
    profile: PersonWorldProfile,
    path: str,
    destinations: tuple[str, ...],
    *,
    state: str,
) -> list[LegacyStatementAudit]:
    return [
        LegacyStatementAudit(
            legacy_path=path,
            legacy_index=index,
            candidate_v2_paths=destinations,
            source_message_ids=tuple(_source_ids(value)),
            state=state,
        )
        for index, value in enumerate(_values_at(profile, path))
    ]


def _values_at(profile: PersonWorldProfile, path: str) -> list[object]:
    value: object = {
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
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return value if isinstance(value, list) else []
        value = value[part]
    return value if isinstance(value, list) else []


def _source_ids(value: object) -> Iterable[str]:
    if not isinstance(value, dict):
        return ()
    source_ids = value.get("source_message_ids")
    if not isinstance(source_ids, list):
        return ()
    return tuple(item for item in source_ids if isinstance(item, str))
