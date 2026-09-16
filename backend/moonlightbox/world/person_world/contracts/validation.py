"""七栏目输出的结构与证据边界校验。

本模块故意不读取、分词、正则匹配或解释消息正文。"主体到底是谁"和"是否真是规律"由
同一个 Section Agent 在完整上下文中判断；这里仅拒绝不存在的证据、越过参与者边界的输出，
以及各 Result 模型可机械验证的时间/关系结构错误。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..schemas import EvidenceMessage
from .sections import SectionResult


def validate_section_result(
    result: SectionResult,
    *,
    evidence: list[EvidenceMessage],
    target_participant_id: str,
    user_participant_id: str,
    timezone: str,
) -> tuple[SectionResult, list[dict[str, object]]]:
    """保留可安全持久化的事实，并返回逐条拒绝的非语义原因。

    每个事实都需要真实的 speaker/evidence message，且通常必须绑定目标人物。用户关系栏目
    记录的是一个 dyad，允许其事实证据由用户或目标人物一端发出，但仍只能绑定这两位参与者。
    """

    by_message_id = {item.message_id: item for item in evidence}
    payload = result.model_dump(mode="python")
    rejections: list[dict[str, object]] = []
    for key, value in list(payload.items()):
        if key in {"section", "unresolved_questions"} or not isinstance(value, list):
            continue
        accepted: list[dict[str, object]] = []
        for index, raw in enumerate(value):
            if not isinstance(raw, dict):
                rejections.append({"field": key, "index": index, "reason": "invalid_fact_shape"})
                continue
            reason = _admission_error(
                raw,
                section=result.section,
                evidence_by_id=by_message_id,
                target_participant_id=target_participant_id,
                user_participant_id=user_participant_id,
                timezone=timezone,
            )
            if reason is None:
                accepted.append(raw)
            else:
                rejections.append({"field": key, "index": index, "reason": reason})
        payload[key] = accepted
    questions = payload.get("unresolved_questions")
    if isinstance(questions, list) and rejections:
        payload["unresolved_questions"] = list(
            dict.fromkeys([*questions, "部分候选未通过来源或结构边界校验。"])
        )[:20]
    return type(result).model_validate(payload), rejections


def _admission_error(
    fact: dict[str, Any],
    *,
    section: str,
    evidence_by_id: dict[str, EvidenceMessage],
    target_participant_id: str,
    user_participant_id: str,
    timezone: str,
) -> str | None:
    speaker = fact.get("speaker_message_id")
    evidence_ids = fact.get("evidence_message_ids")
    if not isinstance(speaker, str) or speaker not in evidence_by_id:
        return "speaker_message_not_in_section_ledger"
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or any(not isinstance(item, str) or item not in evidence_by_id for item in evidence_ids)
    ):
        return "evidence_message_not_in_section_ledger"
    binding = fact.get("subject_binding")
    participant_id = binding.get("resolved_participant_id") if isinstance(binding, dict) else None
    allowed = (
        {target_participant_id, user_participant_id}
        if section == "relationship_with_user"
        else {target_participant_id}
    )
    if not isinstance(participant_id, str) or participant_id not in allowed:
        return "subject_binding_outside_section_scope"
    if fact.get("recurrence_basis") == "observed_pattern":
        observed_dates = {
            _local_date(evidence_by_id[item].timestamp, timezone) for item in evidence_ids
        }
        declared_dates = fact.get("occurrence_dates")
        if len(observed_dates) < 3:
            return "observed_pattern_needs_three_local_dates"
        declared_date_set = (
            set(item for item in declared_dates if isinstance(item, str))
            if isinstance(declared_dates, list)
            else None
        )
        if declared_date_set is None or not declared_date_set.issubset(observed_dates):
            return "occurrence_dates_not_supported_by_evidence"
    if "before_evidence_message_ids" in fact:
        before_ids = fact.get("before_evidence_message_ids")
        after_ids = fact.get("after_evidence_message_ids")
        if not _all_in_evidence(before_ids, evidence_by_id) or not _all_in_evidence(
            after_ids, evidence_by_id
        ):
            return "transition_boundary_evidence_not_in_section_ledger"
    return None


def _all_in_evidence(value: object, evidence_by_id: dict[str, EvidenceMessage]) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and item in evidence_by_id for item in value
    )


def _local_date(value: datetime, timezone: str) -> str:
    return value.astimezone(ZoneInfo(timezone)).date().isoformat()
