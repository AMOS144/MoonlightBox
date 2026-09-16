from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from moonlightbox.world.person_world.contracts import (
    SECTION_CONTRACTS,
    IdentitySectionResult,
    PracticesSectionResult,
    get_section_contract,
    section_result_model,
)
from moonlightbox.world.person_world.contracts.validation import validate_section_result
from moonlightbox.world.person_world.investigation import build_investigation_report
from moonlightbox.world.person_world.schemas import EvidenceMessage
from moonlightbox.world.person_world.tools.current_profile import _profile_sections


def _evidence() -> dict[str, object]:
    return {
        "statement": "我叫洪欣羽",
        "speaker_message_id": "message-1",
        "evidence_message_ids": ["message-1"],
        "subject_binding": {
            "grammatical_subject": "我",
            "resolved_participant_id": "target-1",
            "resolution_basis": "first_person_speaker",
        },
        "assertion_kind": "self_fact",
        "temporal_status": "current",
        "valid_from": datetime(2026, 1, 1, tzinfo=UTC),
    }


def test_catalog_has_exactly_the_seven_top_level_fact_domains() -> None:
    domains = [contract.domain for contract in SECTION_CONTRACTS]

    assert domains == [
        "identity",
        "life_context",
        "social_world",
        "agency",
        "practices",
        "life_course",
        "relationship_with_user",
    ]
    assert get_section_contract("practices").allowed_fact_types == (
        "recurring_activity",
        "temporal_rhythm",
    )
    with pytest.raises(ValueError, match="未知"):
        get_section_contract("routine_summary.workdays")


def test_section_results_are_domain_specific_and_keep_subject_binding() -> None:
    identity = IdentitySectionResult.model_validate(
        {
            "identifiers": [
                {
                    **_evidence(),
                    "identifier": "洪欣羽",
                    "identifier_kind": "name",
                }
            ]
        }
    )
    assert identity.section == "identity"
    assert identity.identifiers[0].subject_binding.resolved_participant_id == "target-1"

    practices = PracticesSectionResult.model_validate(
        {
            "recurring_activities": [
                {
                    **_evidence(),
                    "activity": "下班后跑步",
                    "recurrence_basis": "observed_pattern",
                    "occurrence_dates": ["2026-01-01", "2026-01-03", "2026-01-06"],
                }
            ]
        }
    )
    assert practices.recurring_activities[0].recurrence_basis == "observed_pattern"
    assert section_result_model("identity") is IdentitySectionResult
    assert section_result_model("practices") is PracticesSectionResult
    with pytest.raises(ValueError, match="未知"):
        section_result_model("identity.names")


def test_contextual_inference_requires_multiple_sources_and_an_explicit_rationale() -> None:
    """这是结构审计，不分析中文正文；语义判断仍归七个栏目 Agent。"""

    inferred = IdentitySectionResult.model_validate(
        {
            "identifiers": [
                {
                    **_evidence(),
                    "statement": "多段对话均将洪欣羽指向同一目标人物",
                    "identifier": "洪欣羽",
                    "identifier_kind": "name",
                    "evidence_message_ids": ["message-1", "message-2"],
                    "derivation": "inferred",
                    "inference_rationale": "两条已核对的自指消息在不同上下文中使用同一姓名。",
                }
            ]
        }
    )

    assert inferred.identifiers[0].derivation == "inferred"
    assert len(inferred.identifiers[0].evidence_message_ids) == 2

    with pytest.raises(ValueError, match="至少需要两条"):
        IdentitySectionResult.model_validate(
            {
                "identifiers": [
                    {
                        **_evidence(),
                        "identifier": "洪欣羽",
                        "identifier_kind": "name",
                        "derivation": "inferred",
                        "inference_rationale": "只写一条来源不应通过。",
                    }
                ]
            }
        )




def test_investigation_report_is_a_separate_audit_asset() -> None:
    identity = IdentitySectionResult.model_validate(
        {
            "identifiers": [
                {
                    **_evidence(),
                    "identifier": "洪欣羽",
                    "identifier_kind": "name",
                }
            ],
            "unresolved_questions": ["是否还有常用别名？"],
        }
    )
    report = build_investigation_report([identity], errors={"agency": "cloud_error"})

    states = {item.section: item.state for item in report.section_statuses}
    assert states["identity"] == "completed"
    assert states["agency"] == "cloud_error"
    assert states["practices"] == "not_started"
    assert report.unresolved_questions == ["是否还有常用别名？"]


def test_investigation_report_distinguishes_harness_budget_from_cloud_failure() -> None:
    report = build_investigation_report([], errors={"identity": "tool_result_context_limit"})

    states = {item.section: item.state for item in report.section_statuses}
    assert states["identity"] == "budget_exhausted"


def test_contract_rejects_user_bound_fact_from_target_practices_without_reading_text() -> None:
    """主体归属由 Agent 判断，后端只守住已解析参与者的持久化边界。"""

    practices = PracticesSectionResult.model_validate(
        {
            "recurring_activities": [
                {
                    **_evidence(),
                    "activity": "工作日上午十点上班",
                    "recurrence_basis": "explicit_statement",
                    "occurrence_dates": [],
                    "subject_binding": {
                        "grammatical_subject": "你",
                        "resolved_participant_id": "user-1",
                        "resolution_basis": "second_person_addressee",
                    },
                }
            ]
        }
    )
    evidence = [
        EvidenceMessage.model_validate(
            {
                "message_id": "message-1",
                "document_id": "bundle-1.txt",
                "bundle_id": "bundle-1",
                "ordinal": 1,
                "timestamp": "2026-04-22T17:11:43+00:00",
                "participant_id": "target-1",
                "participant_name": "洪欣羽",
                "participant_role": "target",
                "kind": "text",
                "content": "这个笨入早上十点才上班！",
                "is_primary_match": True,
            },
            strict=False,
        )
    ]

    validated, rejections = validate_section_result(
        practices,
        evidence=evidence,
        target_participant_id="target-1",
        user_participant_id="user-1",
        timezone="Asia/Shanghai",
    )

    assert validated.recurring_activities == []
    assert rejections == [
        {
            "field": "recurring_activities",
            "index": 0,
            "reason": "subject_binding_outside_section_scope",
        }
    ]


def test_current_section_tool_reads_v2_fact_domains_not_legacy_projection() -> None:
    profile = SimpleNamespace(
        profile_schema_version="v2",
        profile_v2={
            "identity": {"identifiers": [{"identifier": "洪欣羽"}]},
            "life_context": {"work_and_learning": [{"detail": "在新公司工作"}]},
        },
    )

    sections = _profile_sections(profile)  # type: ignore[arg-type]

    assert sections["identity"] == {"identifiers": [{"identifier": "洪欣羽"}]}
    assert sections["life_context"] == {"work_and_learning": [{"detail": "在新公司工作"}]}
    assert sections["agency"] == {}
