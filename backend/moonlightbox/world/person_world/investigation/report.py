"""七栏目调查状态到审计报告的确定性汇总。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import Field

from ..contracts.catalog import SECTION_CONTRACTS
from ..contracts.sections import SectionResult
from ..schemas import StrictModel

SectionState = Literal[
    "completed",
    "completed_without_evidence",
    "needs_more_research",
    "cloud_error",
    "schema_error",
    "budget_exhausted",
    "not_started",
]


class SectionInvestigationStatus(StrictModel):
    section: str
    state: SectionState
    accepted_fact_count: int = Field(ge=0)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=20)
    error_code: str | None = Field(default=None, max_length=80)
    trace_refs: list[str] = Field(default_factory=list, max_length=30)


class PersonWorldInvestigationReport(StrictModel):
    """伴随草稿的审计资产；绝不可并入 PersonWorldProfileV2。"""

    section_statuses: list[SectionInvestigationStatus]
    unresolved_questions: list[str] = Field(default_factory=list, max_length=140)
    cloud_and_schema_errors: list[dict[str, str]] = Field(default_factory=list, max_length=20)
    trace_refs: list[str] = Field(default_factory=list, max_length=200)
    # v1 迁移的候选去向和待复核项属于审计资产，绝不成为第八个事实栏目。
    legacy_migration_audit: dict[str, object] | None = None


def build_investigation_report(
    results: Iterable[SectionResult],
    *,
    errors: Mapping[str, str] | None = None,
    trace_refs: Mapping[str, Iterable[str]] | None = None,
) -> PersonWorldInvestigationReport:
    """按固定七栏目汇总，不以是否有事实来伪造调查是否成功。"""

    by_section: dict[str, SectionResult] = {}
    for result in results:
        if result.section in by_section:
            raise ValueError(f"调查审计收到重复栏目: {result.section}")
        by_section[result.section] = result
    error_by_section = dict(errors or {})
    trace_by_section = {
        section: list(values) for section, values in (trace_refs or {}).items()
    }
    statuses: list[SectionInvestigationStatus] = []
    all_questions: list[str] = []
    safe_errors: list[dict[str, str]] = []
    all_traces: list[str] = []
    for contract in SECTION_CONTRACTS:
        section = contract.domain
        result = by_section.get(section)
        error = error_by_section.get(section)
        traces = trace_by_section.get(section, [])
        if error is not None:
            state: SectionState = (
                "schema_error"
                if error == "schema_error"
                else "budget_exhausted"
                if error
                in {
                    "tool_result_context_limit",
                    "tool_call_safety_limit",
                    "emergency_model_step_limit",
                    "wall_deadline",
                }
                else "cloud_error"
            )
            questions: list[str] = []
            fact_count = 0
            safe_errors.append({"section": section, "code": error})
        elif result is None:
            state = "not_started"
            questions = []
            fact_count = 0
        else:
            questions = list(result.unresolved_questions)
            fact_count = _fact_count(result)
            if fact_count:
                state = "completed"
            elif questions:
                state = "needs_more_research"
            else:
                state = "completed_without_evidence"
        statuses.append(
            SectionInvestigationStatus(
                section=section,
                state=state,
                accepted_fact_count=fact_count,
                unresolved_questions=questions,
                error_code=error,
                trace_refs=traces,
            )
        )
        all_questions.extend(questions)
        all_traces.extend(traces)
    return PersonWorldInvestigationReport(
        section_statuses=statuses,
        unresolved_questions=list(dict.fromkeys(all_questions)),
        cloud_and_schema_errors=safe_errors,
        trace_refs=list(dict.fromkeys(all_traces)),
    )


def _fact_count(result: SectionResult) -> int:
    """结果模型只包含事实数组与 unresolved_questions；计数不解释事实正文。"""

    payload = result.model_dump(mode="python")
    return sum(
        len(value)
        for key, value in payload.items()
        if key not in {"section", "unresolved_questions"} and isinstance(value, list)
    )
