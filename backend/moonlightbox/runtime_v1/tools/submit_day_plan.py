"""DayPlan 提案提交工具：统一处理计划参数、锁定块及来源的可修复错误。"""

import json
from typing import Any

from moonlightbox.agent_runtime.contracts import SubmissionContext
from moonlightbox.agent_runtime.input_contracts import input_model
from moonlightbox.agent_runtime.submission import result_submission_tool

from ..plan_context import DayPlanContext
from ..schemas import DayPlanProposal, DayPlanTurn

DayPlanInput = input_model(
    DayPlanTurn, omit=lambda cls, name, field: cls is DayPlanProposal and name == "plan_date"
)


def build_submit_day_plan_tool(context):
    """协作回复直接交还工作流；日程提案先校验再交付 Executor。"""

    def validate(value, validation):
        if value.completion is not None:
            return (
                None
                if context.allow_no_change
                else "本次必须准备日程，不能用 completion 代替；提交计划或说明需要澄清的问题"
            )
        if value.reply is not None:
            return None
        return validate_plan_proposal(
            value.proposal, context=context, validation_context=validation
        )

    def bind(value):
        data = value.model_dump(mode="json")
        if data.get("proposal") is not None:
            data["proposal"]["plan_date"] = context.target_date.isoformat()
        return DayPlanTurn.model_validate_json(json.dumps(data))

    return result_submission_tool("submit_day_plan", DayPlanInput, validate, result_adapter=bind)


def validate_plan_proposal(
    proposal: DayPlanProposal | None,
    *,
    context: DayPlanContext,
    validation_context: SubmissionContext,
) -> str | None:
    """验收只消费 Tool provenance/已确认约束，绝不从聊天正文做正则猜测。"""

    if proposal is None:
        return "invalid_day_plan_proposal"
    available = _context_source_ids(context.model_dump(mode="json"))
    available.update(
        item["source_id"]
        for item in context.origin_projection.get("profile_references", [])
        if isinstance(item, dict) and "source_id" in item
    )
    for ref in validation_context.source_refs:
        if ref.startswith("source_ids:"):
            available.add(ref.removeprefix("source_ids:"))
        elif ref.startswith("evidence_ids:"):
            available.add(ref.removeprefix("evidence_ids:"))
    if proposal.plan_date != context.target_date:
        return "plan_date 必须等于本次目标日期"
    from .plan_validation import validate_plan_structure

    try:
        validate_plan_structure(proposal)
    except ValueError as error:
        return str(error)
    for locked in context.hard_constraints.get("locked_blocks", []):
        fields = ("start", "end", "activity", "location_role", "default_availability")
        if not any(
            all(getattr(block, key) == locked.get(key) for key in fields)
            for block in proposal.blocks
        ):
            return "不能修改 locked_blocks 已发生的时间、活动或状态"
    requested = {
        source_id
        for block in proposal.blocks
        for source_id in block.evidence_ids
        if isinstance(source_id, str)
    }
    if requested - available:
        return (
            "proposal.blocks.evidence_ids 存在未提供引用："
            + ", ".join(sorted(requested - available))
            + "；使用已返回引用，不猜测 ID。"
        )
    return None


def _context_source_ids(context: dict[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    for source_id in context.get("branch_evidence", {}).get("request_source_event_ids", []):
        if isinstance(source_id, str):
            source_ids.add(source_id)
    for commitment in context.get("hard_constraints", {}).get("active_commitments", []):
        if isinstance(commitment, dict):
            source_ids.update(
                source_id
                for source_id in commitment.get("source_ids", [])
                if isinstance(source_id, str)
            )
    for block in context.get("hard_constraints", {}).get("locked_blocks", []):
        if isinstance(block, dict):
            source_ids.update(
                source_id
                for source_id in block.get("evidence_ids", [])
                if isinstance(source_id, str)
            )
    return source_ids
