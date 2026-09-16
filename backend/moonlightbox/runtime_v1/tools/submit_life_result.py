"""生活结果由原生提交工具检查，错误回传模型修正。"""

from datetime import datetime

from moonlightbox.agent_runtime.submission import result_submission_tool

from ..life_events.commit import validate_life_result


def build_submit_life_result_tool(schema, *, context, validate_plan_sources=None):
    def validate(value, validation):
        try:
            validate_life_result(
                value,
                now=datetime.fromisoformat(context["virtual_now"]),
                plans=context.get("plans_by_date", {}),
                requests=context.get("life_requests", []),
            )
            if validate_plan_sources:
                validate_plan_sources(
                    {
                        ref
                        for plan in value.plan_proposals
                        for block in plan.blocks
                        for ref in block.evidence_ids
                    }
                )
        except ValueError as error:
            return str(error)
        return None

    return result_submission_tool("submit_life_result", schema, validate)
