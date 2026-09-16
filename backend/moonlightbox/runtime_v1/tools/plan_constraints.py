"""DayPlanAgent 的已确认计划约束工具。"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.tools import StructuredTool

from ..context_views import day_plan_context_payload
from ..plan_context import DayPlanContext
from ..schemas import StrictModel


class GetPlanConstraintsArgs(StrictModel):
    """目标日期和分支已经由 Runtime 绑定，因此不接受模型参数。"""


_DESCRIPTION = """读取目标日期已经确认的承诺、不可改写时段和可选旧计划参考。

工具不接受日期、分支或用户参数，这些身份由 Runtime 绑定。只有 locked_blocks 和明确
承诺是硬约束；prior_plan 的未发生部分只是参考，并受 reuse_policy 限制。"""


def build_plan_constraints_tool(context: DayPlanContext) -> StructuredTool:
    def invoke(**kwargs: Any) -> dict[str, Any]:
        GetPlanConstraintsArgs.model_validate(kwargs)
        source_ids = list(context.branch_evidence.get("request_source_event_ids", []))
        for item in context.hard_constraints.get("active_commitments", []):
            if isinstance(item, dict) and isinstance(item.get("source_ids"), list):
                source_ids.extend(
                    str(value) for value in item["source_ids"] if isinstance(value, str)
                )
        for block in context.hard_constraints.get("locked_blocks", []):
            if isinstance(block, dict) and isinstance(block.get("evidence_ids"), list):
                source_ids.extend(
                    str(value) for value in block["evidence_ids"] if isinstance(value, str)
                )
        prior_plan = context.prior_plan
        if isinstance(prior_plan, dict) and isinstance(prior_plan.get("blocks"), list):
            for block in prior_plan["blocks"]:
                if isinstance(block, dict) and isinstance(block.get("evidence_ids"), list):
                    source_ids.extend(
                        str(value) for value in block["evidence_ids"] if isinstance(value, str)
                    )
        return {
            "tool_name": "get_plan_constraints",
            "scope": "branch",
            "as_of": context.generated_at.isoformat(),
            "source_ids": list(dict.fromkeys(source_ids)),
            "truncated": False,
            # 复用模型上下文投影，确保计划行 ID、block ID 和版本号不会从工具旁路泄漏。
            "data": {
                "hard_constraints": day_plan_context_payload(context)["hard_constraints"],
                "prior_plan": day_plan_context_payload(context)["prior_plan"],
            },
        }

    return StructuredTool.from_function(
        name="get_plan_constraints",
        description=_DESCRIPTION,
        func=invoke,
        args_schema=cast(Any, GetPlanConstraintsArgs),
    )
