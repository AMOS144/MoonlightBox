"""DayPlan 的领域适配：装配上下文与工具，循环控制统一交给 Harness。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from moonlightbox.agent_runtime import (
    AgentLoopController,
    AgentSpec,
    RunScope,
)
from moonlightbox.agent_runtime.contracts import (
    AgentExecutionRequest,
    ChatModel,
)
from moonlightbox.agent_runtime.policy import (
    DAY_PLAN_RUNTIME_POLICY,
    AgentRuntimePolicy,
    controller_budget,
)

from .agent_catalog import load_agent_definition, select_declared_tools
from .agent_support import (
    prompt_hash,
)
from .context_views import day_plan_context_payload
from .plan_context import DayPlanContext
from .prompting import assemble_agent_prompt
from .schemas import DayPlanProposal, DayPlanTurn, PeerReply
from .tools.plan_work import PlanningToolbox
from .tools.submit_day_plan import build_submit_day_plan_tool


@dataclass(frozen=True)
class DayPlanRun:
    proposal: DayPlanProposal | None
    terminal_reason: str
    draft: str | None = None
    validation_error: str | None = None
    reply: PeerReply | None = None
    working_state: dict | None = None
    completion: str | None = None


class DayPlanAgent:
    """只提出可验证计划，不拥有 RuntimeDayPlan 或数据库写权限。"""

    def advance_life(self, **kwargs):
        """同模型、同 Harness 的生活任务，不是隐藏子 Agent。"""
        from .life_events.agent_mode import run_mode
        from .life_events.contracts import LifeAdvanceDecision

        return run_mode(
            self,
            name="day_planner",
            definition_name="day_planner_life_events",
            schema=LifeAdvanceDecision,
            **kwargs,
        )

    def __init__(
        self,
        model: ChatModel | None = None,
        *,
        runtime_policy: AgentRuntimePolicy | None = None,
        controller: AgentLoopController | None = None,
    ) -> None:
        self.model = model
        self.runtime_policy = runtime_policy or DAY_PLAN_RUNTIME_POLICY
        self.controller = controller or AgentLoopController()

    def run_with_trace(
        self,
        context: DayPlanContext,
        *,
        tools: list[BaseTool] | tuple[BaseTool, ...],
        owner_id: str | None = None,
        project_id: str | None = None,
        input_revision: int = 1,
        input_revision_resolver: Callable[[], int | None] | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
        on_status: Callable[[dict[str, Any]], None] | None = None,
    ) -> DayPlanRun:
        if self.model is None:
            return DayPlanRun(
                proposal=None,
                terminal_reason="guard",
            )
        definition = load_agent_definition("day_planner")
        toolbox = PlanningToolbox(self.runtime_policy)
        toolbox.work = context.working_state or None
        selected_tools = select_declared_tools(definition, [*tools, toolbox.work_tool()])
        registered = toolbox.register(selected_tools)
        registered += (build_submit_day_plan_tool(context),)
        assembly = assemble_agent_prompt(
            "day_planner", DayPlanTurn, tools=registered, submission_tool_name="submit_day_plan"
        )
        prompt = assembly.text
        outcome = self.controller.run(
            spec=AgentSpec(
                name="day_planner",
                prompt_version=prompt_hash(prompt),
                submission_tool_name="submit_day_plan",
                budget=controller_budget(self.runtime_policy),
                tools=registered,
                context_compactor=toolbox.compact,
                restore_tool_results=toolbox.restore,
                resume_messages=toolbox.resume_messages,
            ),
            request=AgentExecutionRequest(
                owner_type="day_plan",
                cancellation_requested=cancellation_requested,
                on_status=on_status,
                owner_id=owner_id or f"{context.branch_id}:{context.target_date.isoformat()}",
                project_id=project_id,
                input_revision=input_revision,
                scope=RunScope(
                    project_id=project_id,
                    branch_id=context.branch_id,
                    allowed_source_snapshot_id=str(context.origin_projection.get("snapshot_id", ""))
                    or None,
                    permissions=frozenset({"runtime.read_plan", "runtime.send_agent_message"}),
                ),
                messages=(
                    assembly.message(),
                    HumanMessage(content=_plan_context_message(context)),
                ),
                input_revision_resolver=input_revision_resolver,
            ),
            model=self.model,
        )
        turn = outcome.value
        proposal = turn.proposal if isinstance(turn, DayPlanTurn) else turn
        work = toolbox.work
        if isinstance(proposal, DayPlanProposal):
            work = {
                **(
                    work
                    or {
                        "established_arrangements": [],
                        "open_questions": [],
                        "next_steps": ["根据 Executor 回执确认提交或修正草稿"],
                    }
                ),
                "draft_blocks": [block.model_dump(mode="json") for block in proposal.blocks],
            }
        return DayPlanRun(
            completion=turn.completion if isinstance(turn, DayPlanTurn) else None,
            working_state=work,
            reply=turn.reply if isinstance(turn, DayPlanTurn) else None,
            proposal=proposal if isinstance(proposal, DayPlanProposal) else None,
            terminal_reason=outcome.terminal_reason,
            draft=next(
                (
                    str(m.content)
                    for m in reversed(outcome.messages)
                    if isinstance(m, AIMessage) and not m.tool_calls
                ),
                None,
            ),
            validation_error=next(
                (
                    str(m.content)
                    for m in reversed(outcome.messages)
                    if isinstance(m, HumanMessage)
                    and str(m.content).startswith("RUNTIME_VALIDATION_FEEDBACK")
                ),
                None,
            ),
        )


def _plan_context_message(context: DayPlanContext) -> str:
    return (
        "<plan_context>"
        + json.dumps(
            day_plan_context_payload(context),
            ensure_ascii=False,
        )
        + "</plan_context>"
    )
