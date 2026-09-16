"""独立 PersonaActor：隔离消息与工具材料，复用统一可恢复 Harness。"""

import json
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

from moonlightbox.agent_runtime import AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest
from moonlightbox.agent_runtime.policy import PERSONA_ACTOR_RUNTIME_POLICY, controller_budget

from .agent_catalog import load_agent_definition, select_declared_tools
from .agent_support import (
    RuntimeToolbox,
    prompt_hash,
)
from .expression_contracts import ExpressionResult
from .prompting import assemble_agent_prompt
from .tools.submit_expression import ExpressionSubmission


@dataclass
class ActorRun:
    result: ExpressionResult | None
    error_code: str | None = None
    asset_sources: dict = field(default_factory=dict)


class PersonaActor:
    def __init__(self, model=None, *, controller=None, runtime_policy=None):
        self.model = model
        self.controller = controller or AgentLoopController()
        self.runtime_policy = runtime_policy or PERSONA_ACTOR_RUNTIME_POLICY

    def run_with_trace(
        self,
        *,
        context,
        tools,
        owner_id,
        project_id,
        branch_id,
        input_revision,
        input_revision_resolver=None,
        asset_validator=None,
        exposed_assets=None,
        cancellation_requested=None,
        on_status=None,
    ):
        if self.model is None:
            return ActorRun(None, "model_unavailable")
        definition = load_agent_definition("persona_actor")
        toolbox = RuntimeToolbox(self.runtime_policy)
        registered = toolbox.register(select_declared_tools(definition, tools))
        submission = ExpressionSubmission(
            context, asset_validator=asset_validator, exposed_assets=exposed_assets
        )

        def restore(results):
            toolbox.restore(results)
            submission.restore(results)

        registered += (submission.build_tool(),)
        assembly = assemble_agent_prompt(
            "persona_actor",
            ExpressionResult,
            tools=registered,
            submission_tool_name="submit_expression",
        )
        prompt = assembly.text
        outcome = self.controller.run(
            spec=AgentSpec(
                name="persona_actor",
                prompt_version=prompt_hash(prompt),
                submission_tool_name="submit_expression",
                budget=controller_budget(self.runtime_policy),
                tools=registered,
                context_compactor=toolbox.compact,
                restore_tool_results=restore,
            ),
            request=AgentExecutionRequest(
                owner_type="runtime_expression",
                cancellation_requested=cancellation_requested,
                on_status=on_status,
                owner_id=owner_id,
                project_id=project_id,
                input_revision=input_revision,
                scope=RunScope(
                    project_id=project_id,
                    branch_id=branch_id,
                    permissions=frozenset({"runtime.read_style", "runtime.read_memory"}),
                ),
                messages=(
                    assembly.message(),
                    HumanMessage(
                        content="<persona_context>"
                        + json.dumps(context, ensure_ascii=False)
                        + "</persona_context>"
                    ),
                ),
                input_revision_resolver=input_revision_resolver,
            ),
            model=self.model,
        )
        if isinstance(outcome.value, ExpressionResult):
            return ActorRun(outcome.value, asset_sources=dict(submission.authorized))
        return ActorRun(None, outcome.terminal_reason)
