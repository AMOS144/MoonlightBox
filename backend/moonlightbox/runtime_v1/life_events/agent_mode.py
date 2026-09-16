"""同一 Agent 的生活任务适配；仍使用已有模型与统一 Harness。"""

import json

from langchain_core.messages import HumanMessage

from moonlightbox.agent_runtime import AgentSpec, RunScope
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest
from moonlightbox.agent_runtime.policy import controller_budget

from ..agent_catalog import load_agent_definition, select_declared_tools
from ..agent_support import (
    RuntimeToolbox,
    prompt_hash,
)
from ..prompting import assemble_agent_prompt
from ..tools.submit_life_result import build_submit_life_result_tool


def run_mode(
    agent,
    *,
    name,
    definition_name,
    schema,
    context,
    tools,
    branch_id,
    project_id,
    owner_id,
    input_revision,
    resolver,
    cancellation_requested=None,
    on_status=None,
    validate_plan_sources=None,
):
    if agent.model is None:
        raise RuntimeError("life_model_unavailable")
    definition = load_agent_definition(definition_name)
    toolbox = RuntimeToolbox(agent.runtime_policy)
    registered = toolbox.register(select_declared_tools(definition, tools))
    registered += (
        build_submit_life_result_tool(
            schema, context=context, validate_plan_sources=validate_plan_sources
        ),
    )
    assembly = assemble_agent_prompt(
        definition_name, schema, tools=registered, submission_tool_name="submit_life_result"
    )
    prompt = assembly.text
    outcome = agent.controller.run(
        spec=AgentSpec(
            name=name,
            prompt_version=prompt_hash(prompt),
            submission_tool_name="submit_life_result",
            budget=controller_budget(agent.runtime_policy),
            tools=registered,
            context_compactor=toolbox.compact,
            restore_tool_results=toolbox.restore,
        ),
        request=AgentExecutionRequest(
            cancellation_requested=cancellation_requested,
            on_status=on_status,
            owner_type="life_opportunity",
            owner_id=owner_id,
            project_id=project_id,
            input_revision=input_revision,
            scope=RunScope(
                project_id=project_id,
                branch_id=branch_id,
                permissions=frozenset(
                    {"runtime.read_memory", "runtime.read_plan", "runtime.send_agent_message"}
                ),
            ),
            messages=(
                assembly.message(),
                HumanMessage(content=json.dumps(context, ensure_ascii=False, default=str)),
            ),
            input_revision_resolver=resolver,
        ),
        model=agent.model,
    )
    if not isinstance(outcome.value, schema) or outcome.terminal_reason not in {
        "success",
    }:
        from moonlightbox.agent_runtime.resilience import ExecutionInterrupted

        # 保留统一 Controller 的错误码，Job 才能区分暂时网络故障与明确终态。
        raise ExecutionInterrupted(outcome.terminal_reason)
    return outcome.value
