"""Director 的领域适配；模型与工具链路由 Phoenix 观测。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from moonlightbox.agent_runtime import (
    AgentLoopController,
    AgentSpec,
    RunScope,
)
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest, ChatModel
from moonlightbox.agent_runtime.policy import (
    DIRECTOR_RUNTIME_POLICY,
    AgentRuntimePolicy,
    controller_budget,
)
from moonlightbox.agent_runtime.skills import build_skill_tool
from moonlightbox.agent_runtime.tool_dispatch import build_execute_tool

from .agent_catalog import load_agent_definition, select_declared_tools
from .agent_support import (
    RuntimeToolbox,
    prompt_hash,
)
from .context_views import director_context_payload
from .expression_profile import expression_material
from .prompting import assemble_agent_prompt
from .schemas import ContextPacket, LifeDecision
from .tools.submit_decision import build_submit_decision_tool
from .tools.submit_expression import ExpressionSubmission


@dataclass(frozen=True)
class DirectorRun:
    decision: LifeDecision
    terminal_reason: str
    asset_sources: dict | None = None


class DirectorAgent:
    """将 ContextPacket 转为 AgentSpec；没有数据库或业务写入权限。"""

    def initialize_branch(self, session, **kwargs):
        """独立起点任务，不复用普通 Director 的消息列表。"""
        from .initialization import initialize_director

        return initialize_director(self, session, **kwargs)

    def __init__(
        self,
        model: ChatModel | None = None,
        *,
        runtime_policy: AgentRuntimePolicy | None = None,
        controller: AgentLoopController | None = None,
    ) -> None:
        self.model = model
        self.runtime_policy = runtime_policy or DIRECTOR_RUNTIME_POLICY
        self.controller = controller or AgentLoopController()

    def run(self, packet: ContextPacket, *, search_tool: BaseTool | None = None) -> LifeDecision:
        return self.run_with_trace(packet, search_tool=search_tool).decision

    def run_with_trace(
        self,
        packet: ContextPacket,
        *,
        search_tool: BaseTool | None = None,
        recent_life_tool: BaseTool | None = None,
        context_tools: list[BaseTool] | None = None,
        skill_tools: list[BaseTool] | None = None,
        owner_id: str | None = None,
        project_id: str | None = None,
        input_revision: int = 1,
        input_revision_resolver: Callable[[], int | None] | None = None,
        reference_validator: Callable[[LifeDecision], str | None] | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
        receive_inputs=None,
        asset_validator=None,
        on_status: Callable[[dict[str, Any]], None] | None = None,
    ) -> DirectorRun:
        if self.model is None:
            return DirectorRun(
                decision=_safe_wait(packet, "模型不可用"),
                terminal_reason="guard",
            )
        tools = [
            build_skill_tool(
                Path(__file__).with_name("skills"),
                business_tools=tuple(skill_tools or []),
                skill_contexts={
                    "speaking": expression_material(packet.origin.get("person_world_profile"))
                },
            )
        ]
        if search_tool is not None:
            tools.append(search_tool)
        if recent_life_tool is not None:
            tools.append(recent_life_tool)
        toolbox = RuntimeToolbox(self.runtime_policy)
        tools.extend(context_tools or [])
        definition = load_agent_definition("director")
        registered = toolbox.register(select_declared_tools(definition, tools))
        registered += (
            build_execute_tool(tuple(toolbox.register_one(t) for t in skill_tools or [])),
        )
        submission = ExpressionSubmission({}, asset_validator=asset_validator)
        registered += (build_submit_decision_tool(packet, reference_validator, submission),)
        assembly = assemble_agent_prompt(
            "director", LifeDecision, tools=registered, submission_tool_name="submit_decision"
        )
        prompt = assembly.text
        read_ids = []

        def restore_inputs(work):
            nonlocal read_ids
            read_ids = list(work.get("input_event_ids", []))
            if receive_inputs is not None and work.get("packet") is not None:
                restored = ContextPacket.model_validate(work["packet"])
                receive_inputs(read_ids, restored)
                for key in type(packet).model_fields:
                    setattr(packet, key, getattr(restored, key))

        def refresh_inputs(messages):
            nonlocal read_ids
            # 输入游标随首轮任务消息进入检查点，压缩时同样保留；不是新的持久化实体。
            first = next(
                i for i, message in enumerate(messages) if isinstance(message, HumanMessage)
            )
            anchor = messages[first]
            ids = list(read_ids)
            previous_ids = list(ids)
            if receive_inputs is not None:
                updated, ids = receive_inputs(ids)
                for key in type(packet).model_fields:
                    setattr(packet, key, getattr(updated, key))
            read_ids = list(ids)
            if previous_ids == read_ids and anchor.additional_kwargs.get(
                "runtime_input_initialized"
            ):
                # 没有新输入就保留压缩后的上下文，不能每轮把大包重新展开。
                return list(messages)
            messages = list(messages)
            messages[first] = anchor.model_copy(
                update={
                    "content": _runtime_context_message(packet),
                    "additional_kwargs": {
                        **anchor.additional_kwargs,
                        "runtime_input_event_ids": ids,
                        "runtime_input_initialized": True,
                    },
                }
            )
            return messages

        spec = AgentSpec(
            name="director",
            prompt_version=prompt_hash(prompt),
            submission_tool_name="submit_decision",
            budget=controller_budget(self.runtime_policy),
            tools=registered,
            context_compactor=toolbox.compact,
            restore_tool_results=lambda results: (
                toolbox.restore(results),
                submission.restore(results),
            ),
            refresh_inputs=refresh_inputs if receive_inputs is not None else None,
            snapshot_work_state=(
                lambda: {
                    "input_event_ids": read_ids,
                    "packet": packet.model_dump(mode="json"),
                }
            )
            if receive_inputs is not None
            else None,
            restore_work_state=restore_inputs if receive_inputs is not None else None,
        )
        outcome = self.controller.run(
            spec=spec,
            request=AgentExecutionRequest(
                owner_type="runtime_cycle",
                cancellation_requested=cancellation_requested,
                on_status=on_status,
                owner_id=owner_id or str(packet.branch.get("branch_id", "runtime")),
                project_id=project_id,
                input_revision=input_revision,
                scope=RunScope(
                    project_id=project_id,
                    branch_id=str(packet.branch.get("branch_id", "")) or None,
                    allowed_source_snapshot_id=str(packet.origin.get("snapshot_id", "")) or None,
                    permissions=frozenset(
                        {"runtime.read_memory", "runtime.read_style", "runtime.send_agent_message"}
                    ),
                ),
                messages=(
                    assembly.message(),
                    HumanMessage(content=_runtime_context_message(packet)),
                ),
                input_revision_resolver=input_revision_resolver,
            ),
            model=self.model,
        )
        decision = outcome.value
        return DirectorRun(
            decision=decision
            if isinstance(decision, LifeDecision)
            else _safe_wait(packet, outcome.terminal_reason),
            terminal_reason="decision"
            if outcome.terminal_reason == "success"
            else outcome.terminal_reason,
            asset_sources=dict(submission.authorized),
        )


def _safe_wait(packet: ContextPacket, reason: str) -> LifeDecision:
    return LifeDecision(action="wait", private_reason=reason[:500])


def _runtime_context_message(packet: ContextPacket) -> str:
    return (
        "<runtime_context>"
        + json.dumps(
            director_context_payload(packet),
            ensure_ascii=False,
        )
        + "</runtime_context>"
    )
