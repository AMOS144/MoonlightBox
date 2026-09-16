"""v3 原生工具 Agent 适配器：模型自主查询，统一 Harness 控制执行与停止。"""

import hashlib
import json
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

from moonlightbox.agent_runtime import AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.context import retain_recent_turns
from moonlightbox.agent_runtime.contracts import (
    AgentExecutionRequest,
    ProgressDelta,
    RegisteredTool,
    ToolContract,
)
from moonlightbox.agent_runtime.policy import (
    model_resilience,
    section_budget,
)
from moonlightbox.agent_runtime.submission import submission_instruction

from .prompt_loader import load_understanding_protocol_v3, validate_registered_tools
from .section_support import (
    _dedupe_evidence,
    _evidence_from_artifacts,
    _retrievals_from_artifacts,
)
from .tools.comparison import comparison_for
from .tools.evidence_page import project_native_messages, project_native_search
from .tools.execution_policy import execution_policy
from .tools.submit_section import build_submit_section


@dataclass
class SectionExecutionV3:
    section: str
    result: object
    status: str
    reason: str | None
    execution_id: str
    evidence: tuple
    retrievals: tuple
    dependencies: list
    section_work: dict = field(default_factory=dict)


def run_section_v3(
    *,
    definition,
    tools,
    compiler,
    context,
    tracker,
    artifacts,
    policy,
    owner_id,
    project_id,
    graph_id,
    target_id,
    input_revision=1,
    input_revision_resolver=None,
    cancellation_requested=None,
    on_status=None,
    checkpoint_path=None,
):
    section = definition.name
    validate_registered_tools(definition, tools)
    # 当前栏目读取也冻结到本回合，避免背景更新或另一候选污染已开始的调查。
    original_current = tools["get_current_profile_section"]

    def current_section(section):
        return {
            **tracker.snapshot.envelope(),
            "section": section,
            "content": context.get("previous_section")
            if section == definition.name
            else context.get("profile_snapshot", {}).get(
                section, {"summary": context.get("section_summaries", {}).get(section, "")}
            ),
        }

    tools = {
        **tools,
        "get_current_profile_section": StructuredTool.from_function(
            current_section,
            name="get_current_profile_section",
            args_schema=original_current.args_schema,
            description="读取上一阶段已接受的完整栏目快照，不包含同批次尚在修改的结果；既有理解可修订，不是独立新证据。",
        ),
    }
    system = load_understanding_protocol_v3() + "\n\n" + definition.system_prompt
    artifacts.section_work = dict(context.get("section_work", {}))
    if context.get("node_scope"):
        from .prompt_loader import load_temporal_protocol

        system += "\n\n" + load_temporal_protocol()
        if context.get("initial_messages"):
            artifacts.add_evidence_set(context["initial_messages"], retrieval_id=None)
    factory = getattr(compiler, "create_agent_chat_model", None)
    if not callable(factory):
        raise ValueError("编译客户端未提供原生工具模型 create_agent_chat_model")
    model = factory()

    submission = build_submit_section(section, context, tracker, artifacts)
    system += "\n\n" + submission_instruction("submit_section")

    def restore_work(state):
        artifacts.restore(state["artifacts"])
        tracker.dependencies[:] = state["dependencies"]

    outcome = AgentLoopController().run(
        spec=AgentSpec(
            name=f"person_world.section.{section}",
            prompt_version=hashlib.sha256(system.encode()).hexdigest(),
            # Harness 的 tool_result_chars 是累计传输量，不是当前上下文长度。
            # 原生回环现在实际传递正文，不能沿用 v2 只传 ID 的 96K 累计额度。
            budget=section_budget(policy),
            resilience=model_resilience(
                model, timeout_seconds=policy.model_request_timeout_seconds
            ),
            submission_tool_name="submit_section",
            context_compactor=lambda messages, **kwargs: _compact_native_context(
                messages, section_work=artifacts.section_work, **kwargs
            ),
            snapshot_work_state=lambda: {
                "artifacts": artifacts.snapshot(),
                "dependencies": list(tracker.dependencies),
            },
            restore_work_state=restore_work,
            tools=tuple(
                RegisteredTool(
                    tool=tool,
                    contract=ToolContract(
                        name=tool.name,
                        comparison_projection=comparison_for(tool.name, artifacts),
                        execution=execution_policy(tool.name, frozen_section=True),
                        max_result_chars=96_000,
                        timeout_seconds=policy.tool_timeout_seconds
                        if tool.name
                        in {"search_world", "get_entity_neighborhood", "list_graph_entities"}
                        else min(30.0, policy.tool_timeout_seconds),
                        progress_evaluator=_native_progress,
                        model_result_projector=project_native_search
                        if tool.name == "search_world"
                        else (lambda result: project_native_messages(artifacts, result))
                        if tool.name in {"locate_source_messages", "get_message_context"}
                        else lambda value: value,
                    ),
                )
                for tool in tools.values()
            )
            + (submission,),
        ),
        request=AgentExecutionRequest(
            checkpoint_path=checkpoint_path,
            input_revision=input_revision,
            input_revision_resolver=input_revision_resolver,
            owner_type="person_world",
            cancellation_requested=cancellation_requested,
            on_status=on_status,
            owner_id=owner_id,
            project_id=project_id,
            scope=RunScope(
                project_id=project_id,
                graph_read_version=graph_id,
                target_person_id=target_id,
                permissions=frozenset({"person_world.read"}),
            ),
            messages=(
                SystemMessage(content=system),
                # 完整跨栏背景按需由工具读取，不重复塞进每个 Agent 的起始消息。
                HumanMessage(
                    content=json.dumps(
                        {key: value for key, value in context.items() if key != "profile_snapshot"},
                        ensure_ascii=False,
                    )
                ),
            ),
        ),
        model=model,
    )
    return SectionExecutionV3(
        section,
        outcome.value,
        outcome.status,
        outcome.terminal_reason,
        outcome.execution_id,
        tuple(_dedupe_evidence(_evidence_from_artifacts(artifacts))),
        tuple(_retrievals_from_artifacts(artifacts)),
        list(tracker.dependencies),
        dict(artifacts.section_work),
    )


def _native_progress(result, *, state_revision, seen_keys):
    """以真正读到的内容判断进展；换查询措辞或 retrieval_id 不能伪造进展。"""
    if not isinstance(result, dict) or result.get("error"):
        return ProgressDelta(summary="没有新的可读结果")
    meaningful = {
        key: result[key]
        for key in ("context", "graph_data", "messages", "values", "content", "modules", "fields")
        if result.get(key)
    }
    if not meaningful:
        return ProgressDelta(summary="返回了目录或空结果")
    digest = hashlib.sha256(
        json.dumps(meaningful, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()
    return ProgressDelta(new_keys=frozenset({digest}), summary="获得新的可读背景或原文")


def _compact_native_context(messages, *, source_refs, unresolved, section_work=None):
    """保留任务与完整的近期工具批次，不能从 ToolMessage 中间截断协议。"""
    summary = HumanMessage(
        content=json.dumps(
            {
                "source_refs": source_refs,
                "unresolved": unresolved,
                "section_work": section_work or {},
                "instruction": (
                    "较早工具批次已离开当前窗口，需要具体原文时使用 read_evidence_page 重新读取。"
                ),
            },
            ensure_ascii=False,
        )
    )
    return retain_recent_turns(
        messages, keep=3, notes=(summary,), tool_turns_only=True
    ), "保留固定任务与完整近期工具批次"
