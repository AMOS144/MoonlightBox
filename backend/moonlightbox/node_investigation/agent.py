"""独立调查节点，模型/工具循环、检查点、压缩与故障策略均交给公共 Controller。"""

import json
from dataclasses import replace

from langchain_core.messages import HumanMessage, SystemMessage

from moonlightbox.agent_runtime.context import retain_recent_turns
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest, AgentSpec, RunScope
from moonlightbox.agent_runtime.controller import AgentLoopController
from moonlightbox.agent_runtime.policy import NODE_INVESTIGATION_POLICY, controller_budget
from moonlightbox.agent_runtime.submission import result_submission_tool, submission_instruction
from moonlightbox.runtime_v1.agent_catalog import load_agent_definition, select_declared_tools

from .schemas import TurnResult
from .store import digest


def _budget():
    """调查原文随时可按游标重读,结论落在 work 笔记;用 128K 实用窗口而不是
    供应商标称的 512K——后者会让每轮请求拖着全部已读原文,又慢又贵。"""
    return replace(
        controller_budget(NODE_INVESTIGATION_POLICY),
        context_window_tokens=128_000,
        model_context_windows={},
    )


def run_investigation(tools, model, *, checkpoint_path=None, cancelled=None):
    definition = load_agent_definition("node_investigator")
    prompt = definition.system_prompt + "\n\n" + submission_instruction("submit_investigation_turn")
    initial = tools.overview()
    bound = tools.store.get()
    tools.seen_input = len(initial["inputs"])

    def refresh(messages):
        state = tools.store.get().state
        new_inputs = [i for i in state["inputs"] if i["seq"] > tools.seen_input]
        if not new_inputs:
            return list(messages)
        tools.seen_input = new_inputs[-1]["seq"]
        # 新信息补入当前推理，不使旧请求失效；Agent 自行决定怎样调整调查。
        return [
            *messages,
            HumanMessage(
                content=json.dumps(
                    {"new_user_inputs": new_inputs, "current_candidates": state["candidates"]},
                    ensure_ascii=False,
                )
            ),
        ]

    def snapshot():
        return {**tools.snapshot(), "seen_input": tools.seen_input}

    def restore(value):
        tools.restore(value)
        tools.seen_input = value.get("seen_input", 0)

    def compact(messages, **kwargs):
        retained = retain_recent_turns(messages)
        retained.append(
            HumanMessage(
                content=json.dumps(
                    {
                        "saved_investigation_work": tools.store.get().state["work"],
                        "note": "完整候选和用户输入可通过 get_record_overview 重读",
                    },
                    ensure_ascii=False,
                )
            )
        )
        return retained, "保留调查笔记和最近完整工具回合，原文和候选可重读"

    registered = (
        *tools.registered(),
        result_submission_tool(
            "submit_investigation_turn",
            TurnResult,
            completion_status=lambda result: (
                result.outcome if result.outcome == "waiting_for_user" else "succeeded"
            ),
        ),
    )
    by_name = {entry.tool.name: entry for entry in registered}
    selected = select_declared_tools(definition, [entry.tool for entry in registered])
    declared = tuple(by_name[tool.name] for tool in selected)

    return AgentLoopController().run(
        spec=AgentSpec(
            name="node_investigator",
            prompt_version=digest(prompt),
            contract_version="1",
            state_version="1",
            submission_tool_name="submit_investigation_turn",
            budget=_budget(),
            tools=declared,
            refresh_inputs=refresh,
            snapshot_work_state=snapshot,
            restore_work_state=restore,
            context_compactor=compact,
        ),
        request=AgentExecutionRequest(
            owner_type="node_investigation",
            owner_id=tools.store.id,
            project_id=tools.store.project_id,
            input_revision=bound.state["turn"],
            scope=RunScope(
                project_id=tools.store.project_id,
                allowed_source_snapshot_id=bound.dataset_version,
                graph_read_version=bound.state["graph_id"],
                extra={"investigation_id": tools.store.id, "job_id": tools.job_id},
            ),
            checkpoint_path=checkpoint_path,
            cancellation_requested=cancelled,
            messages=(
                SystemMessage(content=prompt),
                HumanMessage(content=json.dumps(initial, ensure_ascii=False)),
            ),
        ),
        model=model,
    )
