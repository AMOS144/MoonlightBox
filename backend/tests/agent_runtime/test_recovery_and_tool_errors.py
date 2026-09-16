"""离线验证供应商调用解析、重复保护及检查点重试，不访问云端或业务数据库。"""

import json
from dataclasses import replace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from moonlightbox.agent_runtime import AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.contracts import (
    AgentBudgetPolicy,
    AgentExecutionRequest,
    RegisteredTool,
    ToolContract,
)
from moonlightbox.agent_runtime.persistence import checkpoint_scope
from moonlightbox.agent_runtime.submission import result_submission_tool
from moonlightbox.runtime_v1.cloud_models import (
    RuntimeCloudChatModel,
    RuntimeCloudCompletion,
    RuntimeCloudInferenceError,
    _native_tool_calls,
)
from pydantic import BaseModel


class Result(BaseModel):
    value: str


def spec(*tools):
    return AgentSpec(
        name="review_fix",
        prompt_version="v1",
        submission_tool_name="submit_result",
        tools=(*tools, result_submission_tool("submit_result", Result, lambda *_: None)),
        budget=AgentBudgetPolicy(
            max_wall_seconds=10,
            max_tool_result_chars=100000,
            max_total_tool_calls=30,
            emergency_max_model_steps=30,
        ),
    )


def run(model, definition):
    return AgentLoopController().run(
        spec=definition,
        model=model,
        request=AgentExecutionRequest(
            owner_type="runtime_cycle",
            owner_id="test",
            scope=RunScope(),
            messages=(SystemMessage(content="测试"), HumanMessage(content="开始")),
        ),
    )


def submit(call_id="submit"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "name": "submit_result",
                "args": {"result": {"value": "ok"}},
            }
        ],
    )


@pytest.mark.parametrize(
    "name,arguments,expected",
    [
        ("wrong_name", "{}", "undeclared_tool"),
        ("submit_result", "{broken", "invalid_tool_arguments"),
        ("submit_result", "[]", "invalid_tool_arguments"),
    ],
)
def test_provider_tool_errors_are_returned_then_repaired(name, arguments, expected):
    requests = []
    raw = {"id": "bad", "type": "function", "function": {"name": name, "arguments": arguments}}

    class Client:
        _timeout_seconds = 10
        model_name = "offline"

        def complete(self, **kwargs):
            requests.append(kwargs)
            if len(requests) == 1:
                return RuntimeCloudCompletion(
                    {"content": "", "tool_calls": [raw], "reasoning_details": [{"text": "test"}]},
                    {},
                )
            payload = kwargs["messages"]
            assistant = next(item for item in payload if item["role"] == "assistant")
            assert assistant["tool_calls"] == [raw]
            assert assistant["reasoning_details"] == [{"text": "test"}]
            feedback = json.loads(
                next(item["content"] for item in payload if item["role"] == "tool")
            )
            assert expected in feedback["error"]
            assert feedback["message"]
            return RuntimeCloudCompletion(
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "fixed",
                            "type": "function",
                            "function": {
                                "name": "submit_result",
                                "arguments": '{"result":{"value":"ok"}}',
                            },
                        }
                    ],
                },
                {},
            )

    outcome = run(RuntimeCloudChatModel(Client(), temperature=0), spec())
    assert outcome.status == "succeeded"
    assert len(requests) == 2


def test_missing_or_duplicate_call_id_remains_protocol_error():
    raw = {"id": "same", "function": {"name": "read", "arguments": "{}"}}
    for calls in ([{**raw, "id": None}], [raw, raw]):
        with pytest.raises(RuntimeCloudInferenceError):
            _native_tool_calls(calls, frozenset({"read"}))


def reader():
    tool = StructuredTool.from_function(
        lambda section: {"section": section, "content": "有效材料，没有来源 ID"},
        name="read",
        description="读取栏目",
    )
    return RegisteredTool(tool, ToolContract(name="read"))


def test_distinct_material_and_drafts_do_not_require_new_source_ids():
    class Model:
        calls = 0

        def bind_tools(self, tools, **kwargs):
            assert {tool.name for tool in tools} == {"read", "submit_result"}
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls > 6:
                assert not any("RUNTIME_LOOP_FEEDBACK" in str(m.content) for m in messages)
                return submit()
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(self.calls),
                        "name": "read",
                        "args": {"section": str(self.calls)},
                    }
                ],
            )

    assert run(Model(), spec(reader())).status == "succeeded"


def test_repeated_calls_warn_then_stop_without_infinite_loop():
    class Model:
        calls = 0
        warnings = 0

        def bind_tools(self, tools, **kwargs):
            assert "read" in {tool.name for tool in tools}
            return self

        def invoke(self, messages):
            self.calls += 1
            self.warnings = sum("RUNTIME_LOOP_FEEDBACK" in str(m.content) for m in messages)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(self.calls),
                        "name": "read",
                        "args": {"section": "same"},
                    }
                ],
            )

    model = Model()
    outcome = run(model, spec(reader()))
    assert outcome.terminal_reason == "repeated_tool_cycle"
    assert model.warnings == 1
    assert model.calls < 10


def test_failure_retry_refreshes_time_and_keeps_materials_and_usage():
    class Model:
        calls = 0

        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "read",
                            "name": "read",
                            "args": {"section": "identity"},
                        }
                    ],
                )
            if self.calls == 2:
                raise RuntimeError("temporary failure")
            assert any(isinstance(m, ToolMessage) and "有效材料" in m.content for m in messages)
            return submit()

    model = Model()
    with checkpoint_scope(InMemorySaver(), "retry") as usage:
        with patch("moonlightbox.agent_runtime.controller.time", return_value=1000):
            first = run(model, spec(reader()))
        with patch("moonlightbox.agent_runtime.controller.time", return_value=2000):
            second = run(model, spec(reader()))
        assert usage["model_steps"] == 3
    assert first.status == "failed"
    assert second.status == "succeeded"


def test_retry_attempt_limit_and_cumulative_budget_remain_enforced():
    class Model:
        calls = 0

        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            self.calls += 1
            raise RuntimeError("unavailable")

    model = Model()
    with checkpoint_scope(InMemorySaver(), "attempts"):
        outcomes = [run(model, spec()) for _ in range(4)]
    assert outcomes[-1].terminal_reason == "execution_attempt_limit"
    assert model.calls == 3
    model = Model()
    definition = spec()
    definition = replace(definition, budget=replace(definition.budget, emergency_max_model_steps=1))
    with checkpoint_scope(InMemorySaver(), "budget"):
        run(model, definition)
        outcome = run(model, definition)
    assert outcome.terminal_reason == "emergency_model_step_limit"
    assert model.calls == 1


def test_job_owned_recovery_has_no_second_attempt_cap_but_keeps_usage_guard():
    from moonlightbox.agent_runtime.persistence import job_recovery_scope

    class Model:
        calls = 0

        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            self.calls += 1
            raise RuntimeError("temporary")

    model = Model()
    definition = spec()
    definition = replace(definition, budget=replace(
        definition.budget, max_execution_attempts=1, emergency_max_model_steps=3,
    ))
    with checkpoint_scope(InMemorySaver(), "job-owned"), job_recovery_scope():
        outcomes = [run(model, definition) for _ in range(4)]
    assert model.calls == 3
    assert outcomes[-1].terminal_reason == "emergency_model_step_limit"


def test_changed_draft_with_same_arguments_is_not_a_repeated_cycle():
    revision = [0]

    def save_work():
        revision[0] += 1
        return {"planning_work_checkpoint": {"draft": f"新草稿 {revision[0]}"}}

    tool = StructuredTool.from_function(save_work, name="save_work", description="保存工作")

    class Model:
        calls = 0

        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            self.calls += 1
            assert not any("RUNTIME_LOOP_FEEDBACK" in str(m.content) for m in messages)
            if self.calls > 6:
                return submit()
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(self.calls),
                        "name": "save_work",
                        "args": {},
                    }
                ],
            )

    outcome = run(Model(), spec(RegisteredTool(tool, ToolContract(name="save_work"))))
    assert outcome.status == "succeeded"


def test_interrupted_checkpoint_resumes_pending_node_with_fresh_deadline():
    class Model:
        calls = 0

        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt("模拟进程中断，不发送操作系统信号")
            return submit()

    model = Model()
    with checkpoint_scope(InMemorySaver(), "interrupted"):
        with patch("moonlightbox.agent_runtime.controller.time", return_value=1000):
            with pytest.raises(KeyboardInterrupt):
                run(model, spec())
        with patch("moonlightbox.agent_runtime.controller.time", return_value=2000):
            outcome = run(model, spec())
    assert outcome.status == "succeeded"
    assert model.calls == 2


def test_collaboration_resume_keeps_work_without_own_retry_counter():
    from moonlightbox.runtime_v1.collaboration.graph import resume_window

    saved = {
        "started_at": 1000,
        "handoffs": 7,
        "usage": {"model_steps": 12},
        "planner_work": {"draft": "已调查内容"},
    }
    with patch("moonlightbox.runtime_v1.collaboration.graph.time", return_value=2000):
        state = {**saved, **resume_window()}
    assert state["started_at"] == 2000
    assert "execution_attempt" not in state
    assert state["usage"] == saved["usage"]
    assert state["planner_work"] == saved["planner_work"]
    assert state["handoffs"] == 7
    for _ in range(10):
        state.update(resume_window())
    assert state["usage"] == saved["usage"]
