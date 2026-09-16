"""提交必须经过原生工具，拒绝可修复，接受不等于落库。"""

import json
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from moonlightbox.agent_runtime import AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.contracts import AgentBudgetPolicy, AgentExecutionRequest
from moonlightbox.agent_runtime.submission import result_submission_tool
from pydantic import BaseModel, ConfigDict


class Result(BaseModel):
    model_config = ConfigDict(strict=True)
    day: date
    value: int


def test_native_submission_rejects_then_accepts_without_final_json():
    tool = result_submission_tool(
        "submit_result", Result, lambda value, _: "value 必须大于 0" if value.value <= 0 else None
    )

    class Model:
        calls = 0

        def bind_tools(self, tools):
            assert tools[0].name == "submit_result"
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 2:
                result = json.loads(
                    next(m.content for m in reversed(messages) if isinstance(m, ToolMessage))
                )
                assert result["status"] == "rejected"
                assert result["committed"] is False
                assert "大于" in result["message"]
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(self.calls),
                        "name": "submit_result",
                        "args": {"result": {"day": "2026-05-09", "value": self.calls - 1}},
                    }
                ],
            )

    model = Model()
    outcome = run(model, tool)
    assert outcome.terminal_reason == "success"
    assert outcome.value == Result(day=date(2026, 5, 9), value=1)
    assert model.calls == 2


def test_bare_json_does_not_bypass_submission():
    tool = result_submission_tool("submit_result", Result, lambda *args: None)

    class Model:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(content='{"day":"2026-05-09","value":1}')

    outcome = run(Model(), tool)
    assert outcome.terminal_reason != "success"
    assert outcome.value is None


def test_schema_error_returns_field_paths_for_tool_repair():
    tool = result_submission_tool("submit_result", Result, lambda *args: None)

    class Model:
        calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 2:
                message = next(m for m in reversed(messages) if isinstance(m, ToolMessage))
                assert message.status == "error"
                assert "validation_errors" in message.content
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(self.calls),
                        "name": "submit_result",
                        "args": {
                            "result": {
                                "day": "invalid" if self.calls == 1 else "2026-05-09",
                                "value": 1,
                            }
                        },
                    }
                ],
            )

    assert run(Model(), tool).terminal_reason == "success"


def test_accepted_submission_checkpoint_is_reused():
    from langgraph.checkpoint.memory import InMemorySaver
    from moonlightbox.agent_runtime.persistence import checkpoint_scope

    tool = result_submission_tool("submit_result", Result, lambda *args: None)

    class Model:
        calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "submit",
                        "name": "submit_result",
                        "args": {"result": {"day": "2026-05-09", "value": 1}},
                    }
                ],
            )

    model = Model()
    with checkpoint_scope(InMemorySaver(), "submission-checkpoint"):
        first = run(model, tool)
        second = run(model, tool)
    assert first.value == second.value
    assert model.calls == 1


def test_two_submissions_in_one_batch_are_rejected_before_validation():
    validations = []
    tool = result_submission_tool(
        "submit_result", Result, lambda value, context: validations.append(value)
    )

    class Model:
        calls = 0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 2:
                assert (
                    sum(
                        "submission_must_be_separate" in m.content
                        for m in messages
                        if isinstance(m, ToolMessage)
                    )
                    == 2
                )
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": str(i),
                        "name": "submit_result",
                        "args": {"result": {"day": "2026-05-09", "value": 1}},
                    }
                    for i in range(2 if self.calls == 1 else 1)
                ],
            )

    assert run(Model(), tool).terminal_reason == "success"
    assert len(validations) == 1


def run(model, tool):
    return AgentLoopController().run(
        spec=AgentSpec(
            name="test",
            prompt_version="submission-v1",
            submission_tool_name="submit_result",
            tools=(tool,),
            budget=AgentBudgetPolicy(max_wall_seconds=10, max_tool_result_chars=100000),
        ),
        request=AgentExecutionRequest(
            owner_type="runtime_cycle",
            owner_id="test",
            input_revision=1,
            scope=RunScope(),
            messages=(HumanMessage(content="提交"),),
        ),
        model=model,
    )


def test_tool_returns_same_typed_object_without_second_parser():
    accepted = []
    tool = result_submission_tool(
        "submit_result",
        Result,
        lambda value, _: accepted.append(value),
    )

    class Model:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "s",
                        "name": "submit_result",
                        "args": {"result": {"day": "2026-05-09", "value": 1}},
                    }
                ],
            )

    result = run(Model(), tool)
    assert result.value is accepted[0]
    assert not hasattr(AgentSpec, "output_parser")
    assert not hasattr(AgentSpec, "final_validator")


def test_submission_can_handoff_to_user_without_rejection():
    tool = result_submission_tool(
        "submit_result",
        Result,
        completion_status=lambda _: "waiting_for_user",
    )

    class Model:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "question",
                        "name": "submit_result",
                        "args": {"result": {"day": "2026-05-09", "value": 1}},
                    }
                ],
            )

    model = Model()
    result = run(model, tool)
    assert result.status == "waiting_for_user"
    assert result.value.value == 1
    assert model.calls == 1


def test_plain_text_can_be_corrected_to_explicit_submission():
    tool = result_submission_tool("submit_result", Result)

    class Model:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content='{"day":"2026-05-09","value":1}')
            assert "普通文本不算提交" in messages[-1].content
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "s",
                        "name": "submit_result",
                        "args": {"result": {"day": "2026-05-09", "value": 1}},
                    }
                ],
            )

    model = Model()
    assert run(model, tool).status == "succeeded"
    assert model.calls == 2
