"""真实 Controller、检查点和 MockTransport 回归，不调用云端。"""

from dataclasses import replace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from moonlightbox.agent_runtime.contracts import (
    AgentBudgetPolicy,
    AgentExecutionRequest,
    AgentSpec,
    RegisteredTool,
    RunScope,
    ToolContract,
)
from moonlightbox.agent_runtime.controller import AgentLoopController
from moonlightbox.agent_runtime.deadlines import tool_deadline
from moonlightbox.agent_runtime.persistence import checkpoint_scope
from moonlightbox.agent_runtime.resilience import (
    ResiliencePolicy,
    execution_scope,
    request_timeout,
    run_operation,
)
from moonlightbox.agent_runtime.submission import result_submission_tool
from moonlightbox.agent_runtime.tool_errors import ToolInputError
from moonlightbox.world.client import LightRAGSidecarClient, LightRAGSidecarError
from pydantic import BaseModel


class Result(BaseModel):
    value: int


class Model:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, *args, **kwargs):
        return self

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {"id": str(self.calls), "name": "submit", "args": {"result": {"value": 2}}}
            ],
        )


def spec():
    return AgentSpec(
        name="audit",
        prompt_version="same",
        budget=AgentBudgetPolicy(max_wall_seconds=10, max_tool_result_chars=100000),
        submission_tool_name="submit",
        tools=(result_submission_tool("submit", Result),),
    )


def request():
    return AgentExecutionRequest(
        owner_type="person_world",
        owner_id="test",
        scope=RunScope(),
        messages=(HumanMessage(content="test"),),
    )


def test_same_contract_recovers_and_changed_validator_reexecutes():
    model, checks = Model(), []
    original = spec()
    saver = InMemorySaver()
    with checkpoint_scope(saver, "test"):
        first = AgentLoopController().run(spec=original, request=request(), model=model)
        second = AgentLoopController().run(spec=spec(), request=request(), model=model)
        assert first.value == second.value and model.calls == 1

        def validate(result, context):
            checks.append(result.value)
            return None

        changed = replace(original, tools=(result_submission_tool("submit", Result, validate),))
        AgentLoopController().run(spec=changed, request=request(), model=model)
        assert model.calls == 2 and checks == [2]
        AgentLoopController().run(spec=changed, request=request(), model=model)
        assert model.calls == 2


@pytest.mark.parametrize("field", ["contract_version", "state_version"])
def test_explicit_contract_versions_invalidate_success(field):
    model, original = Model(), spec()
    with checkpoint_scope(InMemorySaver(), "version"):
        AgentLoopController().run(spec=original, request=request(), model=model)
        AgentLoopController().run(
            spec=replace(original, **{field: "2"}), request=request(), model=model
        )
    assert model.calls == 2


def test_schema_and_tool_version_change_contract():
    from moonlightbox.agent_runtime.checkpoint_contract import checkpoint_contract_fingerprint

    class NewResult(BaseModel):
        value: int
        reason: str = "new"

    original = spec()
    schema_changed = replace(original, tools=(result_submission_tool("submit", NewResult),))
    registered = original.tools[0]
    rule_changed = replace(
        original,
        tools=(replace(registered, contract=replace(registered.contract, contract_version="2")),),
    )
    assert (
        len({checkpoint_contract_fingerprint(s) for s in (original, schema_changed, rule_changed)})
        == 3
    )


@pytest.mark.parametrize("status,expected", [(503, 3), (429, 3), (400, 1), (401, 1), (403, 1)])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_sidecar_status_retry_and_real_timeout(status, expected, method):
    seen = []

    def respond(req):
        seen.append(req.extensions["timeout"])
        return httpx.Response(status, json={"error": "test"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        sidecar = LightRAGSidecarClient("http://test", "test", timeout_seconds=300, client=client)

        def operation():
            with tool_deadline(0.2):
                return sidecar._get("/query") if method == "GET" else sidecar._post("/query", {})

        with execution_scope(
            ResiliencePolicy(max_retries=2, backoff_seconds=0), lambda: 10, lambda: None
        ):
            with pytest.raises(LightRAGSidecarError):
                run_operation(operation, kind="tool:search", timeout_seconds=1)
    assert len(seen) == expected
    assert all(0 < t["read"] <= 0.2 for t in seen)


def test_tool_timeout_independent_of_model_timeout():
    observed = []
    with execution_scope(ResiliencePolicy(request_timeout_seconds=0.01), lambda: 20, lambda: None):
        run_operation(
            lambda: observed.append(request_timeout(10, model_request=False)),
            kind="tool:search",
            timeout_seconds=5,
        )
    assert 4 < observed[0] <= 5


def test_non_replay_safe_operation_never_retries_503():
    calls = []

    def fail():
        calls.append(1)
        raise LightRAGSidecarError("lightrag_request_failed", "test", status_code=503)

    with execution_scope(ResiliencePolicy(max_retries=2), lambda: 10, lambda: None):
        with pytest.raises(LightRAGSidecarError):
            run_operation(fail, kind="tool:write", replay_safe=False)
    assert calls == [1]


def test_tool_input_error_is_actionable_and_internal_error_is_hidden():
    def read_page(offset: int):
        """读取分页。"""
        if offset < 0:
            raise ToolInputError("offset 必须大于等于 0，请从 0 重试")
        raise ValueError("internal-private-debug-details")

    reader = StructuredTool.from_function(read_page)
    feedback = []

    class RepairModel(Model):
        def invoke(self, messages):
            feedback[:] = [m.content for m in messages if isinstance(m, ToolMessage)]
            if self.calls < 2:
                self.calls += 1
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": str(self.calls),
                            "name": "read_page",
                            "args": {"offset": -1 if self.calls == 1 else 0},
                        }
                    ],
                )
            return super().invoke(messages)

    original = spec()
    outcome = AgentLoopController().run(
        spec=replace(
            original,
            tools=(RegisteredTool(reader, ToolContract(name=reader.name)), *original.tools),
        ),
        request=request(),
        model=RepairModel(),
    )
    assert outcome.status == "succeeded"
    assert "从 0 重试" in feedback[0] and "invalid_tool_arguments" in feedback[0]
    assert "internal-private-debug-details" not in feedback[1]
