"""无状态 Agent Harness 的冒烟测试。

调用过程由 Phoenix 观测，本测试只锁住 LangGraph 的工具回环、输入版本保护和上下文限制，
不创建任何自建 Agent 账本表。
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from moonlightbox.agent_runtime import AgentBudgetPolicy, AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.contracts import (
    AgentExecutionRequest,
    RegisteredTool,
    ToolContract,
)
from moonlightbox.agent_runtime.submission import result_submission_tool
from pydantic import BaseModel


class _Final(BaseModel):
    value: str


def test_unsubmitted_checkpoint_is_retried_not_replayed():
    from langgraph.checkpoint.memory import InMemorySaver
    from moonlightbox.agent_runtime.persistence import checkpoint_scope

    class Model:
        valid = False
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            return _submit("ok") if self.valid else AIMessage(content="{}")

    model = Model()
    spec = _spec(
        name="retry",
        prompt_version="test",
        budget=AgentBudgetPolicy(max_wall_seconds=60, max_tool_result_chars=100_000),
    )
    with checkpoint_scope(InMemorySaver(), "retry-test"):
        first = AgentLoopController().run(spec=spec, request=_request(), model=model)
        assert first.status != "succeeded"
        before = model.calls
        model.valid = True
        second = AgentLoopController().run(spec=spec, request=_request(), model=model)
        assert model.calls > before
        assert second.status == "succeeded"
        assert second.value.value == "ok"


def _submit(value):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "finish",
                "name": "finish",
                "args": {"result": {"value": value}},
            }
        ],
    )


def _spec(**kwargs):
    tools = kwargs.pop("tools", ())
    return AgentSpec(
        **kwargs,
        submission_tool_name="finish",
        tools=(*tools, result_submission_tool("finish", _Final)),
    )


def _request(*, input_revision: int = 1, resolver: object | None = None) -> AgentExecutionRequest:
    return AgentExecutionRequest(
        owner_type="runtime_cycle",
        owner_id="controller-smoke",
        scope=RunScope(),
        messages=(HumanMessage(content="start"),),
        input_revision=input_revision,
        input_revision_resolver=resolver if callable(resolver) else None,
    )


def test_controller_executes_declared_tool_and_returns_structured_value() -> None:
    calls: list[str] = []

    def read_evidence(key: str) -> dict[str, object]:
        calls.append(key)
        return {"source_ids": [key]}

    tool = StructuredTool.from_function(
        read_evidence,
        name="read_evidence",
        description="test only",
    )

    class Model:
        def invoke(self, messages: list[object]) -> AIMessage:
            if not any(isinstance(item, ToolMessage) for item in messages):
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "read_evidence", "args": {"key": "source-1"}, "id": "1"}],
                )
            return _submit("complete")

    result = AgentLoopController().run(
        spec=_spec(
            name="controller_smoke",
            prompt_version="test-v1",
            budget=AgentBudgetPolicy(max_wall_seconds=10, max_tool_result_chars=100_000),
            tools=(RegisteredTool(tool=tool, contract=ToolContract(name=tool.name)),),
        ),
        request=_request(),
        model=Model(),
    )

    assert result.status == "succeeded"
    assert result.value == _Final(value="complete")
    assert calls == ["source-1"]
    assert result.execution_id


def test_controller_uses_declared_model_projection_for_large_tool_results() -> None:
    """完整结果可留给审计/领域工件，不能挤占模型的下一轮上下文。"""

    def search() -> dict[str, object]:
        return {
            "retrieval_id": "retrieval-1",
            "references": [{"chunk_content": "x" * 20_000}],
        }

    tool = StructuredTool.from_function(search, name="search", description="test only")

    class Model:
        def invoke(self, messages: list[object]) -> AIMessage:
            tool_messages = [item for item in messages if isinstance(item, ToolMessage)]
            if not tool_messages:
                return AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "1"}])
            projection = json.loads(str(tool_messages[-1].content))
            assert projection == {"retrieval_id": "retrieval-1", "reference_count": 1}
            return _submit("complete")

    result = AgentLoopController().run(
        spec=_spec(
            name="projected_tool_result",
            prompt_version="test-v1",
            budget=AgentBudgetPolicy(max_wall_seconds=10, max_tool_result_chars=500),
            tools=(
                RegisteredTool(
                    tool=tool,
                    contract=ToolContract(
                        name=tool.name,
                        model_result_projector=lambda result: {
                            "retrieval_id": result["retrieval_id"],  # type: ignore[index]
                            "reference_count": len(result["references"]),  # type: ignore[index]
                        },
                    ),
                ),
            ),
        ),
        request=_request(),
        model=Model(),
    )

    assert result.status == "succeeded"


def test_controller_rejects_stale_input_before_model_invocation() -> None:
    calls = 0

    class Model:
        def invoke(self, _messages: list[object]) -> AIMessage:
            nonlocal calls
            calls += 1
            return _submit("unexpected")

    result = AgentLoopController().run(
        spec=_spec(
            name="stale_input_smoke",
            prompt_version="test-v1",
            budget=AgentBudgetPolicy(max_wall_seconds=10, max_tool_result_chars=100_000),
        ),
        request=_request(input_revision=4, resolver=lambda: 5),
        model=Model(),
    )

    assert result.status == "stale"
    assert result.terminal_reason == "stale_input_revision"
    assert calls == 0


def test_controller_does_not_make_a_final_model_call_after_wall_deadline() -> None:
    calls = 0

    class Model:
        def invoke(self, _messages: list[object]) -> AIMessage:
            nonlocal calls
            calls += 1
            return _submit("unexpected")

    result = AgentLoopController().run(
        spec=_spec(
            name="deadline_smoke",
            prompt_version="test-v1",
            budget=AgentBudgetPolicy(max_wall_seconds=0, max_tool_result_chars=100_000),
        ),
        request=_request(),
        model=Model(),
    )

    assert result.status == "blocked"
    assert result.terminal_reason == "wall_deadline"
    assert calls == 0


def test_controller_compacts_initial_context_before_first_model_call() -> None:
    compacted = 0

    def compact(messages: list[object], **_kwargs: object) -> tuple[list[object], str]:
        nonlocal compacted
        compacted += 1
        return [HumanMessage(content="compact")], "test compaction"

    class Model:
        def invoke(self, messages: list[object]) -> AIMessage:
            assert messages == [HumanMessage(content="compact")]
            return _submit("complete")

    result = AgentLoopController().run(
        spec=_spec(
            name="initial_compaction_smoke",
            prompt_version="test-v1",
            budget=AgentBudgetPolicy(
                max_wall_seconds=10,
                max_tool_result_chars=100_000,
                compaction_threshold_chars=10,
            ),
            context_compactor=compact,
        ),
        request=AgentExecutionRequest(
            owner_type="runtime_cycle",
            owner_id="initial-compaction",
            scope=RunScope(),
            messages=(HumanMessage(content="a" * 100),),
        ),
        model=Model(),
    )

    assert result.status == "succeeded"
    assert compacted == 1
