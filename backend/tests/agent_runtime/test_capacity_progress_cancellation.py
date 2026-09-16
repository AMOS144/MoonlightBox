"""完整请求容量恢复、稳定结果比较与真实异步请求取消的离线冒烟。"""

import asyncio
from dataclasses import replace
from time import monotonic

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from moonlightbox.agent_runtime import AgentBudgetPolicy, AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.capacity import check_request
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest, RegisteredTool, ToolContract
from moonlightbox.agent_runtime.submission import result_submission_tool
from pydantic import BaseModel


class Final(BaseModel):
    text: str


def submit():
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "finish",
                "name": "finish",
                "args": {"result": {"text": "ok"}},
            }
        ],
    )


def spec(**kwargs):
    return AgentSpec(
        name="smoke",
        prompt_version="1",
        submission_tool_name="finish",
        budget=AgentBudgetPolicy(
            max_wall_seconds=30,
            max_tool_result_chars=100000,
            context_window_tokens=1600,
            context_safety_margin_tokens=20,
        ),
        tools=(result_submission_tool("finish", Final),),
        **kwargs,
    )


def request(content="task"):
    return AgentExecutionRequest(
        owner_type="person_world",
        owner_id="smoke",
        scope=RunScope(),
        messages=(HumanMessage(content=content),),
    )


def test_cancelled_owner_cannot_deliver_successful_checkpoint(tmp_path):
    class Model:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            return submit()

    model = Model()
    execution = replace(request(), checkpoint_path=str(tmp_path / "loop.sqlite"))
    first = AgentLoopController().run(spec=spec(), request=execution, model=model)
    assert first.status == "succeeded"
    cancelled = replace(execution, cancellation_requested=lambda: True)
    second = AgentLoopController().run(spec=spec(), request=cancelled, model=model)
    assert second.status == "cancelled"
    assert second.value is None
    assert model.calls == 1


@pytest.mark.parametrize("fits", [False, True])
def test_native_client_capacity_recovery_sends_only_compacted_request(fits):
    from moonlightbox.runtime_v1.cloud_models import RuntimeCloudChatModel, RuntimeCloudClient

    sent, compacted = [], []

    def transport(request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "finish",
                                    "type": "function",
                                    "function": {
                                        "name": "finish",
                                        "arguments": '{"result":{"text":"ok"}}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    def compact(messages, **kwargs):
        compacted.append(True)
        return [messages[0], HumanMessage(content="summary")], "summary"

    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        client = RuntimeCloudClient(
            endpoint="https://test.invalid/chat/completions",
            model="test",
            api_key="test-only",
            timeout_seconds=10,
            thinking_mode="default",
            max_output_tokens=100,
            client=http,
        )
        execution = replace(
            request(), messages=(SystemMessage(content="system"), HumanMessage(content="中" * 2000))
        )
        configured = spec(context_compactor=compact)
        if fits:
            configured = replace(
                configured,
                budget=replace(
                    configured.budget,
                    context_window_tokens=10000,
                    compaction_threshold_chars=100,
                    max_context_chars=200,
                ),
            )
        result = AgentLoopController().run(
            spec=configured,
            request=execution,
            model=RuntimeCloudChatModel(client, temperature=0),
        )
    assert result.status == "succeeded"
    assert len(sent) == 1
    assert len(compacted) == (0 if fits else 1)
    assert ("summary" in sent[0].content.decode()) is not fits


@pytest.mark.parametrize("effective", [True, False])
def test_complete_request_limit_drives_compaction(effective):
    compressed = []
    sent = []

    def compact(messages, **kwargs):
        compressed.append(True)
        return ([HumanMessage(content="short")] if effective else list(messages)), "summary"

    class Model:
        def invoke(self, messages):
            # 包括 schema、推理字段、输出预留；长度不足旧字符阈值也必须触发压缩。
            check_request(
                {
                    "messages": [m.content for m in messages],
                    "max_tokens": 400,
                    "tools": [{"schema": "中" * 100}],
                    "reasoning": "中" * 100,
                },
                "test",
            )
            sent.append(True)
            return submit()

    result = AgentLoopController().run(
        spec=spec(context_compactor=compact), request=request("中" * 1100), model=Model()
    )
    assert len(compressed) == 1
    assert len(sent) == int(effective)
    assert result.status == ("succeeded" if effective else "blocked")


def test_ineffective_rebuilt_request_does_not_compact_forever():
    count = []

    def compact(messages, **kwargs):
        count.append(True)
        return [HumanMessage(content=str(len(count)))], "smaller"

    class Model:
        def invoke(self, messages):
            # 不可压缩的 schema 本身超限；改变消息不能无限重试。
            check_request({"messages": [m.content for m in messages], "tools": "中" * 3000}, "test")

    result = AgentLoopController().run(
        spec=spec(context_compactor=compact), request=request(), model=Model()
    )
    assert result.status == "blocked"
    assert len(count) <= 2


@pytest.mark.parametrize("new_content", [False, True])
def test_retrieval_ids_do_not_hide_repeated_results(new_content):
    from moonlightbox.world.person_world.investigation_artifacts import InvestigationArtifactStore
    from moonlightbox.world.person_world.tools.comparison import comparison_for

    artifacts = InvestigationArtifactStore()

    def search():
        a = artifacts.add_retrieval(question="same", mode="mix", references=[])
        return {
            "retrieval_id": a.retrieval_id,
            "context": a.retrieval_id if new_content else "same",
        }

    tool = StructuredTool.from_function(search, description="smoke")

    class Model:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            return (
                submit()
                if self.calls > 8
                else AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": str(self.calls),
                            "name": "search",
                            "args": {},
                        }
                    ],
                )
            )

    policy = spec()
    policy = replace(
        policy,
        tools=(
            *policy.tools,
            RegisteredTool(
                tool,
                ToolContract(
                    name="search", comparison_projection=comparison_for("search_world", artifacts)
                ),
            ),
        ),
    )
    result = AgentLoopController().run(spec=policy, request=request(), model=Model())
    assert result.status == ("succeeded" if new_content else "blocked")


def test_cancellation_closes_owned_async_request(monkeypatch):
    from moonlightbox.agent_runtime import http_transport
    from moonlightbox.agent_runtime.resilience import (
        ExecutionInterrupted,
        ResiliencePolicy,
        execution_scope,
    )

    exited = []

    async def slow(request):
        try:
            await asyncio.sleep(30)
        finally:
            exited.append(True)
        return httpx.Response(200, json={})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        http_transport.httpx,
        "AsyncClient",
        lambda: real_client(transport=httpx.MockTransport(slow)),
    )
    start = monotonic()
    with execution_scope(
        ResiliencePolicy(), lambda: 10, lambda: "cancelled" if monotonic() - start > 0.15 else None
    ):
        with pytest.raises(ExecutionInterrupted, match="cancelled"):
            http_transport.post(None, "https://test.invalid", cancellable=True)
    assert exited == [True]
    assert monotonic() - start < 2
