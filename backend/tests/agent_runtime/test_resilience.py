"""同一策略覆盖两种真实客户端，模拟网络不访问外部服务。"""

from threading import Event

import httpx
import pytest
from moonlightbox.agent_runtime.resilience import (
    ExecutionInterrupted,
    ResiliencePolicy,
    classify_failure,
    execution_scope,
    run_operation,
)
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.runtime_v1.cloud_models import RuntimeCloudClient
from pydantic import BaseModel


class Result(BaseModel):
    value: str


@pytest.mark.parametrize("structured", [False, True])
def test_both_clients_use_one_retry_policy_not_multiplied(structured):
    calls, statuses = [], []

    def transport(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503, json={"error": "temporary"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"value":"ok"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        options = dict(
            endpoint="https://example.test/chat/completions",
            model="test",
            api_key="test",
            max_retries=9,
            client=http,
        )
        if structured:
            client = NodeAnalysisCloudClient(enabled=True, **options)

            def invoke():
                return client.create_structured_completion(
                    system_content="test",
                    user_content="test",
                    response_model=Result,
                    request_max_retries=8,
                )
        else:
            client = RuntimeCloudClient(
                timeout_seconds=30, thinking_mode="default", max_output_tokens=100, **options
            )

            def invoke():
                return client.complete(system_prompt="test", messages=[], temperature=0)

        with execution_scope(
            ResiliencePolicy(backoff_seconds=0), lambda: 30, lambda: None, statuses.append
        ):
            invoke()
    assert len(calls) == 3
    assert [s["status"] for s in statuses].count("retrying") == 2
    assert all(request.extensions["timeout"]["read"] <= 30 for request in calls)


def test_authentication_not_retried_and_transport_exhaustion_is_typed():
    for code, count in [(401, 1), (503, 3)]:
        calls = []

        def fail(calls=calls, code=code):
            calls.append(None)
            response = httpx.Response(code, request=httpx.Request("GET", "https://test.invalid"))
            response.raise_for_status()

        with execution_scope(ResiliencePolicy(backoff_seconds=0), lambda: 30, lambda: None):
            with pytest.raises(httpx.HTTPStatusError) as caught:
                run_operation(fail)
        failure = classify_failure(caught.value)
        assert len(calls) == count
        assert failure.attempts == count
        assert failure.retryable == (code == 503)


def test_cancel_during_backoff_stops_before_next_request():
    cancelled = Event()
    calls = []

    def fail():
        calls.append(None)
        raise httpx.ConnectError("offline")

    def status(event):
        if event["status"] == "retrying":
            cancelled.set()

    with execution_scope(
        ResiliencePolicy(), lambda: 30, lambda: "cancelled" if cancelled.is_set() else None, status
    ):
        with pytest.raises(ExecutionInterrupted, match="cancelled"):
            run_operation(fail)
    assert len(calls) == 1


def test_late_result_after_cancel_is_not_delivered():
    cancelled = Event()

    def callback():
        cancelled.set()
        return "late result"

    with execution_scope(
        ResiliencePolicy(), lambda: 30, lambda: "cancelled" if cancelled.is_set() else None
    ):
        with pytest.raises(ExecutionInterrupted, match="cancelled"):
            run_operation(callback)


def test_mutation_is_not_automatically_replayed():
    calls = []

    def mutate():
        calls.append(None)
        raise TimeoutError("outcome unknown")

    with execution_scope(ResiliencePolicy(backoff_seconds=0), lambda: 30, lambda: None):
        with pytest.raises(TimeoutError):
            run_operation(mutate, kind="tool:submit", replay_safe=False)
    assert len(calls) == 1


def test_no_new_request_after_execution_budget_exhausted():
    with execution_scope(ResiliencePolicy(), lambda: 0, lambda: None):
        with pytest.raises(ExecutionInterrupted, match="wall_deadline"):
            run_operation(lambda: pytest.fail("不得启动请求"))


def test_controller_cancellation_has_same_error_and_status_callback():
    from langchain_core.messages import HumanMessage
    from moonlightbox.agent_runtime import (
        AgentBudgetPolicy,
        AgentLoopController,
        AgentSpec,
        RunScope,
    )
    from moonlightbox.agent_runtime.contracts import AgentExecutionRequest
    from moonlightbox.agent_runtime.submission import result_submission_tool

    statuses = []
    result = AgentLoopController().run(
        spec=AgentSpec(
            name="cancel-test",
            submission_tool_name="finish",
            tools=(result_submission_tool("finish", Result),),
            prompt_version="test",
            budget=AgentBudgetPolicy(max_wall_seconds=30, max_tool_result_chars=10000),
        ),
        request=AgentExecutionRequest(
            owner_type="runtime_cycle",
            owner_id="test",
            scope=RunScope(),
            messages=(HumanMessage(content="test"),),
            cancellation_requested=lambda: True,
            on_status=statuses.append,
        ),
        model=None,
    )
    assert result.status == "cancelled"
    assert result.error["code"] == "cancelled"
    assert result.error["retryable"] is False
    assert statuses[-1]["status"] == "cancelled"


def test_nested_model_request_does_not_multiply_tool_retries():
    calls = []

    def network():
        calls.append(None)
        raise httpx.ConnectError("offline")

    with execution_scope(ResiliencePolicy(backoff_seconds=0), lambda: 30, lambda: None):
        with pytest.raises(httpx.ConnectError):
            run_operation(lambda: run_operation(network), kind="tool:read")
    assert len(calls) == 3


@pytest.mark.parametrize("compound", [False, True])
def test_controller_retries_request_not_successful_compound_prefix(compound):
    from time import time
    from types import SimpleNamespace

    from moonlightbox.agent_runtime import AgentLoopController
    from moonlightbox.agent_runtime.resilience import model_request

    calls, prefix = [], []

    @model_request
    def request():
        calls.append(None)
        if len(calls) < 3:
            raise httpx.ConnectError("offline")
        return "ok"

    class Model:
        has_managed_request_boundaries = compound

        def invoke(self, messages):
            prefix.append(None)
            return request()

    context = SimpleNamespace(
        spec=SimpleNamespace(
            resilience=ResiliencePolicy(backoff_seconds=0),
            budget=SimpleNamespace(max_wall_seconds=30),
        ),
        request=SimpleNamespace(
            cancellation_requested=None,
            input_revision_resolver=None,
            on_status=None,
        ),
    )
    result = AgentLoopController()._invoke_model(
        Model(),
        [],
        {"started_wall": time()},
        context,
    )
    assert result == "ok"
    assert len(calls) == 3
    assert len(prefix) == (1 if compound else 3)
