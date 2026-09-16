"""完整请求容量守卫：不访问真实供应商。"""

from dataclasses import replace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from moonlightbox.agent_runtime.capacity import (
    capacity_scope,
    check_request,
    estimate_tokens,
    observe_usage,
)
from moonlightbox.agent_runtime.contracts import AgentBudgetPolicy
from moonlightbox.agent_runtime.resilience import ExecutionInterrupted
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.runtime_v1.cloud_models import RuntimeCloudChatModel, RuntimeCloudClient
from pydantic import BaseModel, Field


def policy(**kwargs):
    return AgentBudgetPolicy(max_wall_seconds=30, max_tool_result_chars=10000, **kwargs)


def test_m3_uses_documented_minimum_window_without_relaxing_unknown_models():
    body = {
        "model": "MiniMax-M3",
        "messages": [{"role": "user", "content": "a" * 440000}],
        "max_tokens": 24576,
    }
    with capacity_scope(policy(), {}):
        assert check_request(body, "endpoint") is not None
        with pytest.raises(ExecutionInterrupted, match="input_context_limit"):
            check_request({**body, "model": "unknown-model"}, "endpoint")


def test_complete_package_and_reserved_output():
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}
    baseline = estimate_tokens(body)
    with capacity_scope(
        policy(context_window_tokens=baseline + 100, context_safety_margin_tokens=0), {}
    ):
        check_request(body, "endpoint")
        for extra in (
            {"tools": [{"description": "参数说明" * 1000}]},
            {"response_format": {"json_schema": {"description": "输出说明" * 1000}}},
            {"messages": [{"role": "assistant", "reasoning_details": ["推理" * 1000]}]},
            {"max_tokens": 101},
        ):
            with pytest.raises(ExecutionInterrupted, match="input_context_limit"):
                check_request({**body, **extra}, "endpoint")


def test_usage_calibration_is_conservative_scoped_and_missing_usage_is_safe():
    body = {"model": "m", "messages": [], "max_tokens": 1}
    calibration = {}
    raw = estimate_tokens(body)
    with capacity_scope(
        policy(context_window_tokens=raw * 2, context_safety_margin_tokens=0), calibration
    ):
        sample = check_request(body, "a")
        observe_usage(sample, {"usage": {"prompt_tokens": raw * 3, "cached_tokens": raw}})
        observe_usage(sample, {"usage": {"prompt_tokens": 1}})
        observe_usage(sample, {})
        assert calibration["a|m"] == 3
        with pytest.raises(ExecutionInterrupted):
            check_request(body, "a")
        check_request(body, "b")
        check_request({**body, "model": "other"}, "a")
    # 同一校正账本在恢复后继续生效。
    with capacity_scope(
        policy(context_window_tokens=raw * 2, context_safety_margin_tokens=0), calibration
    ):
        with pytest.raises(ExecutionInterrupted):
            check_request(body, "a")


def test_model_window_override():
    p = policy(context_window_tokens=20, context_safety_margin_tokens=0)
    with capacity_scope(replace(p, model_context_windows={"large": 1000}), {}):
        check_request({"model": "large", "max_tokens": 10}, "a")
        with pytest.raises(ExecutionInterrupted):
            check_request({"model": "small", "max_tokens": 10}, "a")


@pytest.mark.parametrize("structured", [False, True])
def test_real_client_guards_full_wire_request_before_http(structured):
    calls = []

    def transport(request):
        calls.append(request)
        pytest.fail("超限请求不应到达 HTTP")

    class Output(BaseModel):
        value: str = Field(description="详细输出说明" * 2000)

    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        options = dict(
            endpoint="https://example.test/chat/completions",
            model="test",
            api_key="test",
            client=http,
        )
        with capacity_scope(policy(context_window_tokens=4000, context_safety_margin_tokens=0), {}):
            with pytest.raises(ExecutionInterrupted, match="input_context_limit"):
                if structured:
                    NodeAnalysisCloudClient(
                        enabled=True, response_format="json_schema", **options
                    ).create_structured_completion(
                        system_content="test",
                        user_content="hi",
                        response_model=Output,
                        max_output_tokens=100,
                    )
                else:
                    model = RuntimeCloudChatModel(
                        RuntimeCloudClient(
                            timeout_seconds=30,
                            thinking_mode="default",
                            max_output_tokens=100,
                            **options,
                        ),
                        temperature=0,
                    )
                    model.invoke(
                        [
                            SystemMessage(content="test"),
                            HumanMessage(content="hi"),
                            AIMessage(
                                content="ok",
                                additional_kwargs={"reasoning_content": "推理内容" * 2000},
                            ),
                        ]
                    )
    assert not calls
