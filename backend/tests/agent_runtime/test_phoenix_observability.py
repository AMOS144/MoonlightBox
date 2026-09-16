"""Phoenix 观察层的确定性冒烟测试。

这些测试不要求启动 Phoenix Collector；它们锁住“只以结构化 provenance 判断消费”、
“供应商 usage 不可伪造”以及“结构化编译器不会躲在 Agent decision 黑盒中”三个约定。
"""

from __future__ import annotations

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.observability.execution_context import (
    AgentExecutionTraceContext,
    agent_execution_trace_scope,
)
from moonlightbox.observability.trace_summary import (
    AgentTraceState,
    build_root_cause_snapshot,
    extract_structured_references,
    make_round_observation,
    record_external_provider_call,
    record_model_result,
    record_tool_result,
)
from pydantic import BaseModel


class _StructuredResult(BaseModel):
    value: str


def test_native_requests_include_failed_attempt_without_double_counting_usage():
    from moonlightbox.runtime_v1.cloud_models import RuntimeCloudChatModel, RuntimeCloudClient

    context = AgentExecutionTraceContext(
        execution_id="native",
        agent_name="person_world.section.identity",
        prompt_version="v3",
        owner_type="person_world",
        owner_id="smoke",
        project_id="p",
        branch_id=None,
        target_person_id="t",
        input_revision=1,
    )
    responses = iter(
        [
            httpx.Response(500, json={"error": {"message": "temporary"}}),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
                },
            ),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(lambda request: next(responses))) as http:
        client = RuntimeCloudClient(
            endpoint="https://example.com/v1/chat/completions",
            model="test-model",
            api_key="test-key",
            max_retries=1,
            client=http,
            timeout_seconds=10,
            thinking_mode="default",
            max_output_tokens=1024,
        )
        model = RuntimeCloudChatModel(client, temperature=0.2)
        messages = [SystemMessage(content="system"), HumanMessage(content="user")]
        with agent_execution_trace_scope(context):
            output = model.invoke(messages)
            record_model_result(
                context.state,
                observation=make_round_observation(
                    phase="decision", round_index=0, messages=messages, tool_schemas=[]
                ),
                output=output,
            )
        assert len(context.state.provider_calls) == 2
        assert context.state.total_prompt_tokens == 12
        assert context.state.total_tokens == 19


def test_trace_only_treats_declared_provenance_as_consumed() -> None:
    state = AgentTraceState()
    tool_entry = record_tool_result(
        state,
        tool_name="find_source",
        arguments={"question": "工作在哪里"},
        result={"source_ids": ["message-42"]},
        status="succeeded",
        duration_ms=4,
        retry_count=0,
    )
    observation = make_round_observation(
        phase="decision",
        round_index=0,
        messages=[SystemMessage(content="system"), HumanMessage(content="question")],
        tool_schemas=[],
    )

    # 普通文本即使恰好写出了 ID 也不是可验证的“使用”；必须由模型输出一个协议字段。
    assert extract_structured_references("I saw message-42") == frozenset()
    record_model_result(
        state,
        observation=observation,
        output=AIMessage(content='{"evidence_ids":["message-42"]}'),
    )

    assert tool_entry["consumed_in_next_model_turn"] is True


def test_structured_cloud_call_records_real_usage_in_current_agent_context() -> None:
    trace_context = AgentExecutionTraceContext(
        execution_id="execution-1",
        agent_name="person_world.section.identity",
        prompt_version="prompt-v1",
        owner_type="person_world",
        owner_id="review-1",
        project_id="project-1",
        branch_id=None,
        target_person_id="target-1",
        input_revision=1,
    )
    client = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://example.com/v1/chat/completions",
        model="test-model",
        api_key="test-key",
        max_retries=0,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": '{"value":"ok"}'}}],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 7,
                            "total_tokens": 19,
                        },
                    },
                )
            )
        ),
    )

    with agent_execution_trace_scope(trace_context):
        result = client.create_structured_completion(
            system_content="system",
            user_content="user",
            response_model=_StructuredResult,
            operation_id="test-operation",
        )

    assert result == _StructuredResult(value="ok")
    assert trace_context.state.provider_calls == [
        {
            "provider": "cognition",
            "model_name": "test-model",
            "operation_id": "test-operation",
            "attempts": 1,
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        }
    ]
    summary = build_root_cause_snapshot(trace_context.state)
    assert summary["usage"]["cache_support"] == "not_reported_by_provider"
    assert summary["usage"]["total_tokens"] == 19


def test_root_summary_keeps_direct_provider_calls_separate_from_agent_rounds() -> None:
    state = AgentTraceState()
    record_external_provider_call(
        state,
        provider="cognition",
        model_name="model",
        operation_id="schema-repair",
        usage={"input_tokens": 3, "output_tokens": 5},
        attempts=2,
    )

    summary = build_root_cause_snapshot(state)

    assert summary["execution"]["model_calls"] == 0
    assert summary["execution"]["provider_model_calls"] == 1
    assert summary["usage"]["total_tokens"] == 8


def test_root_summary_preserves_minimax_nested_reasoning_and_cache_usage() -> None:
    state = AgentTraceState()
    record_external_provider_call(
        state,
        provider="cognition",
        model_name="MiniMax-M3",
        operation_id="section-plan",
        usage={
            "prompt_tokens": 20,
            "completion_tokens": 30,
            "total_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 7},
            "completion_tokens_details": {"reasoning_tokens": 19},
        },
        attempts=1,
    )

    summary = build_root_cause_snapshot(state)

    assert summary["usage"]["prompt_tokens"] == 20
    assert summary["usage"]["completion_tokens"] == 30
    assert summary["usage"]["total_tokens"] == 50
    assert summary["usage"]["reasoning_tokens"] == 19
    assert summary["usage"]["cache_read_tokens"] == 7
    assert summary["usage"]["cache_write_tokens"] == 0
    assert summary["usage"]["cache_support"] == "reported"
    assert summary["usage"]["provider_usage_reported"] is True
