"""Phoenix 只读摘要 API 的契约测试。"""

from moonlightbox.observability.router import _agent_execution_summary


def test_agent_execution_summary_reads_tool_counts_from_phoenix_root_span() -> None:
    summary = _agent_execution_summary(
        {
            "context": {"trace_id": "trace-123"},
            "attributes": {
                "moonlightbox.agent.terminal_reason": "success",
                "moonlightbox.agent.tool_calls": 14,
                "moonlightbox.agent.tool_success_calls": 9,
                "moonlightbox.agent.tool_error_calls": 3,
                "moonlightbox.agent.tool_empty_calls": 1,
                "moonlightbox.agent.tool_cancelled_calls": 1,
            },
        },
        execution_id="execution-123",
        phoenix_base_url="http://phoenix:6006",
    )

    assert summary == {
        "execution_id": "execution-123",
        "phoenix_base_url": "http://phoenix:6006",
        "trace_id": "trace-123",
        "terminal_reason": "success",
        "tool_call_count": 14,
        "tool_success_count": 9,
        "tool_error_count": 3,
        "tool_empty_count": 1,
        "tool_cancelled_count": 1,
    }
