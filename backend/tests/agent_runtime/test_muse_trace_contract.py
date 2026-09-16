"""Muse 风格的层次、异常语义和完整快照契约，无真实模型请求。"""

from dataclasses import replace

import pytest
from moonlightbox.observability import phoenix
from moonlightbox.observability.investigation import build_investigation, unpack_snapshots
from moonlightbox.observability.runtime import runtime_cycle_span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(phoenix.trace, "get_tracer", lambda *a, **kw: provider.get_tracer("test"))
    monkeypatch.setattr(
        phoenix,
        "_STATE",
        replace(phoenix._STATE, enabled=True, capture_content=True, max_characters=10000),
    )
    yield exporter
    provider.shutdown()


def test_cycle_error_has_one_root_and_agent_keeps_link(spans):
    with pytest.raises(ValueError, match="invalid decision"):
        with runtime_cycle_span(
            cycle_id="c",
            cycle_key="k",
            project_id="p",
            branch_id="b",
            input_value={"message": "hello"},
        ):
            with phoenix._span(
                "moonlightbox.agent.run",
                kind=phoenix.OpenInferenceSpanKindValues.AGENT,
                detach_ambient_parent=True,
            ):
                pass
            raise ValueError("invalid decision")
    finished = spans.get_finished_spans()
    cycle = next(s for s in finished if s.name == "moonlightbox.runtime.cycle")
    agent = next(s for s in finished if s.name == "moonlightbox.agent.run")
    assert cycle.attributes["openinference.span.kind"] == "CHAIN"
    assert agent.attributes["moonlightbox.cycle.id"] == "c"
    assert agent.links[0].context.span_id == cycle.context.span_id
    assert len(cycle.events) == 1
    assert cycle.status.status_code.name == "ERROR"
    assert not any(s.name.endswith("cycle_failure") for s in finished)


def test_expected_supersession_not_system_error(spans):
    with pytest.raises(RuntimeError):
        with phoenix.chain_span("runtime.test"):
            raise RuntimeError("stale_input_revision")
    assert spans.get_finished_spans()[0].status.status_code.name == "OK"


def test_invalid_output_cannot_be_green_even_from_old_checkpoint(spans):
    with phoenix.chain_span("test") as span:
        phoenix.record_agent_outcome(
            span, status="succeeded", reason="invalid_final_output", execution_id="e", result=None
        )
    assert spans.get_finished_spans()[0].status.status_code.name == "ERROR"


def test_exporter_setup_failure_does_not_block_business(monkeypatch, spans):
    def broken(*a, **kw):
        raise RuntimeError("exporter down")

    monkeypatch.setattr(phoenix.trace, "get_tracer", broken)
    with phoenix.chain_span("test") as span:
        assert span is None
    with pytest.raises(ValueError, match="business"):
        with phoenix.chain_span("test"):
            raise ValueError("business")


def test_snapshots_roundtrip_and_missing_chunk_is_explicit(spans):
    value = {"messages": ["测试" * 40000]}
    with phoenix.chain_span("test") as span:
        phoenix.add_context_snapshot(span, event_name="input", value=value)
    exported = spans.get_finished_spans()[0]
    raw = {"events": [{"name": e.name, "attributes": dict(e.attributes)} for e in exported.events]}
    assert unpack_snapshots(raw)[0]["value"] == value
    raw["events"] = raw["events"][1:]
    assert unpack_snapshots(raw)[0]["complete"] is False
    assert unpack_snapshots(raw)[0]["value"] is None


def test_investigation_deduplicates_propagated_errors():
    attempt = {
        "context": {"trace_id": "t", "span_id": "s"},
        "name": "moonlightbox.provider.http_attempt",
        "status_code": "ERROR",
    }
    parent = {
        "context": {"trace_id": "t", "span_id": "root"},
        "name": "moonlightbox.runtime.cycle",
        "status_code": "ERROR",
    }
    report = build_investigation([attempt, parent, attempt])
    assert report["summary"]["http_failed_attempts"] == 1
    assert len(report["spans"]) == 2


def test_cycle_investigation_collects_independent_agent_trees(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from moonlightbox.config import Settings
    from moonlightbox.observability import router

    root = {"context": {"trace_id": "cycle", "span_id": "c"}, "name": "moonlightbox.runtime.cycle"}
    agent = {"context": {"trace_id": "agent", "span_id": "a"}, "name": "moonlightbox.agent.run"}
    tool = {"context": {"trace_id": "agent", "span_id": "t"}, "parent_id": "a", "name": "tool"}

    def query(**kwargs):
        if kwargs.get("trace_id") == "agent":
            return [agent, tool]
        if kwargs.get("trace_id") == "cycle":
            return [root]
        return [root, agent]

    monkeypatch.setattr(router, "_get_spans", query)
    app = FastAPI()
    app.include_router(router.create_observability_router(Settings(phoenix_enabled=True)))
    response = TestClient(app).get("/api/observability/runtime-cycles/example")
    assert response.status_code == 200
    assert len(response.json()["spans"]) == 3
    assert response.json()["summary"]["agent_executions"] == 1


def test_request_diagnostics_detect_prefix_change():
    from moonlightbox.observability.execution_context import (
        AgentExecutionTraceContext,
        agent_execution_trace_scope,
    )
    from moonlightbox.observability.provider import request_diagnostics

    context = AgentExecutionTraceContext(
        execution_id="e",
        agent_name="test",
        prompt_version="v",
        owner_type="test",
        owner_id="o",
        project_id=None,
        branch_id=None,
        target_person_id=None,
        input_revision=1,
    )
    with agent_execution_trace_scope(context):
        request = {"model": "m", "messages": [{"role": "system", "content": "a"}]}
        assert request_diagnostics(request)["comparison_available"] is False
        assert request_diagnostics(request)["identical_request"] is True
        request["messages"][0]["content"] = "b"
        result = request_diagnostics(request)
        assert result["prefix_changed"] is True
        assert result["parameters_changed"] is False
