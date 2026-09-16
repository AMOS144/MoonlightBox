"""外围任务原生提交、可修复错误、策略注入和恢复的离线冒烟。"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from moonlightbox.agent_runtime.contracts import AgentBudgetPolicy, AgentSpec
from moonlightbox.agent_runtime.policy import inject_policy
from moonlightbox.agent_runtime.resilience import ResiliencePolicy
from moonlightbox.agent_runtime.tasks import AgentTaskError
from moonlightbox.world.client import LightRAGEntity
from moonlightbox.world.merges import generate_merge_candidates
from moonlightbox.world.person_world.jobs import _validate_regressions
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def call(name, args):
    return AIMessage(content="", tool_calls=[{"id": name, "name": name, "args": args}])


class Compiler:
    def __init__(self, invoke):
        self.invoke = invoke
        self.calls = 0

    def create_agent_chat_model(self):
        compiler = self

        class Model:
            def bind_tools(self, tools, **kwargs):
                self.names = {tool.name for tool in tools}
                return self

            def invoke(self, messages):
                compiler.calls += 1
                return compiler.invoke(messages)

        return Model()


def test_alias_repairs_submission_without_heuristic_fallback():
    payload = {"source_entities": ["羽"], "target_entity": "洪欣羽", "reason": "原文中的称呼"}

    def invoke(messages):
        seen = [m for m in messages if isinstance(m, ToolMessage)]
        if not seen:
            return call(
                "submit_alias_resolution",
                {"result": {"proposals": [{**payload, "target_entity": "不存在的人"}]}},
            )
        assert "rejected" in seen[-1].content
        return call("submit_alias_resolution", {"result": {"proposals": [payload]}})

    sidecar = SimpleNamespace(
        list_entities=lambda workspace: [
            LightRAGEntity(entity_name=name, graph_data={"entity_type": "person"})
            for name in ["羽", "洪欣羽"]
        ]
    )
    compiler = Compiler(invoke)
    result = generate_merge_candidates(
        sidecar=sidecar,
        compiler=compiler,
        workspace="world",
        subject_name="洪欣羽",
        user_name="用户",
    )
    assert result[0].target_entity == "洪欣羽"
    assert compiler.calls == 2


def test_alias_model_failure_is_not_silently_converted_to_candidates():
    def fail(messages):
        raise ValueError("provider-invalid")

    sidecar = SimpleNamespace(
        list_entities=lambda workspace: [
            LightRAGEntity(entity_name=name, graph_data={"entity_type": "person"})
            for name in ["羽", "洪欣羽"]
        ]
    )
    with pytest.raises(AgentTaskError):
        generate_merge_candidates(
            sidecar=sidecar,
            compiler=Compiler(fail),
            workspace="world",
            subject_name="洪欣羽",
            user_name="用户",
        )


def test_regression_submission_repairs_missing_query_and_restores_result(tmp_path):
    query = "谁十点上班"

    def invoke(messages):
        seen = [m for m in messages if isinstance(m, ToolMessage)]
        if not seen:
            return call("read_regression_query", {"query": query})
        if len(seen) == 1:
            return call("submit_graph_regression", {"result": {"checks": []}})
        assert "rejected" in seen[-1].content
        return call(
            "submit_graph_regression",
            {
                "result": {
                    "checks": [{"query": query, "passed": True, "explanation": "主体已纠正为用户"}]
                }
            },
        )

    compiler = Compiler(invoke)
    reads = []

    def search(*args, **kwargs):
        reads.append(True)
        return SimpleNamespace(context="用户十点上班", references=[])

    graph = SimpleNamespace(id="graph", workspace_key="world", project_id="project")
    change = SimpleNamespace(
        id="change", regression_queries=[{"query": query, "expected_change": "用户"}]
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'world.db'}")
    with Session(engine) as session:
        args = dict(
            session=session,
            compiler=compiler,
            sidecar=SimpleNamespace(query=search),
            graph=graph,
            change_set=change,
            correction_payloads=[],
        )
        first = _validate_regressions(**args)
        second = _validate_regressions(**args)
    assert first == second
    assert first.checks[0].actual_context_excerpt == "用户十点上班"
    assert compiler.calls == 3
    assert reads == [True]


def test_policy_inherits_model_connection_and_respects_agent_override():
    budget = AgentBudgetPolicy(max_wall_seconds=30, max_tool_result_chars=10000)
    spec = AgentSpec(name="test", prompt_version="v1", budget=budget, submission_tool_name="submit")
    model = SimpleNamespace(
        agent_resilience=ResiliencePolicy(request_timeout_seconds=47, max_retries=4)
    )
    assert inject_policy(spec, model).resilience == model.agent_resilience
    explicit = ResiliencePolicy(request_timeout_seconds=10, max_retries=0)
    assert inject_policy(replace(spec, resilience=explicit), model).resilience == explicit


def test_director_does_not_short_circuit_on_legacy_packet_budget():
    from datetime import UTC, datetime

    from moonlightbox.runtime_v1.director import DirectorAgent
    from moonlightbox.runtime_v1.schemas import ContextPacket

    class Model:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise ValueError("offline model failure")

    now = datetime.now(UTC)
    packet = ContextPacket(
        generated_at=now,
        virtual_now=now,
        timezone="UTC",
        trigger={},
        origin={},
        current={},
        branch={},
        budgets={"budget_exceeded": True},
    )
    model = Model()
    result = DirectorAgent(model).run_with_trace(packet)
    assert model.calls == 1
    assert result.terminal_reason == "model_error"
