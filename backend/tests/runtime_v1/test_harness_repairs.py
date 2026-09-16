"""用真实 Harness 和离线工具验证修复，不发云端请求或改动共享数据。"""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException
from moonlightbox.agent_runtime.contracts import SubmissionContext
from moonlightbox.agent_runtime.deadlines import bounded_timeout, tool_deadline
from moonlightbox.agent_runtime.policy import DAY_PLAN_RUNTIME_POLICY
from moonlightbox.runtime_v1.agent_support import (
    RuntimeToolbox,
    prompt_hash,
)
from moonlightbox.runtime_v1.director import DirectorAgent
from moonlightbox.runtime_v1.executor import _seed_world_memory
from moonlightbox.runtime_v1.plan_context import DayPlanContext
from moonlightbox.runtime_v1.schemas import (
    ContextPacket,
    DayPlanProposal,
    LifeDecision,
)
from moonlightbox.runtime_v1.snapshot_sources import profile_entries
from moonlightbox.runtime_v1.tools.submit_day_plan import (
    validate_plan_proposal as _validate_proposal_for_controller,
)
from pydantic import ValidationError


def context():
    now = datetime(2026, 5, 9, 8, tzinfo=UTC)
    return DayPlanContext(
        generated_at=now,
        virtual_now=now,
        branch_id="b",
        target_date=now.date(),
        timezone="UTC",
        mode="initial",
        request={},
        hard_constraints={},
        origin_projection={},
        date_features={},
        branch_evidence={},
        evidence_requirements={"required_topics": ["work"]},
        budget={},
    )


def proposal(ctx, **overrides):
    return DayPlanProposal(
        plan_date=ctx.target_date,
        blocks=[
            dict(
                start="00:00",
                end="24:00",
                activity="生活安排",
                basis="fallback",
                confidence="fallback",
                **overrides,
            )
        ],
    )


def test_failed_retrieval_does_not_force_more_calls_for_fallback_plan():
    ctx = context()
    validation = SubmissionContext(
        tool_results=({"error": "tool_timeout"},),
        source_refs=("tool:analyze_routine_evidence",),
        state_revision=0,
        input_revision=1,
    )
    assert (
        _validate_proposal_for_controller(proposal(ctx), context=ctx, validation_context=validation)
        is None
    )


def test_empty_director_output_is_not_wait_and_actor_error_has_no_fake_message():
    with pytest.raises(ValidationError):
        LifeDecision.model_validate_json("{}")
    now = datetime.now(UTC)
    packet = ContextPacket(
        generated_at=now,
        virtual_now=now,
        timezone="UTC",
        trigger={},
        origin={},
        current={},
        branch={},
    )
    result = DirectorAgent().run_with_trace(packet)
    assert result.decision.action == "wait" and result.terminal_reason == "guard"


def test_long_tool_result_is_losslessly_pageable_and_prompt_version_changes():
    toolbox = RuntimeToolbox(DAY_PLAN_RUNTIME_POLICY)
    original = {"source_ids": ["source"], "data": "甲" * 35000 + "重要结论在末尾"}
    first = toolbox.project(original)
    pages = [first["page"]]
    while first["next_offset"] is not None:
        first = toolbox.read(first["result_ref"], first["next_offset"])
        pages.append(first["page"])
    assert json.loads("".join(pages)) == original
    with pytest.raises(ToolException):
        toolbox.read("另一个执行的引用")
    assert prompt_hash("版本一") != prompt_hash("版本二")


def test_compaction_preserves_task_and_complete_tool_batches():
    toolbox = RuntimeToolbox(DAY_PLAN_RUNTIME_POLICY)
    messages = [SystemMessage(content="系统"), HumanMessage(content="锁定计划和用户消息")]
    for i in range(4):
        messages += [
            AIMessage(content="", tool_calls=[{"id": str(i), "name": "t", "args": {}}]),
            ToolMessage(content="正文", tool_call_id=str(i)),
        ]
    compact, _ = toolbox.compact(messages, source_refs=[], unresolved=[])
    assert compact[:2] == messages[:2]
    calls = {c["id"] for m in compact if isinstance(m, AIMessage) for c in m.tool_calls}
    assert all(m.tool_call_id in calls for m in compact if isinstance(m, ToolMessage))


def test_profile_inference_has_frozen_reference_and_idempotent_memory():
    snapshot = SimpleNamespace(
        id="snapshot",
        profile={
            "profile_schema_version": "v3",
            "life_context": {
                "primary_engagements": {
                    "status": "described",
                    "basis": "inferred",
                    "description": "目前以灵活时间工作",
                }
            },
        },
    )
    entries = profile_entries(snapshot)

    class Store:
        def __init__(self):
            self.rows = {}

        def get(self, model, key):
            return self.rows.get(key)

        def add(self, row):
            self.rows[row.id] = row

        def flush(self):
            pass

    store = Store()
    _seed_world_memory(store, snapshot)
    _seed_world_memory(store, snapshot)
    assert len(store.rows) == 1
    row = next(iter(store.rows.values()))
    assert len(row.id) == 36 and row.status == "asserted"
    assert row.source_ids == [entries[0]["source_id"]]
    ctx = context()
    ctx.evidence_requirements = {}
    ctx.origin_projection = {"profile_references": entries}
    plan = DayPlanProposal(
        plan_date=ctx.target_date,
        blocks=[
            dict(
                start="00:00",
                end="24:00",
                activity="弹性安排",
                basis="profile_inference",
                confidence="inferred",
                evidence_ids=row.source_ids,
            )
        ],
    )
    valid = SubmissionContext(tool_results=(), source_refs=(), state_revision=0, input_revision=1)
    assert _validate_proposal_for_controller(plan, context=ctx, validation_context=valid) is None


def test_cooperative_network_deadline_is_bounded():
    with tool_deadline(2):
        assert 0 < bounded_timeout(90) <= 2
    with tool_deadline(0):
        with pytest.raises(TimeoutError):
            bounded_timeout(90)


def test_real_controller_returns_field_diagnostic_for_repair():
    class Model:
        def __init__(self):
            self.inputs = []

        def bind_tools(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            self.inputs.append(messages)
            from submission_helpers import submission_message

            return submission_message(
                content="{}" if len(self.inputs) == 1 else '{"action":"wait"}',
                name="submit_decision",
            )

    model = Model()
    now = datetime.now(UTC)
    packet = ContextPacket(
        generated_at=now,
        virtual_now=now,
        timezone="UTC",
        trigger={},
        origin={"person_world_profile": {}},
        current={},
        branch={"branch_id": "b"},
    )
    result = DirectorAgent(model).run_with_trace(packet)
    assert result.decision.action == "wait"
    feedback = str(model.inputs[1][-1].content)
    assert '"action"' in feedback and "missing" in feedback and "validation_errors" in feedback


def test_retrieval_failure_is_not_an_empty_success():
    from moonlightbox.agent_runtime.tool_errors import ToolServiceError
    from moonlightbox.config import Settings
    from moonlightbox.runtime_v1.tools.routine_evidence import query_frozen_history
    from moonlightbox.world.client import LightRAGSidecarError

    class Store:
        def get(self, *args):
            return SimpleNamespace(status="superseded", workspace_key="frozen")

    class Sidecar:
        def query(self, workspace, *args, **kwargs):
            assert workspace == "frozen"
            raise LightRAGSidecarError("network", "offline")

    with pytest.raises(ToolServiceError) as failed:
        query_frozen_history(
            Store(),
            SimpleNamespace(graph_version_id="old", snapshot_mode="latest_profile"),
            "当前工作安排",
            settings=Settings(lightrag_enabled=True),
            client=Sidecar(),
        )
    assert failed.value.code == "network"


def test_context_hard_limit_does_not_send_oversized_request():
    from moonlightbox.runtime_v1.actor import PersonaActor
    from moonlightbox.runtime_v1.context_views import actor_context_payload

    class Model:
        def invoke(self, messages):
            raise AssertionError("超出边界时不应请求供应商")

    now = datetime.now(UTC)
    packet = ContextPacket(
        generated_at=now,
        virtual_now=now,
        timezone="UTC",
        trigger={},
        origin={"person_world_profile": {"overview": "大" * 310_000}},
        current={},
        branch={},
    )
    result = PersonaActor(Model()).run_with_trace(
        context=actor_context_payload(packet),
        tools=[],
        owner_id="test",
        project_id="p",
        branch_id="b",
        input_revision=1,
    )
    assert result.result is None and result.error_code == "input_context_limit"
