"""宿主字段不进入工具 Schema，也不能由 LLM 偷渡覆盖；绑定后仍做领域校验。"""

from datetime import date
from types import SimpleNamespace

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from moonlightbox.agent_runtime.submission import submission_context
from moonlightbox.agent_runtime.tool_errors import ToolInputError, tool_error_result
from moonlightbox.runtime_v1.tools.submit_day_plan import build_submit_day_plan_tool
from moonlightbox.world.person_world.review.graph_input import GraphPatchInput, bind_graph_patch
from moonlightbox.world.person_world.review.submit_turn import build_submit_revision_turn
from pydantic import ValidationError


def plan_tool():
    context = SimpleNamespace(
        target_date=date(2026, 5, 12),
        origin_projection={},
        hard_constraints={},
        model_dump=lambda **kwargs: {},
    )
    return build_submit_day_plan_tool(context).tool


def plan():
    return {
        "proposal": {
            "blocks": [
                {
                    "start": "00:00",
                    "end": "24:00",
                    "activity": "在家休息",
                    "basis": "fallback",
                    "confidence": "fallback",
                    "evidence_ids": [],
                }
            ]
        }
    }


def test_date_is_bound_not_echoed_and_validators_still_run():
    tool = plan_tool()
    with submission_context(SimpleNamespace(source_refs=set())) as receipt:
        assert tool.invoke({"result": plan()})["status"] == "accepted"
        assert receipt.value.proposal.plan_date == date(2026, 5, 12)
        invalid = plan()
        invalid["proposal"]["blocks"][0]["start"] = "01:00"
        assert tool.invoke({"result": invalid})["status"] == "rejected"
    malicious = plan()
    malicious["proposal"]["plan_date"] = "2040-01-01"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        tool.invoke({"result": malicious})


def test_graph_operation_identity_is_assigned_after_input_validation():
    data = {
        "graph_operations": [
            {
                "operation_type": "UPDATE_ENTITY",
                "entity_name": "目标",
                "after_description": "修正描述",
                "reason": "用户已确认",
            },
            {"operation_type": "UPDATE_ENTITY", "entity_name": "另一实体", "reason": "用户已确认"},
        ]
    }
    bound = bind_graph_patch(GraphPatchInput.model_validate(data))
    assert [op.operation_id for op in bound.graph_operations] == ["operation-1", "operation-2"]
    assert all(op.precondition_hash is None for op in bound.graph_operations)
    data["graph_operations"][0]["precondition_hash"] = "a" * 64
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GraphPatchInput.model_validate(data)


def test_question_options_get_ids_but_do_not_approve_anything():
    tool = build_submit_revision_turn().tool
    data = {
        "kind": "question",
        "question": {
            "premise": "需要确认修改范围",
            "decision_key": "scope",
            "question": "只改这处吗？",
            "options": [{"label": "只改这处", "effect": "不调整其他内容"}],
        },
    }
    with submission_context(SimpleNamespace()) as receipt:
        result = tool.invoke({"result": data})
        assert result["status"] == "accepted" and result["committed"] is False
        assert receipt.status == "waiting_for_user"
        assert receipt.value.question.options[0].id == "option-1"


def test_graph_missing_target_is_a_repairable_field_error():
    data = {"graph_operations": [{"operation_type": "DELETE_ENTITY", "reason": "已确认"}]}
    with pytest.raises(ToolInputError) as caught:
        bind_graph_patch(GraphPatchInput.model_validate(data))
    result = tool_error_result(
        caught.value, tool_name="submit_graph_patch", arguments={"result": data}
    )
    assert result["recoverable"] is True
    assert result["retryable"] is False
    assert result["validation_errors"][0]["loc"] == [
        "result",
        "graph_operations",
        "0",
        "entity_name",
    ]


def test_removed_parameter_reports_exact_path_and_delete_guidance():
    tool = plan_tool()
    data = plan()
    data["proposal"]["plan_date"] = "2040-01-01"
    with pytest.raises(ValidationError) as caught:
        tool.invoke({"result": data})
    result = tool_error_result(caught.value, tool_name=tool.name, arguments={"result": data})
    assert tuple(result["validation_errors"][0]["loc"]) == ("result", "proposal", "plan_date")
    assert result["code"] == "invalid_tool_arguments"
    assert "删除" in result["message"]


def test_all_published_section_schemas_remove_owned_fields():
    from moonlightbox.world.person_world.context_module_snapshots import (
        ModuleReadTracker,
        ModuleSnapshot,
    )
    from moonlightbox.world.person_world.contracts.profile_v3 import SECTION_RESULT_MODELS
    from moonlightbox.world.person_world.tools.submit_section import build_submit_section

    def inspect(node):
        if isinstance(node, dict):
            props = node.get("properties", {})
            assert "revision" not in props
            assert "schema_version" not in props
            for child in node.values():
                inspect(child)
        elif isinstance(node, list):
            for child in node:
                inspect(child)

    for section in SECTION_RESULT_MODELS:
        tracker = ModuleReadTracker(section, ModuleSnapshot.create("p"))
        tool = build_submit_section(section, {}, tracker, None).tool
        inspect(convert_to_openai_tool(tool)["function"]["parameters"])
