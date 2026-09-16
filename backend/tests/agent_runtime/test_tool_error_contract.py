"""针对真实 DayPlan Trace 的离线回归，不消耗云端调用。"""

import pytest
from moonlightbox.agent_runtime.tool_errors import tool_error_result
from moonlightbox.runtime_v1.schemas import DayPlanProposalBlock
from pydantic import ValidationError


def block(evidence):
    return dict(
        start="08:30",
        end="09:30",
        activity="通勤",
        basis="simulation_assumption",
        evidence_ids=evidence,
        confidence="inferred",
        assumption="估计通勤时长",
    )


@pytest.mark.parametrize(
    "evidence,actual_type", [("", "str"), (["placeholder"], "list"), (None, "NoneType")]
)
def test_bad_evidence_reports_actual_field_and_repair_action(evidence, actual_type):
    with pytest.raises(ValidationError) as caught:
        DayPlanProposalBlock.model_validate(block(evidence))
    result = tool_error_result(caught.value, tool_name="submit_day_plan", arguments={})
    assert result["error"] == "invalid_tool_arguments"
    assert result["failure"]["category"] == "tool_input"
    assert result["recoverable"] and not result["retryable"]
    field = result["validation_errors"][0]
    assert field["loc"] == ("evidence_ids",)
    assert field["actual_type"] == actual_type
    assert field["actual_value"] == evidence
    assert DayPlanProposalBlock.model_validate(block([])).evidence_ids == []


def test_schema_explains_cross_field_contract_before_first_call():
    schema = DayPlanProposalBlock.model_json_schema()["properties"]
    assert "simulation_assumption" in schema["evidence_ids"]["description"]
    assert "[]" in schema["evidence_ids"]["description"]
    assert schema["evidence_ids"]["examples"] == [[]]
    assert "fallback" in schema["confidence"]["description"]


def test_internal_exception_is_not_exposed_as_repairable_input():
    result = tool_error_result(
        RuntimeError("secret internal path"), tool_name="search_world", arguments={}
    )
    assert not result["recoverable"]
    assert "secret internal path" not in str(result)


def test_dispatcher_preserves_target_and_nested_argument_path():
    with pytest.raises(ValidationError) as caught:
        DayPlanProposalBlock.model_validate(block(""))
    caught.value.target_tool_name = "inner_tool"
    result = tool_error_result(caught.value, tool_name="execute_tool", arguments={})
    assert result["target_tool_name"] == "inner_tool"
    assert result["validation_errors"][0]["loc"] == ["arguments", "evidence_ids"]


@pytest.mark.parametrize(
    "code,repairable",
    [
        ("undeclared_tool", True),
        ("invalid_tool_arguments", True),
        ("tool_not_authorized", False),
        ("tool_call_safety_limit", False),
    ],
)
def test_pre_execution_rejections_share_recovery_contract(code, repairable):
    from moonlightbox.agent_runtime.tool_errors import rejected_call_result

    result = rejected_call_result({"error": code}, tool_name="missing", available_tools={"read"})
    assert result["recoverable"] is repairable
    assert result["retryable"] is False
    assert result["failure"]["code"] == code
    assert result["available_tools"] == ["read"]


def test_missing_dependency_is_not_correctable_input():
    from moonlightbox.agent_runtime.tool_errors import ToolServiceError

    result = tool_error_result(ToolServiceError("missing binding"), tool_name="read", arguments={})
    assert result["error"] == "tool_unavailable"
    assert result["failure"]["category"] == "configuration"
    assert not result["recoverable"] and not result["retryable"]


def test_transport_failure_reports_actual_attempts_without_parameter_repair():
    result = tool_error_result(TimeoutError(), tool_name="search_world", arguments={}, attempts=3)
    assert result["failure"]["attempts"] == 3
    assert result["error"] == "timeout"
    assert result["retryable"] and not result["recoverable"]


def test_native_schema_keeps_empty_array_guidance():
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schema = convert_to_openai_tool(DayPlanProposalBlock)["function"]["parameters"]
    assert schema["properties"]["evidence_ids"]["examples"] == [[]]


def test_invalid_evidence_reference_is_error_not_empty_result():
    from moonlightbox.agent_runtime.tool_errors import ToolInputError
    from moonlightbox.world.person_world.tools.evidence_page import evidence_page

    class Artifacts:
        def evidence_set_rows(self, reference):
            raise ValueError("unknown reference")

    with pytest.raises(ToolInputError, match="evidence_set_id"):
        evidence_page(Artifacts(), "unknown")
