"""逐个调用 Director 的真实工具入口；失败留在工具回执，不访问云端。"""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from moonlightbox.agent_runtime.policy import DIRECTOR_RUNTIME_POLICY
from moonlightbox.agent_runtime.skills import build_skill_tool
from moonlightbox.agent_runtime.tool_dispatch import build_execute_tool
from moonlightbox.agent_runtime.tool_errors import (
    ToolInputError,
    ToolServiceError,
    tool_error_result,
)
from moonlightbox.runtime_v1.agent_support import RuntimeToolbox
from moonlightbox.runtime_v1.tools.director_context import build_director_context_tools
from moonlightbox.runtime_v1.tools.memory_search import build_memory_search_tool
from moonlightbox.runtime_v1.tools.read_runtime_result import build_read_runtime_result_tool
from moonlightbox.runtime_v1.tools.recent_life_events import build_recent_life_events_tool
from moonlightbox.runtime_v1.tools.style_examples import StyleService
from moonlightbox.runtime_v1.tools.submit_decision import build_submit_decision_tool
from pydantic import ValidationError


@pytest.fixture
def inventory():
    session = MagicMock()
    packet = SimpleNamespace(
        branch={"branch_id": "b", "working_window": {}},
        origin={"snapshot_id": "s", "person_world_profile": {}},
        current={"life_state": {"subjective_state": {"concerns": []}}},
        virtual_now=datetime.now(UTC),
    )
    style = StyleService(session).tool(branch_id="b", model_version_id="v")
    skills = Path(__file__).parents[2] / "moonlightbox/runtime_v1/skills"
    with patch("moonlightbox.runtime_v1.memory.MemoryService.current_index", return_value=None):
        memory = build_memory_search_tool(session, branch_id="b", snapshot_id="s")
    tools = [
        *build_director_context_tools(session, packet),
        memory,
        style,
        build_recent_life_events_tool(session, "b", packet.virtual_now),
        build_read_runtime_result_tool({"known": "abcdef"}),
        build_skill_tool(skills, business_tools=(style,)),
        build_execute_tool((RuntimeToolbox(DIRECTOR_RUNTIME_POLICY).register_one(style),)).tool,
        build_submit_decision_tool(packet).tool,
    ]
    return {tool.name: tool for tool in tools}


@pytest.mark.parametrize(
    "name,args",
    [
        ("search_memory", {"query": "  "}),
        ("search_conversation", {"query": " "}),
        ("read_conversation", {"before_ref": "a", "around_ref": "b"}),
        ("get_subjective_state", {"concern_ref": "unknown"}),
        ("get_profile_section", {"section": "unknown"}),
        ("get_recent_life_events", {"before": "tomorrow"}),
        ("read_runtime_result", {"result_ref": "unknown"}),
        ("read_skill", {"skill": "speaking", "resource": "../secrets"}),
        ("execute_tool", {"tool_name": "missing", "arguments": {}}),
        ("get_style_examples", {"situation": "", "intent": ""}),
        ("submit_decision", {"result": {"action": "speak"}}),
    ],
)
def test_each_tool_returns_correctable_error(inventory, name, args):
    with pytest.raises((ValidationError, ToolInputError)) as caught:
        inventory[name].invoke(args)
    result = tool_error_result(caught.value, tool_name=name, arguments=args)
    assert result["recoverable"] and not result["retryable"]
    assert result["failure"]["category"] == "tool_input"
    assert result["next_action"] == "correct_arguments"


def test_nested_style_error_preserves_inner_name_and_field(inventory):
    args = {"tool_name": "get_style_examples", "arguments": {"limit": 99}}
    with pytest.raises(ValidationError) as caught:
        inventory["execute_tool"].invoke(args)
    result = tool_error_result(caught.value, tool_name="execute_tool", arguments=args)
    assert result["target_tool_name"] == "get_style_examples"
    assert result["validation_errors"][0]["loc"] == ["arguments", "limit"]


def test_pagination_error_and_end_are_distinct(inventory):
    reader = inventory["read_runtime_result"]
    assert reader.invoke({"result_ref": "known"})["next_offset"] is None
    with pytest.raises(ToolInputError) as caught:
        reader.invoke({"result_ref": "known", "offset": 100})
    result = tool_error_result(caught.value, tool_name=reader.name, arguments={})
    assert result["validation_errors"][0]["loc"] == ["offset"]


def test_nested_schema_documents_every_field(inventory):
    def check(node):
        if isinstance(node, dict):
            for name, field in node.get("properties", {}).items():
                assert field.get("description"), name
            for child in node.values():
                check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)

    for tool in inventory.values():
        check(convert_to_openai_tool(tool)["function"]["parameters"])


def test_service_failure_is_not_parameter_error():
    result = tool_error_result(
        ToolServiceError("快照不可用"), tool_name="read_conversation", arguments={}
    )
    assert not result["recoverable"] and result["next_action"] == "report_failure"


def test_error_guidance_does_not_force_nullable_fields_to_empty_arrays():
    from moonlightbox.runtime_v1.tools.style_examples import GetStyleExamplesArgs

    with pytest.raises(ValidationError) as caught:
        GetStyleExamplesArgs.model_validate({"situation": "接话", "intent": "安慰", "limit": None})
    result = tool_error_result(caught.value, tool_name="get_style_examples", arguments={})
    assert "可空字段允许 null" in result["message"]
    assert GetStyleExamplesArgs(situation="接话", intent="安慰", usage_cursor=None)


def test_omitted_object_is_not_reported_as_actual_null(inventory):
    with pytest.raises(ValidationError) as caught:
        inventory["read_runtime_result"].invoke({"result_ref": "known", "offset": {"item": 1}})
    result = tool_error_result(caught.value, tool_name="read_runtime_result", arguments={})
    detail = result["validation_errors"][0]
    assert detail["actual_type"] == "dict"
    assert detail["actual_value"] == "[value omitted: object]"
