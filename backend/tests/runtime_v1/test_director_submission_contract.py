"""对真实 LangChain 工具的嵌套 Schema 和提交路径做回归。"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from moonlightbox.agent_runtime.submission import submission_context
from moonlightbox.runtime_v1.schemas import LifeDecision
from moonlightbox.runtime_v1.tools.submit_decision import build_submit_decision_tool
from pydantic import ValidationError


def tool():
    packet = SimpleNamespace(
        branch={"working_window": {"pending_message_refs": ["m1"]}},
        virtual_now=datetime.now(UTC),
        current={"life_state": {}},
    )
    return build_submit_decision_tool(packet).tool


def proposal():
    return {
        "result": {
            "action": "speak",
            "reply": {"messages": [{"kind": "text", "text": "九点半呢"}]},
            "input_resolutions": [{"message_ref": "m1", "status": "completed"}],
        }
    }


def test_real_nested_tool_schema_has_no_legacy_fields():
    schema = convert_to_openai_tool(tool())["function"]["parameters"]
    fields = schema["properties"]["result"]["properties"]
    assert not {"expression_task", "communication_intent", "content_points"} & fields.keys()
    state = fields["state_patch"]["properties"]
    assert not {"mood", "attention", "current_goal", "open_conversation_threads"} & state.keys()


def test_minimal_reply_delivers_domain_decision_without_legacy_input():
    with submission_context(SimpleNamespace()) as receipt:
        result = tool().invoke(proposal())
        assert result["status"] == "accepted"
        assert isinstance(receipt.value, LifeDecision)
        assert receipt.value.expression_task.respond_to_refs == ["m1"]


def test_missing_reply_reports_current_contract_not_expression_task():
    value = proposal()
    del value["result"]["reply"]
    with pytest.raises(ValidationError) as caught:
        tool().invoke(value)
    assert "result.reply" in str(caught.value)
    assert "expression_task" not in str(caught.value)


def test_legacy_field_rejected_at_exact_path():
    value = proposal()
    value["result"]["state_patch"] = {"open_conversation_threads": []}
    with pytest.raises(ValidationError) as caught:
        tool().invoke(value)
    assert caught.value.errors()[0]["loc"] == ("result", "state_patch", "open_conversation_threads")


def test_optional_null_is_valid_but_empty_string_is_not():
    value = proposal()
    value["result"]["plan_request"] = None
    with submission_context(SimpleNamespace()):
        assert tool().invoke(value)["status"] == "accepted"
    value["result"]["plan_request"] = ""
    with pytest.raises(ValidationError) as caught:
        tool().invoke(value)
    assert caught.value.errors()[0]["loc"] == ("result", "plan_request")


def test_misplaced_top_level_fields_are_not_silently_ignored():
    value = proposal()
    value["reply"] = value["result"]["reply"]
    with pytest.raises(ValidationError) as caught:
        tool().invoke(value)
    assert caught.value.errors()[0]["loc"] == ("reply",)
