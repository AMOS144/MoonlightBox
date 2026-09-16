"""固定执行入口的原生工具回环、缓存前缀、权限及恢复冒烟，不请求云端。"""

import json
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from moonlightbox.agent_runtime import AgentBudgetPolicy, AgentLoopController, AgentSpec, RunScope
from moonlightbox.agent_runtime.contracts import AgentExecutionRequest
from moonlightbox.agent_runtime.policy import DIRECTOR_RUNTIME_POLICY
from moonlightbox.agent_runtime.skills import build_skill_tool
from moonlightbox.agent_runtime.submission import result_submission_tool
from moonlightbox.agent_runtime.tool_dispatch import build_execute_tool
from moonlightbox.agent_runtime.tool_errors import ToolInputError
from moonlightbox.runtime_v1.agent_support import RuntimeToolbox
from pydantic import BaseModel, Field, ValidationError


class Query(BaseModel):
    query: str = Field(min_length=1)


class Result(BaseModel):
    text: str


def business(callback=None):
    return StructuredTool.from_function(
        name="style_lookup",
        description="查真实措辞",
        args_schema=Query,
        metadata={"required_permissions": ["runtime.read_style"]},
        func=callback or (lambda query: {"source_ids": ["m1"], "text": query}),
    )


def test_skill_discloses_schema_only_in_read_result(tmp_path):
    folder = tmp_path / "speaking"
    folder.mkdir()
    path = folder / "SKILL.md"
    path.write_text(
        "---\nname: speaking\ndescription: 表达\ntools: [style_lookup]\n---\n说话", encoding="utf-8"
    )
    reader = build_skill_tool(tmp_path, business_tools=(business(),))
    native = convert_to_openai_tool(reader)
    assert "style_lookup" not in json.dumps(native)
    assert "minLength" in reader.invoke({"skill": "speaking"})["contents"]
    path.write_text(path.read_text(encoding="utf-8") + "\n新版正文", encoding="utf-8")
    updated = build_skill_tool(tmp_path, business_tools=(business(),))
    assert convert_to_openai_tool(updated) == native
    assert updated.metadata != reader.metadata


def test_internal_contract_changes_without_changing_visible_schema():
    toolbox = RuntimeToolbox(DIRECTOR_RUNTIME_POLICY)
    original = toolbox.register_one(business())
    first = build_execute_tool((original,))
    second = build_execute_tool(
        (replace(original, contract=replace(original.contract, contract_version="2")),)
    )
    assert convert_to_openai_tool(first.tool) == convert_to_openai_tool(second.tool)
    assert first.contract.contract_version != second.contract.contract_version
    with pytest.raises(ToolInputError):
        first.tool.invoke({"tool_name": "submit_decision", "arguments": {}})
    with pytest.raises(ValidationError):
        first.tool.invoke({"tool_name": "style_lookup", "arguments": {"query": 8}})
    with pytest.raises(ValueError, match="只读"):
        build_execute_tool(
            (replace(original, contract=replace(original.contract, side_effect="proposal")),)
        )


def test_controller_keeps_schema_fixed_and_returns_parameter_errors(tmp_path):
    calls = []
    toolbox = RuntimeToolbox(DIRECTOR_RUNTIME_POLICY)
    hidden = toolbox.register_one(
        business(lambda query: calls.append(query) or {"source_ids": ["m1"]})
    )
    dispatcher = build_execute_tool((hidden,))
    folder = tmp_path / "speaking"
    folder.mkdir()
    (folder / "SKILL.md").write_text(
        "---\nname: speaking\ndescription: 表达\ntools: [style_lookup]\n---\n说话",
        encoding="utf-8",
    )
    reader = toolbox.register_one(build_skill_tool(tmp_path, business_tools=(hidden.tool,)))

    class Model:
        def __init__(self):
            self.bindings = []
            self.round = 0

        def bind_tools(self, tools, **kwargs):
            self.bindings.append(tools)
            return self

        def invoke(self, messages):
            self.round += 1
            if self.round == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "skill",
                            "name": "read_skill",
                            "args": {"skill": "speaking"},
                        }
                    ],
                )
            if self.round == 2:
                receipt = json.loads(
                    [m for m in messages if isinstance(m, ToolMessage)][-1].content
                )
                assert "style_lookup" in receipt["contents"]
                args = {"tool_name": "style_lookup", "arguments": {}}
            elif self.round == 3:
                receipt = json.loads(
                    [m for m in messages if isinstance(m, ToolMessage)][-1].content
                )
                assert receipt["validation_errors"][0]["loc"] == ["arguments", "query"]
                assert receipt["target_tool_name"] == "style_lookup"
                assert receipt["error"] == "invalid_tool_arguments"
                args = {"tool_name": "style_lookup", "arguments": {"query": "开心时怎么接话"}}
            else:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "done", "name": "finish", "args": {"result": {"text": "完成"}}}
                    ],
                )
            return AIMessage(
                content="",
                tool_calls=[{"id": str(self.round), "name": "execute_tool", "args": args}],
            )

    model = Model()
    spec = AgentSpec(
        name="dispatch",
        prompt_version="test",
        submission_tool_name="finish",
        tools=(reader, dispatcher, result_submission_tool("finish", Result)),
        budget=AgentBudgetPolicy(max_wall_seconds=60, max_tool_result_chars=100_000),
    )
    request = AgentExecutionRequest(
        owner_type="test",
        owner_id="dispatch",
        messages=(HumanMessage(content="测试"),),
        scope=RunScope(permissions=frozenset({"runtime.read_style"})),
    )
    outcome = AgentLoopController().run(spec=spec, request=request, model=model)
    assert outcome.terminal_reason == "success"
    assert calls == ["开心时怎么接话"]
    assert len(model.bindings) == 4
    assert all(binding == model.bindings[0] for binding in model.bindings)
    definitions = {item["function"]["name"]: item["function"] for item in model.bindings[0]}
    assert "style_lookup" not in definitions
    assert definitions["execute_tool"]["strict"] is False
    assert (
        definitions["execute_tool"]["parameters"]["properties"]["arguments"]["additionalProperties"]
        is True
    )
    assert definitions["finish"]["strict"] is True


def test_dispatch_result_restores_original_paging_and_hooks():
    restored = []
    tool = business(lambda query: {"text": "长" * 25_000})
    tool.metadata["restore_tool_result"] = restored.append
    box = RuntimeToolbox(DIRECTOR_RUNTIME_POLICY)
    dispatch = build_execute_tool((box.register_one(tool),))
    full = dispatch.tool.invoke({"tool_name": "style_lookup", "arguments": {"query": "test"}})
    projected = dispatch.contract.model_result_projector(full)
    ref = projected["result"]["result_ref"]
    fresh = RuntimeToolbox(DIRECTOR_RUNTIME_POLICY)
    fresh.register_one(tool)
    fresh.restore([full])
    assert fresh.read(ref) == box.read(ref)
    assert restored == [full]


def test_phoenix_marks_business_target_without_extra_tool_span(monkeypatch):
    from moonlightbox.agent_runtime import tool_dispatch

    attributes = []
    monkeypatch.setattr(
        tool_dispatch, "record_span_attributes", lambda span, attrs: attributes.append(attrs)
    )
    box = RuntimeToolbox(DIRECTOR_RUNTIME_POLICY)
    dispatch = build_execute_tool((box.register_one(business()),))
    dispatch.tool.invoke({"tool_name": "style_lookup", "arguments": {"query": "你好"}})
    assert attributes[0]["moonlightbox.tool.target_name"] == "style_lookup"
