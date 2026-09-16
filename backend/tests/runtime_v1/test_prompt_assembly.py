"""Prompt 作者文件、实际拼装与预览保持一致，无云端请求。"""

import json

import pytest
from moonlightbox.runtime_v1.agent_catalog import load_agent_definition
from moonlightbox.runtime_v1.prompting import assemble_agent_prompt
from moonlightbox.runtime_v1.schemas import LifeDecision


@pytest.mark.parametrize("name", ["director", "day_planner", "persona_actor"])
def test_preview_uses_explicit_submission_tool(name, monkeypatch, capsys):
    from moonlightbox.runtime_v1.prompting import main

    monkeypatch.setattr("sys.argv", ["preview", name])
    main()
    preview = json.loads(capsys.readouterr().out)
    assert preview["submission_tool_name"] in preview["tools"]
    assert preview["output_mode"] == "tool_submission"


def test_assembly_keeps_protocol_separate_without_repeating_tool_instructions():
    plain = assemble_agent_prompt(
        "director", LifeDecision, tools=("deliver",), submission_tool_name="deliver"
    )
    paged = assemble_agent_prompt(
        "director",
        LifeDecision,
        tools=("deliver", "read_runtime_result"),
        submission_tool_name="deliver",
    )
    assert plain.behavior == load_agent_definition("director").system_prompt
    assert plain.output_schema == LifeDecision.model_json_schema()
    assert plain.text == paged.text
    assert plain.manifest()["capabilities"] == []
    assert paged.manifest()["capabilities"] == ["tool_result_paging"]
    assert paged.preview()["rendered_system"] == paged.message().content
    assert paged.message().additional_kwargs["prompt_manifest"] == paged.manifest()
    assert plain.manifest()["submission_tool_name"] == "deliver"
    assert "最终输出必须符合" not in plain.text
    with pytest.raises(ValueError, match="未注册"):
        assemble_agent_prompt(
            "director", LifeDecision, tools=("submit_fake",), submission_tool_name="deliver"
        )


def test_all_runtime_roles_load_from_agent_directories():
    for name in (
        "director",
        "day_planner",
        "persona_actor",
        "day_planner_life_events",
    ):
        definition = load_agent_definition(name)
        assert definition.tool_names
        assert definition.system_prompt.startswith("#")
        # 工具白名单仍在 frontmatter，正文不再维护逐个工具的操作手册。
        assert all(name not in definition.system_prompt for name in definition.tool_names)


def test_native_tool_definition_contains_paging_instructions():
    from langchain_core.utils.function_calling import convert_to_openai_function
    from moonlightbox.agent_runtime.policy import DIRECTOR_RUNTIME_POLICY
    from moonlightbox.runtime_v1.agent_support import RuntimeToolbox

    reader = RuntimeToolbox(DIRECTOR_RUNTIME_POLICY).register([])[0].tool
    native = convert_to_openai_function(reader)
    assert native["name"] == "read_runtime_result"
    assert "不是独立 JSON" in native["description"]
    assert "next_offset" in native["parameters"]["properties"]["offset"]["description"]
    assert "不能自行构造" in native["parameters"]["properties"]["result_ref"]["description"]
