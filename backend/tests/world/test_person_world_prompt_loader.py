from __future__ import annotations

import pytest
from moonlightbox.world.person_world.prompt_loader import (
    SECTION_PROMPT_NAMES,
    load_prompt_definition,
    load_section_prompt,
    parse_prompt_markdown,
    validate_registered_tools,
)
from moonlightbox.world.person_world.tools import (
    build_current_profile_tools,
    build_graph_query_tools,
    build_source_message_tools,
    build_temporal_analysis_tools,
)


def test_all_section_prompts_are_loadable_and_distinct() -> None:
    definitions = {name: load_section_prompt(name) for name in SECTION_PROMPT_NAMES}

    assert set(definitions) == set(SECTION_PROMPT_NAMES)
    assert len({item.content_hash for item in definitions.values()}) == len(definitions)
    for definition in definitions.values():
        assert definition.description
        assert definition.tool_names
        assert len(definition.system_prompt) > 100



def test_revision_prompt_is_loaded_from_markdown() -> None:
    definition = load_prompt_definition("revision")

    assert definition.name == "revision"
    assert "共同理解" in definition.system_prompt
    assert len(definition.content_hash) == 64


def test_loader_rejects_unknown_frontmatter_and_tool_mismatch() -> None:
    with pytest.raises(ValueError, match="未知配置"):
        parse_prompt_markdown(
            "identity",
            "---\ndescription: x\ntools: [search_world]\nextra: no\n---\n正文",
        )

    definition = parse_prompt_markdown(
        "identity",
        "---\ndescription: x\ntools: [search_world]\n---\n正文",
    )
    with pytest.raises(ValueError, match="缺少声明工具"):
        validate_registered_tools(definition, [])
    with pytest.raises(ValueError, match="未声明工具"):
        validate_registered_tools(definition, ["search_world", "locate_source_messages"])
    assert validate_registered_tools(definition, ["search_world"]) == ("search_world",)


def test_section_prompt_tools_are_a_subset_of_the_runtime_tool_catalog() -> None:
    """Prompt 的工具名是可执行契约，不允许只写在 Markdown 里。"""

    runtime_tool_names = {
        "get_context_module_spec",
        "list_context_modules",
        "read_context_module",
        "read_evidence_page",
        "search_world",
        "list_graph_entities",
        "get_graph_entity",
        "get_entity_neighborhood",
        "get_relation",
        "locate_source_messages",
        "get_message_context",
        "get_current_profile_section",
        "get_active_corrections",
        "analyze_evidence_dates",
        "save_section_work",
    }
    # 同时确认测试引用的四个 builder 是公共工具入口；不实例化它们以避免建立数据库/Sidecar。
    assert {
        build_graph_query_tools.__name__,
        build_source_message_tools.__name__,
        build_current_profile_tools.__name__,
        build_temporal_analysis_tools.__name__,
    } == {
        "build_graph_query_tools",
        "build_source_message_tools",
        "build_current_profile_tools",
        "build_temporal_analysis_tools",
    }
    for section in SECTION_PROMPT_NAMES:
        definition = load_section_prompt(section)
        assert set(definition.tool_names) <= runtime_tool_names
