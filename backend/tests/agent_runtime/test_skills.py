"""技能入口冒烟：真实 LangChain Schema、错误回传、快照和检查点契约。"""

from datetime import UTC

import pytest
from langchain_core.utils.function_calling import convert_to_openai_function
from moonlightbox.agent_runtime.policy import DIRECTOR_RUNTIME_POLICY
from moonlightbox.agent_runtime.skills import build_skill_tool
from moonlightbox.agent_runtime.tool_errors import ToolInputError
from moonlightbox.runtime_v1.agent_support import RuntimeToolbox
from pydantic import ValidationError


def make_skill(root, name="speaking", body="表达指南"):
    directory = root / name
    directory.mkdir(exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} 的用途\n---\n{body}", encoding="utf-8"
    )


def test_schema_lists_all_skills_and_reading_rules(tmp_path):
    make_skill(tmp_path)
    make_skill(tmp_path, "planning")
    tool = build_skill_tool(tmp_path)
    native = convert_to_openai_function(tool)
    assert native["name"] == "read_skill"
    assert native["parameters"]["properties"]["skill"]["enum"] == ["planning", "speaking"]
    assert "speaking 的用途" in native["description"]
    assert "不启动子 Agent" in native["description"]
    assert tool.invoke({"skill": "speaking"})["contents"].endswith("表达指南")
    with pytest.raises(ValidationError):
        tool.invoke({"skill": "missing"})
    with pytest.raises(ToolInputError, match="可用资源"):
        tool.invoke({"skill": "speaking", "resource": "../../.env"})
    with pytest.raises(ValidationError):
        tool.invoke({"skill": "speaking", "execute": True})


def test_resources_are_frozen_and_changes_invalidate_contract(tmp_path):
    make_skill(tmp_path)
    refs = tmp_path / "speaking" / "references"
    refs.mkdir()
    reference = refs / "examples.md"
    reference.write_text("旧参考", encoding="utf-8")
    old = build_skill_tool(tmp_path)
    same = build_skill_tool(tmp_path)
    assert same.metadata == old.metadata
    reference.write_text("新参考", encoding="utf-8")
    new = build_skill_tool(tmp_path)
    assert new.metadata != old.metadata
    assert (
        old.invoke({"skill": "speaking", "resource": "references/examples.md"})["contents"]
        == "旧参考"
    )
    registered = RuntimeToolbox(DIRECTOR_RUNTIME_POLICY).register([new])[0]
    assert registered.contract.contract_version == new.metadata["contract_version"]


def test_reference_symlinks_cannot_escape_skill(tmp_path):
    make_skill(tmp_path)
    refs = tmp_path / "speaking" / "references"
    refs.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("not a skill", encoding="utf-8")
    (refs / "escape.md").symlink_to(outside)
    with pytest.raises(ValueError, match="越界"):
        build_skill_tool(tmp_path)


def test_director_registers_reader_without_changing_submission():
    from types import SimpleNamespace

    from moonlightbox.runtime_v1.director import DirectorAgent
    from moonlightbox.runtime_v1.schemas import ContextPacket

    class Controller:
        def run(self, *, spec, **kwargs):
            assert spec.submission_tool_name == "submit_decision"
            readers = [item for item in spec.tools if item.tool.name == "read_skill"]
            assert len(readers) == 1
            assert readers[0].tool.invoke({"skill": "speaking"})["resource"] == "SKILL.md"
            return SimpleNamespace(value=None, terminal_reason="test")

    from datetime import datetime

    now = datetime.now(UTC)
    packet = ContextPacket(
        generated_at=now,
        virtual_now=now,
        timezone="Asia/Shanghai",
        trigger={},
        origin={},
        current={},
        branch={},
    )
    DirectorAgent(model=object(), controller=Controller()).run(packet)
