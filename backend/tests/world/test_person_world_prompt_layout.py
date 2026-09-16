"""当前作者入口与历史兼容入口不能因同名文件而混用。"""

from pathlib import Path

from moonlightbox.world.person_world import prompt_loader


def test_current_prompts_live_at_the_author_root():
    root = Path(prompt_loader.__file__).with_name("subagents")
    for section in prompt_loader.SECTION_PROMPT_NAMES:
        current = prompt_loader.load_section_prompt(section)
        expected = prompt_loader.parse_prompt_markdown(
            section, (root / f"{section}.md").read_text(encoding="utf-8")
        )
        assert current == expected == prompt_loader.load_section_prompt_v3(section)
        assert not (root / "v3" / f"{section}.md").exists()
    assert "Big Five" in prompt_loader.load_section_prompt("identity").system_prompt
    assert prompt_loader.load_understanding_protocol_v3()
    assert not hasattr(prompt_loader, "load_legacy_section_prompt_v2")
    assert not (root / "legacy" / "v2" / "identity.md").exists()
    assert prompt_loader.load_prompt_definition("revision").system_prompt
