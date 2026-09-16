"""加载 PersonWorld 的作者维护 Markdown Prompt。

Prompt 是运行时行为的真源；工具参数 schema 仍由 LangChain ``StructuredTool`` 注册。
这里故意只校验文件结构和工具白名单，不把工具 JSON Schema 拼回模型上下文。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_SUBAGENT_DIR = Path(__file__).with_name("subagents")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ALLOWED_KEYS = {
    "description",
    "tools",
}
SECTION_PROMPT_NAMES = frozenset(
    {
        "identity",
        "life_context",
        "social_world",
        "agency",
        "practices",
        "life_course",
        "relationship_with_user",
    }
)
ALL_PROMPT_NAMES = SECTION_PROMPT_NAMES | {"revision"}


@dataclass(frozen=True, slots=True)
class PersonWorldPromptDefinition:
    """一个可审计的子 Agent Prompt，不包含任何工具实现。"""

    name: str
    description: str
    tool_names: tuple[str, ...]
    system_prompt: str
    content_hash: str


def parse_prompt_markdown(name: str, text: str) -> PersonWorldPromptDefinition:
    """严格解析一个 Prompt 文件，避免静默接受拼写错误或越权配置。"""

    if name not in ALL_PROMPT_NAMES:
        raise ValueError(f"未知 PersonWorld Prompt: {name}")
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError(f"PersonWorld Prompt {name} 缺少 YAML frontmatter")
    raw = yaml.safe_load(match.group(1)) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"PersonWorld Prompt {name} 的 frontmatter 必须是对象")
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"PersonWorld Prompt {name} 包含未知配置: {sorted(unknown)}")

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"PersonWorld Prompt {name} 缺少 description")
    tools = raw.get("tools")
    if not isinstance(tools, list) or not tools or not all(isinstance(item, str) for item in tools):
        raise ValueError(f"PersonWorld Prompt {name} 必须声明非空 tools 列表")
    tool_names = tuple(dict.fromkeys(item.strip() for item in tools if item.strip()))
    if not tool_names:
        raise ValueError(f"PersonWorld Prompt {name} 没有有效工具名")

    prompt = text[match.end() :].strip()
    if not prompt:
        raise ValueError(f"PersonWorld Prompt {name} 缺少 system prompt 正文")
    definition = PersonWorldPromptDefinition(
        name=name,
        description=description.strip(),
        tool_names=tool_names,
        system_prompt=prompt,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    return definition


@lru_cache(maxsize=len(ALL_PROMPT_NAMES))
def load_prompt_definition(name: str) -> PersonWorldPromptDefinition:
    """从唯一的 Markdown 文件加载 Prompt。"""

    if name not in ALL_PROMPT_NAMES:
        raise ValueError(f"未知 PersonWorld Prompt: {name}")
    path = _SUBAGENT_DIR / f"{name}.md"
    if not path.is_file():
        raise ValueError(f"PersonWorld Prompt 文件不存在: {path.name}")
    return parse_prompt_markdown(name, path.read_text(encoding="utf-8"))


def load_section_prompt(section: str) -> PersonWorldPromptDefinition:
    """按七个顶层事实栏目加载其唯一的行为 Prompt。"""

    if section not in SECTION_PROMPT_NAMES:
        raise ValueError(f"{section!r} 不是 PersonWorld 的顶层栏目")
    return load_prompt_definition(section)


def load_section_prompt_v3(section: str) -> PersonWorldPromptDefinition:
    """当前七栏作者文件位于主目录；保留版本化函数名供 v3 调用方使用。"""
    if section not in SECTION_PROMPT_NAMES:
        raise ValueError("未知 v3 栏目")
    return load_section_prompt(section)


def load_understanding_protocol_v3() -> str:
    return (_SUBAGENT_DIR / "_understanding.md").read_text(encoding="utf-8")


def load_temporal_protocol() -> str:
    return (_SUBAGENT_DIR / "_temporal.md").read_text(encoding="utf-8")


def validate_registered_tools(
    definition: PersonWorldPromptDefinition,
    registered_names: Iterable[str],
    *,
    require_all: bool = True,
) -> tuple[str, ...]:
    """校验 Prompt 白名单与实际 LangChain 注册结果，防止静默失配。"""

    names = tuple(registered_names)
    if any(not name for name in names):
        raise ValueError(f"PersonWorld Prompt {definition.name} 收到空工具名")
    if len(set(names)) != len(names):
        raise ValueError(f"PersonWorld Prompt {definition.name} 收到重复工具名")
    declared = set(definition.tool_names)
    actual = set(names)
    undeclared = actual - declared
    if undeclared:
        raise ValueError(
            f"PersonWorld Prompt {definition.name} 收到未声明工具: {sorted(undeclared)}"
        )
    missing = declared - actual
    if require_all and missing:
        raise ValueError(f"PersonWorld Prompt {definition.name} 缺少声明工具: {sorted(missing)}")
    return tuple(name for name in definition.tool_names if name in actual)
