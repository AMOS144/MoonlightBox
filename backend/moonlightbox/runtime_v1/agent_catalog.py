"""加载 Runtime Agent 的 Markdown 定义。

借鉴 muse-api 的组织方式：YAML frontmatter 只声明 Agent 身份和工具白名单，正文只写
行为规则。运行预算属于运行时策略，由后端统一注入；工具 schema 仍由 LangChain
``StructuredTool`` 自己提供。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool

_AGENT_DIR = Path(__file__).with_name("prompts")
_DEFINITION_FILES = {
    "director": "director/system.md",
    "day_planner": "day_planner/system.md",
    "persona_actor": "persona_actor/system.md",
    "day_planner_life_events": "day_planner/life_events.md",
    "node_investigator": "node_investigator/system.md",
}
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ALLOWED_KEYS = {
    "description",
    "tools",
}


@dataclass(frozen=True, slots=True)
class RuntimeAgentDefinition:
    """一个Agent 的作者配置，不包含工具实现或 JSON schema。"""

    name: str
    description: str
    tool_names: tuple[str, ...]
    system_prompt: str


def parse_agent_markdown(name: str, text: str) -> RuntimeAgentDefinition:
    """严格解析声明文件，让配置错误在启动/测试阶段直接暴露。"""

    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError(f"Agent {name} 缺少 YAML frontmatter")
    raw = yaml.safe_load(match.group(1)) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Agent {name} 的 frontmatter 必须是对象")
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Agent {name} 包含未知配置: {sorted(unknown)}")
    tools = raw.get("tools")
    if not isinstance(tools, list) or not tools or not all(isinstance(item, str) for item in tools):
        raise ValueError(f"Agent {name} 必须声明非空 tools 列表")
    tool_names = tuple(dict.fromkeys(item.strip() for item in tools if item.strip()))
    if not tool_names:
        raise ValueError(f"Agent {name} 没有有效工具名")
    prompt = text[match.end() :].strip()
    if not prompt:
        raise ValueError(f"Agent {name} 缺少行为 Prompt")
    definition = RuntimeAgentDefinition(
        name=name,
        description=str(raw.get("description", "")).strip(),
        tool_names=tool_names,
        system_prompt=prompt,
    )
    return definition


@lru_cache(maxsize=8)
def load_agent_definition(name: str) -> RuntimeAgentDefinition:
    """按文件名加载定义；不做全局可变注册。"""

    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError(f"无效的Agent 名称: {name}")
    relative = _DEFINITION_FILES.get(name)
    if relative is None:
        raise ValueError(f"未知 Runtime Agent: {name}")
    path = _AGENT_DIR / relative
    if not path.is_file():
        raise ValueError(f"未知 Runtime Agent: {name}")
    return parse_agent_markdown(name, path.read_text(encoding="utf-8"))


def select_declared_tools(
    definition: RuntimeAgentDefinition,
    tools: Sequence[BaseTool],
    *,
    require_all: bool = False,
) -> list[BaseTool]:
    """按 Markdown 白名单裁出实际 LangChain 工具，并对越权工具 fail-fast。

    数据库或 Sidecar 不可用时，上层可以不提供某个声明工具，因此默认允许白名单的
    子集；启动期需要检查完整装配时可传 ``require_all=True``。
    """

    by_name: dict[str, BaseTool] = {}
    for tool in tools:
        tool_name = getattr(tool, "name", None)
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError(f"Agent {definition.name} 收到没有名称的工具")
        if tool_name in by_name:
            raise ValueError(f"Agent {definition.name} 收到重复工具: {tool_name}")
        by_name[tool_name] = tool
    undeclared = set(by_name) - set(definition.tool_names)
    if undeclared:
        raise ValueError(f"Agent {definition.name} 收到未声明工具: {sorted(undeclared)}")
    missing = set(definition.tool_names) - set(by_name)
    if require_all and missing:
        raise ValueError(f"Agent {definition.name} 缺少声明工具: {sorted(missing)}")
    return [by_name[name] for name in definition.tool_names if name in by_name]
