"""统一技能读取入口：目录生成 Schema，正文按需加载，不创建第二套 Agent 循环。"""

import hashlib
import json
from enum import Enum
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ConfigDict, Field, create_model

from .tool_errors import ToolInputError


def build_skill_tool(
    root: Path,
    *,
    business_tools: tuple[BaseTool, ...] = (),
    skill_contexts: dict[str, object] | None = None,
) -> StructuredTool:
    """每次装配冻结目录和正文，保证本轮读取一致；下一次装配自动读取文件更新。

    仅公开 SKILL.md 和 references 下的 Markdown，不提供任意文件读取或代码执行。
    技能正文指纹交给调用方的工具契约，避免改技能后复用旧检查点结论。
    """
    root = root.resolve()
    catalog = {}
    descriptions = []
    tool_catalog = {tool.name: tool for tool in business_tools}
    if len(tool_catalog) != len(business_tools):
        raise ValueError("技能业务工具名称重复")
    for entry in sorted(root.glob("*/SKILL.md")):
        folder = entry.parent.resolve()
        if not folder.is_relative_to(root) or not entry.resolve().is_relative_to(folder):
            raise ValueError(f"技能目录越界：{entry.name}")
        text = entry.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0] != "---" or "---" not in lines[1:]:
            raise ValueError(f"技能 {folder.name} 缺少 YAML frontmatter")
        end = lines.index("---", 1)
        meta = yaml.safe_load("\n".join(lines[1:end]))
        if not isinstance(meta, dict):
            raise ValueError(f"技能 {folder.name} 的 frontmatter 必须是对象")
        name, description = meta.get("name"), meta.get("description")
        if name != folder.name or not isinstance(description, str) or not description.strip():
            raise ValueError(f"技能 {folder.name} 必须声明同名 name 和非空 description")
        if not "\n".join(lines[end + 1 :]).strip():
            raise ValueError(f"技能 {name} 正文为空")
        # 说明只进入读取回执，不拼入模型侧工具 Schema；参数以实际 LangChain 定义为准。
        dependencies = meta.get("tools", [])
        if not isinstance(dependencies, list) or not all(isinstance(t, str) for t in dependencies):
            raise ValueError(f"技能 {name} 的 tools 必须是名称列表")
        for dependency in dependencies:
            tool = tool_catalog.get(dependency)
            if tool is None:
                text += f"\n\n### {dependency}\n本次运行未提供此工具，不可调用。\n"
                continue
            schema = tool.get_input_schema().model_json_schema()
            text += (
                f"\n\n### {dependency}\n{tool.description}\n"
                f"通过 execute_tool 调用，tool_name 为 {dependency}。arguments 格式：\n"
                "```json\n" + json.dumps(schema, ensure_ascii=False, indent=2) + "\n```\n"
            )
        placeholder = meta.get("context_placeholder")
        has_context = skill_contexts is not None and name in skill_contexts
        if placeholder is not None or has_context:
            if not isinstance(placeholder, str) or not placeholder.isidentifier():
                raise ValueError(f"技能 {name} 必须声明有效的 context_placeholder")
            token = "{{ " + placeholder + " }}"
            if text.count(token) != 1:
                raise ValueError(f"技能 {name} 必须恰好包含一次 {token}")
            # 只替换作者指定的位置，不递归渲染资料中的文本，也不改共享文件或 Schema。
            material = (
                skill_contexts[name]
                if has_context
                else {"status": "not_provided", "explanation": "本次未提供此技能的动态资料。"}
            )
            text = text.replace(
                token,
                "```json\n"
                + json.dumps(material, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n```",
                1,
            )
        resources = {"SKILL.md": text}
        for reference in sorted((folder / "references").rglob("*.md")):
            if not reference.resolve().is_relative_to(folder):
                raise ValueError(f"技能 {name} 的参考资源越界")
            resources[reference.relative_to(folder).as_posix()] = reference.read_text(
                encoding="utf-8"
            )
        catalog[name] = resources
        descriptions.append(f"{name}：{description.strip()}；资源：{', '.join(resources)}")
    if not catalog:
        raise ValueError("技能目录为空")
    if set(skill_contexts or {}) - set(catalog):
        raise ValueError("动态资料指定了不存在的技能")

    version = hashlib.sha256(
        json.dumps(catalog, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    names = Enum("SkillName", {name: name for name in catalog}, type=str)
    args = create_model(
        "ReadSkillArgs",
        __config__=ConfigDict(extra="forbid"),
        skill=(names, Field(description="选择技能。" + "\n".join(descriptions))),
        resource=(
            str,
            Field(
                default="SKILL.md",
                description="首次读取 SKILL.md；后续只读取该技能列出的参考资源路径。",
            ),
        ),
    )

    def read_skill(skill, resource="SKILL.md"):
        name = skill.value if isinstance(skill, Enum) else skill
        if name not in catalog:
            raise ToolInputError(f"未知 skill；可选值：{', '.join(catalog)}", field="skill")
        resources = catalog[name]
        if resource not in resources:
            raise ToolInputError(
                f"未知 resource；{name} 可用资源：{', '.join(resources)}", field="resource"
            )
        return {
            "skill": name,
            "resource": resource,
            "contents": resources[resource],
            "available_resources": list(resources),
            "version": version,
        }

    return StructuredTool.from_function(
        name="read_skill",
        func=read_skill,
        args_schema=args,
        metadata={"contract_version": f"skills-v1:{version}"},
        description=(
            "统一读取技能说明。根据下列用途选择技能，首次省略 resource 读取完整 SKILL.md，"
            "再按说明按需读取参考资源。返回 contents、available_resources 和 version。"
            "读取后由当前 Agent 继续执行，不启动子 Agent、不注册新工具、不发送消息；"
            "技能不改变当前角色职责、工具权限或结果提交协议。压缩后需要原文可再次读取。\n"
            "若统一运行时将结果分页，按回执的 result_ref 和 next_offset 读取后续页，"
            "不能只读首段。\n" + "\n".join(descriptions)
        ),
    )
