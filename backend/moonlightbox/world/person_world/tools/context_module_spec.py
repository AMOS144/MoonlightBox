"""模块结构查询：工具、输出校验共用固定类型定义。"""

from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import Field

from ..contracts.context_module_details import MODULE_FIELDS, MODULE_LABELS
from ..contracts.context_modules import MODULE_MODELS, ProfileModel


class GetContextModuleSpecArgs(ProfileModel):
    kind: Literal[tuple(MODULE_FIELDS)] = Field(description="要查看的情境类型")


def module_read_paths(kind: str) -> list[str]:
    """工具与最终引用 Schema 共用根路径，避免模型误把 details 子字段当根字段。"""
    return [
        "kind",
        "title",
        "summary",
        "status",
        "period",
        "basis",
        "related_module_ids",
        "details",
        *[f"details.{group}" for group in MODULE_FIELDS[kind]],
        *[
            f"details.{group}.{name}"
            for group, names in MODULE_FIELDS[kind].items()
            for name in names.split()
        ],
    ]


def get_context_module_spec(kind: str):
    from moonlightbox.agent_runtime.tool_errors import ToolInputError

    if kind not in MODULE_MODELS:
        raise ToolInputError(f"未知情境类型；可用类型: {', '.join(MODULE_MODELS)}")
    return {
        "kind": kind,
        "label": MODULE_LABELS[kind],
        "schema_version": f"{kind}_v1",
        "fields": {group: names.split() for group, names in MODULE_FIELDS[kind].items()},
        "read_paths": module_read_paths(kind),
        "editing_guidance": (
            "每个叶子填写 status、basis、value、description；"
            "推断可填 inferred，未知写 unknown。描述用简明中文，约20–80字是软目标。"
            "未知公司、学校、收入和精确时间不要编造。一次提交整个模块。"
        ),
    }


def build_context_module_spec_tool():
    return StructuredTool.from_function(
        get_context_module_spec,
        name="get_context_module_spec",
        description="查看一种生活情境的固定分组和字段要求；不查询聊天、不写入模块。",
        args_schema=GetContextModuleSpecArgs,
    )
