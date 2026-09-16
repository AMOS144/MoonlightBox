"""Run/Publication 绑定的只读模块工具。Agent 不能传入另一个项目或版本。"""

from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import Field

from moonlightbox.agent_runtime.tool_errors import ToolInputError

from ..context_module_snapshots import ModuleReadTracker, read_path
from ..contracts.context_module_details import MODULE_FIELDS
from ..contracts.context_modules import ProfileModel
from .context_module_spec import build_context_module_spec_tool


class ListContextModulesArgs(ProfileModel):
    kinds: list[Literal[tuple(MODULE_FIELDS)]] | None = Field(
        default=None, description="模块类型数组；省略表示全部类型"
    )
    statuses: list[Literal["current", "planned", "paused", "past", "unknown"]] | None = Field(
        default=None, description="状态数组；省略查询 current/planned/paused"
    )


class ReadContextModuleArgs(ProfileModel):
    module_id: str = Field(description="list_context_modules 返回的实例 id，不是 kind 或栏目名")
    field_paths: list[str] | None = Field(
        default=None,
        description=(
            "省略可读取完整模块。指定时从模块根开始：summary、status、period、details，"
            "或 details.working_arrangement.schedule 等。业务字段必须保留 details. 前缀；"
            "get_context_module_spec 返回可直接使用的 read_paths。"
        ),
    )


def build_context_module_tools(tracker: ModuleReadTracker, *, project_id: str):
    snapshot = tracker.snapshot
    if snapshot.project_id != project_id:
        raise ValueError("模块快照不属于当前项目")

    def list_context_modules(kinds=None, statuses=None):
        statuses = statuses if statuses is not None else ["current", "planned", "paused"]
        tracker.record(directory=True)
        items = [
            item
            for item in snapshot.modules
            if item["status"] in statuses and (kinds is None or item["kind"] in kinds)
        ]
        for item in items:
            tracker.record(item, ["summary", "kind", "status", "basis"])
        return {
            **snapshot.envelope(),
            "statuses": statuses,
            "modules": [
                {
                    key: item[key]
                    for key in ("id", "kind", "title", "status", "revision", "summary", "basis")
                }
                for item in items
            ],
        }

    def read_context_module(module_id, field_paths=None):
        if snapshot.availability != "ready":
            return {**snapshot.envelope(), "module": None}
        module = next((item for item in snapshot.modules if item["id"] == module_id), None)
        if module is None:
            raise ToolInputError("模块不属于本次冻结快照，请先 list_context_modules 获取实例 id", field="module_id")
        try:
            values = (
                {path: read_path(module, path) for path in field_paths} if field_paths else module
            )
        except ValueError as error:
            raise ToolInputError(
                f"{error}。业务路径从 details. 开始；可省略 field_paths 读取完整模块，"
                "或使用 get_context_module_spec.read_paths 中的完整路径。",
                field="field_paths",
            ) from error
        tracker.record(module, field_paths or [])
        return {
            **snapshot.envelope(),
            "module_id": module_id,
            "revision": module["revision"],
            "values": values,
        }

    spec = build_context_module_spec_tool()
    return {
        spec.name: spec,
        "list_context_modules": StructuredTool.from_function(
            list_context_modules,
            name="list_context_modules",
            args_schema=ListContextModulesArgs,
            description=(
                "列出本次固定快照中的生活模块。默认当前/计划/暂停，可显式读取过去；"
                "pending/failed 与空目录不同。"
            ),
        ),
        "read_context_module": StructuredTool.from_function(
            read_context_module,
            name="read_context_module",
            args_schema=ReadContextModuleArgs,
            description=(
                "按实例 ID 读取生活模块或指定字段，返回 basis 和 revision。背景不是人格定论。"
            ),
        ),
    }
