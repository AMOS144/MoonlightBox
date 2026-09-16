"""本轮工具结果分页读取：参数定义、引用权限和分页边界在工具内维护。"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from moonlightbox.agent_runtime.tool_errors import ToolInputError


class ReadRuntimeResultArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result_ref: str = Field(min_length=1, description="已返回的工具结果引用，不能自行构造")
    offset: int = Field(default=0, ge=0, description="首次为 0，续页使用上一页 next_offset")


def read_runtime_result(results, result_ref, offset=0):
    if result_ref not in results:
        raise ToolInputError(
            "result_ref 不属于当前运行或恢复的检查点；"
            "使用工具返回的 result_ref，不是消息 source_ref。",
            field="result_ref",
        )
    text = results[result_ref]
    if offset > len(text):
        raise ToolInputError(
            f"offset 超过正文长度 {len(text)}；首次为 0，续页原样使用 next_offset，null 表示结束。",
            field="offset",
        )
    end = min(offset + 12_000, len(text))
    return {
        "result_ref": result_ref,
        "offset": offset,
        "page": text[offset:end],
        "next_offset": end if end < len(text) else None,
        "document_ids": [f"{result_ref}:{offset}"],
        "total_chars": len(text),
    }


def build_read_runtime_result_tool(results):
    def read(result_ref, offset=0):
        return read_runtime_result(results, result_ref, offset)

    return StructuredTool.from_function(
        name="read_runtime_result",
        func=read,
        args_schema=ReadRuntimeResultArgs,
        description=(
            "只读分页恢复本次调查中保存的工具结果。结果被截断或已移出窗口时使用；"
            "返回 page、next_offset、total_chars。page 是原始 JSON 文本片段，不是独立 JSON；"
            "next_offset 非空表示还有正文，不能把第一页未出现的内容当作不存在。"
            "引用在恢复的检查点内可复用；同一引用与 offset 可重复读取。"
        ),
    )
