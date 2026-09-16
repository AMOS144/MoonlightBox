"""保存栏目自己的工作笔记；不是发布画像，也不是另一套调查状态机。"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field


class SectionWorkArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    draft_notes: str = Field(
        max_length=12000, description="当前栏目草稿及适用时期，保留最有用的理解，不复制整批原文"
    )
    review_questions: list[str] = Field(
        max_length=20, description="仍值得用后续或前段材料区分的解释；没有则空列表"
    )
    findings: str = Field(
        max_length=8000, description="后续复核带来的保留、缩小、修订或时期变化；未查到时如实记录"
    )
    next_action: str = Field(max_length=2000, description="下一步准备做什么；可明确已可提交")


def build_section_work_tool(artifacts):
    def save_section_work(**kwargs):
        value = SectionWorkArgs.model_validate(kwargs).model_dump()
        artifacts.section_work = value
        return {"saved": True, "work": value, "published": False}

    return StructuredTool.from_function(
        save_section_work,
        args_schema=SectionWorkArgs,
        description="保存当前栏目草稿、复核问题与下一步，随统一检查点恢复；不会发布或修改图谱。",
    )
