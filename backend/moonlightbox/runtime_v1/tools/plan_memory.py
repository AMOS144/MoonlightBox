"""DayPlanAgent 的结构化计划记忆工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from langchain_core.tools import StructuredTool
from pydantic import Field
from sqlalchemy.orm import Session

from ..db_models import RuntimeSnapshotRow
from ..memory import MemoryService
from ..schemas import StrictModel
from .memory_search import SearchMemoryArgs, add_semantic_history, model_memory_result


class SearchPlanMemoryArgs(StrictModel):
    """查询与目标日期生活安排相关的结构化记忆。"""

    query: str = Field(min_length=2, max_length=300, description="一个明确的排程问题。")
    limit: int = Field(
        default=6, ge=1, le=10, description="最多返回的记忆候选条数；不是要生成的日程块数"
    )


_DESCRIPTION = """查询与计划有关的冻结人物记忆和当前分支事实。

适合核实具体承诺和已经结构化的生活事实。工具固定同时查询当前分支和冻结人物世界，
避免把历史作息误查到空的分支范围。recent_context 是近期上下文候选，不保证与问题相关；
图谱部分为语义召回，需要结合原文理解。结果没有 source_ids 时只能作为弱提示。
读取不提交日程；服务异常或 partial 不等于没有资料。"""


def build_plan_memory_tool(
    session: Session,
    *,
    branch_id: str,
    snapshot: RuntimeSnapshotRow,
    virtual_now: datetime,
) -> StructuredTool:
    service = MemoryService(session)

    def invoke(**kwargs: Any) -> dict[str, Any]:
        args = SearchPlanMemoryArgs.model_validate(kwargs)
        result = service.search(
            branch_id=branch_id,
            snapshot_id=snapshot.id,
            args=SearchMemoryArgs(
                scope="both",
                query=args.query,
                limit=args.limit,
                include_original=False,
            ),
            as_of=virtual_now,
        )
        result = add_semantic_history(session, snapshot, args.query, args.limit, result)
        return {**model_memory_result(result), "tool_name": "search_plan_memory"}

    return StructuredTool.from_function(
        name="search_plan_memory",
        description=_DESCRIPTION,
        func=invoke,
        args_schema=cast(Any, SearchPlanMemoryArgs),
    )
