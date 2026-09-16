"""Director 的 ``search_memory`` LangChain 工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast

from langchain_core.tools import StructuredTool
from pydantic import Field, field_validator
from sqlalchemy.orm import Session

from ..schemas import StrictModel


class MemorySearchQuery(StrictModel):
    """模型只能描述查询；分支、快照和时间边界由 Runtime 绑定。"""

    query: str = Field(
        min_length=2,
        max_length=300,
        description="需要核实的具体人物事实、关系、承诺或历史情境。",
    )
    limit: int = Field(default=8, ge=1, le=12, description="最多返回多少条证据。")
    include_original: bool = Field(
        default=False,
        description="只有确实需要核对措辞时才返回少量原文。",
    )

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("query 不能只有空白；请描述需要核实的事实或情境")
        return value


class SearchMemoryArgs(MemorySearchQuery):
    """仅供内部服务选择数据范围，不暴露给模型。"""

    scope: Literal["branch", "world", "both"] = Field(
        default="both",
        description=(
            "内部绑定的记忆范围；模型不接收此参数。"
            "branch 为分支，world 为冻结人物资料，both 为两者"
        ),
    )


_DESCRIPTION = """只读查询当前虚拟分支与冻结历史中的记忆证据。

当 runtime_context 缺少完成当前判断所需的具体事实时使用；不要重复查询已经预加载的
最近对话、当前生活状态或当天计划。返回值包含 source_ids、资料时间边界和匹配记录；
没有 source_ids 的结果只能当作弱背景，不能据此创造事实。
图谱部分为语义召回；recent_context 是近期上下文候选，不保证与当前问题相关。
服务异常或 partial 不等于没有历史，已有信息不足时可以调整具体问题重查。"""


def build_memory_search_tool(
    session: Session,
    *,
    branch_id: str,
    snapshot_id: str | None,
    excluded_source_ids: set[str] | None = None,
    as_of: datetime | None = None,
    as_of_resolver=None,
) -> StructuredTool:
    """构造单个 Cycle 的工具；所有身份参数都封装在闭包中。"""

    # 局部导入避免 MemoryService 为类型参数导入本模块时形成循环。
    from ..memory import MemoryService

    service = MemoryService(session)
    index = service.current_index(branch_id)
    active_ids = set(index.active_record_ids or []) if index is not None else set()

    def invoke(**kwargs: Any) -> dict[str, Any]:
        visible_now = as_of_resolver() if as_of_resolver else as_of
        args = SearchMemoryArgs(
            scope="both", **MemorySearchQuery.model_validate(kwargs).model_dump()
        )
        result = service.search(
            branch_id=branch_id,
            snapshot_id=snapshot_id,
            args=args,
            excluded_source_ids=excluded_source_ids,
            as_of=visible_now,
            branch_record_ids=active_ids,
        )
        if args.scope in {"branch", "both"}:
            from datetime import UTC

            from ..conversation_index import search_branch

            semantic = search_branch(
                session, branch_id, args.query, visible_now or datetime.now(UTC), args.limit
            )
            result["data"].append({"selection": "branch_semantic_history", **semantic})
        if args.scope in {"world", "both"} and snapshot_id:
            from ..db_models import RuntimeSnapshotRow

            snapshot = session.get(RuntimeSnapshotRow, snapshot_id)
            if snapshot is not None:
                from langchain_core.tools import ToolException

                from moonlightbox.agent_runtime.tool_errors import ToolServiceError

                try:
                    result = add_semantic_history(session, snapshot, args.query, args.limit, result)
                except (ToolException, ToolServiceError):
                    # 部分服务失败不能吞掉另一侧已找到的内容，也不解释成没有历史。
                    result["retrieval_status"] = "partial"
                    result["historical_status"] = "unavailable"
        return model_memory_result(result)

    return StructuredTool.from_function(
        name="search_memory",
        description=_DESCRIPTION,
        func=invoke,
        args_schema=cast(Any, MemorySearchQuery),
    )


def model_memory_result(result: dict[str, Any]) -> dict[str, Any]:
    """记录 ID 留在账本内部，Agent 只得到事实摘要和可审计来源。"""

    data = result.get("data", [])
    clean_data = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                clean = {key: value for key, value in item.items() if key != "record_id"}
                if item.get("record_id"):
                    clean["memory_ref"] = item["record_id"]
                clean_data.append(clean)
    return {**result, "data": clean_data}


def add_semantic_history(session, snapshot, query, limit, result):
    from .routine_evidence import query_frozen_history

    history = query_frozen_history(session, snapshot, query, limit=limit)
    return {
        **result,
        "source_ids": list(dict.fromkeys([*result.get("source_ids", []), *history["source_ids"]])),
        "data": [
            *result.get("data", []),
            {
                "selection": "lightrag_semantic_history",
                "summary": history["context"],
                "messages": history["messages"],
                "source_ids": history["source_ids"],
                "retrieval_status": history["retrieval_status"],
            },
        ],
    }
