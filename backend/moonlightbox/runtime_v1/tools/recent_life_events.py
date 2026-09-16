"""只读分支经历；候选不进入此工具，来源不可丢失。"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select

from ..db_models import RuntimeLifeEventRow


class RecentLifeEventsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: int = Field(default=10, ge=1, le=30, description="本页最多读取的已提交模拟经历数")
    before: str | None = Field(default=None, description="上一页最早的 occurred_at，向前翻页")

    @field_validator("before")
    @classmethod
    def valid_cursor(cls, value):
        if value is not None:
            from datetime import datetime

            try:
                datetime.fromisoformat(value)
            except ValueError as error:
                raise ValueError(
                    "before 必须原样使用返回的 ISO occurred_at；首次省略或 null"
                ) from error
        return value


def recent_life_events(session, branch_id, now, limit=10, before=None):
    from datetime import datetime

    query = select(RuntimeLifeEventRow).where(
        RuntimeLifeEventRow.branch_id == branch_id,
        RuntimeLifeEventRow.event_type == "simulated_life",
        RuntimeLifeEventRow.occurred_at <= now,
    )
    if before:
        from moonlightbox.agent_runtime.tool_errors import ToolInputError

        try:
            boundary = datetime.fromisoformat(before)
        except ValueError as error:
            raise ToolInputError(
                "before 必须是上一页返回的 ISO occurred_at 时间；首次查询省略，不填自然语言时间",
                field="before",
            ) from error
        # 数据库存储 UTC；带偏移的等价游标先归一化，避免 SQL 丢弃时区后错页。
        if boundary.tzinfo is not None:
            from datetime import UTC

            boundary = boundary.astimezone(UTC).replace(tzinfo=None)
        query = query.where(RuntimeLifeEventRow.occurred_at < boundary)
    rows = session.scalars(query.order_by(RuntimeLifeEventRow.occurred_at.desc()).limit(limit))
    return [
        {
            "source_id": row.id,
            "origin": "simulation",
            "occurred_at": row.occurred_at.isoformat(),
            "event": row.payload,
        }
        for row in rows
    ]


def build_recent_life_events_tool(session, branch_id, now):
    def invoke(limit=10, before=None):
        visible_now = now() if callable(now) else now
        events = recent_life_events(session, branch_id, visible_now, limit, before)
        return {"events": events, "source_ids": [item["source_id"] for item in events]}

    return StructuredTool.from_function(
        name="get_recent_life_events",
        func=invoke,
        args_schema=RecentLifeEventsArgs,
        description=(
            "只读查询当前虚拟时间之前已提交的分支模拟生活经历及其影响。"
            "需要回顾生活变化、检查重复事件或此前影响时使用。返回 events、source_ids；"
            "按 occurred_at 倒序，用本页最早时间继续向前翻页。"
            "不含未提交候选或未来结果；origin=simulation，不是原人物真实史。"
        ),
    )
