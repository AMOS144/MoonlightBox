"""只做证据日期分布统计，不对聊天正文作语义判定。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message
from moonlightbox.world.models import ConversationBundle, ConversationBundleMessage


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AnalyzeEvidenceDatesArgs(_StrictArgs):
    message_ids: list[str] = Field(
        min_length=1,
        max_length=80,
        description="已读原始消息 UUID；统计这些消息发送时间，不能当作活动发生时间",
    )


_DESCRIPTION = """对已经选定的原始消息做本地日期、星期和时段统计。它不读取或匹配消息正文，
不把发送时间当活动发生时间，也不判断某项模式是否属于目标人物。"""


def build_temporal_analysis_tools(
    session: Session,
    *,
    project_id: str,
    graph_version_id: str,
    timezone_name: str = "Asia/Shanghai",
) -> dict[str, StructuredTool]:
    # 时区是本次资料分析的宿主配置，不让模型每次改写统计口径。
    timezone = ZoneInfo(timezone_name)
    def analyze_evidence_dates(**kwargs: Any) -> dict[str, object]:
        args = AnalyzeEvidenceDatesArgs.model_validate(kwargs)
        requested = list(dict.fromkeys(args.message_ids))
        rows = list(
            session.execute(
                select(Message.id, Message.timestamp)
                .join(
                    ConversationBundleMessage,
                    ConversationBundleMessage.message_id == Message.id,
                )
                .join(
                    ConversationBundle,
                    ConversationBundle.id == ConversationBundleMessage.bundle_id,
                )
                .where(
                    ConversationBundle.project_id == project_id,
                    ConversationBundle.graph_version_id == graph_version_id,
                    Message.id.in_(requested),
                )
            )
        )
        by_id = {message_id: timestamp for message_id, timestamp in rows}
        local_values = [
            _to_local(timestamp, timezone)
            for message_id in requested
            if (timestamp := by_id.get(message_id)) is not None
        ]
        date_counts = Counter(item.date().isoformat() for item in local_values)
        weekday_counts = Counter(str(item.weekday()) for item in local_values)
        hour_counts = Counter(str(item.hour) for item in local_values)
        return {
            "timezone": timezone_name,
            "message_ids": [item for item in requested if item in by_id],
            "missing_message_ids": [item for item in requested if item not in by_id],
            "distinct_local_dates": sorted(date_counts),
            "date_counts": dict(sorted(date_counts.items())),
            "weekday_counts": dict(sorted(weekday_counts.items())),
            "hour_counts": dict(sorted(hour_counts.items(), key=lambda item: int(item[0]))),
        }

    return {
        "analyze_evidence_dates": StructuredTool.from_function(
            name="analyze_evidence_dates",
            description=_DESCRIPTION,
            func=analyze_evidence_dates,
            args_schema=cast(Any, AnalyzeEvidenceDatesArgs),
        )
    }


def _to_local(timestamp: datetime, timezone: ZoneInfo) -> datetime:
    if timestamp.tzinfo is None:
        # 导入旧数据若没有 offset，项目历史约定按 UTC 存储；这里只统一做展示统计，
        # 不从其数值推断任何事件事实。
        return timestamp.replace(tzinfo=ZoneInfo("UTC")).astimezone(timezone)
    return timestamp.astimezone(timezone)
