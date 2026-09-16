"""Director 的连续原文、处境和绑定画像工具；身份及时间边界封装在闭包中。"""

from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import Field, field_validator, model_validator

from moonlightbox.agent_runtime.tool_errors import ToolInputError, ToolServiceError

from ..conversation_history import ConversationHistory
from ..db_models import RuntimeSnapshotRow
from ..director_contracts import Contract


class ReadConversation(Contract):
    before_ref: str | None = Field(default=None, description="从该消息之前继续向前翻页")
    around_ref: str | None = Field(default=None, description="展开搜索命中消息的前后原文")
    limit: int = Field(
        default=40,
        ge=1,
        le=100,
        description="最多读取的消息条数；不是对话轮数。首次不传两个引用，翻页和展开只能选择一种。",
    )

    @field_validator("before_ref", "around_ref")
    @classmethod
    def nonempty_reference(cls, value):
        if value is not None and not value.strip():
            raise ValueError("引用不能为空字符串；首次读取请省略或传 null")
        return value

    @model_validator(mode="after")
    def exclusive_cursor(self):
        if self.before_ref is not None and self.around_ref is not None:
            raise ValueError("before_ref 与 around_ref 只能选择一个；首次均省略")
        return self


class ReadConcerns(Contract):
    concern_ref: str | None = Field(
        default=None, description="当前状态中已有的 concern_ref；省略读取全部关切，不自行构造 ID"
    )

    @field_validator("concern_ref")
    @classmethod
    def nonempty_reference(cls, value):
        if value is not None and not value.strip():
            raise ValueError("concern_ref 不能为空字符串；读取全部关切请省略或 null")
        return value


class ReadProfile(Contract):
    section: Literal[
        "identity",
        "life_context",
        "social_world",
        "agency",
        "practices",
        "life_course",
        "relationship_with_user",
    ] = Field(
        description="identity/life_context/social_world/agency/practices/life_course/relationship_with_user"
    )


class SearchConversation(Contract):
    query: str = Field(
        min_length=1,
        max_length=1200,
        description="描述要找的对话内容或互动情境；同时搜索原始历史与当前分支，不传数据库筛选表达式",
    )
    limit: int = Field(
        default=8, ge=1, le=20, description="最多返回的命中消息数；使用命中引用继续展开原文"
    )

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value):
        if not value.strip():
            raise ValueError("query 必须描述要查找的互动或内容，不能只有空白")
        return value


def build_director_context_tools(session, packet):
    branch_id = packet.branch["branch_id"]

    def history():
        # 输入续接会推进 packet.virtual_now，不能继续使用首次调用时缓存的消息索引。
        snapshot = session.get(RuntimeSnapshotRow, packet.origin["snapshot_id"])
        if snapshot is None:
            raise ToolServiceError("当前分支历史快照不可用；不是没有聊天记录")
        return ConversationHistory(
            session,
            branch_id,
            snapshot,
            packet.virtual_now,
        )

    def conversation(before_ref=None, around_ref=None, limit=40):
        try:
            return history().read(before_ref=before_ref, around_ref=around_ref, limit=limit)
        except ValueError as error:
            if str(error) != "conversation_reference_out_of_scope":
                raise
            raise ToolInputError(
                "消息引用不在当前可见对话中；使用 source_ref 或 next_before_ref，"
                "不要使用 result_ref、事件 ID 或其他分支引用。首次读取可省略引用。",
                field="around_ref" if around_ref else "before_ref",
            ) from error

    def concerns(concern_ref=None):
        state = packet.current.get("life_state", {}).get("subjective_state", {})
        if concern_ref is not None and not any(
            item["concern_ref"] == concern_ref for item in state.get("concerns", [])
        ):
            raise ToolInputError(
                "concern_ref 不在当前关切中；省略此参数读取全部现有关切，再使用返回的引用",
                field="concern_ref",
            )
        return {
            **state,
            "concerns": [
                item
                for item in state.get("concerns", [])
                if concern_ref is None or item["concern_ref"] == concern_ref
            ],
        }

    def profile(section):
        # 栏目名称已由 Literal 校验；绑定丢失不应返回空内容冒充未调查。
        profile = packet.origin.get("person_world_profile")
        if not isinstance(profile, dict) or not profile:
            raise ToolServiceError("本分支绑定的人物画像不可用；不是该栏目没有资料")
        return {
            "section": section,
            "content": profile.get(section),
            "status": "available" if profile.get(section) is not None else "not_compiled",
            "binding": "current_branch_snapshot",
        }

    def search(query, limit=8):
        return history().search(query, limit)

    return [
        StructuredTool.from_function(
            search,
            name="search_conversation",
            args_schema=SearchConversation,
            description=(
                "统一搜索以前的聊天，自动合并可见历史与当前对话。"
                "按 source_ref 用 read_conversation 展开前后文；partial 表示检索范围尚不完整。"
            ),
        ),
        StructuredTool.from_function(
            conversation,
            name="read_conversation",
            args_schema=ReadConversation,
            description=(
                "读取连续聊天原文，自动衔接历史与当前对话。"
                "用 around_ref 展开命中，用 before_ref 向前翻页。"
            ),
        ),
        StructuredTool.from_function(
            concerns,
            name="get_subjective_state",
            args_schema=ReadConcerns,
            description="读取已持久化处境状态，包含暂放和已解决事项；可引用 concern_ref 重新打开。",
        ),
        StructuredTool.from_function(
            profile,
            name="get_profile_section",
            args_schema=ReadProfile,
            description="展开本分支已经绑定的人物画像栏目，不切换到其他画像版本。",
        ),
    ]
