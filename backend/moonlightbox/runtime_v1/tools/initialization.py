"""初始化工具：调查笔记与提案检查留在工具层，不管理分支生命周期。"""

from datetime import UTC, datetime

from langchain_core.tools import StructuredTool

from moonlightbox.agent_runtime.contracts import RegisteredTool, ToolContract
from moonlightbox.agent_runtime.input_contracts import input_model
from moonlightbox.agent_runtime.submission import result_submission_tool
from moonlightbox.agent_runtime.tool_execution import serial

from ..director_contracts import Concern, InitialState, InvestigationWork
from .director_context import build_director_context_tools
from .memory_search import build_memory_search_tool


def build_initialization_tools(session, packet, snapshot, row, box, allowed):
    branch_id = packet.branch["branch_id"]

    def save_work(**kwargs):
        row.work = InvestigationWork.model_validate(kwargs).model_dump(mode="json")
        row.updated_at = datetime.now(UTC)
        session.commit()
        return {"saved": True, "work": row.work}

    tools = box.register(
        [
            t
            for t in build_director_context_tools(session, packet)
            if t.name != "get_subjective_state"
        ]
    )
    tools += (
        box.register_one(
            build_memory_search_tool(
                session,
                branch_id=branch_id,
                snapshot_id=snapshot.id,
                as_of=packet.virtual_now,
            )
        ),
    )
    work_tool = StructuredTool.from_function(
        save_work,
        name="update_initialization_work",
        args_schema=InvestigationWork,
        description="保存已读范围、暂定理解与待查问题，替换本任务调查笔记；不是正式状态。压缩或恢复后继续使用。",
    )
    tools += (
        RegisteredTool(
            tool=work_tool,
            contract=ToolContract(
                name=work_tool.name,
                side_effect="proposal",
                execution=serial(
                    "sqlalchemy_session", "initialization_work", reason="只保存本任务草稿，串行提交"
                ),
            ),
        ),
    )

    def validate(value, _context):
        # 一次列出所有错误位置和值，不让模型修完第一个才看到下一个。
        errors = []
        for index, concern in enumerate(value.subjective_state.concerns):
            for field in ("concern_ref", "action_ref", "depends_on_expression"):
                actual = getattr(concern, field)
                if actual:
                    expected = "false" if field == "depends_on_expression" else "null"
                    errors.append(
                        f"subjective_state.concerns[{index}].{field}={actual!r}：起点初始化应为 {expected}"
                    )
        groups = {
            "subjective_state.concerns": value.subjective_state.concerns,
            "open_conversation_threads": value.open_conversation_threads,
            "active_commitments": value.active_commitments,
        }
        for group, entries in groups.items():
            for index, item in enumerate(entries):
                for position, ref in enumerate(item.source_refs):
                    if ref not in allowed:
                        errors.append(
                            f"{group}[{index}].source_refs[{position}]={ref!r}：不是本分支可见的原始消息引用"
                        )
        if errors:
            return (
                "；".join(errors)
                + "。source_refs 只能原样使用 read_conversation 或预读消息的 source_ref；不能填写 result_ref、画像标签或自造 UUID。没有可定位的消息引用可填 []，保留你的理解，但不要编造来源。"
            )
        return None

    initial_input = input_model(
        InitialState,
        omit=lambda cls, name, field: (
            cls is Concern and name in {"concern_ref", "action_ref", "depends_on_expression"}
        ),
    )
    submission = result_submission_tool(
        "submit_initial_state",
        initial_input,
        validate,
        result_adapter=lambda value: InitialState.model_validate(value.model_dump()),
    )
    submission.tool.description += (
        "本工具只初始化新状态；身份与初始执行标记由后端绑定。"
        "所有 source_refs 仅接受预读或 read_conversation 消息中的 source_ref，不接受分页 result_ref、"
        "profile 等标签或自造 UUID。source_refs 允许 []；只有画像理解而无具体消息引用时不要伪造来源。"
    )
    return tools + (submission,)
