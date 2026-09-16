"""规划工作笔记：属于可恢复调查状态，不是正式日程或已确认事实。"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from ..agent_support import RuntimeToolbox


class PlanWork(BaseModel):
    model_config = ConfigDict(extra="forbid")
    established_arrangements: list[str] = Field(
        description="已确定的安排及其依据，区分承诺与模拟选择"
    )
    open_questions: list[str] = Field(description="尚未解决、确实影响计划的问题")
    draft_blocks: list[dict] = Field(description="当前部分或完整日程草稿；尚未经过 Executor 验收")
    next_steps: list[str] = Field(description="恢复后接着处理的具体事项，避免重复已有调查")


class PlanningToolbox(RuntimeToolbox):
    """笔记结果随 LangGraph tool_results 持久化，恢复和压缩均保留最新版本。"""

    def __init__(self, policy):
        super().__init__(policy)
        self.work = None

    def update(self, **fields):
        work = PlanWork.model_validate(fields).model_dump(mode="json")
        self.work = work
        return {"planning_work_checkpoint": work}

    def project(self, result):
        if isinstance(result, dict) and isinstance(result.get("planning_work_checkpoint"), dict):
            self.work = PlanWork.model_validate(result["planning_work_checkpoint"]).model_dump(
                mode="json"
            )
        return super().project(result)

    def compact(self, messages, *, source_refs, unresolved):
        import json

        from langchain_core.messages import HumanMessage

        compacted, reason = super().compact(
            messages, source_refs=source_refs, unresolved=unresolved
        )
        if self.work is not None:
            compacted.insert(
                2,
                HumanMessage(
                    content="当前规划工作笔记（不是正式计划）："
                    + json.dumps(self.work, ensure_ascii=False)
                ),
            )
        return compacted, reason

    def work_tool(self):
        return StructuredTool.from_function(
            name="update_plan_work",
            func=self.update,
            args_schema=PlanWork,
            description=(
                "完整替换调查工作笔记，保存已确定安排、待解决问题、计划草稿和下一步。"
                "有实质进展或形成草稿时保存，不必每轮调用，也不要等最终提交才保存。"
                "返回 planning_work_checkpoint，随检查点恢复；不是正式日程或已确认事实。"
                "每次提供完整笔记，相同内容重复提交只替换笔记，不会创建正式日程。"
            ),
        )

    def resume_messages(self, saved, fresh):
        from langchain_core.messages import HumanMessage

        messages = list(saved)
        current_task = next((m for m in fresh if isinstance(m, HumanMessage)), None)
        if current_task is not None:
            for index, message in enumerate(messages):
                if isinstance(message, HumanMessage):
                    # 只换任务锚点：日期、承诺和已开始块重新取值；其余调查顺序不动。
                    messages[index] = current_task
                    break
        return messages
