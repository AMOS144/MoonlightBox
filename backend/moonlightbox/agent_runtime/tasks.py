"""短任务也通过原生提交工具交付；没有裸 JSON、修复 API 或第二套循环。"""

import hashlib
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .context import retain_recent_turns
from .contracts import (
    AgentExecutionRequest,
    AgentSpec,
    RegisteredTool,
    RunScope,
    ToolContract,
)
from .controller import AgentLoopController
from .persistence import checkpoint_path
from .policy import task_budget
from .submission import result_submission_tool, submission_instruction
from .tool_errors import ToolInputError


class AgentTaskError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class ReadTaskMaterialArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    offset: int = Field(
        default=0, ge=0, description="完整任务 JSON 文本的字符偏移；首次 0，续页使用 next_offset"
    )
    limit: int = Field(
        default=12000,
        ge=1,
        le=16000,
        description="本页最多字符数；片段不一定是独立有效 JSON，next_offset=null 表示读完",
    )


def run_submission_task(
    *,
    compiler,
    name,
    system_prompt,
    payload,
    result_model,
    owner_id,
    session=None,
    project_id=None,
    tools=(),
    validator=None,
    input_revision=1,
    input_revision_resolver=None,
    snapshot_work_state=None,
    restore_work_state=None,
    input_model=None,
    result_adapter=None,
):
    """固定资料任务与可检索任务共用入口；数据库只保存检查点，不代替业务批准。"""
    model = compiler.create_agent_chat_model()
    submit_name = f"submit_{name}"
    prompt = system_prompt + "\n\n" + submission_instruction(submit_name)
    # 固定材料变化必须形成新任务，不能重放另一份输入的成功产物。
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)

    def read_task_material(offset: int = 0, limit: int = 12000):
        """分页读取本任务固定输入，offset 为字符偏移，limit 范围 1 至 16000。"""
        if offset < 0 or offset > len(serialized) or not 1 <= limit <= 16000:
            raise ToolInputError(
                f"offset 必须在 0 至 {len(serialized)} 之间，limit 必须在 1 至 16000 之间。"
            )
        return {
            "text": serialized[offset : offset + limit],
            "offset": offset,
            "total_chars": len(serialized),
            "next_offset": offset + limit if offset + limit < len(serialized) else None,
        }

    reader = StructuredTool.from_function(read_task_material, args_schema=ReadTaskMaterialArgs)
    registered = (
        *tools,
        RegisteredTool(reader, ToolContract(name=reader.name, max_result_chars=24000)),
        result_submission_tool(
            submit_name, input_model or result_model, validator, result_adapter=result_adapter
        ),
    )
    visible = (
        serialized
        if len(serialized) <= 20000
        else json.dumps(
            {
                "task_material": read_task_material(),
                "instruction": "完整输入可调用 read_task_material 分页读取。",
            },
            ensure_ascii=False,
        )
    )

    def compact(messages, **kwargs):
        return retain_recent_turns(messages), "保留固定任务与最近完整回合，原始输入可分页重读"

    fingerprint = hashlib.sha256((prompt + serialized).encode()).hexdigest()
    result = AgentLoopController().run(
        spec=AgentSpec(
            name=name,
            prompt_version=fingerprint,
            submission_tool_name=submit_name,
            budget=task_budget(),
            tools=registered,
            context_compactor=compact,
            snapshot_work_state=snapshot_work_state,
            restore_work_state=restore_work_state,
        ),
        request=AgentExecutionRequest(
            owner_type="person_world",
            owner_id=f"{name}:{owner_id}",
            project_id=project_id,
            input_revision=input_revision,
            input_revision_resolver=input_revision_resolver,
            checkpoint_path=checkpoint_path(session) if session is not None else None,
            scope=RunScope(project_id=project_id),
            messages=(SystemMessage(content=prompt), HumanMessage(content=visible)),
        ),
        model=model,
    )
    if not isinstance(result.value, result_model):
        raise AgentTaskError(result.terminal_reason)
    return result.value
