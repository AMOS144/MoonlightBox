"""共用通信工具。身份与事务由宿主绑定，模型不能跨分支或冒充发送者。"""

from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import update
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.tool_errors import ToolInputError

from ..branch_models import Branch
from ..collaboration.transport import enqueue_peer_message


class SendAgentMessageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    recipient: Literal["director", "day_planner"] = Field(
        description="接收协作消息的同级 Agent；不能选择自己"
    )
    content: str = Field(
        min_length=1,
        max_length=8000,
        description="发给对方的正文：说明问题、已知约束或需要同步的信息；区分提议和已生效事实",
    )


def build_send_agent_message_tool(
    session, *, branch_id, sender, task_id, cancellation_requested=None
):
    bind = session.get_bind()

    def send(recipient, content):
        if recipient == sender:
            raise ToolInputError("不能给自己发送协作消息；请选择另一位 Agent", field="recipient")
        if cancellation_requested and cancellation_requested():
            from moonlightbox.agent_runtime.resilience import ExecutionInterrupted

            raise ExecutionInterrupted("cancelled")
        # 独立短事务，不把调用者的草稿/尚未验证的状态顺带提交。
        with Session(bind) as writer:
            locked = writer.execute(
                update(Branch)
                .where(
                    Branch.id == branch_id,
                    Branch.lifecycle_status == "active",
                )
                .values(runtime_input_revision=Branch.runtime_input_revision)
            )
            if locked.rowcount != 1:
                raise ToolInputError("当前分支不可通信，请结束本次工作")
            from ..service import RuntimeService

            branch = writer.get(Branch, branch_id)
            clock = RuntimeService(writer).get_clock(branch.project_id, branch_id)
            event = enqueue_peer_message(
                writer,
                branch=branch,
                sender=sender,
                recipient=recipient,
                content=content,
                task_id=task_id,
                occurred_at=clock.now(),
            )
            result = {
                "status": "queued" if event.status == "queued" else event.status,
                "message_id": event.id,
                "recipient": recipient,
                "meaning": "消息已持久化；不是已读、同意或计划提交回执",
            }
            writer.commit()
            return result

    return StructuredTool.from_function(
        name="send_agent_message",
        func=send,
        args_schema=SendAgentMessageArgs,
        description="向同一分支的另一位 Agent 发送协作消息，由现有调度器交付。不会嵌套启动对方，"
        "不会给用户发聊天，也不会修改计划。同一任务重复发送相同正文复用原回执。"
        "接收方按需处理，不必对每条通知回复；计划已生效以 Executor 回执为准。",
        metadata={
            "side_effect": "communication",
            "contract_version": "peer-message-v1",
            "required_permissions": ["runtime.send_agent_message"],
        },
    )
