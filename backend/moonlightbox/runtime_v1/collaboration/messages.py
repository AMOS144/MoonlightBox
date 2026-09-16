"""只追加的共享发言；模型私有工具协议不会混入另一位 Agent 的会话。"""

import json
from typing import Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage, convert_to_messages
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class PeerMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sender: Literal["system", "director", "day_planner", "executor", "persona_actor"]
    recipient: Literal["director", "day_planner", "executor", "persona_actor", "broadcast"]
    kind: Literal[
        "event",
        "request",
        "clarification",
        "answer",
        "proposal",
        "objection",
        "submit",
        "commit_result",
        "notification",
        "failure",
    ]
    content: str
    task_id: str
    payload: dict = Field(default_factory=dict)


def shared_message(sender, recipient, kind, content, task_id, payload=None):
    envelope = PeerMessage(
        sender=sender,
        recipient=recipient,
        kind=kind,
        content=content,
        task_id=task_id,
        payload=payload or {},
    )
    return HumanMessage(id=str(uuid4()), content=envelope.model_dump_json())


def merge_shared_messages(left, right):
    """禁止 add_messages 默认的同 ID 覆写，重放只能完全相同。"""
    old, new = convert_to_messages(left or []), convert_to_messages(right or [])
    known = {message.id: message.model_dump() for message in old}
    for message in new:
        if not message.id:
            raise ValueError("共享消息必须由运行时分配 ID")
        if message.id in known and known[message.id] != message.model_dump():
            raise ValueError("共享消息不可覆盖")
        known[message.id] = message.model_dump()
    return add_messages(old, new)


def discussion_payload(messages):
    """以带发送者的数据装配，不伪装成接收者曾经发出的 assistant 消息。"""
    return [json.loads(message.content) for message in messages]
