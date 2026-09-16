"""离线模型显式生成原生提交调用；生产环境绝不把裸 JSON 自动转换为提交。"""

import json
from uuid import uuid4

from langchain_core.messages import AIMessage


def submission_message(*, content="", tool_calls=None, name=None):
    if tool_calls:
        return AIMessage(content=content, tool_calls=tool_calls)
    from moonlightbox.runtime_v1.cloud_models import _unwrap_json

    value = json.loads(_unwrap_json(content)) if isinstance(content, str) else content
    if name is None:
        name = (
            "submit_decision"
            if "action" in value
            else ("submit_expression" if "messages" in value else "submit_day_plan")
        )
    if name == "submit_day_plan" and not {"proposal", "reply", "completion"}.intersection(value):
        value = {"proposal": value}
    if name == "submit_day_plan" and value.get("proposal"):
        value["proposal"].pop("plan_date", None)
    if name == "submit_decision" and value.get("reply"):
        value["reply"].pop("status", None)
        value["reply"].pop("clarification", None)
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": uuid4().hex,
                "name": name,
                "args": {"result": value},
            }
        ],
    )
