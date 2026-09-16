"""Director 当前提交契约：历史读取字段不进入原生工具 Schema。

不能靠覆盖领域模型的 model_json_schema 隐藏字段：Pydantic 嵌套模型生成
Schema 时不保证调用该覆盖方法。这里从共享字段定义构造真实的窄输入模型，
Schema 和参数校验使用同一份字段；历史 LifeDecision 仅在验收后适配。
"""

from copy import deepcopy

from pydantic import Field, create_model, model_validator

from moonlightbox.agent_runtime.input_contracts import input_model

from ..expression_contracts import ExpressionResult
from ..schemas import LifeDecision, StatePatch, StrictModel


def _fields(model, excluded):
    return {
        name: (field.annotation, deepcopy(field))
        for name, field in model.model_fields.items()
        if name not in excluded
    }


DirectorStatePatch = create_model(
    "DirectorStatePatch",
    __base__=StrictModel,
    **_fields(StatePatch, {"mood", "attention", "current_goal", "open_conversation_threads"}),
)
DirectorReply = input_model(
    ExpressionResult,
    omit=lambda cls, name, field: cls is ExpressionResult and name in {"status", "clarification"},
)


class _SubmissionRules(StrictModel):
    @model_validator(mode="after")
    def check_current_action(self):
        if self.action == "speak" and self.reply is None:
            raise ValueError(
                "result.reply 缺失：speak 必须提交 reply={messages: [...]}；"
                "reply 必须直接放在 result 下，不要嵌套进 plan_request 或 state_patch。"
            )
        if self.reply is not None and self.action != "speak":
            raise ValueError("result.reply 仅用于 action=speak；不发送时省略 reply")
        if self.action == "schedule" and self.next_wakeup_at is None:
            raise ValueError("result.next_wakeup_at：schedule 必须提供带时区的未来虚拟时间")
        refs = [item.message_ref for item in self.input_resolutions]
        if len(refs) != len(set(refs)):
            raise ValueError("result.input_resolutions：同一 message_ref 只能提交一次处理状态")
        return self


_current_fields = _fields(
    LifeDecision,
    {"state_patch", "reply", "communication_intent", "content_points", "expression_task"},
)
DirectorSubmission = create_model(
    "DirectorSubmission",
    __base__=_SubmissionRules,
    state_patch=(
        DirectorStatePatch,
        Field(
            default_factory=DirectorStatePatch,
            description=LifeDecision.model_fields["state_patch"].description,
        ),
    ),
    **_current_fields,
    reply=(
        DirectorReply | None,
        Field(
            default=None,
            description="speak 时提交 messages 数组，按发送顺序提供真实文字或可用表情；其他动作省略。完成状态由后端生成。",
        ),
    ),
)


def to_life_decision(value):
    """保留执行器和历史读取兼容，不让兼容字段重新成为模型输入。"""
    data = value.model_dump()
    if data.get("reply") is not None:
        data["reply"]["status"] = "ready"
    return LifeDecision.model_validate(data)
