"""Director 提案提交工具：参数语义与引用检查在工具交互内完成。"""

from datetime import UTC, timedelta

from moonlightbox.agent_runtime.submission import result_submission_tool

from ..schemas import LifeDecision
from .director_submission_contract import DirectorSubmission, to_life_decision


def build_submit_decision_tool(packet, reference_validator=None, expression_submission=None):
    """绑定本轮上下文；只验收提案，不写入人物状态或发送消息。"""

    def validate(value, context):
        if value.action == "speak" and value.reply is None:
            return "speak 必须通过 reply 提交实际回复，不再交给其他 Agent 代写"
        if value.reply is not None and expression_submission is not None:
            error = expression_submission.validate(value.reply, context)
            if error:
                return error
        return validate_decision(value, packet) or (
            reference_validator(value) if reference_validator else None
        )

    return result_submission_tool(
        "submit_decision", DirectorSubmission, validate, result_adapter=to_life_decision
    )


def validate_decision(value, packet):
    if not isinstance(value, LifeDecision):
        return "invalid_life_decision"
    if not (value.plan_request or value.peer_reply):
        pending = set(packet.branch.get("working_window", {}).get("pending_message_refs", []))
        answered = {item.message_ref for item in value.input_resolutions}
        if not pending.issubset(answered):
            return (
                "result.input_resolutions 缺少待处理消息："
                + ", ".join(sorted(pending - answered))
                + "；需要稍后回应用 awaiting_response 并设置 next_wakeup_at。"
            )
        if (
            any(item.status == "completed" for item in value.input_resolutions)
            and value.action != "speak"
        ):
            return "未产生表达不能标记回复完成；确实不需回复用 no_response_needed"
        if (
            any(item.status == "awaiting_response" for item in value.input_resolutions)
            and value.next_wakeup_at is None
        ):
            return "延迟回应需要 next_wakeup_at，避免用户消息一直等待"
    if (value.plan_request or value.peer_reply) and (
        value.subjective_state_updates or value.memory_proposals or value.input_resolutions
    ):
        return "协作中先交接请求；状态与记忆更新留到最终决策，避免中间提案未提交就被覆盖"
    conflicts = [
        f"state_patch.{field}"
        for field in ("mood", "attention", "current_goal", "open_conversation_threads")
        if getattr(value.state_patch, field) is not None
    ]
    if conflicts:
        return (
            "新 Director 不允许写旧字段："
            + ", ".join(conflicts)
            + "。请删除这些键；心理变化和意向使用 subjective_state_updates，空列表也请删除。"
        )
    if value.peer_reply is not None and value.plan_request is not None:
        return "同一轮只能回答协作问题或发起修改请求，不能同时交接两个任务"
    if value.next_wakeup_at is not None:
        now = (
            packet.virtual_now.replace(tzinfo=UTC)
            if packet.virtual_now.tzinfo is None
            else packet.virtual_now
        )
        if value.next_wakeup_at.tzinfo is None:
            return "next_wakeup_at 必须含时区，例如 +08:00 或 Z"
        if value.next_wakeup_at <= now:
            return "next_wakeup_at 必须晚于 virtual_now"
    if value.action != "speak" and value.speech_mode is not None:
        return "只有 speak 可以设置 speech_mode"
    if value.plan_request and value.plan_request.target_date not in {
        None,
        packet.virtual_now.date(),
        packet.virtual_now.date() + timedelta(days=1),
    }:
        return "只接受虚拟本地日期的当天或明天的计划修订"
    current = packet.current.get("life_state", {}).get("activity")
    if value.state_patch.activity and value.state_patch.activity != current:
        if value.plan_request is None and not value.state_patch.source_event_ids:
            return "改变 activity 必须提供计划修订请求或已发生事件来源"
    return None
