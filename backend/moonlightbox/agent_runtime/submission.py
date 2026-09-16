"""原生结果提交工具：校验失败返回可修复回执，不产生领域写入副作用。"""

import json
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.tools import StructuredTool
from pydantic import ConfigDict, Field, SerializeAsAny, create_model, field_validator

from .contracts import ProgressDelta, RegisteredTool, SubmissionReceipt, ToolContract
from .tool_execution import submission_policy

_CONTEXT = ContextVar("agent_submission_context", default=None)


@contextmanager
def submission_context(context):
    receipt = SubmissionReceipt()
    token = _CONTEXT.set((context, receipt))
    try:
        yield receipt
    finally:
        _CONTEXT.reset(token)


def result_submission_tool(
    name, model, validator=None, *, completion_status=None, result_adapter=None
):
    """结构由 args_schema 校验，领域条件由注入的无副作用服务校验。"""

    @field_validator("result", mode="before")
    def parse_json_result(cls, value):
        # 工具参数本来来自 JSON；按 JSON 语义解析 date/datetime，保持领域模型 strict。
        return (
            model.model_validate_json(json.dumps(value, ensure_ascii=False))
            if isinstance(value, dict)
            else value
        )

    @field_validator("result", mode="after")
    def bind_result(cls, value):
        # 绑定与领域校验仍在 LangChain 参数解析阶段执行；Pydantic 自动加 result 路径。
        return result_adapter(value) if result_adapter else value

    args = create_model(
        f"{model.__name__}Submission",
        __config__=ConfigDict(extra="forbid"),
        __validators__={"parse_json_result": parse_json_result, "bind_result": bind_result},
        result=(
            SerializeAsAny[model],
            Field(description="本次任务的完整提案；通过后交给工作流执行，不代表已经落库"),
        ),
    )

    def submit(result):
        scope = _CONTEXT.get()
        if scope is None:
            raise RuntimeError("submission_context_missing")
        context, receipt = scope
        reason = validator(result, context) if validator else None
        if reason:
            return {
                "status": "rejected",
                "code": "invalid_proposal",
                "error": "invalid_proposal",
                "target_tool_name": name,
                "validation_errors": [],
                "failure": {
                    "code": "invalid_proposal",
                    "category": "tool_input",
                    "retryable": False,
                    "message": reason,
                    "attempts": 1,
                },
                "message": reason,
                "recoverable": True,
                "retryable": False,
                "tool_name": name,
                "next_action": "correct_arguments",
                "committed": False,
            }
        receipt.value = result
        receipt.status = completion_status(result) if completion_status else "succeeded"
        if receipt.status not in {"succeeded", "waiting_for_user"}:
            raise ValueError("invalid_submission_completion_status")
        return {"status": "accepted", "committed": False, "result": result.model_dump(mode="json")}

    tool = StructuredTool.from_function(
        func=submit,
        name=name,
        args_schema=args,
        description=(
            "提交本次任务结果，必须单独调用，不能与其他工具并行。"
            "rejected 表示未被接受，按 message 修正后再次提交；"
            "accepted 表示提案已交付并结束本次 Agent 执行，committed=false，"
            "实际状态写入、消息发送及最终版本检查仍由工作流中的 Executor 负责。"
            "参数错误返回 validation_errors，其中 loc 是字段路径，"
            "actual_type/actual_value 是实际收到的错误值。"
            "根据这些字段修正后重交完整提案，不要仅改说明文字、改依据类型或伪造引用绕过错误。"
        ),
    )
    return RegisteredTool(
        tool=tool,
        is_submission=True,
        contract=ToolContract(
            name=name,
            side_effect="proposal",
            execution=submission_policy(name),
            max_result_chars=96000,
            progress_evaluator=lambda result, **_: ProgressDelta(
                new_keys=frozenset({"submission:accepted"})
                if result.get("status") == "accepted"
                else frozenset(),
                summary=result.get("message", "结果已交付"),
            ),
        ),
    )


def submission_instruction(name):
    """显式完成约定，不根据工具命名推断工具职责。"""
    return f"通过 {name} 交付本次产物；普通文本不是提交。"
