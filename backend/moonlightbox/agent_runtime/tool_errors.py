"""工具作者显式声明的可修复输入错误，不公开未知内部异常。"""

from langchain_core.tools import ToolException
from pydantic import ValidationError


class ToolInputError(ToolException):
    """message 应说明错误字段、有效范围或下一步修正方式。"""

    code = "invalid_tool_arguments"

    def __init__(self, message, *, field=None):
        super().__init__(message)
        self.field = field


class ToolServiceError(RuntimeError):
    """依赖不可用不是模型参数错误；保留服务代码交给统一重试策略。"""

    def __init__(self, message, *, code="tool_unavailable"):
        self.code = code
        super().__init__(message)


def tool_error_result(error, *, tool_name, arguments, attempts=1):
    """统一面向模型的错误回执；仅回显出错字段，不泄露整个内部异常或上下文。

    retryable 只表示传输层可否原样重试；模型能否修正由 recoverable 明确表达。
    缺失字段与 null、空字符串、空数组必须可区分，不能把模型的错误猜测当实参。
    """
    from dataclasses import asdict, replace

    from .resilience import classify_failure

    failure = replace(classify_failure(error), attempts=attempts)
    if failure.code == "model_error":
        failure = replace(failure, code="tool_execution_error")
    details = []
    if isinstance(error, ToolInputError) and error.field:
        path = error.field.split(".")
        if getattr(error, "target_tool_name", None):
            path = ["arguments", *path]
        details.append({"loc": path, "type": "value_error", "msg": str(error)})
    if isinstance(error, ValidationError):
        for item in error.errors(include_url=False, include_context=False, include_input=True):
            actual = item.pop("input", None)
            # 模型级校验可能包含整份提案；仅展示类型，不重复大段正文或凭据。
            missing = item["type"] == "missing"
            value_type = "missing" if missing else type(actual).__name__
            if missing:
                actual = None
            elif isinstance(actual, dict):
                # 省略对象不能写成 null，否则模型会把“对象类型错”误解成“传了 null”。
                actual = "[value omitted: object]"
            elif isinstance(actual, str) and len(actual) > 120:
                actual = "[value omitted: too long]"
            elif isinstance(actual, list):
                # 数组可能含整份 Profile；只回显有界的标量数组（如错误引用）。
                import json

                if any(
                    not isinstance(value, (str, int, float, bool, type(None))) for value in actual
                ):
                    actual = "[value omitted: nested input]"
                elif len(actual) > 120 or len(json.dumps(actual, ensure_ascii=False)) > 512:
                    actual = "[value omitted: too long]"
            elif not isinstance(actual, (str, int, float, bool, list, type(None))):
                actual = "[value omitted: non-JSON input]"
            if getattr(error, "target_tool_name", None):
                item["loc"] = ["arguments", *item["loc"]]
            details.append({**item, "actual_type": value_type, "actual_value": actual})
    recoverable = isinstance(error, (ValidationError, ToolException))
    message = str(error) if isinstance(error, ToolException) else failure.message
    if details:
        message = (
            "按 validation_errors 的 loc、msg 和实际类型修正对应字段，其他正确字段保持不变。"
            "数组使用 JSON 数组；可空字段允许 null，不等于空字符串或 []；"
            "不需要的可选字段可以省略。不要用占位对象代替 null。"
        )
        if any(item.get("type") == "extra_forbidden" for item in details):
            message += "extra_forbidden 表示该键不是当前工具参数；删除对应键，不要照抄内部状态或历史产物。"
    return {
        "status": "rejected" if recoverable else "error",
        "error": failure.code,
        "code": failure.code,
        "tool_name": tool_name,
        "target_tool_name": getattr(error, "target_tool_name", tool_name),
        "source_ids": [],
        "failure": asdict(failure),
        "message": message,
        "recoverable": recoverable,
        "retryable": failure.retryable,
        "next_action": "correct_arguments"
        if recoverable
        else "retry_transport"
        if failure.retryable
        else "report_failure",
        "validation_errors": details,
    }


def rejected_call_result(planned, *, tool_name, available_tools):
    """工具执行前的协议、授权和预算拒绝也使用相同回执形状。"""
    code = str(planned.get("error", "undeclared_tool"))
    messages = {
        "submission_must_be_separate": (
            "提交工具必须单独调用；先完成所需查询，再单独提交完整结果。此批未执行。"
        ),
        "invalid_tool_arguments": "参数必须是合法 JSON 对象；修正语法、顶层类型后重新调用。",
        "undeclared_tool_or_invalid_arguments": (
            "工具未提供或参数不是对象；核对 available_tools，使用正确名称和 JSON 对象。"
        ),
        "undeclared_tool": "工具未提供；从 available_tools 选择已声明工具，不猜测其他工具名。",
        "tool_not_authorized": "本次执行没有此工具权限；不能通过改参数绕过授权，请报告配置问题。",
        "tool_call_safety_limit": "本次工具调用预算已耗尽，不应继续重试同一操作。",
    }
    recoverable = code not in {"tool_not_authorized", "tool_call_safety_limit"}
    message = str(planned.get("message") or messages.get(code, "调用被拒绝，请检查工具契约"))
    return {
        "status": "rejected",
        "error": code,
        "code": code,
        "tool_name": tool_name,
        "target_tool_name": tool_name,
        "source_ids": [],
        "message": message,
        "recoverable": recoverable,
        "retryable": False,
        "next_action": "correct_call" if recoverable else "report_failure",
        "failure": {
            "code": code,
            "category": "tool_input"
            if recoverable
            else "budget"
            if code == "tool_call_safety_limit"
            else "configuration",
            "retryable": False,
            "message": message,
            "attempts": 0,
        },
        "validation_errors": [],
        "available_tools": sorted(available_tools) if tool_name not in available_tools else [],
        **(
            {"raw_arguments": str(planned["raw_arguments"])[:1000]}
            if "raw_arguments" in planned
            else {}
        ),
    }
