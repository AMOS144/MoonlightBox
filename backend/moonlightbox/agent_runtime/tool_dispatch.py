"""固定模型侧入口，业务工具契约通过技能渐进提供；没有独立重试或 Agent 循环。"""

from collections.abc import Sequence
from types import SimpleNamespace

from langchain_core.tools import StructuredTool
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from moonlightbox.observability.phoenix import record_span_attributes

from .checkpoint_contract import checkpoint_contract_fingerprint
from .contracts import RegisteredTool, ToolContract
from .resilience import check_interruption
from .tool_errors import ToolInputError
from .tool_execution import serial


class ExecuteToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1, description="技能说明中提供的业务工具名称")
    arguments: dict = Field(description="按技能中的参数格式填写 JSON 对象，不传 JSON 字符串")


def build_execute_tool(tools: Sequence[RegisteredTool]) -> RegisteredTool:
    """只接受调用方已绑定身份与资源范围的只读工具，不做任意函数查找或代码执行。

    参数校验仍由内部 LangChain 工具完成。外层 Controller 负责唯一的调用计数、
    重试、取消和错误回执；保守合并资源、权限及预算，不能经通用入口绕过原契约。
    """
    registry = {item.tool.name: item for item in tools}
    if len(registry) != len(tools):
        raise ValueError("execute_tool 内部工具名称重复")
    if any(
        item.is_submission
        or item.contract.side_effect != "read_only"
        or item.tool.name in {"execute_tool", "read_skill"}
        for item in tools
    ):
        raise ValueError("execute_tool 只允许只读业务工具，提交及递归调用不开放")
    version = checkpoint_contract_fingerprint(
        SimpleNamespace(
            tools=tuple(tools),
            contract_version="execute-tool-v1",
            state_version="1",
            submission_tool_name="",
        )
    )

    def execute_tool(tool_name, arguments):
        selected = registry.get(tool_name)
        if selected is None:
            error = ToolInputError(
                "tool_name 未提供；可用名称："
                + (", ".join(sorted(registry)) or "无")
                + "。读取技能确认参数，不猜测工具名。",
                field="tool_name",
            )
            # 此处错误在分发器自身，不给路径添加内层 arguments 前缀。
            raise error
        record_span_attributes(
            trace.get_current_span(),
            {
                "moonlightbox.tool.dispatcher": "execute_tool",
                "moonlightbox.tool.target_name": tool_name,
                "moonlightbox.tool.target_arguments": arguments,
            },
        )
        check_interruption()
        try:
            result = selected.contract.result_normalizer(selected.tool.invoke(arguments))
        except Exception as error:
            # 保留原异常类型与传输重试语义，同时指出真正失败的内部工具。
            error.target_tool_name = tool_name
            raise
        check_interruption()
        return {"tool_name": tool_name, "result": result}

    def selected_result(value):
        if isinstance(value, dict) and "result" in value:
            selected = registry.get(value.get("tool_name"))
            if selected is not None:
                return selected, value["result"]
        return None, value

    def compare(value):
        selected, raw = selected_result(value)
        return (
            {
                "tool_name": selected.tool.name,
                "result": selected.contract.comparison_projection(raw),
            }
            if selected
            else value
        )

    def project(value):
        selected, raw = selected_result(value)
        return (
            {
                "tool_name": selected.tool.name,
                "result": selected.contract.model_result_projector(raw),
            }
            if selected
            else value
        )

    def progress(value, *, state_revision, seen_keys):
        from .contracts import source_reference_progress

        selected, raw = selected_result(value)
        evaluator = selected.contract.progress_evaluator if selected else source_reference_progress
        return evaluator(raw, state_revision=state_revision, seen_keys=seen_keys)

    contract = ToolContract(
        name="execute_tool",
        contract_version=version,
        required_permissions=frozenset(
            p for item in tools for p in item.contract.required_permissions
        ),
        timeout_seconds=min((item.contract.timeout_seconds for item in tools), default=30),
        max_result_chars=min((item.contract.max_result_chars for item in tools), default=6000),
        execution=serial(
            *sorted({r for item in tools for r in item.contract.execution.exclusive_resources}),
            reason="单次分发，沿用业务工具资源约束；共享会话及结果缓存保守串行",
        ),
        model_result_projector=project,
        comparison_projection=compare,
        progress_evaluator=progress,
    )
    return RegisteredTool(
        tool=StructuredTool.from_function(
            name="execute_tool",
            func=execute_tool,
            args_schema=ExecuteToolArgs,
            # 通用 arguments 是开放对象；strict=true 会把它变成只能传 {} 的对象。
            # 业务参数仍在内层工具中严格校验，并将错误交回模型修正。
            metadata={"native_strict": False},
            description=(
                "执行技能说明中列出的一个只读业务工具。先读取技能了解名称及参数，"
                "再传 tool_name 和 arguments。返回实际工具名称和 result；"
                "参数错误可按回执修正。不执行代码、不启动 Agent、不提交或发送消息。"
            ),
        ),
        contract=contract,
    )
