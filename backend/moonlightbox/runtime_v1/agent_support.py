"""Runtime 三个 Agent 共用的结构输出、工具分页与上下文维护。

完整工具结果保存在内部图状态及 Phoenix；启用检查点时可恢复分页正文，不静默丢正文。
这里不创建第二套 Agent 循环，调度仍由 AgentLoopController 完成。
"""

from __future__ import annotations

import hashlib
import json

from langchain_core.messages import HumanMessage

from moonlightbox.agent_runtime import ToolContract
from moonlightbox.agent_runtime.context import retain_recent_turns
from moonlightbox.agent_runtime.contracts import RegisteredTool

from .tools.execution_policy import execution_policy
from .tools.read_runtime_result import build_read_runtime_result_tool, read_runtime_result


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


class RuntimeToolbox:
    def __init__(self, policy):
        self.policy = policy
        self.results = {}
        self.restore_callbacks = []
        self.tool_names = set()

    def restore(self, results):
        """恢复分页附件，不能只恢复 ToolMessage 却丢失 result_ref 指向的正文。"""
        for result in results:
            for callback in self.restore_callbacks:
                callback(result)
            self.project(result)
            if (
                isinstance(result, dict)
                and result.get("tool_name") in self.tool_names
                and "result" in result
            ):
                # 分发回执保留完整内层结果；重建其原始分页引用，不只缓存外层封套。
                self.project(result["result"])

    def project(self, result):
        serialized = json.dumps(result, ensure_ascii=False, default=str)
        ref = hashlib.sha256(serialized.encode()).hexdigest()
        self.results[ref] = serialized
        if len(serialized) <= 20_000:
            return result
        return self.read(result_ref=ref)

    def read(self, result_ref, offset=0):
        return read_runtime_result(self.results, result_ref, offset)

    def register(self, tools):
        reader = build_read_runtime_result_tool(self.results)
        return tuple(self.register_one(tool) for tool in [*tools, reader])

    def register_one(self, tool):
        """内外工具复用相同契约，内部工具不自动暴露给模型。"""
        self.tool_names.add(tool.name)
        callback = (tool.metadata or {}).get("restore_tool_result")
        if callback is not None:
            self.restore_callbacks.append(callback)
            # 恢复钩子是宿主代码，不作为 LangChain 调用元数据发送给观测回调。
            tool = tool.model_copy(
                update={
                    "metadata": {
                        key: value
                        for key, value in tool.metadata.items()
                        if key != "restore_tool_result"
                    }
                }
            )
        return RegisteredTool(
            tool=tool,
            contract=ToolContract(
                name=tool.name,
                contract_version=(tool.metadata or {}).get("contract_version", "1"),
                required_permissions=frozenset(
                    (tool.metadata or {}).get("required_permissions", [])
                ),
                execution=execution_policy(tool.name),
                side_effect=(tool.metadata or {}).get(
                    "side_effect", "proposal" if tool.name == "update_plan_work" else "read_only"
                ),
                timeout_seconds=min(self.policy.tool_timeout_seconds, self.policy.deadline_seconds),
                max_result_chars=self.policy.max_single_tool_result_chars,
                model_result_projector=self.project,
            ),
        )

    def compact(self, messages, *, source_refs, unresolved):
        # 锚点保存所有已确认状态、用户输入与完整画像；保留整批工具协议，不能留下孤立 tool 消息。
        note = HumanMessage(
            content=json.dumps(
                {
                    "instruction": (
                        "较早调查正文已移出窗口，可用 read_runtime_result 恢复；不要重做相同检索。"
                    ),
                    "results": [
                        {"result_ref": ref, "total_chars": len(text)}
                        for ref, text in self.results.items()
                    ],
                    "source_refs": source_refs,
                    "unresolved": unresolved,
                },
                ensure_ascii=False,
            )
        )
        return retain_recent_turns(
            messages, notes=(note,), tool_turns_only=True
        ), "保留固定上下文与最近完整工具批次；旧正文可分页恢复"
