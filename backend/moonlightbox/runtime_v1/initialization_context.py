"""起点材料预读与压缩：不访问准备状态，不执行 Agent 或提交正式状态。"""

import json

from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from moonlightbox.agent_runtime.capacity import estimate_tokens
from moonlightbox.agent_runtime.policy import (
    DIRECTOR_INITIALIZATION_POLICY,
    INITIALIZATION_HISTORY_SHARE,
    INITIALIZATION_OUTPUT_RESERVE,
)

from .agent_support import RuntimeToolbox


class InitializationToolbox(RuntimeToolbox):
    def __init__(self, base, work):
        super().__init__(DIRECTOR_INITIALIZATION_POLICY)
        self.base, self.work = base, work

    @staticmethod
    def resume_messages(saved, fresh):
        """恢复起点预读锚点，保留已执行工具和调查过程；不重置累计预算。"""
        from langchain_core.messages import SystemMessage

        result = list(saved)
        for kind in (SystemMessage, HumanMessage):
            replacement = next((m for m in fresh if isinstance(m, kind)), None)
            index = next((i for i, m in enumerate(result) if isinstance(m, kind)), None)
            if replacement is not None and index is not None:
                result[index] = replacement
        return result

    def compact(self, messages, *, source_refs, unresolved):
        # 不保留含大量预读原文的首轮 HumanMessage；原文已缓存，工具可继续向前读取。
        result, _ = super().compact(messages, source_refs=source_refs, unresolved=unresolved)
        result[1] = HumanMessage(
            content=json.dumps(
                {
                    "initialization_context": self.base,
                    "investigation_work": self.work(),
                },
                ensure_ascii=False,
                default=str,
            )
        )
        return result, "保留起点任务与调查笔记；预读原文和旧工具结果可分页恢复"


def preload_history(history, base, assembly, tools, budget, total_messages, cancellation_requested):
    from .service import RuntimeModelExecutionError

    # 初次预读使用同一启发式估算，扣除固定输入、工具和输出空间；最终由 Controller 校正准入。
    fixed = estimate_tokens(
        {
            "system": assembly.text,
            "context": base,
            "tools": [convert_to_openai_tool(t.tool) for t in tools],
        }
    )
    room = max(
        0,
        int(
            (
                budget.context_window_tokens
                - budget.context_safety_margin_tokens
                - INITIALIZATION_OUTPUT_RESERVE
                - fixed
            )
            * INITIALIZATION_HISTORY_SHARE
        ),
    )
    preload, cursor, used = [], None, 0
    while room > used:
        if cancellation_requested and cancellation_requested():
            raise RuntimeModelExecutionError("cancelled", "起点调查已取消")
        page = history.read(before_ref=cursor, limit=100)
        if not page["messages"]:
            break
        chosen = []
        for message in reversed(page["messages"]):
            size = estimate_tokens(message)
            if used + size > room:
                break
            chosen.insert(0, message)
            used += size
        preload = chosen + preload
        cursor = preload[0]["source_ref"] if preload else None
        if len(chosen) != len(page["messages"]) or not page["next_before_ref"]:
            break
    raw = {"messages": preload, "next_before_ref": cursor, "total_messages": total_messages}
    return raw
