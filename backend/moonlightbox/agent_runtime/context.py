"""共享上下文裁剪结构；领域只提供摘要和需要保留的近期回合数。"""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def retain_recent_turns(
    messages: Sequence[BaseMessage],
    *,
    keep: int = 2,
    notes: Sequence[BaseMessage] = (),
    tool_turns_only: bool = False,
) -> list[BaseMessage]:
    """保留首个系统/任务消息和完整 AI 回合，不从一批工具回执中间截断。

    摘要、分页引用与工作草稿由调用方提供；本函数不理解证据或决定业务完成。
    使用消息位置去重，而不是按正文去重，避免误删相同措辞的不同回合。
    """
    if keep < 1:
        raise ValueError("至少保留一个近期回合")
    anchors = []
    for kind in (SystemMessage, HumanMessage):
        index = next((i for i, item in enumerate(messages) if isinstance(item, kind)), None)
        if index is not None:
            anchors.append(index)
    starts = [
        i
        for i, item in enumerate(messages)
        if isinstance(item, AIMessage) and (not tool_turns_only or item.tool_calls)
    ]
    start = starts[-keep] if len(starts) >= keep else starts[0] if starts else len(messages)
    return [
        *(messages[i] for i in anchors),
        *notes,
        *(item for i, item in enumerate(messages) if i >= start and i not in anchors),
    ]
