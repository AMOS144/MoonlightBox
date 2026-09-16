"""各领域共用同一完整回合裁剪机制。"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from moonlightbox.agent_runtime.context import retain_recent_turns


@pytest.mark.parametrize("keep,tool_only", [(2, False), (2, True), (3, True)])
def test_keeps_complete_parallel_tool_batches_and_task(keep, tool_only):
    system, task = SystemMessage(content="规则"), HumanMessage(content="任务")
    messages = [system, task]
    for turn in range(4):
        calls = [{"id": f"{turn}-{n}", "name": "read", "args": {}} for n in range(2)]
        messages.append(AIMessage(content="", tool_calls=calls))
        messages.extend(ToolMessage(content="正文", tool_call_id=call["id"]) for call in calls)
    note = HumanMessage(content="领域草稿与分页引用")
    kept = retain_recent_turns(messages, keep=keep, notes=(note,), tool_turns_only=tool_only)
    assert kept[:3] == [system, task, note]
    assert kept[3:] == messages[-keep * 3 :]
    ids = {call["id"] for item in kept if isinstance(item, AIMessage) for call in item.tool_calls}
    assert {item.tool_call_id for item in kept if isinstance(item, ToolMessage)} == ids


def test_no_model_turn_does_not_duplicate_task_or_keep_old_notes():
    system, task = SystemMessage(content="规则"), HumanMessage(content="任务")
    assert retain_recent_turns([system, task, HumanMessage(content="过期摘要")]) == [system, task]
    assert retain_recent_turns([]) == []
    with pytest.raises(ValueError):
        retain_recent_turns([system, task], keep=0)


def test_text_turn_is_kept_for_tasks_and_revision():
    messages = [
        SystemMessage(content="规则"),
        HumanMessage(content="任务"),
        AIMessage(content="修正"),
    ]
    assert retain_recent_turns(messages) == messages
