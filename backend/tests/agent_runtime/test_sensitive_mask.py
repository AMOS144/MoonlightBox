"""遮罩注册表与最小遮罩降级的离线契约；不向云端发消息。"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from moonlightbox.agent_runtime.sensitive_mask import (
    CHUNK_PLACEHOLDER,
    MASK_PLACEHOLDER,
    MaskRegistry,
    SensitiveContentHandler,
    StillRejected,
    collect_candidates,
    mask_messages,
)
from moonlightbox.db import Database
from moonlightbox.projects.models import Project
from sqlalchemy.orm import Session


def _tool_result(*entries):
    return json.dumps({"messages": list(entries)}, ensure_ascii=False)


def _entry(ref, content):
    return {"message_ref": ref, "sent_at": "2026-05-01T10:00:00", "speaker": "她", "content": content}


def test_mask_json_entries_and_keep_metadata():
    messages = [
        ToolMessage(
            content=_tool_result(_entry("m1", "正常内容"), _entry("m2", "敏感内容")),
            tool_call_id="c1",
        )
    ]
    masked = mask_messages(messages, {"m2"})
    obj = json.loads(masked[0].content)
    assert obj["messages"][0]["content"] == "正常内容"
    assert obj["messages"][1]["content"] == MASK_PLACEHOLDER
    assert obj["messages"][1]["message_ref"] == "m2"
    assert obj["messages"][1]["sent_at"] == "2026-05-01T10:00:00"
    # 原消息不被改动;重复遮罩幂等
    assert json.loads(messages[0].content)["messages"][1]["content"] == "敏感内容"
    assert mask_messages(masked, {"m2"})[0].content == masked[0].content


def test_mask_fragment_chunk_and_context_when_ref_hit():
    result = {
        "mode": "naive",
        "context": "原始拼接文本",
        "fragments": [
            {"chunk": "敏感片段", "message_refs": ["m9"], "messages": [_entry("m9", "敏感内容")]},
            {"chunk": "正常片段", "message_refs": ["m1"], "messages": [_entry("m1", "正常内容")]},
        ],
    }
    masked = mask_messages([ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id="s")], {"m9"})
    obj = json.loads(masked[0].content)
    assert obj["fragments"][0]["chunk"] == CHUNK_PLACEHOLDER
    assert obj["fragments"][0]["messages"][0]["content"] == MASK_PLACEHOLDER
    assert obj["fragments"][1]["chunk"] == "正常片段"
    assert obj["context"] == CHUNK_PLACEHOLDER


def test_non_json_content_only_masked_as_raw_unit():
    messages = [HumanMessage(content="plain text")]
    assert mask_messages(messages, {"m1"})[0].content == "plain text"
    assert mask_messages(messages, set(), {0})[0].content == MASK_PLACEHOLDER
    assert collect_candidates(messages) == [("raw", 0)]


def test_collect_candidates_skips_masked_and_duplicates():
    messages = [
        ToolMessage(content=_tool_result(_entry("m1", "a"), _entry("m1", "a"), _entry("m2", MASK_PLACEHOLDER)), tool_call_id="c"),
    ]
    assert collect_candidates(messages) == [("ref", "m1")]
    assert collect_candidates([AIMessage(content="")]) == []


def test_registry_add_and_load(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'mask.db'}")
    database.create_schema()
    with Session(database.engine) as session:
        session.add(Project(id="p", name="test"))
        session.commit()
    registry = MaskRegistry(database.engine, "p")
    assert registry.load() == frozenset()
    assert registry.add(["m1", "m2"], source="node_investigation") == ["m1", "m2"]
    assert registry.add(["m2", "m3"], source="node_investigation") == ["m3"]
    assert registry.load() == frozenset({"m1", "m2", "m3"})


class _FakeCall:
    """含 marker 的上下文一律拒绝,否则接受;记录每次实际发送的内容。"""

    def __init__(self, marker):
        self.marker = marker
        self.sent = []

    def __call__(self, messages):
        text = json.dumps([str(m.content) for m in messages], ensure_ascii=False)
        self.sent.append(text)
        if self.marker in text:
            raise StillRejected()
        return "accepted"


def _handler(tmp_path, call):
    database = Database(f"sqlite:///{tmp_path / 'mask.db'}")
    database.create_schema()
    with Session(database.engine) as session:
        session.add(Project(id="p", name="test"))
        session.commit()
    registry = MaskRegistry(database.engine, "p")
    return SensitiveContentHandler(registry, source="test"), registry


def test_recover_masks_only_the_offending_ref(tmp_path):
    handler, registry = _handler(tmp_path, None)
    call = _FakeCall("敏感内容")
    messages = [
        ToolMessage(
            content=_tool_result(_entry("m1", "正常一"), _entry("m2", "敏感内容"), _entry("m3", "正常三")),
            tool_call_id="c1",
        )
    ]
    result = handler.recover(messages, call)
    assert result == "accepted"
    assert registry.load() == frozenset({"m2"})
    # 最后一次发送:m2 被遮,m1/m3 原文保留
    assert MASK_PLACEHOLDER in call.sent[-1]
    assert "正常一" in call.sent[-1] and "正常三" in call.sent[-1]
    assert "敏感内容" not in call.sent[-1]


def test_recover_uses_registry_without_new_trials(tmp_path):
    handler, registry = _handler(tmp_path, None)
    registry.add(["m2"], source="earlier")
    handler.known = {"m2"}
    call = _FakeCall("敏感内容")
    messages = [ToolMessage(content=_tool_result(_entry("m2", "敏感内容")), tool_call_id="c")]
    view = handler.masked_view(messages)
    assert "敏感内容" not in str(view[0].content)
    assert call(view) == "accepted"


def test_recover_gives_up_when_nothing_maskable(tmp_path):
    handler, _ = _handler(tmp_path, None)
    call = _FakeCall("敏感内容")
    # 全部是已遮罩或无原文的消息,没有候选单元
    messages = [ToolMessage(content=_tool_result(_entry("m2", MASK_PLACEHOLDER)), tool_call_id="c")]
    assert handler.recover(messages, call) is None


def test_recover_bisects_multiple_offenders(tmp_path):
    handler, registry = _handler(tmp_path, None)
    handler._max_trials = 12

    class TwoMarkers:
        def __call__(self, messages):
            text = json.dumps([str(m.content) for m in messages], ensure_ascii=False)
            if "敏感甲" in text or "敏感乙" in text:
                raise StillRejected()
            return "accepted"

    messages = [
        ToolMessage(
            content=_tool_result(
                _entry("m1", "敏感甲"), _entry("m2", "正常"), _entry("m3", "敏感乙"), _entry("m4", "正常")
            ),
            tool_call_id="c",
        )
    ]
    assert handler.recover(messages, TwoMarkers()) == "accepted"
    assert registry.load() == frozenset({"m1", "m3"})


def test_recover_finds_offender_in_already_accepted_history(tmp_path):
    """供应商按整次请求判定:历史里被接受过的消息也可能在新一轮触发。"""

    handler, registry = _handler(tmp_path, None)
    call = _FakeCall("敏感内容")
    history = [
        ToolMessage(content=_tool_result(_entry("m1", "敏感内容")), tool_call_id="c1"),
        AIMessage(content='{"nodes": []}'),
    ]
    # 模拟这两条消息此前已被接受;新增的消息本身没问题。
    handler.note_accepted(history)
    messages = [*history, ToolMessage(content=_tool_result(_entry("m2", "正常")), tool_call_id="c2")]
    assert handler.recover(messages, call) == "accepted"
    assert registry.load() == frozenset({"m1"})


def test_session_raw_mask_persists_across_calls(tmp_path):
    """自由文本 offender 遮住后,后续追加了消息的调用仍按绝对下标避开它。"""

    handler, registry = _handler(tmp_path, None)
    call = _FakeCall("敏感原文")
    messages = [
        HumanMessage(content="正常问题"),
        AIMessage(content="敏感原文"),
    ]
    assert handler.recover(messages, call) == "accepted"
    assert registry.load() == frozenset()
    assert handler._session_raw == {1}
    handler.note_accepted(messages)
    followup = [*messages, HumanMessage(content="继续")]
    view = handler.masked_view(followup)
    assert view[1].content == MASK_PLACEHOLDER
    assert view[2].content == "继续"
    # 历史被压缩改写后下标失效,session raw 遮罩作废
    handler.note_accepted([HumanMessage(content="压缩后的摘要")])
    assert handler._session_raw == set()
