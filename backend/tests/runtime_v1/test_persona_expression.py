"""表达工具与独立表达组件冒烟；正式 Runtime 由 Director 直接交付回复。"""

# ruff: noqa: F811
import json
from datetime import timedelta

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from moonlightbox.config import Settings
from moonlightbox.imports.models import Message
from moonlightbox.media.service import MediaStore
from moonlightbox.runtime_v1.actor import PersonaActor
from moonlightbox.runtime_v1.branch_models import Branch, BranchMessage
from moonlightbox.runtime_v1.conversation_index import index_branch
from moonlightbox.runtime_v1.executor import RuntimeExecutor
from moonlightbox.runtime_v1.expression_contracts import ExpressionResult
from moonlightbox.runtime_v1.persona_context import PersonaContextAssembler
from moonlightbox.runtime_v1.schemas import LifeDecision
from moonlightbox.runtime_v1.service import RuntimeService
from moonlightbox.runtime_v1.style_history import validate_asset
from moonlightbox.runtime_v1.tools.director_context import ReadConversation
from moonlightbox.runtime_v1.tools.style_examples import StyleService
from sqlalchemy import select
from test_peer_collaboration import session  # noqa: F401
from test_unified_conversation import history


class Encoder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def resources(session, tmp_path, monkeypatch):
    service = history(session)
    settings = Settings(data_dir=tmp_path, lightrag_enabled=False)
    asset = MediaStore(tmp_path).save(
        session, "p", kind="sticker", filename="a.gif", content=b"GIF89a"
    )
    original = session.get(Message, "old1")
    original.kind, original.content, original.media_asset_id = "sticker", "[表情]", asset.id
    session.add(
        Message(
            id="again",
            project_id="p",
            import_id="i",
            participant_id="person",
            source_id="again",
            timestamp=original.timestamp + timedelta(hours=1),
            kind="sticker",
            content="[表情]",
            media_asset_id=asset.id,
            raw={},
        )
    )
    service.snapshot.source_message_ids = ["old0", "old1", "again"]
    session.commit()
    monkeypatch.setattr("moonlightbox.runtime_v1.style_history.Settings", lambda: settings)
    monkeypatch.setattr("moonlightbox.runtime_v1.conversation_index.embedder", lambda: Encoder())
    index_branch(session, "b", Encoder())
    return service, asset, settings


def test_contextual_sticker_search_and_paging(session, tmp_path, monkeypatch):
    history, asset, settings = resources(session, tmp_path, monkeypatch)
    tool = StyleService(session, settings=settings).tool(branch_id="b", model_version_id="m")
    result = tool.invoke({"situation": "一起调侃", "intent": "自然接话"})
    assert result["retrieval_status"] == "partial"  # LightRAG 关闭，但文字使用索引仍能召回。
    assert result["coverage"]["usage_index"]["indexed_usages"] == 2
    assert result["sticker_candidates"][0]["asset_ref"] == asset.id
    assert result["sticker_candidates"][0]["used_by"] == "target"
    first = tool.invoke({"asset_ref": asset.id, "limit": 1})
    second = tool.invoke(
        {"asset_ref": asset.id, "usage_cursor": first["more_usage_cursor"], "limit": 1}
    )
    assert second["more_usage_cursor"] is None
    assert first["examples"][0]["around_ref"] != second["examples"][0]["around_ref"]
    from langchain_core.tools import ToolException

    with pytest.raises(ToolException, match="asset_ref 不在已提供候选中"):
        tool.invoke({"asset_ref": "not-a-real-asset"})
    MediaStore(tmp_path).path_for(asset).unlink()
    with pytest.raises(ValueError, match="unavailable"):
        validate_asset(session, session.get(Branch, "b"), history.snapshot, asset.id, settings)


def test_actor_multitool_isolated_and_mixed_commit(session, tmp_path, monkeypatch):
    history, asset, settings = resources(session, tmp_path, monkeypatch)
    style = StyleService(session, settings=settings).tool(branch_id="b", model_version_id="m")
    read = StructuredTool.from_function(
        history.read, name="read_conversation", description="回读原文", args_schema=ReadConversation
    )
    reply = {
        "status": "ready",
        "messages": [
            {"kind": "text", "text": "你可真行"},
            {"kind": "sticker", "asset_ref": asset.id},
            {"kind": "text", "text": "快去睡觉"},
        ],
    }

    class Model:
        calls = 0

        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            self.calls += 1
            steps = [
                ("get_style_examples", {"situation": "轻松调侃", "intent": "劝睡"}),
                ("read_conversation", {"around_ref": "old1"}),
                ("get_style_examples", {"asset_ref": asset.id, "limit": 1}),
            ]
            if self.calls <= len(steps):
                name, args = steps[self.calls - 1]
                return AIMessage(
                    content="", tool_calls=[{"id": str(self.calls), "name": name, "args": args}]
                )
            from submission_helpers import submission_message

            return submission_message(content=json.dumps(reply), name="submit_expression")

    context = {"expression_task": {"purpose": "劝睡"}, "recent_messages": []}
    model = Model()
    run_args = dict(
        context=context,
        tools=[style, read],
        owner_id="test",
        project_id="p",
        branch_id="b",
        input_revision=1,
        asset_validator=lambda ref: validate_asset(
            session, session.get(Branch, "b"), history.snapshot, ref, settings
        ),
    )
    from langgraph.checkpoint.memory import InMemorySaver
    from moonlightbox.agent_runtime.persistence import checkpoint_scope

    saver = InMemorySaver()
    with checkpoint_scope(saver, "expression-test"):
        run = PersonaActor(model).run_with_trace(**run_args)
    with checkpoint_scope(saver, "expression-test"):
        restored = PersonaActor(model).run_with_trace(**run_args)
    assert restored.asset_sources == run.asset_sources
    assert model.calls == 4 and run.result.status == "ready"
    assert context == {"expression_task": {"purpose": "劝睡"}, "recent_messages": []}
    assert asset.id in run.asset_sources
    bootstrap = RuntimeService(session).bootstrap("p", "b")
    decision = LifeDecision(action="speak", expression_task={"purpose": "劝睡"})
    args = dict(
        project_id="p",
        branch_id="b",
        decision=decision,
        expected_version=bootstrap["state"].version,
        virtual_now=history.now,
        trigger_event_ids=[],
        expression_result=run.result,
        asset_sources=run.asset_sources,
        idempotency_key="mixed",
    )
    executor = RuntimeExecutor(session)
    result = executor.commit(**args)
    session.commit()
    executor.commit(**args)
    rows = list(
        session.scalars(
            select(BranchMessage)
            .where(BranchMessage.turn_id == result["event"].id)
            .order_by(BranchMessage.bubble_index)
        )
    )
    assert [row.type for row in rows] == ["text", "sticker", "text"]
    assert rows[1].media_asset_id == asset.id
    assert rows[1].generation_metadata["sticker_usage_refs"]
    with pytest.raises(ValueError, match="not_supplied"):
        executor.commit(
            **{
                **args,
                "asset_sources": {},
                "idempotency_key": "forged",
                "expected_version": result["state"].version,
            }
        )


def test_expression_contract_and_context_isolation():
    with pytest.raises(ValueError):
        ExpressionResult(
            status="ready", messages=[{"kind": "sticker", "asset_ref": "x", "text": "fake"}]
        )
    view = {
        "origin": {"person_world_profile": {"identity": {}, "life_course": {"large": True}}},
        "current": {"day_plans": {"large": True}, "life_state": {}},
        "branch": {},
        "virtual_now": "now",
        "collaboration": ["private discussion"],
        "memory": {"tools": "large"},
    }
    context = PersonaContextAssembler().assemble(view, task={"purpose": "接话"})
    assert not {"collaboration", "day_plans", "memory"}.intersection(context)
    assert "life_course" not in context["person_world_profile"]
    schema = LifeDecision.model_json_schema()["properties"]
    assert "reply" in schema and "expression_task" not in schema
    assert "content_points" not in schema


def test_runtime_director_submits_reply_without_actor_roundtrip(session):
    from test_peer_collaboration import Model, speak

    director = Model([speak()])
    actor = Model(
        [
            {
                "status": "needs_clarification",
                "clarification": {"question": "这次需要主动追问吗？", "related_refs": []},
            },
            {"status": "ready", "messages": [{"kind": "text", "text": "你好呀"}]},
        ]
    )
    service = RuntimeService(session, director_model=director, actor_model=actor)
    service.submit_user_message(
        project_id="p", branch_id="b", content="你好", idempotency_key="ask"
    )
    service.process_next(project_id="p", branch_id="b")
    # 保留独立组件不意味着主流程应再调用它；技能内化后不再有澄清往返。
    assert len(director.inputs) == 1 and len(actor.inputs) == 0
    payload = "\n".join(str(m.content) for m in director.inputs[-1])
    assert "这次需要主动追问吗" not in payload
    assert "# PersonaActor" not in payload
    rows = list(session.scalars(select(BranchMessage).where(BranchMessage.role == "assistant")))
    assert len(rows) == 1 and rows[0].content == "你好"
