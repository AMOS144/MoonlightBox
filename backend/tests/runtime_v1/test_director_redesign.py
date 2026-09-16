"""Director 新闭环的离线冒烟：真实事务与向量排序，不请求供应商。"""

# pytest 跨文件复用数据夹具。
# ruff: noqa: F811
from datetime import UTC, datetime, timedelta

import pytest
from moonlightbox.runtime_v1.branch_models import BranchMessage
from moonlightbox.runtime_v1.context import ContextAssembler
from moonlightbox.runtime_v1.conversation_index import index_branch, search_branch
from moonlightbox.runtime_v1.conversation_maintenance import summarize
from moonlightbox.runtime_v1.db_models import RuntimeMemoryRow
from moonlightbox.runtime_v1.executor import RuntimeExecutor
from moonlightbox.runtime_v1.schemas import ActorMessage, LifeDecision
from moonlightbox.runtime_v1.service import RuntimeService
from sqlalchemy import select
from test_peer_collaboration import Model, session  # noqa: F401


def add_message(session, sequence, text, now, status="completed"):
    row = BranchMessage(
        branch_id="b",
        sequence=sequence,
        role="user",
        content=text,
        observed_at=now,
        generation_metadata={"input_status": status},
    )
    session.add(row)
    session.flush()
    return row


def test_new_schema_hides_legacy_psychology_and_rejects_duplicate_resolutions():
    schema = LifeDecision.model_json_schema()
    assert not {"mood", "attention", "current_goal", "open_conversation_threads"} & set(
        schema["$defs"]["StatePatch"]["properties"]
    )
    with pytest.raises(ValueError, match="只能提交一次"):
        LifeDecision(
            action="wait",
            input_resolutions=[
                {"message_ref": "m", "status": "completed"},
                {"message_ref": "m", "status": "no_response_needed"},
            ],
        )


def test_wakeup_and_message_references_are_scoped_without_one_to_one_rule(session):
    from moonlightbox.runtime_v1.db_models import RuntimeWakeupRow

    now = datetime.now(UTC)
    first = add_message(session, 0, "第一句", now, "pending")
    second = add_message(session, 1, "第二句", now, "pending")
    session.add(
        RuntimeWakeupRow(
            id="wake",
            branch_id="b",
            wake_at=now,
            reason="起床",
            trigger_type="plan_transition",
            idempotency_key="wake",
        )
    )
    session.flush()
    decision = LifeDecision(
        action="speak",
        state_patch={"source_event_ids": ["wake"]},
        expression_task={"purpose": "一起回应", "respond_to_refs": [second.id]},
        input_resolutions=[
            {"message_ref": first.id, "status": "completed"},
            {"message_ref": second.id, "status": "completed"},
        ],
    )
    executor = RuntimeExecutor(session)
    assert executor.validate_decision_references("b", decision, now) is None
    decision.expression_task.respond_to_refs = ["missing"]
    assert "无效消息引用" in executor.validate_decision_references("b", decision, now)
    decision.expression_task.respond_to_refs = [second.id]
    wake = session.get(RuntimeWakeupRow, "wake")
    wake.wake_at = now + timedelta(hours=1)
    session.flush()
    assert executor.validate_decision_references("b", decision, now) is not None


def test_subjective_memory_expression_commit_once(session):
    runtime = RuntimeService(session)
    bootstrap = runtime.bootstrap("p", "b")
    now = datetime.now(UTC)
    message = add_message(session, 0, "明天面试", now, "pending")
    decision = LifeDecision.model_validate(
        {
            "action": "speak",
            "expression_task": {"purpose": "接住对方的分享", "content_points": ["问准备情况"]},
            "subjective_state_updates": {
                "mood": "平静",
                "concerns": [
                    {
                        "focus": "用户明天面试",
                        "subjects": ["user"],
                        "appraisal": {"interpretation": "尚不清楚是否紧张"},
                        "source_refs": [message.id],
                    }
                ],
            },
            "memory_proposals": [
                {
                    "subject": "user",
                    "summary": "用户说次日面试",
                    "basis": "user_report",
                    "source_refs": [message.id],
                }
            ],
            "input_resolutions": [{"message_ref": message.id, "status": "completed"}],
        }
    )
    executor = RuntimeExecutor(session)
    args = dict(
        project_id="p",
        branch_id="b",
        decision=decision,
        expected_version=bootstrap["state"].version,
        virtual_now=now,
        trigger_event_ids=[],
        actor_message=ActorMessage(text="准备得怎么样呀"),
        idempotency_key="director-test",
    )
    result = executor.commit(**args)
    session.commit()
    again = executor.commit(**args)
    assert result["event"].id == again["event"].id
    assert again["state"].state["subjective_state"]["concerns"][0]["subjects"] == ["user"]
    assert message.generation_metadata["input_status"] == "completed"
    memories = list(
        session.scalars(select(RuntimeMemoryRow).where(RuntimeMemoryRow.scope == "branch"))
    )
    assert len(memories) == 1 and memories[0].status == "asserted"


def test_failed_actor_does_not_write_subjective_state(session):
    bootstrap = RuntimeService(session).bootstrap("p", "b")
    decision = LifeDecision(
        action="speak",
        communication_intent="回复",
        content_points=["问好"],
        subjective_state_updates={"mood": "放松"},
    )
    with pytest.raises(ValueError, match="可提交回复"):
        RuntimeExecutor(session).commit(
            project_id="p",
            branch_id="b",
            decision=decision,
            expected_version=bootstrap["state"].version,
            virtual_now=datetime.now(UTC),
            trigger_event_ids=[],
        )
    assert "subjective_state" not in bootstrap["state"].state


def test_overnight_and_uncovered_messages_not_lost(session):
    runtime = RuntimeService(session)
    bootstrap = runtime.bootstrap("p", "b")
    now = datetime.now(UTC)
    for i in range(110):
        add_message(session, i, f"昨天的事情{i}", now - timedelta(days=1))
    packet = ContextAssembler(session).assemble(
        branch_id="b", trigger={}, clock=bootstrap["clock"], snapshot=bootstrap["snapshot"], now=now
    )
    assert len(packet.branch["working_window"]["messages"]) == 110
    assert packet.branch["working_window"]["messages"][0]["content"] == "昨天的事情0"


class Encoder:
    def embed(self, texts):
        return [[1.0, 0.0] if "面试" in text else [0.0, 1.0] for text in texts]


def test_semantic_index_and_summary_recovery(session):
    now = datetime.now(UTC)
    first = add_message(session, 0, "明天面试", now)
    for i in range(1, 105):
        add_message(session, i, "今天吃饭", now)
    session.commit()
    assert search_branch(session, "b", "面试", now, encoder=Encoder())["status"] == "index_pending"
    index_branch(session, "b", Encoder())
    result = search_branch(session, "b", "面试", now, encoder=Encoder())
    assert result["data"][0]["source_ref"] == first.id
    assert search_branch(session, "another-branch", "面试", now, encoder=Encoder())["data"] == []
    model = Model([{"summary": "用户提到次日面试，之后聊吃饭", "unresolved_items": []}])
    assert summarize(session, "b", "p", model)
    assert len(model.inputs) == 1
