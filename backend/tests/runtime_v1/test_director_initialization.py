"""真实统一循环与 SQLite 的起点初始化冒烟，不调用云端。"""

# ruff: noqa: F811

import json
from datetime import timedelta

import pytest
from langchain_core.messages import AIMessage
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.runtime_v1.branch_models import Branch, BranchMessage
from moonlightbox.runtime_v1.context import ContextAssembler
from moonlightbox.runtime_v1.context_views import director_context_payload
from moonlightbox.runtime_v1.db_models import (
    RuntimeInitializationRow,
    RuntimeLifeStateRow,
    RuntimeSnapshotRow,
)
from moonlightbox.runtime_v1.director import DirectorAgent
from moonlightbox.runtime_v1.initialization_context import InitializationToolbox
from moonlightbox.runtime_v1.service import RuntimeModelExecutionError, RuntimeService
from sqlalchemy import select
from test_peer_collaboration import session  # noqa: F401


class Model:
    def __init__(self, fail=False):
        self.inputs, self.fail = [], fail

    def bind_tools(self, tools, **kwargs):
        self.tools = {t.name for t in tools}
        assert "submit_decision" not in self.tools
        assert "submit_initial_state" in self.tools
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        if len(self.inputs) == 1:
            assert len(json.loads(messages[1].content)["recent_conversation"]["messages"]) == 120
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "notes",
                        "name": "update_initialization_work",
                        "args": {
                            "tentative_understanding": "只属于初始化调查的临时笔记",
                            "open_questions": [],
                            "read_ranges": ["已读末端"],
                        },
                    }
                ],
            )
        if self.fail:
            raise RuntimeError("temporary_failure")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "done",
                    "name": "submit_initial_state",
                    "args": {
                        "result": {
                            "subjective_state": {"mood": "有些期待", "concerns": []},
                            "open_conversation_threads": [
                                {
                                    "topic": "面试",
                                    "summary": "在等后续",
                                    "resume_when": "有结果时",
                                    "source_refs": ["m119"],
                                }
                            ],
                            "active_commitments": [],
                        }
                    },
                }
            ],
        )


def prepared(session):
    snapshot = session.get(RuntimeSnapshotRow, "snapshot")
    branch = session.get(Branch, "b")
    branch.lifecycle_status = "preparing"
    session.add(
        ImportSource(
            id="i",
            project_id="p",
            preview_id="x",
            source_path="test",
            message_count=120,
            confirmed_at=snapshot.cutoff_at,
        )
    )
    session.add(Participant(id="target", project_id="p", name="目标", role="target"))
    session.flush()
    for index in range(120):
        session.add(
            Message(
                id=f"m{index}",
                project_id="p",
                import_id="i",
                participant_id="target",
                source_id=str(index),
                timestamp=snapshot.cutoff_at - timedelta(seconds=120 - index),
                kind="text",
                content=f"历史原文{index}",
                raw={},
            )
        )
    snapshot.source_message_ids = [f"m{i}" for i in range(120)]
    session.commit()
    service = RuntimeService(session)
    bootstrap = service.bootstrap("p", "b")
    bootstrap["plan"].generation_metadata = {"status": "agent"}
    bootstrap["plan"].blocks = []
    session.commit()
    return bootstrap


def test_preparation_submits_only_state_and_never_sends_message(session):  # noqa: F811
    bootstrap = prepared(session)
    model = Model()
    service = RuntimeService(session, director_model=model)
    service.process_next(project_id="p", branch_id="b", prepare_branch=True)
    assert session.get(Branch, "b").lifecycle_status == "active"
    assert session.get(RuntimeInitializationRow, "b").status == "ready"
    assert not list(session.scalars(select(BranchMessage)))
    state = session.scalar(
        select(RuntimeLifeStateRow).where(RuntimeLifeStateRow.is_current.is_(True))
    )
    assert state.state["subjective_state"]["mood"] == "有些期待"
    packet = ContextAssembler(session).assemble(
        branch_id="b", trigger={}, clock=bootstrap["clock"], snapshot=bootstrap["snapshot"]
    )
    visible = json.dumps(director_context_payload(packet), ensure_ascii=False)
    assert "有些期待" in visible and "面试" in visible
    assert "只属于初始化调查" not in visible and "submit_initial_state" not in visible
    service.director.initialize_branch(session, project_id="p", branch_id="b", bootstrap=bootstrap)
    assert len(model.inputs) == 2


def test_initialization_reports_each_bad_reference_and_model_repairs(session):
    prepared(session)

    class RepairModel(Model):
        def invoke(self, messages):
            output = super().invoke(messages)
            if len(self.inputs) == 2:
                output.tool_calls[0]["args"]["result"]["open_conversation_threads"][0][
                    "source_refs"
                ] = ["result-hash", "profile"]
            elif len(self.inputs) == 3:
                feedback = str(messages[-1].content)
                assert "open_conversation_threads[0].source_refs[0]" in feedback
                assert "open_conversation_threads[0].source_refs[1]" in feedback
                assert "result-hash" in feedback and "profile" in feedback
                assert "可填 []" in feedback
            return output

    model = RepairModel()
    RuntimeService(session, director_model=model).process_next(
        project_id="p", branch_id="b", prepare_branch=True
    )
    assert len(model.inputs) == 3
    assert session.get(RuntimeInitializationRow, "b").status == "ready"


def test_failed_initialization_resumes_notes_and_work_not_normal_context(session):  # noqa: F811
    bootstrap = prepared(session)
    model = Model(fail=True)
    agent = DirectorAgent(model)
    with pytest.raises(RuntimeModelExecutionError):
        agent.initialize_branch(session, project_id="p", branch_id="b", bootstrap=bootstrap)
    assert session.get(RuntimeInitializationRow, "b").status == "failed"
    assert "临时笔记" in session.get(RuntimeInitializationRow, "b").work["tentative_understanding"]
    model.fail = False
    agent.initialize_branch(session, project_id="p", branch_id="b", bootstrap=bootstrap)
    assert session.get(RuntimeInitializationRow, "b").status == "ready"
    assert len(model.inputs) == 3  # 恢复后直接接着提交，不重跑第一次笔记工具。


def test_dayplan_then_initialization_then_chat_ready(session):
    from test_peer_collaboration import Model as PlannerModel
    from test_peer_collaboration import proposal

    bootstrap = prepared(session)
    bootstrap["plan"].generation_metadata = {"status": "pending"}
    session.commit()
    model = Model()
    planner = PlannerModel([proposal(bootstrap["plan"].plan_date)])
    service = RuntimeService(session, director_model=model, planner_model=planner)
    result = service.process_next(project_id="p", branch_id="b", prepare_branch=True)
    assert result["trace"].status == "succeeded"
    assert planner.inputs and len(model.inputs) == 2
    assert session.get(RuntimeInitializationRow, "b").status == "ready"
    assert session.get(Branch, "b").lifecycle_status == "active"
    assert not list(session.scalars(select(BranchMessage)))


def test_compaction_drops_preload_but_keeps_recoverable_reference():
    from langchain_core.messages import HumanMessage, SystemMessage

    box = InitializationToolbox({"task": "起点理解"}, lambda: {"tentative_understanding": "笔记"})
    box.project({"messages": ["大量预读原文"]})
    compacted, _ = box.compact(
        [SystemMessage(content="规则"), HumanMessage(content="大量预读原文")],
        source_refs=[],
        unresolved=[],
    )
    assert "大量预读原文" not in str(compacted)
    assert "笔记" in str(compacted) and "result_ref" in str(compacted)


def test_cancelled_or_stale_initialization_cannot_commit(session):
    bootstrap = prepared(session)
    model = Model()
    agent = DirectorAgent(model)
    with pytest.raises(RuntimeModelExecutionError, match="取消"):
        agent.initialize_branch(
            session,
            project_id="p",
            branch_id="b",
            bootstrap=bootstrap,
            cancellation_requested=lambda: True,
        )
    assert not model.inputs
    assert session.get(RuntimeInitializationRow, "b").status == "failed"

    class Changed(Model):
        def invoke(self, messages):
            response = super().invoke(messages)
            bootstrap["snapshot"].profile = {"overview": "另一个起点版本"}
            session.commit()
            return response

    with pytest.raises(RuntimeModelExecutionError):
        DirectorAgent(Changed()).initialize_branch(
            session, project_id="p", branch_id="b", bootstrap=bootstrap
        )
    current = session.scalar(
        select(RuntimeLifeStateRow).where(RuntimeLifeStateRow.is_current.is_(True))
    )
    assert not current.state.get("subjective_state")
    assert session.get(RuntimeInitializationRow, "b").status == "failed"


def test_initialization_failure_is_recorded_once(session, monkeypatch):
    """材料装配和模型失败共用一个错误边界；已提交 ready 不能被后续异常降级。"""
    from moonlightbox.runtime_v1 import initialization

    bootstrap = prepared(session)
    row = RuntimeInitializationRow(branch_id="b", input_hash="test", status="running", work={})
    session.add(row)
    session.commit()
    writes = []
    commit = session.commit

    def record_commit():
        writes.append(row.status)
        return commit()

    def fail(*args, **kwargs):
        raise RuntimeError("装配失败")

    monkeypatch.setattr(session, "commit", record_commit)
    monkeypatch.setattr(initialization, "_initialize_director", fail)
    with pytest.raises(RuntimeError, match="装配失败"):
        initialization.initialize_director(
            DirectorAgent(Model()), session, project_id="p", branch_id="b", bootstrap=bootstrap
        )
    assert writes == ["failed"]
    row.status = "ready"
    commit()
    writes.clear()
    with pytest.raises(RuntimeError, match="装配失败"):
        initialization.initialize_director(
            DirectorAgent(Model()), session, project_id="p", branch_id="b", bootstrap=bootstrap
        )
    assert row.status == "ready" and writes == []


def test_resume_restores_preload_without_losing_completed_tool_turns():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from moonlightbox.runtime_v1.initialization_context import InitializationToolbox

    tail = [AIMessage(content="已调查"), ToolMessage(content="已有原文", tool_call_id="read")]
    saved = [SystemMessage(content="规则"), HumanMessage(content="压缩过的起点"), *tail]
    fresh = [SystemMessage(content="规则"), HumanMessage(content="完整起点预读")]
    restored = InitializationToolbox.resume_messages(saved, fresh)
    assert restored[:2] == fresh
    assert restored[2:] == tail
