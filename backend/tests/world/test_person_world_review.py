import json
from datetime import timedelta
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from moonlightbox.config import Settings
from moonlightbox.db import Base
from moonlightbox.jobs.service import JobService
from moonlightbox.world.client import LightRAGMetadata, LightRAGRetrieval
from moonlightbox.world.models import (
    AtomicWorldClaim,
    PersonWorldRevisionContextSnapshot,
    PersonWorldRevisionMessage,
    PersonWorldRevisionSession,
    WorldChangeApproval,
    WorldCorrection,
    WorldGraphVersion,
    WorldPublication,
)
from moonlightbox.world.person_world.prompt_loader import load_prompt_definition
from moonlightbox.world.person_world.review import (
    PersonWorldReviewService,
    RevisionBaseStaleError,
    RevisionStateError,
    create_revision_turn_handler,
    enqueue_revision_turn_job,
)
from moonlightbox.world.person_world.review.context import RevisionContextAssembler
from moonlightbox.world.person_world.review.revision_graph import (
    RevisionAgentGraph,
)
from moonlightbox.world.person_world.review.service import (
    _graph_state_description,
)
from moonlightbox.world.person_world.schemas import (
    GraphPatchDraft,
    RevisionAgentTurn,
    RevisionQuestionTurn,
    RevisionScopeProposal,
    RevisionUnderstanding,
    SelectedProfileStatement,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


class NativeReviewModel:
    """测试供应商的原生工具响应，不模拟已删除的文本完成协议。"""

    def __init__(self, compiler):
        self.compiler = compiler

    def bind_tools(self, tools, **kwargs):
        self.tools = {tool.name for tool in tools}
        assert {"submit_revision_turn", "submit_graph_patch"} & self.tools
        return self

    def invoke(self, messages):
        if "submit_graph_patch" in self.tools:
            payload = next(m.content for m in messages if m.type == "human")
            result = self.compiler.create_structured_completion(
                response_model=GraphPatchDraft, user_content=payload
            )
            submitted = result.model_dump(mode="json")
            for operation in submitted["graph_operations"]:
                for key in ("operation_id", "before_description", "precondition_hash"):
                    operation.pop(key, None)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_graph_patch",
                        "id": "graph-submit",
                        "args": {"result": submitted},
                    }
                ],
            )
        seen = {m.name for m in messages if isinstance(m, ToolMessage)}
        if isinstance(self.compiler, ResearchingTurnCompiler):
            sequence = [
                ("search_world", {"question": "这条十点上班陈述的事实主体是谁？"}),
                (
                    "locate_source_messages",
                    {"retrieval_id": "retrieval-1", "context_turns": 6, "limit": 120},
                ),
                ("get_message_context", {"message_ids": ["source-1"], "context_turns": 6}),
            ]
            for name, args in sequence:
                if name not in seen:
                    return AIMessage(
                        content="", tool_calls=[{"name": name, "args": args, "id": name}]
                    )
        # 仅复用测试样例构造器，不调用真实结构化请求客户端。
        turn = self.compiler.create_structured_completion(response_model=RevisionAgentTurn)
        submitted = turn.model_dump(mode="json")
        for option in (submitted.get("question") or {}).get("options", []):
            option.pop("id", None)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_revision_turn",
                    "id": "submit",
                    "args": {"result": submitted},
                }
            ],
        )


class FakeReviewCompiler:
    def create_agent_chat_model(self):
        return NativeReviewModel(self)

    def create_structured_completion(self, **kwargs: Any) -> Any:
        response_model = kwargs["response_model"]
        if response_model is RevisionAgentTurn:
            return RevisionAgentTurn(
                kind="understanding",
                understanding=RevisionUnderstanding(
                    wrong_interpretation="把用户的上班时间归给目标人物",
                    corrected_interpretation="十点上班描述的是用户",
                    affected_dimensions=["subject"],
                    source_message_ids=[],
                    open_question=None,
                    summary_for_user="删除目标人物十点上班的错误结论，并纠正图谱主体。",
                ),
            )
        if response_model is GraphPatchDraft:
            payload = json.loads(kwargs["user_content"])
            assert payload["understanding"]["corrected_interpretation"] == "十点上班描述的是用户"
            assert "approved_profile_patch" in payload
            return GraphPatchDraft.model_validate(
                {
                    "graph_operations": [
                        {
                            "operation_id": "fix-subject",
                            "operation_type": "UPDATE_ENTITY",
                            "entity_name": "用户",
                            "after_description": "用户通常十点上班",
                            "entity_type": "PERSON",
                            "reason": "纠正事实主体",
                            "source_message_ids": [],
                        }
                    ],
                    "regression_queries": ["聊天中十点上班的人是谁？"],
                }
            )
        raise AssertionError(f"unexpected response model: {response_model}")


class QuestionReviewCompiler(FakeReviewCompiler):
    def create_structured_completion(self, **kwargs: Any) -> Any:
        if kwargs["response_model"] is RevisionAgentTurn:
            return RevisionAgentTurn(
                kind="question",
                question=RevisionQuestionTurn(
                    premise="已知被圈选的陈述把十点上班归给目标人物。",
                    decision_key="time",
                    question="这条信息只应删除，还是还要记录为过去的用户情况？",
                    options=[],
                    source_message_ids=[],
                ),
            )
        return super().create_structured_completion(**kwargs)


class ScopeProposalCompiler(FakeReviewCompiler):
    """验证相关 Claim 只能先以用户确认的范围提案进入会话。"""

    def create_structured_completion(self, **kwargs: Any) -> Any:
        if kwargs["response_model"] is RevisionAgentTurn:
            return RevisionAgentTurn(
                kind="scope_proposal",
                scope_proposal=RevisionScopeProposal.model_validate(
                    {
                        "summary": "还有一条同类作息结论可能受这次主体纠正影响。",
                        "items": [
                            {
                                "candidate_item": 1,
                                "reason": "与已选择事实主体和事实类型相同。",
                                "effect": "纳入后会一并核对是否同样误归因。",
                            }
                        ],
                    }
                ),
            )
        return super().create_structured_completion(**kwargs)


class FakeReviewLightRAG:
    def __init__(self) -> None:
        self.mutations: list[object] = []

    def query(self, _workspace: str, _query: str, **_: object) -> LightRAGRetrieval:
        return LightRAGRetrieval(
            context="没有可直接引用的图谱上下文",
            references=[],
            metadata=LightRAGMetadata(
                lightrag_version="test",
                embedding_model="test",
                embedding_dimension=3,
                extraction_model="test",
                chunking_strategy="fixed",
                chunk_token_size=100,
                chunk_overlap_token_size=10,
                entity_prompt_version="test",
            ),
        )

    def get_entity(self, _workspace: str, entity_name: str) -> dict[str, object]:
        return {"entity_name": entity_name, "description": "旧描述", "entity_type": "PERSON"}


def test_graph_preview_description_comes_from_server_precondition_state() -> None:
    assert _graph_state_description({"description": "当前关系描述"}) == "当前关系描述"
    assert _graph_state_description({"description": ""}) is None
    assert _graph_state_description({"unexpected": "value"}) is None
    assert _graph_state_description("not-a-graph-state") is None


class ResearchingTurnCompiler(FakeReviewCompiler):
    """验证探索 Agent 自己选择查询，而不是服务端固定检索。"""

    def __init__(self) -> None:
        self.models: list[type[object]] = []

    def create_structured_completion(self, **kwargs: Any) -> Any:
        response_model = kwargs["response_model"]
        self.models.append(response_model)
        assert response_model is RevisionAgentTurn
        return RevisionAgentTurn(
            kind="understanding",
            understanding=RevisionUnderstanding(
                wrong_interpretation="把用户的上班时间归给目标人物",
                corrected_interpretation="十点上班描述的是用户",
                affected_dimensions=["subject"],
                source_message_ids=["source-1"],
                open_question=None,
                summary_for_user="删除错误归因，不新增用户的人物档案。",
            ),
        )


def _graph() -> WorldGraphVersion:
    return WorldGraphVersion(
        id="graph-1",
        project_id="project-1",
        trigger_import_id="import-1",
        workspace_key="world_graph_1",
        status="awaiting_profile_review",
        source_fingerprint="s" * 64,
        config_fingerprint="c" * 64,
        source_import_ids=["import-1"],
        compiler_version="person-world-agent-v2",
    )


def test_revision_graph_uses_agent_selected_tool_node_and_freezes_retrieved_sources(
    tmp_path,
) -> None:
    engine = create_engine(f"sqlite:///{tmp_path}/revision.db")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        graph = _graph()
        session.add(graph)
        revision = PersonWorldRevisionSession(
            project_id="project-1",
            base_graph_version_id=graph.id,
            status="exploring",
            session_revision=1,
            scope={"selected_claim_ids": []},
        )
        session.add(revision)
        session.flush()
        calls: list[str] = []
        source = {
            "message_id": "source-1",
            "document_id": "bundle-1.txt",
            "bundle_id": "bundle-1",
            "ordinal": 1,
            "timestamp": "2026-05-01T09:00:00+00:00",
            "participant_id": "target-1",
            "participant_name": "洪欣羽",
            "participant_role": "target",
            "kind": "text",
            "content": "这个笨人早上十点才上班！",
            "is_primary_match": True,
        }

        def search(question: str) -> dict[str, object]:
            calls.append("search_world")
            assert question == "这条十点上班陈述的事实主体是谁？"
            return {
                "retrieval_id": "retrieval-1",
                "context": "候选图谱材料",
                "references": [
                    {"document_name": "bundle-1.txt", "chunk_content": source["content"]}
                ],
            }

        def locate(**_: object) -> list[dict[str, object]]:
            calls.append("locate_source_messages")
            return [source]

        def context(**_: object) -> list[dict[str, object]]:
            calls.append("get_message_context")
            return [source]

        tools = {
            "search_world": StructuredTool.from_function(
                search,
                name="search_world",
                description="test",
            ),
            "locate_source_messages": StructuredTool.from_function(
                locate, name="locate_source_messages", description="test"
            ),
            "get_message_context": StructuredTool.from_function(
                context, name="get_message_context", description="test"
            ),
            "list_graph_entities": StructuredTool.from_function(
                lambda: [], name="list_graph_entities", description="test"
            ),
            "get_graph_entity": StructuredTool.from_function(
                lambda entity_name: {"entity_name": entity_name},
                name="get_graph_entity",
                description="test",
            ),
        }
        compiler = ResearchingTurnCompiler()
        agent = RevisionAgentGraph(
            definition=load_prompt_definition("revision"),
            tools=tools,
            compiler=compiler,  # type: ignore[arg-type]
            context_assembler=RevisionContextAssembler(session),
            revision=revision,
            graph=graph,
            profile=None,
            current_profile={},
            conversation=[{"role": "user", "content": "十点上班说的是我，不是她"}],
            selected_statements=[],
            explicit_source_ids=[],
        )
        execution = agent.run()

        assert calls == ["search_world", "locate_source_messages", "get_message_context"]
        assert execution.snapshot.snapshot.evidence_message_ids == ["source-1"]
        assert execution.turn.understanding is not None
        assert execution.turn.understanding.source_message_ids == ["source-1"]
        assert compiler.models == [RevisionAgentTurn]
        # 已接受的回合恢复时复用原上下文快照，不再检索、请求模型或插入重复快照。
        replayed = agent.run()
        assert calls == ["search_world", "locate_source_messages", "get_message_context"]
        assert compiler.models == [RevisionAgentTurn]
        assert replayed.turn == execution.turn
        assert replayed.snapshot.snapshot.id == execution.snapshot.snapshot.id


def test_revision_turn_job_runs_after_http_has_only_queued_user_input() -> None:
    """回合推理必须由 Worker 执行，排队 API 只保存用户输入和稳定范围。"""

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_graph())
        session.flush()
        review = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )
        revision = review.create_session(
            project_id="project-1",
            user_message="十点上班说的是我，不是她",
            idempotency_key="queued-turn-001",
            run_agent=False,
        )
        assert revision.status == "agent_queued"
        assert revision.understanding_payload is None

        jobs = JobService(session)
        queued = enqueue_revision_turn_job(jobs, settings=Settings(), revision=revision)
        running = jobs.start(
            queued.id,
            worker_token="revision-worker",
            lease_duration=timedelta(minutes=2),
        )
        create_revision_turn_handler(
            Settings(),
            compiler_client=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag_client=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )(jobs, running)

        session.expire_all()
        completed = session.get(PersonWorldRevisionSession, revision.id)
        assert completed is not None
        assert completed.status == "understanding_ready"
        assistant = session.scalar(
            select(PersonWorldRevisionMessage)
            .where(
                PersonWorldRevisionMessage.session_id == revision.id,
                PersonWorldRevisionMessage.role == "assistant",
            )
            .order_by(PersonWorldRevisionMessage.created_at.desc())
        )
        assert assistant is not None
        assert assistant.payload is not None
        assert isinstance(assistant.payload["trace"], list)
        checkpoint = session.get(type(queued), queued.id)
        assert checkpoint is not None
        assert checkpoint.checkpoint is not None
        assert checkpoint.checkpoint["stage"] == "completed"
        assert completed.scope["agent_stage"] == "completed"
        assert completed.scope["agent_stage_progress"] == 1.0


def test_cancelled_revision_rejects_late_agent_turn_before_any_tool_or_model_call() -> None:
    """取消是后端写入边界，不能只靠前端隐藏按钮。"""

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_graph())
        session.flush()
        review = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )
        revision = review.create_session(
            project_id="project-1",
            user_message="这条归因不对",
            idempotency_key="cancelled-turn-001",
            run_agent=False,
        )
        revision.status = "cancelled"
        session.flush()

        try:
            review.advance(revision)
        except RevisionStateError as error:
            assert "可推进" in str(error)
        else:
            raise AssertionError("已取消会话不得接收迟到的 Agent 回合")

        assert (
            session.scalar(
                select(PersonWorldRevisionMessage).where(
                    PersonWorldRevisionMessage.session_id == revision.id,
                    PersonWorldRevisionMessage.role == "assistant",
                )
            )
            is None
        )


def test_retry_agent_turn_keeps_same_scope_and_creates_auditable_retry() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        graph = _graph()
        session.add(graph)
        revision = PersonWorldRevisionSession(
            project_id="project-1",
            base_graph_version_id=graph.id,
            status="agent_failed",
            session_revision=4,
            scope={"selected_claim_ids": ["claim-1"], "active_agent_job_id": "old-job"},
        )
        session.add(revision)
        session.flush()
        review = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )

        retried = review.retry_agent_turn(
            revision,
            expected_session_revision=4,
            idempotency_key="retry-agent-turn-001",
        )

        assert retried.status == "agent_queued"
        assert retried.session_revision == 5
        assert retried.scope == {
            "selected_claim_ids": ["claim-1"],
            "agent_stage": "queued",
            "agent_stage_progress": 0.0,
        }
        message = session.scalar(
            select(PersonWorldRevisionMessage).where(
                PersonWorldRevisionMessage.session_id == revision.id,
                PersonWorldRevisionMessage.kind == "agent_retry",
            )
        )
        assert message is not None


def test_revision_context_uses_claim_scope_without_exposing_claim_uuid_to_model() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        graph = _graph()
        session.add(graph)
        session.add(
            AtomicWorldClaim(
                id="claim-internal-1",
                project_id="project-1",
                graph_version_id=graph.id,
                agent_run_id="run-1",
                profile_section="practices.temporal_rhythms",
                subject_kind="target_person",
                predicate="temporal_rhythm",
                object_text="目标人物通常十点上班",
                normalized_text="目标人物通常十点上班",
                evidence_ids=[],
            )
        )
        revision = PersonWorldRevisionSession(
            project_id="project-1",
            base_graph_version_id=graph.id,
            status="exploring",
            session_revision=1,
            scope={"selected_claim_ids": ["claim-internal-1"]},
        )
        session.add(revision)
        session.flush()

        assembled = RevisionContextAssembler(session).assemble(
            revision=revision,
            graph=graph,
            profile=None,
            current_profile={},
            conversation=[{"role": "user", "content": "这条不对"}],
            selected_statements=[
                {
                    "claim_id": "claim-internal-1",
                    "section": "practices.temporal_rhythms",
                    "section_label": "实践与规律",
                    "text": "目标人物通常十点上班",
                    "source_message_ids": [],
                }
            ],
            source_messages=[],
            related_graph_context="",
            graph_references=[],
        )

        assert assembled.snapshot.related_claim_ids == ["claim-internal-1"]
        assert assembled.model_payload["selected_claims"] == [
            {
                "section": "practices.temporal_rhythms",
                "primary_domain": "practices",
                "fact_type": "temporal_rhythm",
                "statement": "目标人物通常十点上班",
                "assertion_kind": "unknown",
                "derivation": "inferred",
                "temporal_status": "unknown",
                "valid_from": None,
                "valid_to": None,
                "evidence_count": 0,
            }
        ]
        assert "claim_id" not in str(assembled.model_payload)


def test_scope_proposal_requires_explicit_include_before_related_claim_enters_scope() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        graph = _graph()
        session.add(graph)
        for claim_id, statement in [
            ("claim-1", "目标人物通常十点上班"),
            ("claim-2", "目标人物周末也通常十点上班"),
        ]:
            session.add(
                AtomicWorldClaim(
                    id=claim_id,
                    project_id="project-1",
                    graph_version_id=graph.id,
                    agent_run_id="run-1",
                    profile_section="practices.temporal_rhythms",
                    subject_kind="target_person",
                    predicate="temporal_rhythm",
                    object_text=statement,
                    normalized_text=statement,
                    evidence_ids=[],
                )
            )
        session.flush()
        review = PersonWorldReviewService(
            session,
            compiler=ScopeProposalCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )
        revision = review.create_session(
            project_id="project-1",
            user_message="这条作息的主体不对",
            idempotency_key="scope-proposal-001",
            selected_statements=[
                SelectedProfileStatement(
                    claim_id="claim-1",
                    section="practices.temporal_rhythms",
                    section_label="实践与规律",
                    text="目标人物通常十点上班",
                )
            ],
        )

        assert revision.status == "waiting_for_scope"
        assert revision.scope["selected_claim_ids"] == ["claim-1"]
        proposal = revision.scope["pending_scope_proposal"]
        assert proposal["items"] == [{"proposal_item": 1, "claim_id": "claim-2"}]

        review.change_scope(
            revision,
            action="include",
            claim_ids=[],
            graph_object_refs=[],
            proposal_item_ids=[1],
            expected_session_revision=revision.session_revision,
            idempotency_key="scope-proposal-include-001",
            run_agent=False,
        )
        assert revision.status == "agent_queued"
        assert revision.scope["selected_claim_ids"] == ["claim-1", "claim-2"]


def test_confirmed_correction_keeps_structural_claim_scope_for_recompilation() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        graph = _graph()
        session.add(graph)
        session.add(
            AtomicWorldClaim(
                id="claim-1",
                project_id="project-1",
                graph_version_id=graph.id,
                agent_run_id="run-1",
                profile_section="practices.temporal_rhythms",
                subject_kind="target_person",
                predicate="temporal_rhythm",
                object_text="目标人物通常十点上班",
                normalized_text="目标人物通常十点上班",
                evidence_ids=[],
            )
        )
        session.flush()
        service = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )
        revision = service.create_session(
            project_id="project-1",
            user_message="这条作息的主体不是她",
            idempotency_key="create-claim-scope-001",
            selected_statements=[
                SelectedProfileStatement(
                    claim_id="claim-1",
                    section="routine_summary.other_patterns",
                    section_label="生活规律",
                    text="目标人物通常十点上班",
                )
            ],
        )
        # 历史 Claim 仍能定位和展示，但不再启动旧 Profile Patch 生成器。
        scope = service._correction_scope(revision)
        assert scope["claim_ids"] == ["claim-1"]
        assert scope["primary_domains"] == ["practices"]
        import pytest

        with pytest.raises(RevisionStateError, match="重新生成 v3"):
            service.confirm_understanding(
                revision,
                understanding_revision=revision.understanding_revision,
                understanding_payload_hash=revision.understanding_payload_hash or "",
                expected_session_revision=revision.session_revision,
            )
        assert session.scalar(select(WorldCorrection)) is None


def test_revision_requires_each_confirmation_before_graph_execution(monkeypatch) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_graph())
        session.flush()
        lightrag = FakeReviewLightRAG()
        service = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=lightrag,  # type: ignore[arg-type]
        )

        revision = service.create_session(
            project_id="project-1",
            user_message="十点上班说的是我，不是她",
            idempotency_key="create-revision-001",
        )
        assert revision.status == "understanding_ready"
        assert revision.context_snapshot_id is not None
        snapshot = session.get(PersonWorldRevisionContextSnapshot, revision.context_snapshot_id)
        assert snapshot is not None
        assert snapshot.session_revision == 1
        assert snapshot.base_graph_version_id == "graph-1"
        assert len(snapshot.context_hash) == 64
        assert revision.understanding_payload_hash is not None

        _prepare_v3_confirmation(service, revision, monkeypatch)
        change_set = service.confirm_understanding(
            revision,
            understanding_revision=revision.understanding_revision,
            understanding_payload_hash=revision.understanding_payload_hash or "",
            expected_session_revision=revision.session_revision,
        )
        assert revision.status == "profile_review"
        assert change_set.status == "awaiting_profile_approval"
        assert change_set.graph_operations == []
        assert lightrag.mutations == []

        service.approve_profile(
            revision,
            change_set_id=change_set.id,
            revision_number=change_set.revision,
            payload_hash=change_set.canonical_payload_hash,
            expected_session_revision=revision.session_revision,
        )
        assert revision.status == "graph_review"
        assert change_set.status == "awaiting_graph_approval"
        assert change_set.graph_operations[0]["precondition_hash"]
        assert lightrag.mutations == []

        service.approve_graph(
            revision,
            change_set_id=change_set.id,
            revision_number=change_set.revision,
            payload_hash=change_set.canonical_payload_hash,
            expected_session_revision=revision.session_revision,
        )
        assert revision.status == "approved"
        assert change_set.status == "approved"
        approvals = list(
            session.scalars(
                select(WorldChangeApproval).where(
                    WorldChangeApproval.change_set_id == change_set.id
                )
            )
        )
        assert {item.approval_stage for item in approvals} == {"profile", "graph"}
        # 审核服务只记录已批准 ChangeSet，真正写图必须交给候选图后台任务。
        assert lightrag.mutations == []


def test_revision_invalidates_pending_understanding_when_active_base_changes() -> None:
    """版本竞争必须作废旧理解，而不是允许它继续生成 Patch。"""

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        base = _graph()
        session.add(base)
        session.flush()
        service = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )
        revision = service.create_session(
            project_id="project-1",
            user_message="十点上班说的是我，不是她",
            idempotency_key="create-stale-base-001",
        )
        current_graph = WorldGraphVersion(
            id="graph-new",
            project_id="project-1",
            trigger_import_id="import-1",
            workspace_key="world_graph_new",
            status="ready",
            source_fingerprint="n" * 64,
            config_fingerprint="d" * 64,
            source_import_ids=["import-1"],
            compiler_version="person-world-agent-v3",
        )
        session.add_all(
            [
                current_graph,
                WorldPublication(
                    id="publication-new",
                    project_id="project-1",
                    graph_version_id=current_graph.id,
                    # 本测试只验证稳定 ID 比较；Profile 正文不参与这项乐观锁。
                    profile_id="profile-new",
                    correction_head_hash="h" * 64,
                    status="active",
                    published_by="test",
                ),
            ]
        )
        session.flush()

        try:
            service.confirm_understanding(
                revision,
                understanding_revision=revision.understanding_revision,
                understanding_payload_hash=revision.understanding_payload_hash or "",
                expected_session_revision=revision.session_revision,
            )
        except RevisionBaseStaleError:
            pass
        else:
            raise AssertionError("活动基线变更后不得继续生成 Profile Patch")

        assert revision.status == "stale"
        assert revision.scope["stale_base"]["reason"] == "active_graph_changed"
        assert revision.pending_turn_id is None
        message = session.scalar(
            select(PersonWorldRevisionMessage).where(
                PersonWorldRevisionMessage.session_id == revision.id,
                PersonWorldRevisionMessage.kind == "base_stale",
            )
        )
        assert message is not None


def test_revision_rejects_stale_reply_and_replays_idempotency_key() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_graph())
        session.flush()
        service = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )
        revision = service.create_session(
            project_id="project-1",
            user_message="十点上班说的是我，不是她",
            idempotency_key="create-revision-idempotent",
        )
        existing = service.create_session(
            project_id="project-1",
            user_message="不应再次创建会话",
            idempotency_key="create-revision-idempotent",
        )
        assert existing.id == revision.id
        current_revision = revision.session_revision
        service.add_user_message(
            revision,
            "补充：这是过去的情况。",
            expected_session_revision=current_revision,
            idempotency_key="reply-idempotent-001",
        )
        after_reply = revision.session_revision
        replay = service.add_user_message(
            revision,
            "重复网络请求不应生成第二条用户消息。",
            expected_session_revision=after_reply,
            idempotency_key="reply-idempotent-001",
        )
        assert replay.id == revision.id
        assert revision.session_revision == after_reply
        try:
            service.add_user_message(
                revision,
                "过期回答",
                expected_session_revision=current_revision,
                idempotency_key="reply-stale-001",
            )
        except RuntimeError as error:
            assert "会话已经更新" in str(error)
        else:
            raise AssertionError("过期会话版本必须被拒绝")


def test_revision_question_turn_requires_the_current_turn_id() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_graph())
        session.flush()
        service = PersonWorldReviewService(
            session,
            compiler=QuestionReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )
        revision = service.create_session(
            project_id="project-1",
            user_message="十点上班的归属不对",
            idempotency_key="create-question-turn-001",
        )
        assert revision.status == "waiting_for_user"
        assert revision.pending_turn_id is not None
        assert revision.understanding_payload is None
        question_message = session.scalar(
            select(PersonWorldRevisionMessage).where(
                PersonWorldRevisionMessage.turn_id == revision.pending_turn_id
            )
        )
        assert question_message is not None
        assert question_message.kind == "question"
        assert question_message.payload is not None
        assert question_message.payload["question"]["decision_key"] == "time"
        try:
            service.add_user_message(
                revision,
                "只删除。",
                expected_session_revision=revision.session_revision,
                in_reply_to_turn_id="not-the-pending-turn",
                idempotency_key="wrong-question-001",
            )
        except RuntimeError as error:
            assert "当前正在等待" in str(error)
        else:
            raise AssertionError("回答不对应 pending turn 时必须拒绝")


def test_scope_changes_use_stable_references_and_invalidate_prior_understanding() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_graph())
        session.flush()
        service = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )
        revision = service.create_session(
            project_id="project-1",
            user_message="我要核对这条事实。",
            idempotency_key="create-scope-001",
        )
        old_snapshot = revision.context_snapshot_id
        service.change_scope(
            revision,
            action="include",
            claim_ids=["claim-1"],
            graph_object_refs=[{"kind": "entity", "id": "entity-1"}],
            expected_session_revision=revision.session_revision,
            idempotency_key="scope-include-001",
        )
        assert revision.scope["selected_claim_ids"] == ["claim-1"]
        assert revision.scope["included_graph_objects"] == [{"id": "entity-1", "kind": "entity"}]
        assert revision.context_snapshot_id != old_snapshot
        assert revision.status == "understanding_ready"
        service.change_scope(
            revision,
            action="exclude",
            claim_ids=["claim-1"],
            graph_object_refs=[{"kind": "entity", "id": "entity-1"}],
            expected_session_revision=revision.session_revision,
            idempotency_key="scope-exclude-001",
        )
        assert revision.scope["selected_claim_ids"] == []
        assert revision.scope["excluded_claim_ids"] == ["claim-1"]


def test_revision_persists_structured_profile_selection() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_graph())
        session.flush()
        service = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )

        revision = service.create_session(
            project_id="project-1",
            user_message="这条工作时间的主体不对",
            idempotency_key="create-revision-002",
            selected_statements=[
                SelectedProfileStatement(
                    section="routine_summary.workdays",
                    section_label="生活规律",
                    text="目标人物通常十点上班",
                    source_message_ids=[],
                )
            ],
        )

        first_message = session.scalar(
            select(PersonWorldRevisionMessage).where(
                PersonWorldRevisionMessage.session_id == revision.id,
                PersonWorldRevisionMessage.role == "user",
            )
        )
        assert first_message is not None
        assert first_message.payload is not None
        assert first_message.payload["selected_statements"] == [
            {
                "section": "routine_summary.workdays",
                "section_label": "生活规律",
                "text": "目标人物通常十点上班",
                "source_message_ids": [],
            }
        ]


def test_user_can_revise_after_candidate_validation_before_publish(monkeypatch) -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(_graph())
        session.flush()
        service = PersonWorldReviewService(
            session,
            compiler=FakeReviewCompiler(),  # type: ignore[arg-type]
            lightrag=FakeReviewLightRAG(),  # type: ignore[arg-type]
        )
        revision = service.create_session(
            project_id="project-1",
            user_message="十点上班说的是我，不是她",
            idempotency_key="create-revision-003",
        )
        _prepare_v3_confirmation(service, revision, monkeypatch)
        change_set = service.confirm_understanding(
            revision,
            understanding_revision=revision.understanding_revision,
            understanding_payload_hash=revision.understanding_payload_hash or "",
            expected_session_revision=revision.session_revision,
        )
        candidate = WorldGraphVersion(
            id="graph-2",
            project_id="project-1",
            trigger_import_id="import-1",
            workspace_key="world_graph_2",
            status="publish_ready",
            source_fingerprint="s" * 64,
            config_fingerprint="d" * 64,
            source_import_ids=["import-1"],
            compiler_version="person-world-agent-v2",
            parent_version_id="graph-1",
            revision=2,
        )
        session.add(candidate)
        session.flush()
        change_set.candidate_graph_version_id = candidate.id
        change_set.status = "publish_ready"
        revision.status = "publish_ready"
        session.flush()

        service.add_user_message(revision, "还要说明这是以前的时间，不是现在的规律")

        assert revision.status == "understanding_ready"
        assert revision.graph_change_set_id is None
        assert change_set.status == "rejected"
        assert candidate.status == "rejected"


def _prepare_v3_confirmation(service, revision, monkeypatch):
    """只替换栏目编译，审批与原生 Graph Patch Agent 仍运行真实服务代码。"""
    from moonlightbox.world.models import PersonWorldProfile
    from moonlightbox.world.person_world.coordinator_v3 import empty_legacy_projection
    from moonlightbox.world.person_world.review import profile_v3

    profile = PersonWorldProfile(
        id="v3-profile",
        project_id=revision.project_id,
        subject_person_id="target",
        graph_version_id=revision.base_graph_version_id,
        profile_schema_version="v3",
        profile_v3={},
        source_message_ids=[],
        retrieval_manifest=[],
        compiler_version="v3",
        **empty_legacy_projection().model_dump(mode="json"),
    )
    service.session.add(profile)
    service.session.flush()
    revision.base_profile_id = profile.id
    revision.understanding_payload = {
        **revision.understanding_payload,
        "graph_change_requested": True,
    }
    monkeypatch.setattr(profile_v3, "build_merged_preview", lambda *args: [])
