"""用户纠正 PersonWorldProfile 时的多步探索和确认服务。"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime import AgentLoopController
from moonlightbox.agent_runtime.tasks import run_submission_task
from moonlightbox.imports.models import Message, Participant
from moonlightbox.world.client import LightRAGSidecarClient
from moonlightbox.world.compiler import AgentCompilerClient
from moonlightbox.world.models import (
    AtomicWorldClaim,
    PersonWorldProfile,
    PersonWorldRevisionMessage,
    PersonWorldRevisionSession,
    WorldChangeApproval,
    WorldCorrection,
    WorldGraphChangeSet,
    WorldGraphVersion,
    WorldPublication,
)

from ..graph_executor import (
    canonical_change_set_hash,
    canonical_graph_state_hash,
    canonical_profile_patch_hash,
    operation_precondition_state,
)
from ..investigation_artifacts import InvestigationArtifactStore
from ..prompt_loader import load_prompt_definition
from ..schemas import (
    GraphPatchDraft,
    SelectedProfileStatement,
)
from ..tools import build_graph_query_tools, build_source_message_tools
from .context import RevisionContextAssembler, canonical_context_hash
from .revision_graph import RevisionAgentGraph


class RevisionStateError(RuntimeError):
    pass


class RevisionBaseStaleError(RevisionStateError):
    """本轮纠正固定的 Graph/Profile 基线已不再是项目活动版本。"""


class PersonWorldReviewService:
    def __init__(
        self,
        session: Session,
        *,
        compiler: AgentCompilerClient,
        lightrag: LightRAGSidecarClient,
    ) -> None:
        self.session = session
        self.compiler = compiler
        self.lightrag = lightrag

    def create_session(
        self,
        *,
        project_id: str,
        user_message: str,
        idempotency_key: str,
        source_message_ids: list[str] | None = None,
        selected_statements: list[SelectedProfileStatement] | None = None,
        base_profile_id: str | None = None,
        run_agent: bool = True,
    ) -> PersonWorldRevisionSession:
        existing = self.session.scalar(
            select(PersonWorldRevisionSession)
            .join(
                PersonWorldRevisionMessage,
                PersonWorldRevisionMessage.session_id == PersonWorldRevisionSession.id,
            )
            .where(
                PersonWorldRevisionSession.project_id == project_id,
                PersonWorldRevisionMessage.idempotency_key == idempotency_key,
            )
            .order_by(PersonWorldRevisionSession.created_at.asc())
        )
        if existing is not None:
            return existing
        profile, graph = _active_profile_and_graph(self.session, project_id)
        if base_profile_id:
            selected_profile = self.session.get(PersonWorldProfile, base_profile_id)
            selected_graph = (
                self.session.get(WorldGraphVersion, selected_profile.graph_version_id)
                if selected_profile
                else None
            )
            if (
                selected_profile is None
                or selected_profile.project_id != project_id
                or selected_graph is None
                or selected_graph.status
                not in (
                    {"ready", "awaiting_profile_review", "superseded"}
                    if selected_profile.node_boundary_hash
                    else {"ready", "awaiting_profile_review"}
                )
            ):
                raise RevisionStateError("所选人物画像不属于当前项目的可审核版本")
            profile, graph = selected_profile, selected_graph
        if graph is None:
            raise RevisionStateError("人物世界尚未生成，不能开始纠正")
        if profile is not None and profile.profile_schema_version == "v3":
            from .profile_v3 import validate_selection

            for selected in selected_statements or []:
                validate_selection(profile.profile_v3, selected)
        revision = PersonWorldRevisionSession(
            project_id=project_id,
            base_graph_version_id=graph.id,
            base_profile_id=profile.id if profile is not None else None,
            status="exploring" if run_agent else "agent_queued",
            session_revision=1,
            input_revision=1,
            scope={
                "node_scope": (
                    (profile.generation_summary or {}).get("node_scope") if profile else None
                ),
                "base_profile_hash": canonical_context_hash(profile.profile_v3)
                if profile is not None and profile.profile_schema_version == "v3"
                else None,
                "selected_claim_ids": [
                    item.claim_id for item in selected_statements or [] if item.claim_id is not None
                ],
                "selected_statements": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in selected_statements or []
                ],
                "included_graph_objects": [],
                "excluded_refs": [],
                "provisional_target": user_message.strip(),
                "agent_stage": "queued",
                "agent_stage_progress": 0.0,
            },
            understanding_revision=0,
        )
        self.session.add(revision)
        self.session.flush()
        self.session.add(
            PersonWorldRevisionMessage(
                session_id=revision.id,
                role="user",
                kind="user_message",
                session_revision=revision.session_revision,
                idempotency_key=idempotency_key,
                content=user_message.strip(),
                payload={
                    "source_message_ids": list(dict.fromkeys(source_message_ids or [])),
                    "selected_statements": [
                        item.model_dump(mode="json", exclude_none=True)
                        for item in selected_statements or []
                    ],
                },
            )
        )
        self.session.flush()
        return self.advance(revision) if run_agent else revision

    def add_user_message(
        self,
        revision: PersonWorldRevisionSession,
        content: str,
        *,
        expected_session_revision: int | None = None,
        in_reply_to_turn_id: str | None = None,
        idempotency_key: str | None = None,
        run_agent: bool = True,
    ) -> PersonWorldRevisionSession:
        self._ensure_base_is_current(revision)
        if revision.status in {
            "approved",
            "executing",
            "validating",
            "published",
            "cancelled",
            "agent_queued",
            "agent_running",
            "agent_failed",
            "stale",
        }:
            raise RevisionStateError("当前纠正会话不能继续修改；请创建新的修订会话")
        if (
            expected_session_revision is not None
            and expected_session_revision != revision.session_revision
        ):
            raise RevisionStateError("会话已经更新，请刷新后再提交回答")
        if revision.pending_turn_id is not None and in_reply_to_turn_id != revision.pending_turn_id:
            raise RevisionStateError("该回答没有对应当前正在等待的 Agent 问题")
        if idempotency_key:
            replay = self.session.scalar(
                select(PersonWorldRevisionMessage).where(
                    PersonWorldRevisionMessage.session_id == revision.id,
                    PersonWorldRevisionMessage.idempotency_key == idempotency_key,
                )
            )
            if replay is not None:
                return revision
        previous_change_set = (
            self.session.get(WorldGraphChangeSet, revision.graph_change_set_id)
            if revision.graph_change_set_id
            else None
        )
        if previous_change_set is not None:
            previous_change_set.status = "rejected"
            candidate = (
                self.session.get(WorldGraphVersion, previous_change_set.candidate_graph_version_id)
                if previous_change_set.candidate_graph_version_id
                else None
            )
            if candidate is not None and candidate.status != "ready":
                # 用户在最终发布前仍可继续修改；已验证候选只作废，不碰活动图。
                candidate.status = "rejected"
            for correction in self.session.scalars(
                select(WorldCorrection).where(
                    WorldCorrection.approved_change_set_id == previous_change_set.id,
                    WorldCorrection.status.in_(["proposed", "approved"]),
                )
            ):
                correction.status = "revoked"
        scope = dict(revision.scope or {})
        scope.pop("active_agent_job_id", None)
        scope["agent_stage"] = "queued"
        scope["agent_stage_progress"] = 0.0
        revision.scope = scope
        revision.session_revision += 1
        revision.input_revision += 1
        revision.pending_turn_id = None
        self.session.add(
            PersonWorldRevisionMessage(
                session_id=revision.id,
                role="user",
                kind="user_message",
                in_reply_to_turn_id=in_reply_to_turn_id,
                context_snapshot_id=revision.context_snapshot_id,
                session_revision=revision.session_revision,
                idempotency_key=idempotency_key,
                content=content.strip(),
                payload=None,
            )
        )
        revision.status = "exploring" if run_agent else "agent_queued"
        revision.understanding_payload = None
        revision.understanding_payload_hash = None
        revision.profile_change_set_id = None
        revision.graph_change_set_id = None
        self.session.flush()
        return self.advance(revision) if run_agent else revision

    def change_scope(
        self,
        revision: PersonWorldRevisionSession,
        *,
        action: str,
        claim_ids: list[str],
        graph_object_refs: list[dict[str, str]],
        proposal_item_ids: list[int] | None = None,
        expected_session_revision: int,
        idempotency_key: str,
        run_agent: bool = True,
    ) -> PersonWorldRevisionSession:
        """显式扩大或排除 Scope；不通过 Profile 展示文字反查 Claim。"""

        self._ensure_base_is_current(revision)
        if action not in {"include", "exclude"}:
            raise RevisionStateError("范围操作必须是 include 或 exclude")
        if expected_session_revision != revision.session_revision:
            raise RevisionStateError("会话已经更新，请刷新后再修改范围")
        existing = self.session.scalar(
            select(PersonWorldRevisionMessage).where(
                PersonWorldRevisionMessage.session_id == revision.id,
                PersonWorldRevisionMessage.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return revision
        scope = dict(revision.scope or {})
        proposed_claim_ids = _resolve_scope_proposal_items(scope, proposal_item_ids or [])
        claim_ids = list(dict.fromkeys([*claim_ids, *proposed_claim_ids]))
        selected_claim_ids = _string_list(scope.get("selected_claim_ids"))
        excluded_claim_ids = _string_list(scope.get("excluded_claim_ids"))
        graph_objects = _object_refs(scope.get("included_graph_objects"))
        excluded_refs = _object_refs(scope.get("excluded_refs"))
        if action == "include":
            selected_claim_ids = _merge_values(selected_claim_ids, claim_ids)
            excluded_claim_ids = [item for item in excluded_claim_ids if item not in claim_ids]
            graph_objects = _merge_object_refs(graph_objects, graph_object_refs)
            excluded_refs = _remove_object_refs(excluded_refs, graph_object_refs)
        else:
            excluded_claim_ids = _merge_values(excluded_claim_ids, claim_ids)
            selected_claim_ids = [item for item in selected_claim_ids if item not in claim_ids]
            excluded_refs = _merge_object_refs(excluded_refs, graph_object_refs)
            graph_objects = _remove_object_refs(graph_objects, graph_object_refs)
        revision.scope = {
            **scope,
            "selected_claim_ids": selected_claim_ids,
            "excluded_claim_ids": excluded_claim_ids,
            "included_graph_objects": graph_objects,
            "excluded_refs": excluded_refs,
            "pending_scope_proposal": None,
            "agent_stage": "queued",
            "agent_stage_progress": 0.0,
        }
        self._invalidate_unapproved_outputs(revision)
        revision.session_revision += 1
        revision.input_revision += 1
        revision.pending_turn_id = None
        revision.understanding_payload = None
        revision.understanding_payload_hash = None
        revision.status = "exploring" if run_agent else "agent_queued"
        self.session.add(
            PersonWorldRevisionMessage(
                session_id=revision.id,
                role="system",
                kind="scope_change",
                context_snapshot_id=revision.context_snapshot_id,
                session_revision=revision.session_revision,
                idempotency_key=idempotency_key,
                content="已更新本轮核对范围，正在重新核对。",
                payload={
                    "action": action,
                    "claim_ids": list(dict.fromkeys(claim_ids)),
                    "proposal_item_ids": list(dict.fromkeys(proposal_item_ids or [])),
                    "graph_object_refs": graph_object_refs,
                },
            )
        )
        self.session.flush()
        return self.advance(revision) if run_agent else revision

    def retry_agent_turn(
        self,
        revision: PersonWorldRevisionSession,
        *,
        expected_session_revision: int,
        idempotency_key: str,
    ) -> PersonWorldRevisionSession:
        """重新排队失败的同一轮调查，不复制用户输入也不改变已冻结的 Scope。

        失败不是“没有证据”。因此重试只允许从 ``agent_failed`` 进入排队态，并把这次
        操作写进审计消息；Worker 会重新从原始消息、当前 Scope 和当前活动纠正装配快照。
        """

        self._ensure_base_is_current(revision)
        if revision.status != "agent_failed":
            raise RevisionStateError("当前会话没有可重试的 Agent 回合")
        if expected_session_revision != revision.session_revision:
            raise RevisionStateError("会话已经更新，请刷新后再重试")
        duplicate = self.session.scalar(
            select(PersonWorldRevisionMessage).where(
                PersonWorldRevisionMessage.session_id == revision.id,
                PersonWorldRevisionMessage.idempotency_key == idempotency_key,
            )
        )
        if duplicate is not None:
            return revision
        scope = dict(revision.scope or {})
        scope.pop("active_agent_job_id", None)
        scope["agent_stage"] = "queued"
        scope["agent_stage_progress"] = 0.0
        revision.scope = scope
        revision.status = "agent_queued"
        revision.session_revision += 1
        revision.input_revision += 1
        self.session.add(
            PersonWorldRevisionMessage(
                session_id=revision.id,
                role="system",
                kind="agent_retry",
                context_snapshot_id=revision.context_snapshot_id,
                session_revision=revision.session_revision,
                idempotency_key=idempotency_key,
                content="正在重试本轮 PersonWorld Agent 调查。",
                payload={"retry_of_status": "agent_failed"},
            )
        )
        self.session.flush()
        return revision

    def advance(
        self,
        revision: PersonWorldRevisionSession,
        *,
        on_stage: Callable[[str], None] | None = None,
    ) -> PersonWorldRevisionSession:
        # 取消、已批准或已失效的会话没有资格接收迟到的模型输出。这个检查必须在
        # 读取图谱、调用模型之前完成，不能只依赖前端按钮禁用。
        if revision.status not in {"exploring", "agent_queued", "agent_running"}:
            raise RevisionStateError("当前纠正会话不处于可推进的调查状态")
        self._ensure_base_is_current(revision)
        graph = self.session.get(WorldGraphVersion, revision.base_graph_version_id)
        if graph is None:
            raise RevisionStateError("纠正会话绑定的图版本不存在")
        profile = (
            self.session.get(PersonWorldProfile, revision.base_profile_id)
            if revision.base_profile_id
            else None
        )
        conversation = list(
            self.session.scalars(
                select(PersonWorldRevisionMessage)
                .where(PersonWorldRevisionMessage.session_id == revision.id)
                .order_by(
                    PersonWorldRevisionMessage.created_at.asc(), PersonWorldRevisionMessage.id.asc()
                )
            )
        )
        evidence_ids = list(
            dict.fromkeys(
                str(value)
                for item in conversation
                if isinstance(item.payload, dict)
                for value in item.payload.get("source_message_ids", [])
                if isinstance(value, str)
            )
        )
        selected_statements: list[dict[str, object]] = next(
            (
                item.payload.get("selected_statements", [])
                for item in conversation
                if item.role == "user"
                and isinstance(item.payload, dict)
                and isinstance(item.payload.get("selected_statements"), list)
            ),
            [],
        )
        artifacts = InvestigationArtifactStore()
        from ..node_scope import inherited_node_scope

        node_scope = inherited_node_scope(
            self.session,
            graph,
            revision.scope,
            profile.generation_summary if profile is not None else {},
        )
        graph_tools = build_graph_query_tools(
            self.lightrag,
            workspace=graph.workspace_key,
            allowed_document_names={
                item.source_name for item in graph_bundles(self.session, graph.id)
            },
            artifacts=artifacts,
            top_k=16,
            chunk_top_k=8,
            max_total_tokens=6000,
            temporal_scope=node_scope.retrieval_scope() if node_scope else None,
        )
        source_tools = build_source_message_tools(
            self.session,
            project_id=revision.project_id,
            graph_version_id=graph.id,
            artifacts=artifacts,
            message_periods=node_scope.message_periods() if node_scope else None,
        )
        # 由 Revision Agent 决定是否补查；服务端只绑定工具到本项目、当前图版本和预算，
        # 不再以“用户消息 → 固定 LightRAG 搜索”替代探索过程。
        definition = load_prompt_definition("revision")
        registered_tools = {**graph_tools, **source_tools}
        execution = RevisionAgentGraph(
            definition=definition,
            tools={name: registered_tools[name] for name in definition.tool_names},
            compiler=self.compiler,
            context_assembler=RevisionContextAssembler(self.session),
            revision=revision,
            graph=graph,
            profile=profile,
            current_profile=_profile_payload(profile),
            conversation=[{"role": item.role, "content": item.content} for item in conversation],
            selected_statements=selected_statements,
            explicit_source_ids=evidence_ids,
            on_stage=on_stage,
            controller=AgentLoopController(),
            artifacts=artifacts,
        ).run()
        agent_turn = execution.turn
        assembled = execution.snapshot
        # 只能引用本轮实际展示给模型的原始消息，不能仅因为某个 UUID 恰好
        # 属于当前项目就把它接受为证据。
        allowed_ids = {
            item for item in assembled.snapshot.evidence_message_ids if isinstance(item, str)
        }
        if agent_turn.kind == "question":
            question = agent_turn.question
            if question is None:  # Pydantic 联合已校验；此处保留运行时防线。
                raise RevisionStateError("纠正 Agent 没有提供有效的问题回合")
            question = question.model_copy(
                update={
                    "source_message_ids": list(
                        dict.fromkeys(
                            item for item in question.source_message_ids if item in allowed_ids
                        )
                    )
                }
            )
            revision.understanding_payload = None
            revision.understanding_payload_hash = None
            revision.status = "waiting_for_user"
            assistant_text = question.question
            assistant_payload: dict[str, object] = {
                "kind": "question",
                "question": question.model_dump(mode="json"),
            }
        elif agent_turn.kind == "scope_proposal":
            proposal = agent_turn.scope_proposal
            if proposal is None:
                raise RevisionStateError("纠正 Agent 没有提供有效的范围提案")
            candidates = _scope_proposal_candidates(
                revision.scope, assembled.snapshot.related_claim_ids
            )
            accepted_items = [
                item for item in proposal.items if 1 <= item.candidate_item <= len(candidates)
            ]
            if not accepted_items:
                raise RevisionStateError("范围提案没有指向当前快照中的候选事实")
            related_details = {
                item.get("candidate_item"): item
                for item in assembled.model_payload.get("related_claims", [])
                if isinstance(item, dict) and isinstance(item.get("candidate_item"), int)
            }
            revision.scope = {
                **dict(revision.scope or {}),
                "pending_scope_proposal": {
                    "snapshot_id": assembled.snapshot.id,
                    "items": [
                        {
                            "proposal_item": item.candidate_item,
                            "claim_id": candidates[item.candidate_item - 1],
                        }
                        for item in accepted_items
                    ],
                },
            }
            revision.understanding_payload = None
            revision.understanding_payload_hash = None
            revision.status = "waiting_for_scope"
            assistant_text = proposal.summary
            assistant_payload = {
                "kind": "scope_proposal",
                "scope_proposal": {
                    "summary": proposal.summary,
                    "items": [
                        {
                            **item.model_dump(mode="json"),
                            # 候选事实来自同一 Snapshot 的结构化投影；模型只选择展示
                            # 序号，不能自行改写待纳入的对象或泄露 Claim UUID。
                            "section": related_details.get(item.candidate_item, {}).get("section"),
                            "statement": related_details.get(item.candidate_item, {}).get(
                                "statement"
                            ),
                            "structural_relations": related_details.get(
                                item.candidate_item, {}
                            ).get("structural_relations", []),
                        }
                        for item in accepted_items
                    ],
                },
            }
        else:
            understanding = agent_turn.understanding
            if understanding is None:  # Pydantic 联合已校验；此处保留运行时防线。
                raise RevisionStateError("纠正 Agent 没有提供有效的共同理解")
            understanding = understanding.model_copy(
                update={
                    "source_message_ids": list(
                        dict.fromkeys(
                            item for item in understanding.source_message_ids if item in allowed_ids
                        )
                    )
                }
            )
            revision.understanding_revision += 1
            revision.understanding_payload = understanding.model_dump(mode="json")
            revision.understanding_payload_hash = canonical_context_hash(
                revision.understanding_payload
            )
            revision.status = "understanding_ready"
            assistant_text = understanding.summary_for_user
            assistant_payload = {
                "kind": "understanding",
                "understanding_revision": revision.understanding_revision,
                "understanding_payload_hash": revision.understanding_payload_hash,
            }
        revision.session_revision += 1
        turn = PersonWorldRevisionMessage(
            session_id=revision.id,
            role="assistant",
            kind=agent_turn.kind,
            context_snapshot_id=assembled.snapshot.id,
            session_revision=revision.session_revision,
            content=assistant_text,
            payload={
                **assistant_payload,
                "context_snapshot_id": assembled.snapshot.id,
                # 只保存有界错误摘要；正常情况下 Agent 已在回合中明确“材料暂不可读”，
                # 不能把工具失败伪装成没有历史证据。
                "tool_errors": list(execution.tool_errors),
                # LangGraph 的节点摘要与冻结快照一起保存，供失败重试、界面解释和
                # 审计回放使用；不包含未受限的模型原始思维或工具参数密文。
                "trace": list(execution.trace),
            },
        )
        self.session.add(turn)
        # ``turn_id`` 由 ORM 默认值生成，必须先 flush 才能把它写为下一条用户消息
        # 必须回答的稳定 pending ID。
        self.session.flush()
        revision.pending_turn_id = turn.turn_id if agent_turn.kind == "question" else None
        self.session.flush()
        return revision

    def confirm_understanding(
        self,
        revision: PersonWorldRevisionSession,
        *,
        understanding_revision: int,
        understanding_payload_hash: str,
        expected_session_revision: int,
        defer_v3: bool = False,
    ) -> WorldGraphChangeSet | None:
        self._ensure_base_is_current(revision)
        if revision.status != "understanding_ready":
            raise RevisionStateError("Agent 尚未形成可确认的完整理解")
        if understanding_revision != revision.understanding_revision:
            raise RevisionStateError("理解摘要已经更新，请确认最新版本")
        if expected_session_revision != revision.session_revision:
            raise RevisionStateError("会话已经更新，请刷新后再确认理解")
        if understanding_payload_hash != revision.understanding_payload_hash:
            raise RevisionStateError("理解摘要内容已经变化，请重新查看并确认")
        if not isinstance(revision.understanding_payload, dict):
            raise RevisionStateError("纠正会话缺少理解摘要")
        graph = self.session.get(WorldGraphVersion, revision.base_graph_version_id)
        if graph is None:
            raise RevisionStateError("基础图版本不存在")
        profile = (
            self.session.get(PersonWorldProfile, revision.base_profile_id)
            if revision.base_profile_id
            else None
        )
        if profile is not None and profile.profile_schema_version == "v3":
            if defer_v3:
                revision.status = "profile_compiling"
                revision.session_revision += 1
                revision.scope = {
                    **revision.scope,
                    "profile_preview_attempt": int(revision.scope.get("profile_preview_attempt", 0))
                    + 1,
                    "profile_preview_error": None,
                    "agent_stage": "profile_preview_queued",
                    "agent_stage_progress": 0.0,
                }
                return None
            from .profile_v3 import create_v3_change_set

            return create_v3_change_set(self, revision, profile, graph)
        raise RevisionStateError("历史画像只读，请先重新生成 v3 再发起纠正。")

    def approve_profile(
        self,
        revision: PersonWorldRevisionSession,
        *,
        change_set_id: str,
        revision_number: int,
        payload_hash: str,
        expected_session_revision: int,
    ) -> WorldGraphChangeSet:
        self._ensure_base_is_current(revision)
        if expected_session_revision != revision.session_revision:
            raise RevisionStateError("会话已经更新，请刷新后再批准 Profile Patch")
        change_set = self._change_set(revision, change_set_id, revision_number, payload_hash)
        if revision.status != "profile_review" or change_set.status != "awaiting_profile_approval":
            raise RevisionStateError("当前状态不能批准 Profile Patch")
        # Graph Patch 必须建立在用户已经看到并批准的 Profile Patch 之上；这里才允许
        # 为最小图操作读取候选实体与关系。
        # 用户请求已通过批准哈希检查；先完成只读提案，不让 Approval 的 autoflush
        # 在长时间 Agent 调查期间占用数据库写锁。最终审批记录与候选状态一起提交。
        self._generate_graph_patch(revision, change_set)
        self.session.add(
            WorldChangeApproval(
                change_set_id=change_set.id,
                approval_stage="profile",
                revision=change_set.revision,
                # 图提案生成后总哈希已改变；本阶段批准的仍是 Profile 本身。
                payload_hash=canonical_profile_patch_hash(
                    profile_patch=change_set.profile_patch,
                    base_graph_version_id=change_set.base_graph_version_id,
                    revision=change_set.revision,
                ),
                approved_by="local_user",
            )
        )
        change_set.status = "awaiting_graph_approval"
        revision.status = "graph_review"
        revision.session_revision += 1
        self.session.flush()
        return change_set

    def approve_graph(
        self,
        revision: PersonWorldRevisionSession,
        *,
        change_set_id: str,
        revision_number: int,
        payload_hash: str,
        expected_session_revision: int,
    ) -> WorldGraphChangeSet:
        self._ensure_base_is_current(revision)
        if expected_session_revision != revision.session_revision:
            raise RevisionStateError("会话已经更新，请刷新后再批准 Graph Patch")
        change_set = self._change_set(revision, change_set_id, revision_number, payload_hash)
        if revision.status != "graph_review" or change_set.status != "awaiting_graph_approval":
            raise RevisionStateError("当前状态不能批准 Graph Patch")
        profile_approval = self.session.scalar(
            select(WorldChangeApproval).where(
                WorldChangeApproval.change_set_id == change_set.id,
                WorldChangeApproval.approval_stage == "profile",
                WorldChangeApproval.revision == change_set.revision,
                WorldChangeApproval.payload_hash
                == canonical_profile_patch_hash(
                    profile_patch=change_set.profile_patch,
                    base_graph_version_id=change_set.base_graph_version_id,
                    revision=change_set.revision,
                ),
            )
        )
        if profile_approval is None:
            raise RevisionStateError("必须先批准 Profile Patch")
        self.session.add(
            WorldChangeApproval(
                change_set_id=change_set.id,
                approval_stage="graph",
                revision=change_set.revision,
                payload_hash=change_set.canonical_payload_hash,
                approved_by="local_user",
            )
        )
        for correction in self.session.scalars(
            select(WorldCorrection).where(WorldCorrection.approved_change_set_id == change_set.id)
        ):
            correction.status = "approved"
            correction.approved_at = datetime.now(UTC)
        change_set.status = "approved"
        revision.status = "approved"
        revision.session_revision += 1
        self.session.flush()
        return change_set

    def _generate_graph_patch(
        self,
        revision: PersonWorldRevisionSession,
        change_set: WorldGraphChangeSet,
    ) -> None:
        """仅在 Profile Patch 获批后生成最小、可独立审核的 Graph Patch。"""

        self._ensure_base_is_current(revision)
        if not isinstance(revision.understanding_payload, dict):
            raise RevisionStateError("纠正会话缺少已确认理解")
        graph = self.session.get(WorldGraphVersion, change_set.base_graph_version_id)
        if graph is None:
            raise RevisionStateError("基础图版本不存在")
        profile = (
            self.session.get(PersonWorldProfile, revision.base_profile_id)
            if revision.base_profile_id
            else None
        )
        source_ids = [
            value
            for value in revision.understanding_payload.get("source_message_ids", [])
            if isinstance(value, str)
        ]
        context = _source_message_context(self.session, revision.project_id, source_ids)
        if (
            profile is not None
            and profile.profile_schema_version == "v3"
            and not revision.understanding_payload.get("graph_change_requested", False)
        ):
            change_set.graph_operations = []
            change_set.regression_queries = []
            change_set.canonical_payload_hash = canonical_change_set_hash(
                profile_patch=change_set.profile_patch,
                graph_operations=[],
                regression_queries=[],
                base_graph_version_id=change_set.base_graph_version_id,
                revision=change_set.revision,
            )
            return
        from pathlib import Path

        from langchain_core.tools import StructuredTool

        from moonlightbox.agent_runtime.contracts import RegisteredTool, ToolContract
        from moonlightbox.agent_runtime.tool_execution import serial

        def search_graph(question: str):
            """检索当前待纠正图谱的相关实体、关系与上下文；只读。"""
            from ..node_scope import inherited_node_scope

            scope = inherited_node_scope(
                self.session,
                graph,
                revision.scope,
                profile.generation_summary if profile is not None else {},
            )
            result = self.lightrag.query(
                graph.workspace_key,
                question,
                mode="mix",
                top_k=16,
                chunk_top_k=8,
                max_total_tokens=6000,
                **({"temporal": {**scope.retrieval_scope(), "period": "before"}} if scope else {}),
            )
            return {
                "context": result.context,
                "references": [item.model_dump(mode="json") for item in result.references],
                "temporal_diagnostics": getattr(result, "temporal_diagnostics", None),
                "global_graph_clues": getattr(result, "global_graph_clues", None),
            }

        from ..tools.graph_query import GraphPatchSearchArgs

        tool = StructuredTool.from_function(search_graph, args_schema=GraphPatchSearchArgs)

        def validate_patch(result, submission_context):
            for operation in result.graph_operations:
                if not set(operation.source_message_ids).issubset(source_ids):
                    return "source_message_ids 只能引用当前确认范围提供的消息。"
            return None

        from .graph_input import GraphPatchInput, bind_graph_patch

        draft = run_submission_task(
            compiler=self.compiler,
            name="graph_patch",
            system_prompt=Path(__file__).parents[1].joinpath("prompts/graph_patch.md").read_text(),
            payload={
                "understanding": revision.understanding_payload,
                "approved_profile_patch": change_set.profile_patch,
                "current_profile": _profile_payload(profile),
                "source_messages": context,
            },
            result_model=GraphPatchDraft,
            input_model=GraphPatchInput,
            result_adapter=bind_graph_patch,
            owner_id=f"{revision.id}:{change_set.revision}",
            session=self.session,
            project_id=revision.project_id,
            input_revision=revision.input_revision,
            input_revision_resolver=lambda: self.session.scalar(
                select(PersonWorldRevisionSession.input_revision).where(
                    PersonWorldRevisionSession.id == revision.id,
                    PersonWorldRevisionSession.status != "cancelled",
                )
            ),
            validator=validate_patch,
            tools=(
                RegisteredTool(
                    tool,
                    ToolContract(
                        name=tool.name,
                        execution=serial("revision_graph_read", reason="共享图谱只读客户端"),
                        timeout_seconds=300,
                        max_result_chars=32000,
                    ),
                ),
            ),
        )
        allowed_ids = set(source_ids)
        operations: list[dict[str, object]] = []
        for operation in draft.graph_operations:
            sanitized = operation.model_copy(
                update={
                    "source_message_ids": [
                        item for item in operation.source_message_ids if item in allowed_ids
                    ]
                }
            )
            # 模型只描述意图；服务端读取当前图状态后签发 precondition hash，
            # 防止用户批准后的 workspace 被其他操作悄然改写。
            precondition = operation_precondition_state(
                self.lightrag, graph.workspace_key, sanitized
            )
            # 用户审核的 diff 必须展示服务端刚读取到的图状态，而不是让模型凭记忆
            # 填“旧描述”。这两个展示字段不参与 precondition hash，也不作为后续查询
            # 条件；真正执行仍由同一个 precondition hash 守护。
            before_description = _graph_state_description(precondition)
            updates: dict[str, object] = {}
            if before_description and sanitized.before_description is None:
                updates["before_description"] = before_description
            if (
                sanitized.operation_type.endswith("RELATION")
                and sanitized.relation_description
                and sanitized.after_description is None
            ):
                updates["after_description"] = sanitized.relation_description
            if updates:
                sanitized = sanitized.model_copy(update=updates)
            sanitized = sanitized.model_copy(
                update={"precondition_hash": canonical_graph_state_hash(precondition)}
            )
            operations.append(sanitized.model_dump(mode="json"))
        regression_queries = [
            {"query": item, "expected_change": "按用户确认的纠正回答"}
            for item in draft.regression_queries
        ]
        change_set.graph_operations = operations
        change_set.affected_entities = _affected_entities(operations)
        change_set.affected_relations = _affected_relations(operations)
        change_set.regression_queries = regression_queries
        change_set.canonical_payload_hash = canonical_change_set_hash(
            profile_patch=change_set.profile_patch,
            graph_operations=operations,
            regression_queries=regression_queries,
            base_graph_version_id=change_set.base_graph_version_id,
            revision=change_set.revision,
        )

    def cancel(
        self,
        revision: PersonWorldRevisionSession,
        *,
        expected_session_revision: int,
        idempotency_key: str,
    ) -> None:
        if revision.session_revision != expected_session_revision:
            raise RevisionStateError("会话已经更新，请刷新后再取消")
        existing = self.session.scalar(
            select(PersonWorldRevisionMessage).where(
                PersonWorldRevisionMessage.session_id == revision.id,
                PersonWorldRevisionMessage.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return
        if revision.status == "published":
            raise RevisionStateError("已发布的纠正不能取消；请创建撤销纠正")
        if revision.status in {"approved", "executing", "validating"}:
            raise RevisionStateError("候选图正在构建，完成验证后才能拒绝发布")
        change_set = (
            self.session.get(WorldGraphChangeSet, revision.graph_change_set_id)
            if revision.graph_change_set_id
            else None
        )
        if change_set is not None:
            change_set.status = "rejected"
            candidate = (
                self.session.get(WorldGraphVersion, change_set.candidate_graph_version_id)
                if change_set.candidate_graph_version_id
                else None
            )
            if candidate is not None and candidate.status != "ready":
                candidate.status = "rejected"
            for correction in self.session.scalars(
                select(WorldCorrection).where(
                    WorldCorrection.approved_change_set_id == change_set.id,
                    WorldCorrection.status.in_(["proposed", "approved"]),
                )
            ):
                correction.status = "revoked"
        revision.status = "cancelled"
        # 正在执行的回合会在返回边界读取 ``input_revision``，发现用户取消后不得再
        # 提交 Profile Patch 或 Graph Patch；不维护独立的通用 Agent 取消账本。
        revision.scope = {
            **dict(revision.scope or {}),
            "agent_stage": "cancelled",
            "agent_stage_progress": 1.0,
        }
        revision.session_revision += 1
        self.session.add(
            PersonWorldRevisionMessage(
                session_id=revision.id,
                role="system",
                kind="cancelled",
                session_revision=revision.session_revision,
                idempotency_key=idempotency_key,
                content="用户已取消本次纠正。",
                payload=None,
            )
        )
        self.session.flush()

    def _ensure_base_is_current(self, revision: PersonWorldRevisionSession) -> None:
        """用发布指针而非展示文字验证本轮纠正的乐观锁基线。

        这项检查只比较固定的 Graph/Profile UUID。它不阅读聊天正文、不会借助正则判断
        用户意图；一旦活动版本改变，旧 ContextSnapshot、Patch 和 Approval 都不得继续
        使用，必须由新会话在新基线上重新调查。
        """

        base_graph = self.session.get(WorldGraphVersion, revision.base_graph_version_id)
        publication = self.session.scalar(
            select(WorldPublication)
            .where(
                WorldPublication.project_id == revision.project_id,
                WorldPublication.status == "active",
                WorldPublication.node_boundary_hash
                == (revision.scope.get("node_scope") or {}).get("preview_hash"),
            )
            .order_by(WorldPublication.published_at.desc(), WorldPublication.id.desc())
        )
        stale_reason: str | None = None
        active_graph_id: str | None = None
        active_profile_id: str | None = None
        if revision.scope.get("base_profile_hash"):
            from .profile_v3 import revision_base_is_current

            if revision_base_is_current(self.session, revision, publication):
                return
            stale_reason = "v3_profile_or_publication_changed"
        elif base_graph is None:
            stale_reason = "base_graph_missing"
        elif publication is not None:
            active_graph_id = publication.graph_version_id
            active_profile_id = publication.profile_id
            if publication.graph_version_id != revision.base_graph_version_id:
                stale_reason = "active_graph_changed"
            elif publication.profile_id != revision.base_profile_id:
                stale_reason = "active_profile_changed"
        elif base_graph.status == "superseded":
            # 尚未创建 Publication 的旧项目兼容路径：一旦基础 Graph 已退役，同样
            # 不能让纠正沿用它的证据窗口。
            stale_reason = "base_graph_superseded"
        if stale_reason is None:
            return
        if revision.status != "stale":
            self._invalidate_stale_base(
                revision,
                reason=stale_reason,
                active_graph_id=active_graph_id,
                active_profile_id=active_profile_id,
            )
        raise RevisionBaseStaleError(
            "人物世界的活动版本已变化；本次纠正及其批准已失效，请基于最新版本重新开始"
        )

    def _invalidate_stale_base(
        self,
        revision: PersonWorldRevisionSession,
        *,
        reason: str,
        active_graph_id: str | None,
        active_profile_id: str | None,
    ) -> None:
        """原子作废旧基线的派生产物，不改动已经发布的 Graph/Profile。"""

        now = datetime.now(UTC)
        change_sets = list(
            self.session.scalars(
                select(WorldGraphChangeSet).where(
                    WorldGraphChangeSet.revision_session_id == revision.id,
                    WorldGraphChangeSet.status != "published",
                )
            )
        )
        for change_set in change_sets:
            change_set.status = "stale"
            candidate = (
                self.session.get(WorldGraphVersion, change_set.candidate_graph_version_id)
                if change_set.candidate_graph_version_id
                else None
            )
            if candidate is not None and candidate.status not in {"ready", "superseded"}:
                candidate.status = "rejected"
                candidate.completed_at = now
        for correction in self.session.scalars(
            select(WorldCorrection).where(
                WorldCorrection.revision_session_id == revision.id,
                WorldCorrection.status.in_(["proposed", "approved"]),
            )
        ):
            correction.status = "revoked"
        revision.scope = {
            **dict(revision.scope or {}),
            "stale_base": {
                "reason": reason,
                "base_graph_version_id": revision.base_graph_version_id,
                "base_profile_id": revision.base_profile_id,
                "active_graph_version_id": active_graph_id,
                "active_profile_id": active_profile_id,
                "invalidated_at": now.isoformat(),
            },
        }
        revision.status = "stale"
        revision.pending_turn_id = None
        revision.understanding_payload = None
        revision.understanding_payload_hash = None
        revision.session_revision += 1
        self.session.add(
            PersonWorldRevisionMessage(
                session_id=revision.id,
                role="system",
                kind="base_stale",
                session_revision=revision.session_revision,
                content="活动人物世界版本已变化，原纠正范围、理解与批准已失效。请基于最新版本重新开始。",
                payload={
                    "reason": reason,
                    "base_graph_version_id": revision.base_graph_version_id,
                    "base_profile_id": revision.base_profile_id,
                    "active_graph_version_id": active_graph_id,
                    "active_profile_id": active_profile_id,
                },
            )
        )
        self.session.flush()

    def _invalidate_unapproved_outputs(self, revision: PersonWorldRevisionSession) -> None:
        """用户改变意图或范围时作废下游草稿，不触碰活动图谱。"""

        previous_change_set = (
            self.session.get(WorldGraphChangeSet, revision.graph_change_set_id)
            if revision.graph_change_set_id
            else None
        )
        if previous_change_set is None:
            return
        previous_change_set.status = "rejected"
        candidate = (
            self.session.get(WorldGraphVersion, previous_change_set.candidate_graph_version_id)
            if previous_change_set.candidate_graph_version_id
            else None
        )
        if candidate is not None and candidate.status != "ready":
            candidate.status = "rejected"
        for correction in self.session.scalars(
            select(WorldCorrection).where(
                WorldCorrection.approved_change_set_id == previous_change_set.id,
                WorldCorrection.status.in_(["proposed", "approved"]),
            )
        ):
            correction.status = "revoked"

    def _correction_scope(self, revision: PersonWorldRevisionSession) -> dict[str, object]:
        """把 Revision Scope 中的稳定 Claim 转成可供重编译筛选的结构边界。

        这里不理解或匹配任何中文陈述。数据库的 ``primary_domain``、``fact_type`` 和
        ``subject_kind`` 是 Claim 创建时已经固化的结构字段；没有 Claim 的自由纠正保持
        ``provisional``，不会伪装成可自动归类的已知人物事实。
        """

        scope = dict(revision.scope or {})
        claim_ids = _string_list(scope.get("selected_claim_ids"))
        claims = (
            list(
                self.session.scalars(
                    select(AtomicWorldClaim).where(
                        AtomicWorldClaim.project_id == revision.project_id,
                        AtomicWorldClaim.graph_version_id == revision.base_graph_version_id,
                        AtomicWorldClaim.id.in_(claim_ids),
                    )
                )
            )
            if claim_ids
            else []
        )
        domains = list(
            dict.fromkeys(
                claim.primary_domain
                if claim.primary_domain and claim.primary_domain != "unknown"
                else claim.profile_section.split(".", 1)[0]
                for claim in claims
                if claim.profile_section
            )
        )
        return {
            "claim_ids": [claim.id for claim in claims],
            "node_scope": scope.get("node_scope"),
            "primary_domains": domains,
            "fact_types": list(
                dict.fromkeys(
                    claim.fact_type
                    if claim.fact_type and claim.fact_type != "unknown"
                    else claim.predicate
                    for claim in claims
                )
            ),
            "subject_kinds": list(dict.fromkeys(claim.subject_kind for claim in claims)),
            "graph_object_refs": _object_refs(scope.get("included_graph_objects")),
            "provisional": not claims,
        }

    def _change_set(
        self,
        revision: PersonWorldRevisionSession,
        change_set_id: str,
        revision_number: int,
        payload_hash: str,
    ) -> WorldGraphChangeSet:
        change_set = self.session.get(WorldGraphChangeSet, change_set_id)
        if (
            change_set is None
            or change_set.revision_session_id != revision.id
            or change_set.project_id != revision.project_id
        ):
            raise RevisionStateError("变更集不存在")
        if (
            change_set.revision != revision_number
            or change_set.canonical_payload_hash != payload_hash
        ):
            raise RevisionStateError("变更集已经变化，请重新查看并确认")
        return change_set


def _active_profile_and_graph(
    session: Session, project_id: str
) -> tuple[PersonWorldProfile | None, WorldGraphVersion | None]:
    publication = session.scalar(
        select(WorldPublication)
        .where(
            WorldPublication.project_id == project_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash.is_(None),
        )
        .order_by(WorldPublication.published_at.desc())
    )
    if publication is not None:
        return (
            session.get(PersonWorldProfile, publication.profile_id),
            session.get(WorldGraphVersion, publication.graph_version_id),
        )
    graph = session.scalar(
        select(WorldGraphVersion)
        .where(
            WorldGraphVersion.project_id == project_id,
            WorldGraphVersion.status.in_(["ready", "awaiting_profile_review"]),
        )
        .order_by(WorldGraphVersion.created_at.desc())
    )
    profile = (
        session.scalar(
            select(PersonWorldProfile)
            .where(PersonWorldProfile.node_boundary_hash.is_(None))
            .where(
                PersonWorldProfile.project_id == project_id,
                PersonWorldProfile.graph_version_id == graph.id,
            )
        )
        if graph is not None
        else None
    )
    return profile, graph


def graph_bundles(session: Session, graph_version_id: str) -> list[Any]:
    from moonlightbox.world.models import ConversationBundle

    return list(
        session.scalars(
            select(ConversationBundle).where(
                ConversationBundle.graph_version_id == graph_version_id
            )
        )
    )


def _profile_payload(profile: PersonWorldProfile | None) -> dict[str, object]:
    if profile is None:
        return {}
    if profile.profile_schema_version == "v3":
        return dict(profile.profile_v3 or {})
    if profile.profile_schema_version == "v2" and isinstance(profile.profile_v2, dict):
        # v2 是 Revision 的唯一人物事实读模型。去掉内部 Claim UUID 后再交给模型：具体
        # 修改对象仅由服务器根据 selected_scope_items 的稳定位置绑定，不能让模型复制 ID。
        return _without_claim_ids(profile.profile_v2)
    return {
        "identity": profile.identity,
        "work_and_education": profile.work_and_education,
        "places": profile.places,
        "social_relationships": profile.social_relationships,
        "preferences": profile.preferences,
        "recurring_activities": profile.recurring_activities,
        "routine_summary": profile.routine_summary,
        "life_phases": profile.life_phases,
        "relationship_with_user": profile.relationship_with_user,
        "important_events": profile.important_events,
        "unresolved_candidates": profile.unresolved_candidates,
    }


def _without_claim_ids(value: object) -> dict[str, object]:
    """递归移除模型不应看到的数据库事实身份；不改变任何可读事实内容。"""

    def sanitize(item: object) -> object:
        if isinstance(item, dict):
            return {key: sanitize(value) for key, value in item.items() if key != "claim_id"}
        if isinstance(item, list):
            return [sanitize(child) for child in item]
        return item

    cleaned = sanitize(value)
    return dict(cleaned) if isinstance(cleaned, dict) else {}


def _scope_proposal_candidates(scope: object, related_claim_ids: list[str]) -> list[str]:
    """从 Snapshot 的有序相关 Claim 中移除已选项，得到模型序号的后端绑定。"""

    selected = set(_string_list(scope.get("selected_claim_ids")) if isinstance(scope, dict) else [])
    return [item for item in related_claim_ids if item not in selected]


def _resolve_scope_proposal_items(scope: dict[str, object], item_ids: list[int]) -> list[str]:
    pending = scope.get("pending_scope_proposal")
    if not isinstance(pending, dict):
        if item_ids:
            raise RevisionStateError("没有可供确认的范围提案")
        return []
    items = pending.get("items")
    if not isinstance(items, list):
        raise RevisionStateError("范围提案已损坏，请重新发起调查")
    requested = set(item_ids)
    return [
        item["claim_id"]
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("proposal_item"), int)
        and item["proposal_item"] in requested
        and isinstance(item.get("claim_id"), str)
    ]


def _source_message_context(
    session: Session, project_id: str, message_ids: list[str]
) -> list[dict[str, object]]:
    if not message_ids:
        return []
    return [
        {
            "message_id": message.id,
            "timestamp": message.timestamp.isoformat(),
            "participant": participant.name,
            "role": participant.role,
            "content": message.content,
        }
        for message, participant in session.execute(
            select(Message, Participant)
            .join(Participant, Participant.id == Message.participant_id)
            .where(Message.project_id == project_id, Message.id.in_(message_ids))
            .order_by(Message.timestamp.asc(), Message.id.asc())
        ).all()
    ]


def _dedupe_message_payloads(items: list[Any]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("message_id"), str):
            continue
        if item["message_id"] in seen:
            continue
        seen.add(item["message_id"])
        result.append(item)
    return result


def _affected_entities(operations: list[dict[str, object]]) -> list[str]:
    values: list[str] = []
    for operation in operations:
        for key in ("entity_name", "source_entity", "target_entity"):
            value = operation.get(key)
            if isinstance(value, str) and value:
                values.append(value)
        sources = operation.get("source_entities")
        if isinstance(sources, list):
            values.extend(item for item in sources if isinstance(item, str) and item)
    return list(dict.fromkeys(values))


def _affected_relations(operations: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {"source_entity": source, "target_entity": target}
        for item in operations
        for source, target in [(item.get("source_entity"), item.get("target_entity"))]
        if isinstance(source, str) and isinstance(target, str) and source and target
    ]


def _graph_state_description(value: object) -> str | None:
    """从已签发 precondition 的结构状态提取可读旧描述，不解释消息或图谱语义。"""

    if not isinstance(value, dict):
        return None
    description = value.get("description")
    return description if isinstance(description, str) and description.strip() else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str)))


def _object_refs(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    return [
        {key: item[key] for key in sorted(item) if isinstance(item[key], str)}
        for item in value
        if isinstance(item, dict) and all(isinstance(key, str) for key in item)
    ]


def _merge_values(existing: list[str], incoming: list[str]) -> list[str]:
    return list(dict.fromkeys([*existing, *(item for item in incoming if isinstance(item, str))]))


def _object_ref_key(value: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(value.items()))


def _merge_object_refs(
    existing: list[dict[str, str]], incoming: list[dict[str, str]]
) -> list[dict[str, str]]:
    result = list(existing)
    seen = {_object_ref_key(item) for item in existing}
    for item in incoming:
        normalized = {key: value for key, value in sorted(item.items()) if isinstance(value, str)}
        key = _object_ref_key(normalized)
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def _remove_object_refs(
    existing: list[dict[str, str]], removed: list[dict[str, str]]
) -> list[dict[str, str]]:
    removed_keys = {
        _object_ref_key({key: value for key, value in item.items() if isinstance(value, str)})
        for item in removed
    }
    return [item for item in existing if _object_ref_key(item) not in removed_keys]
