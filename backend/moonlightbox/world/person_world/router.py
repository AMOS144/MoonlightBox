"""PersonWorldAgent 调查结果审核、用户纠正和候选图发布 API。"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.imports.models import Message, Participant
from moonlightbox.jobs.service import InvalidJobTransitionError, JobNotFoundError, JobService
from moonlightbox.world.client import LightRAGSidecarClient
from moonlightbox.world.models import (
    PersonWorldAgentRun,
    PersonWorldProfile,
    PersonWorldRevisionContextSnapshot,
    PersonWorldRevisionMessage,
    PersonWorldRevisionSession,
    PersonWorldSectionTask,
    WorldEvidence,
    WorldGraphChangeSet,
    WorldGraphVersion,
    WorldPublication,
)

from .jobs import enqueue_graph_patch_job, enqueue_profile_recompile_job
from .publication import (
    WorldPublicationError,
    WorldPublicationStaleBaseError,
    approve_initial_profile,
    enqueue_publication_analysis,
    publish_change_set,
)
from .review import (
    REVISION_TURN_JOB_KIND,
    PersonWorldReviewService,
    RevisionBaseStaleError,
    RevisionStateError,
    enqueue_revision_turn_job,
)
from .review.profile_preview_jobs import enqueue_profile_preview
from .schemas import SelectedProfileStatement
from .section_retry import SectionRetryStateError, enqueue_section_retry_job

_CONTEXT_MESSAGE_IDS_QUERY = Query(default_factory=list)


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class InitialProfileApproval(StrictRequest):
    graph_version_id: str
    profile_id: str


class NodeProfileCompilation(StrictRequest):
    graph_version_id: str = Field(min_length=1, max_length=36)
    investigation_id: str = Field(min_length=1, max_length=36)
    preview_hash: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=8, max_length=120)


class NodeProfileApproval(StrictRequest):
    approval_hash: str = Field(min_length=64, max_length=64)


class RevisionCreate(StrictRequest):
    base_profile_id: str | None = None
    message: str = Field(min_length=1, max_length=5000)
    idempotency_key: str = Field(min_length=8, max_length=120)
    source_message_ids: list[str] = Field(default_factory=list, max_length=80)
    selected_statements: list[SelectedProfileStatement] = Field(default_factory=list, max_length=40)


class RevisionMessageCreate(StrictRequest):
    message: str = Field(min_length=1, max_length=5000)
    session_revision: int = Field(ge=1)
    in_reply_to_turn_id: str | None = Field(default=None, max_length=36)
    idempotency_key: str = Field(min_length=8, max_length=120)


class RevisionScopeChange(StrictRequest):
    """范围只接收稳定 Claim/图对象引用，绝不接收展示文本反查。"""

    action: str = Field(pattern=r"^(include|exclude)$")
    claim_ids: list[str] = Field(default_factory=list, max_length=40)
    proposal_item_ids: list[int] = Field(default_factory=list, max_length=8)
    graph_object_refs: list[dict[str, str]] = Field(default_factory=list, max_length=40)
    session_revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=120)


class UnderstandingConfirmation(StrictRequest):
    understanding_revision: int = Field(ge=1)
    understanding_payload_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    session_revision: int = Field(ge=1)


class ChangeSetApproval(StrictRequest):
    change_set_id: str
    revision: int = Field(ge=1)
    payload_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    session_revision: int = Field(ge=1)


class PublishRequest(StrictRequest):
    change_set_id: str
    session_revision: int = Field(ge=1)


class RevisionCancel(StrictRequest):
    session_revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=120)


class RevisionAgentRetry(StrictRequest):
    """失败回合重试只引用会话版本，不能由客户端拼接或覆盖 Agent 上下文。"""

    session_revision: int = Field(ge=1)
    idempotency_key: str = Field(min_length=8, max_length=120)


class SectionRetryRequest(StrictRequest):
    """重试只引用稳定运行/栏目身份，不允许客户端提交或篡改调查内容。"""

    idempotency_key: str = Field(min_length=8, max_length=120)


def create_person_world_agent_router(database: Database, settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/projects/{project_id}/world-agent", tags=["world-agent"])

    @router.get("/profile-v3/catalog")
    def get_profile_v3_catalog(project_id: str):
        from .contracts.display import profile_editor_catalog

        return profile_editor_catalog()

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.post("/node-drafts/{draft_id}/review")
    def prepare_node_review(project_id: str, draft_id: str, session: SessionDependency):
        from .node_review import node_review_payload, review_node_draft

        try:
            profile = review_node_draft(session, project_id=project_id, draft_id=draft_id)
            session.commit()
            return node_review_payload(session, profile)
        except LookupError as error:
            raise HTTPException(404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(409, detail=str(error)) from error

    @router.get("/node-profiles/{profile_id}")
    def read_node_profile(project_id: str, profile_id: str, session: SessionDependency):
        from .node_review import node_review_payload

        profile = session.get(PersonWorldProfile, profile_id)
        if profile is None or profile.project_id != project_id or not profile.node_boundary_hash:
            raise HTTPException(404, detail="节点画像不存在")
        return node_review_payload(session, profile)

    @router.post("/node-profiles/{profile_id}/approve")
    def approve_node(
        project_id: str, profile_id: str, payload: NodeProfileApproval, session: SessionDependency
    ):
        from .node_review import approve_node_profile

        try:
            publication = approve_node_profile(
                session,
                project_id=project_id,
                profile_id=profile_id,
                approval_hash=payload.approval_hash,
            )
            session.commit()
            return {"publication_id": publication.id, "profile_id": publication.profile_id}
        except LookupError as error:
            raise HTTPException(404, detail=str(error)) from error
        except (ValueError, WorldPublicationError) as error:
            session.rollback()
            raise HTTPException(409, detail=str(error)) from error

    @router.post("/node-compilations", status_code=status.HTTP_202_ACCEPTED)
    def compile_node(project_id: str, request: NodeProfileCompilation, session: SessionDependency):
        from .node_jobs import enqueue_node_compilation

        try:
            job = enqueue_node_compilation(
                JobService(session),
                project_id=project_id,
                graph_id=request.graph_version_id,
                investigation_id=request.investigation_id,
                preview_hash=request.preview_hash,
                idempotency_key=request.idempotency_key,
            )
        except LookupError as error:
            raise HTTPException(404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(409, detail=str(error)) from error
        return {"job_id": job.id, "status": job.status}

    @router.get("/node-compilations")
    def latest_node_compilation(project_id: str, preview_hash: str, session: SessionDependency):
        from moonlightbox.jobs.models import Job

        from .node_jobs import NODE_PROFILE_JOB_KIND

        job = session.scalar(
            select(Job)
            .where(
                Job.kind == NODE_PROFILE_JOB_KIND,
                Job.payload["project_id"].as_string() == project_id,
                Job.payload["preview_hash"].as_string() == preview_hash,
            )
            .order_by(Job.created_at.desc(), Job.id.desc())
        )
        return {"job_id": job.id} if job else None

    @router.get("/node-compilations/{job_id}")
    def read_node_compilation(project_id: str, job_id: str, session: SessionDependency):
        from moonlightbox.jobs.models import Job
        from moonlightbox.world.models import (
            PersonWorldAgentRun,
            PersonWorldProfileDraft,
            PersonWorldSectionTask,
        )

        from .node_jobs import NODE_PROFILE_JOB_KIND

        job = session.get(Job, job_id)
        if (
            job is None
            or job.kind != NODE_PROFILE_JOB_KIND
            or job.payload.get("project_id") != project_id
        ):
            raise HTTPException(404, detail="节点编译不存在")
        draft_id = (job.checkpoint or {}).get("profile_draft_id")
        draft = session.get(PersonWorldProfileDraft, draft_id) if draft_id else None
        # 栏目任务挂在编译运行上;checkpoint 里的 agent_run_id 要等阶段回调才写入,
        # 运行前期用 resume_key 反查,让前端从第一节就能看到逐栏进度。
        agent_run_id = (job.checkpoint or {}).get("agent_run_id")
        if agent_run_id is None:
            agent_run_id = session.scalar(
                select(PersonWorldAgentRun.id)
                .where(
                    PersonWorldAgentRun.mode == "node_compile",
                    PersonWorldAgentRun.state["resume_key"].as_string() == f"job:{job.id}",
                )
                .order_by(PersonWorldAgentRun.created_at.desc())
                .limit(1)
            )
        section_tasks = []
        if agent_run_id is not None:
            section_tasks = [
                {
                    "section": task.section,
                    "status": task.status,
                    "research_round": task.research_round,
                    "unresolved_questions": list(task.unresolved_questions or []),
                    "error_code": task.error_code,
                }
                for task in session.scalars(
                    select(PersonWorldSectionTask)
                    .where(PersonWorldSectionTask.agent_run_id == agent_run_id)
                    .order_by(PersonWorldSectionTask.section.asc())
                )
            ]
        return {
            "job_id": job.id,
            "status": job.status,
            "progress": job.progress,
            "error_code": job.error_code,
            "checkpoint": job.checkpoint,
            "node_scope": job.payload["node_scope"],
            "section_tasks": section_tasks,
            "draft": {
                "id": draft.id,
                "status": draft.status,
                "profile_v3": draft.profile_v3,
                "investigation_report": draft.investigation_report,
                "generation_summary": draft.generation_summary,
            }
            if draft
            else None,
        }

    def raise_revision_conflict(error: RevisionStateError, session: Session) -> None:
        """陈旧基线需要持久化作废状态；普通乐观锁冲突则保持原有回滚语义。"""

        if isinstance(error, RevisionBaseStaleError):
            session.commit()
        else:
            session.rollback()
        raise HTTPException(status_code=409, detail=str(error)) from error

    @router.post("/recompile", status_code=status.HTTP_202_ACCEPTED)
    def recompile_from_frozen_graph(
        project_id: str,
        session: SessionDependency,
    ) -> dict[str, str]:
        """基于当前完整图谱创建独立候选，不重新抽取聊天或改写活动图谱。"""

        publication = session.scalar(
            select(WorldPublication)
            .where(
                WorldPublication.project_id == project_id,
                WorldPublication.status == "active",
                WorldPublication.node_boundary_hash.is_(None),
            )
            .order_by(WorldPublication.published_at.desc(), WorldPublication.id.desc())
        )
        base = (
            session.get(WorldGraphVersion, publication.graph_version_id)
            if publication is not None
            else session.scalar(
                select(WorldGraphVersion)
                .where(
                    WorldGraphVersion.project_id == project_id,
                    WorldGraphVersion.status == "ready",
                )
                .order_by(WorldGraphVersion.completed_at.desc(), WorldGraphVersion.id.desc())
            )
        )
        legacy_candidate = session.scalar(
            select(WorldGraphVersion)
            .join(PersonWorldProfile, PersonWorldProfile.graph_version_id == WorldGraphVersion.id)
            .where(
                WorldGraphVersion.project_id == project_id,
                WorldGraphVersion.status == "awaiting_profile_review",
                PersonWorldProfile.profile_schema_version.in_(["v1", "v2"]),
            )
            .order_by(WorldGraphVersion.created_at.desc())
        )
        if legacy_candidate is not None:
            base = legacy_candidate
        if base is None:
            raise HTTPException(status_code=409, detail="没有可冻结的已完成 LightRAG 图谱")
        try:
            job, candidate = enqueue_profile_recompile_job(
                JobService(session),
                settings=settings,
                base=base,
            )
        except ValueError as error:
            session.rollback()
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "job_id": job.id,
            "candidate_graph_version_id": candidate.id,
            "status": candidate.status,
        }

    @router.get("/draft")
    def get_draft(project_id: str, session: SessionDependency) -> dict[str, object]:
        graph = session.scalar(
            select(WorldGraphVersion)
            .where(
                WorldGraphVersion.project_id == project_id,
                WorldGraphVersion.status == "awaiting_profile_review",
            )
            .order_by(WorldGraphVersion.created_at.desc(), WorldGraphVersion.id.desc())
        )
        if graph is None:
            raise HTTPException(status_code=404, detail="没有等待审核的 PersonWorldProfile")
        profile = session.scalar(
            select(PersonWorldProfile)
            .where(PersonWorldProfile.node_boundary_hash.is_(None))
            .where(
                PersonWorldProfile.project_id == project_id,
                PersonWorldProfile.graph_version_id == graph.id,
            )
        )
        if profile is None:
            raise HTTPException(status_code=404, detail="PersonWorldAgent 草稿不存在")
        return _profile_draft_payload(session, profile, graph)

    @router.post("/draft/approve", status_code=status.HTTP_200_OK)
    def approve_draft(
        project_id: str,
        payload: InitialProfileApproval,
        session: SessionDependency,
    ) -> dict[str, object]:
        try:
            publication = approve_initial_profile(
                session,
                project_id=project_id,
                graph_version_id=payload.graph_version_id,
                profile_id=payload.profile_id,
            )
            graph = session.get(WorldGraphVersion, publication.graph_version_id)
            if graph is None:
                raise WorldPublicationError("发布后的图版本不存在")
            analysis_job = (
                None
                if publication.node_boundary_hash
                else enqueue_publication_analysis(session, settings=settings, graph=graph)
            )
            session.commit()
        except WorldPublicationError as error:
            if isinstance(error, WorldPublicationStaleBaseError):
                session.commit()
            else:
                session.rollback()
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "publication_id": publication.id,
            "graph_version_id": publication.graph_version_id,
            "profile_id": publication.profile_id,
            "analysis_job_id": analysis_job.id if analysis_job else None,
            "status": publication.status,
        }

    @router.post(
        "/runs/{agent_run_id}/sections/{section}/retry",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def retry_failed_section(
        project_id: str,
        agent_run_id: str,
        section: str,
        payload: SectionRetryRequest,
        session: SessionDependency,
    ) -> dict[str, object]:
        """仅在候选审核阶段重新调查一个失败栏目。"""

        run = session.get(PersonWorldAgentRun, agent_run_id)
        if run is None or run.project_id != project_id:
            raise HTTPException(status_code=404, detail="PersonWorld 调查运行不存在")
        task = session.scalar(
            select(PersonWorldSectionTask).where(
                PersonWorldSectionTask.agent_run_id == run.id,
                PersonWorldSectionTask.section == section,
            )
        )
        if task is None:
            raise HTTPException(status_code=404, detail="PersonWorld 栏目任务不存在")
        try:
            job = enqueue_section_retry_job(
                JobService(session),
                settings=settings,
                run=run,
                task=task,
                idempotency_key=payload.idempotency_key,
            )
        except SectionRetryStateError as error:
            session.rollback()
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "job_id": job.id,
            "agent_run_id": run.id,
            "section": task.section,
            "status": task.status,
            "retry_attempt": dict(task.result_summary or {}).get("retry_attempt", 0),
        }

    @router.post("/sessions", status_code=status.HTTP_201_CREATED)
    def create_revision(
        project_id: str,
        payload: RevisionCreate,
        session: SessionDependency,
    ) -> dict[str, object]:
        with _review_service(session, settings) as service:
            try:
                revision = service.create_session(
                    project_id=project_id,
                    user_message=payload.message,
                    idempotency_key=payload.idempotency_key,
                    source_message_ids=payload.source_message_ids,
                    selected_statements=payload.selected_statements,
                    base_profile_id=payload.base_profile_id,
                    run_agent=False,
                )
                session.flush()
                job = enqueue_revision_turn_job(
                    JobService(session), settings=settings, revision=revision
                )
                session.commit()
            except RevisionStateError as error:
                raise_revision_conflict(error, session)
        return {**_revision_payload(session, revision), "agent_job_id": job.id}

    @router.get("/sessions")
    def list_revisions(
        project_id: str,
        session: SessionDependency,
        include_terminal: bool = False,
    ) -> list[dict[str, object]]:
        statement = (
            select(PersonWorldRevisionSession)
            .where(PersonWorldRevisionSession.project_id == project_id)
            .order_by(
                PersonWorldRevisionSession.updated_at.desc(),
                PersonWorldRevisionSession.id.desc(),
            )
        )
        if not include_terminal:
            statement = statement.where(
                PersonWorldRevisionSession.status.not_in(
                    ["published", "cancelled", "failed", "stale"]
                )
            )
        return [_revision_payload(session, item) for item in session.scalars(statement)]

    @router.get("/sessions/{revision_session_id}")
    def get_revision(
        project_id: str,
        revision_session_id: str,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        return _revision_payload(session, revision)

    @router.get("/sessions/{revision_session_id}/events")
    async def revision_events(
        project_id: str,
        revision_session_id: str,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        """推送修订会话的版本化状态；断线客户端可用 Last-Event-ID 无损续接。

        事件正文复用普通会话读模型，客户端仍以 ``session_revision`` 做乐观锁。这里不
        持有请求 Session 跨越等待周期，避免慢客户端长期占用数据库连接。
        """

        try:
            seen_revision = int(last_event_id) if last_event_id is not None else -1
        except ValueError:
            seen_revision = -1

        async def stream() -> AsyncIterator[str]:
            nonlocal seen_revision
            heartbeat = 0
            while True:
                with Session(database.engine) as event_session:
                    revision = event_session.get(PersonWorldRevisionSession, revision_session_id)
                    if revision is None or revision.project_id != project_id:
                        yield "event: closed\ndata: {}\n\n"
                        return
                    if revision.session_revision > seen_revision:
                        payload = _revision_payload(event_session, revision)
                        seen_revision = revision.session_revision
                        yield (
                            f"id: {seen_revision}\n"
                            "event: revision\n"
                            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        )
                        heartbeat = 0
                    else:
                        heartbeat += 1
                        if heartbeat >= 15:
                            yield ": keepalive\n\n"
                            heartbeat = 0
                await asyncio.sleep(1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/sessions/{revision_session_id}/context-snapshots/{snapshot_id}")
    def get_context_snapshot(
        project_id: str,
        revision_session_id: str,
        snapshot_id: str,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        snapshot = session.get(PersonWorldRevisionContextSnapshot, snapshot_id)
        if snapshot is None or snapshot.session_id != revision.id:
            raise HTTPException(status_code=404, detail="纠正上下文快照不存在")
        # 此接口给右侧工作栏解释 Agent 读了哪些来源；原消息正文仍通过现有的
        # 受项目和 GraphVersion 约束的来源接口按需展开，不能在快照元数据中广播。
        return {
            "id": snapshot.id,
            "session_id": snapshot.session_id,
            "session_revision": snapshot.session_revision,
            "base_graph_version_id": snapshot.base_graph_version_id,
            "base_profile_id": snapshot.base_profile_id,
            "scope": snapshot.scope,
            "evidence_message_ids": snapshot.evidence_message_ids,
            "related_claim_ids": snapshot.related_claim_ids,
            "graph_manifest": snapshot.graph_manifest,
            "context_hash": snapshot.context_hash,
            "created_at": snapshot.created_at.isoformat(),
        }

    @router.get("/sessions/{revision_session_id}/context-snapshots/{snapshot_id}/messages")
    def get_context_snapshot_messages(
        project_id: str,
        revision_session_id: str,
        snapshot_id: str,
        session: SessionDependency,
        message_ids: list[str] = _CONTEXT_MESSAGE_IDS_QUERY,
    ) -> list[dict[str, object]]:
        """按冻结快照读取审核证据，不能用该接口任意枚举项目消息。"""

        revision = _revision(session, project_id, revision_session_id)
        snapshot = session.get(PersonWorldRevisionContextSnapshot, snapshot_id)
        if snapshot is None or snapshot.session_id != revision.id:
            raise HTTPException(status_code=404, detail="纠正上下文快照不存在")
        allowed = list(dict.fromkeys(snapshot.evidence_message_ids or []))
        allowed_set = set(allowed)
        requested = list(dict.fromkeys(message_ids)) if message_ids else allowed
        if any(item not in allowed_set for item in requested):
            raise HTTPException(status_code=403, detail="请求的证据不属于当前纠正快照")
        if not requested:
            return []
        # Snapshot 已冻结的相邻消息用于判断指代、引述和事件时间。只从同一 Graph
        # Version 的 Evidence 账本取前后窗口，再与 Snapshot 白名单求交集；不能按
        # 文字、时间或任意项目消息扩展范围。
        context_ids = {
            context_id
            for evidence in session.scalars(
                select(WorldEvidence).where(
                    WorldEvidence.graph_version_id == snapshot.base_graph_version_id,
                    WorldEvidence.message_id.in_(requested),
                )
            )
            for context_id in [
                *list(evidence.context_before_ids or []),
                *list(evidence.context_after_ids or []),
            ]
            if context_id in allowed_set
        }
        expanded_ids = list(dict.fromkeys([*requested, *context_ids]))
        requested_set = set(requested)
        rows = session.execute(
            select(Message, Participant)
            .join(Participant, Participant.id == Message.participant_id)
            .where(Message.project_id == project_id, Message.id.in_(expanded_ids))
            .order_by(Message.timestamp.asc(), Message.id.asc())
        ).all()
        return [
            {
                "message_id": message.id,
                "timestamp": message.timestamp.isoformat(),
                "participant": participant.name,
                "role": participant.role,
                "kind": message.kind,
                "content": message.content,
                "is_focus": message.id in requested_set,
            }
            for message, participant in rows
        ]

    @router.post("/sessions/{revision_session_id}/messages")
    def add_revision_message(
        project_id: str,
        revision_session_id: str,
        payload: RevisionMessageCreate,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        with _review_service(session, settings) as service:
            try:
                revision = service.add_user_message(
                    revision,
                    payload.message,
                    expected_session_revision=payload.session_revision,
                    in_reply_to_turn_id=payload.in_reply_to_turn_id,
                    idempotency_key=payload.idempotency_key,
                    run_agent=False,
                )
                session.flush()
                job = enqueue_revision_turn_job(
                    JobService(session), settings=settings, revision=revision
                )
                session.commit()
            except RevisionStateError as error:
                raise_revision_conflict(error, session)
        return {**_revision_payload(session, revision), "agent_job_id": job.id}

    @router.post("/sessions/{revision_session_id}/confirm-understanding")
    def confirm_understanding(
        project_id: str,
        revision_session_id: str,
        payload: UnderstandingConfirmation,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        with _review_service(session, settings) as service:
            try:
                change_set = service.confirm_understanding(
                    revision,
                    understanding_revision=payload.understanding_revision,
                    understanding_payload_hash=payload.understanding_payload_hash,
                    expected_session_revision=payload.session_revision,
                    defer_v3=True,
                )
                if change_set is None:
                    job = enqueue_profile_preview(JobService(session), revision=revision)
                    revision.scope = {**revision.scope, "active_agent_job_id": job.id}
                session.commit()
            except RevisionStateError as error:
                raise_revision_conflict(error, session)
        return (
            _revision_payload(session, revision)
            if change_set is None
            else _change_set_payload(change_set)
        )

    @router.post("/sessions/{revision_session_id}/scope")
    def change_revision_scope(
        project_id: str,
        revision_session_id: str,
        payload: RevisionScopeChange,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        with _review_service(session, settings) as service:
            try:
                revision = service.change_scope(
                    revision,
                    action=payload.action,
                    claim_ids=payload.claim_ids,
                    graph_object_refs=payload.graph_object_refs,
                    proposal_item_ids=payload.proposal_item_ids,
                    expected_session_revision=payload.session_revision,
                    idempotency_key=payload.idempotency_key,
                    run_agent=False,
                )
                session.flush()
                job = enqueue_revision_turn_job(
                    JobService(session), settings=settings, revision=revision
                )
                session.commit()
            except RevisionStateError as error:
                raise_revision_conflict(error, session)
        return {**_revision_payload(session, revision), "agent_job_id": job.id}

    @router.get("/sessions/{revision_session_id}/profile-diff")
    def profile_diff(
        project_id: str,
        revision_session_id: str,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        change_set = _revision_change_set(session, revision)
        if change_set is None:
            raise HTTPException(status_code=404, detail="共同理解尚未确认，暂无 Profile Patch")
        return {
            "session_revision": revision.session_revision,
            "status": revision.status,
            "change_set": _change_set_payload(change_set),
            "profile_patch": change_set.profile_patch,
        }

    @router.get("/sessions/{revision_session_id}/graph-diff")
    def graph_diff(
        project_id: str,
        revision_session_id: str,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        change_set = _revision_change_set(session, revision)
        if change_set is None or revision.status in {
            "exploring",
            "waiting_for_user",
            "profile_review",
        }:
            raise HTTPException(status_code=404, detail="Profile Patch 批准前不生成 Graph Patch")
        return {
            "session_revision": revision.session_revision,
            "status": revision.status,
            "change_set": _change_set_payload(change_set),
            "graph_operations": change_set.graph_operations,
            "affected_entities": change_set.affected_entities,
            "affected_relations": change_set.affected_relations,
        }

    @router.get("/sessions/{revision_session_id}/validation")
    def validation_report(
        project_id: str,
        revision_session_id: str,
        session: SessionDependency,
    ) -> dict[str, object]:
        """独立返回候选图验证结果，前端无需从完整 Session 状态猜测。"""

        revision = _revision(session, project_id, revision_session_id)
        change_set = (
            session.get(WorldGraphChangeSet, revision.graph_change_set_id)
            if revision.graph_change_set_id
            else None
        )
        execution = dict(change_set.execution_result or {}) if change_set is not None else {}
        validation = execution.get("validation")
        return {
            "revision_session_id": revision.id,
            "graph_change_set_id": change_set.id if change_set is not None else None,
            "status": change_set.status if change_set is not None else "not_started",
            "validation": validation if isinstance(validation, dict | list) else None,
        }

    @router.post("/sessions/{revision_session_id}/approve-profile")
    def approve_profile_patch(
        project_id: str,
        revision_session_id: str,
        payload: ChangeSetApproval,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        with _review_service(session, settings) as service:
            try:
                change_set = service.approve_profile(
                    revision,
                    change_set_id=payload.change_set_id,
                    revision_number=payload.revision,
                    payload_hash=payload.payload_hash,
                    expected_session_revision=payload.session_revision,
                )
                session.commit()
            except RevisionStateError as error:
                raise_revision_conflict(error, session)
        return _change_set_payload(change_set)

    @router.post("/sessions/{revision_session_id}/approve-graph", status_code=202)
    def approve_graph_patch(
        project_id: str,
        revision_session_id: str,
        payload: ChangeSetApproval,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        with _review_service(session, settings) as service:
            try:
                change_set = service.approve_graph(
                    revision,
                    change_set_id=payload.change_set_id,
                    revision_number=payload.revision,
                    payload_hash=payload.payload_hash,
                    expected_session_revision=payload.session_revision,
                )
                session.flush()
                job = enqueue_graph_patch_job(
                    JobService(session), settings=settings, change_set=change_set
                )
                session.commit()
            except RevisionStateError as error:
                raise_revision_conflict(error, session)
        return {**_change_set_payload(change_set), "job_id": job.id}

    @router.post("/sessions/{revision_session_id}/publish")
    def publish_revision(
        project_id: str,
        revision_session_id: str,
        payload: PublishRequest,
        session: SessionDependency,
    ) -> dict[str, object]:
        try:
            publication = publish_change_set(
                session,
                project_id=project_id,
                revision_session_id=revision_session_id,
                change_set_id=payload.change_set_id,
                expected_session_revision=payload.session_revision,
            )
            graph = session.get(WorldGraphVersion, publication.graph_version_id)
            if graph is None:
                raise WorldPublicationError("发布后的图版本不存在")
            # 节点纠正只推进节点版本，不能触发全量起点调查或改变其他节点。
            analysis_job = (
                None
                if publication.node_boundary_hash
                else enqueue_publication_analysis(session, settings=settings, graph=graph)
            )
            session.commit()
        except WorldPublicationError as error:
            session.rollback()
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "publication_id": publication.id,
            "graph_version_id": publication.graph_version_id,
            "profile_id": publication.profile_id,
            "analysis_job_id": analysis_job.id if analysis_job else None,
            "status": publication.status,
        }

    @router.post("/sessions/{revision_session_id}/cancel")
    def cancel_revision(
        project_id: str,
        revision_session_id: str,
        payload: RevisionCancel,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        active_job_id = (
            revision.scope.get("active_agent_job_id") if isinstance(revision.scope, dict) else None
        )
        with _review_service(session, settings) as service:
            try:
                service.cancel(
                    revision,
                    expected_session_revision=payload.session_revision,
                    idempotency_key=payload.idempotency_key,
                )
                # 先持久化会话的 cancelled 状态，再撤销关联 Job 的租约。即使模型请求已经
                # 在飞，handler 重新写入前也会看见取消状态并拒绝迟到结果。
                job_service = JobService(session)
                if not isinstance(active_job_id, str) or not active_job_id:
                    queued_or_running = job_service.find_latest_by_payload(
                        REVISION_TURN_JOB_KIND,
                        "revision_session_id",
                        revision.id,
                    )
                    active_job_id = queued_or_running.id if queued_or_running is not None else None
                if isinstance(active_job_id, str) and active_job_id:
                    try:
                        job_service.cancel(active_job_id)
                    except (InvalidJobTransitionError, JobNotFoundError):
                        # Job 可能已自然完成；会话状态仍是最终的安全写入边界。
                        pass
                session.commit()
            except RevisionStateError as error:
                raise_revision_conflict(error, session)
        return _revision_payload(session, revision)

    @router.post("/sessions/{revision_session_id}/retry-agent-turn", status_code=202)
    def retry_agent_turn(
        project_id: str,
        revision_session_id: str,
        payload: RevisionAgentRetry,
        session: SessionDependency,
    ) -> dict[str, object]:
        revision = _revision(session, project_id, revision_session_id)
        with _review_service(session, settings) as service:
            try:
                revision = service.retry_agent_turn(
                    revision,
                    expected_session_revision=payload.session_revision,
                    idempotency_key=payload.idempotency_key,
                )
                session.flush()
                job = enqueue_revision_turn_job(
                    JobService(session), settings=settings, revision=revision
                )
                session.commit()
            except RevisionStateError as error:
                raise_revision_conflict(error, session)
        return {**_revision_payload(session, revision), "agent_job_id": job.id}

    return router


class _ReviewServiceContext:
    def __init__(self, session: Session, settings: Settings) -> None:
        if settings.node_analysis_api_key is None:
            raise HTTPException(status_code=409, detail="人物世界 Agent 缺少认知模型 API key")
        self.compiler = NodeAnalysisCloudClient(
            enabled=settings.node_analysis_enabled,
            endpoint=settings.node_analysis_endpoint,
            model=settings.node_analysis_model,
            api_key=settings.node_analysis_api_key,
            timeout_seconds=settings.world_compiler_timeout_seconds,
            max_retries=settings.node_analysis_max_retries,
            backoff_seconds=settings.node_analysis_backoff_seconds,
            max_backoff_seconds=settings.node_analysis_max_backoff_seconds,
            max_retry_after_seconds=settings.node_analysis_max_retry_after_seconds,
            response_format=settings.node_analysis_response_format,
            thinking_mode=settings.node_analysis_thinking_mode,
            max_output_tokens=settings.node_analysis_max_output_tokens,
        )
        self.sidecar = LightRAGSidecarClient(
            settings.lightrag_sidecar_url,
            settings.lightrag_sidecar_token.get_secret_value(),
            timeout_seconds=settings.lightrag_timeout_seconds,
        )
        self.service = PersonWorldReviewService(
            session, compiler=self.compiler, lightrag=self.sidecar
        )

    def __enter__(self) -> PersonWorldReviewService:
        return self.service

    def __exit__(self, *_: object) -> None:
        self.compiler.close()
        self.sidecar.close()


def _review_service(session: Session, settings: Settings) -> _ReviewServiceContext:
    return _ReviewServiceContext(session, settings)


def _revision(
    session: Session, project_id: str, revision_session_id: str
) -> PersonWorldRevisionSession:
    revision = session.get(PersonWorldRevisionSession, revision_session_id)
    if revision is None or revision.project_id != project_id:
        raise HTTPException(status_code=404, detail="PersonWorld 纠正会话不存在")
    return revision


def _revision_change_set(
    session: Session, revision: PersonWorldRevisionSession
) -> WorldGraphChangeSet | None:
    return (
        session.get(WorldGraphChangeSet, revision.graph_change_set_id)
        if revision.graph_change_set_id
        else None
    )


def _revision_payload(session: Session, revision: PersonWorldRevisionSession) -> dict[str, object]:
    messages = list(
        session.scalars(
            select(PersonWorldRevisionMessage)
            .where(PersonWorldRevisionMessage.session_id == revision.id)
            .order_by(
                PersonWorldRevisionMessage.created_at.asc(), PersonWorldRevisionMessage.id.asc()
            )
        )
    )
    change_set = (
        session.get(WorldGraphChangeSet, revision.graph_change_set_id)
        if revision.graph_change_set_id
        else None
    )
    selected_statements: list[dict[str, object]] = next(
        (
            item.payload.get("selected_statements", [])
            for item in messages
            if item.role == "user"
            and isinstance(item.payload, dict)
            and isinstance(item.payload.get("selected_statements"), list)
        ),
        [],
    )
    return {
        "id": revision.id,
        "project_id": revision.project_id,
        "base_graph_version_id": revision.base_graph_version_id,
        "base_profile_id": revision.base_profile_id,
        "status": revision.status,
        "session_revision": revision.session_revision,
        "scope": revision.scope,
        "agent_stage": (
            revision.scope.get("agent_stage") if isinstance(revision.scope, dict) else None
        ),
        "agent_stage_progress": (
            revision.scope.get("agent_stage_progress") if isinstance(revision.scope, dict) else None
        ),
        "context_snapshot_id": revision.context_snapshot_id,
        "pending_turn_id": revision.pending_turn_id,
        "understanding_revision": revision.understanding_revision,
        "understanding": revision.understanding_payload,
        "understanding_payload_hash": revision.understanding_payload_hash,
        "selected_statements": selected_statements,
        "messages": [
            {
                "id": item.id,
                "turn_id": item.turn_id,
                "role": item.role,
                "kind": item.kind,
                "in_reply_to_turn_id": item.in_reply_to_turn_id,
                "context_snapshot_id": item.context_snapshot_id,
                "session_revision": item.session_revision,
                "content": item.content,
                "payload": item.payload,
                "created_at": item.created_at.isoformat(),
            }
            for item in messages
        ],
        "change_set": _change_set_payload(change_set) if change_set is not None else None,
        "created_at": revision.created_at.isoformat(),
        "updated_at": revision.updated_at.isoformat(),
    }


def _change_set_payload(change_set: WorldGraphChangeSet) -> dict[str, object]:
    return {
        "id": change_set.id,
        "project_id": change_set.project_id,
        "revision_session_id": change_set.revision_session_id,
        "base_graph_version_id": change_set.base_graph_version_id,
        "revision": change_set.revision,
        "status": change_set.status,
        "profile_patch": change_set.profile_patch,
        "graph_operations": change_set.graph_operations,
        "affected_entities": change_set.affected_entities,
        "affected_relations": change_set.affected_relations,
        "regression_queries": change_set.regression_queries,
        "payload_hash": change_set.canonical_payload_hash,
        "candidate_graph_version_id": change_set.candidate_graph_version_id,
        "execution_result": change_set.execution_result,
    }


def _profile_draft_payload(
    session: Session,
    profile: PersonWorldProfile,
    graph: WorldGraphVersion,
) -> dict[str, object]:
    """候选档案除事实投影外，还返回每栏可显示的调查审计摘要。"""

    tasks = (
        list(
            session.scalars(
                select(PersonWorldSectionTask)
                .where(PersonWorldSectionTask.agent_run_id == profile.agent_run_id)
                .order_by(PersonWorldSectionTask.section.asc())
            )
        )
        if profile.agent_run_id
        else []
    )
    return {
        "profile": {
            "id": profile.id,
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
            "source_message_ids": profile.source_message_ids,
            "agent_run_id": profile.agent_run_id,
            "generation_summary": profile.generation_summary,
            "profile_v2": profile.profile_v2,
            "profile_v3": profile.profile_v3,
            "profile_schema_version": profile.profile_schema_version,
            "investigation_report": profile.investigation_report,
        },
        "graph": {
            "id": graph.id,
            "status": graph.status,
            "revision": graph.revision,
            "parent_version_id": graph.parent_version_id,
            "message_count": graph.message_count,
            "bundle_count": graph.bundle_count,
        },
        # 仅暴露用户审核所需的查询问题、轮数、未解决项和安全错误码；不返回
        # tool arguments、完整检索上下文或内部诊断，避免把私密原文带到页面。
        "section_tasks": [
            {
                "section": task.section,
                "status": task.status,
                "attempted_queries": list(task.attempted_queries or []),
                "research_round": task.research_round,
                # 调用/失败次数只由前端通过该关联键从 Phoenix 根 Span 读取，业务
                # 数据库不镜像一份会与 Trace 脱节的通用 Agent 调用账本。
                "phoenix_execution_id": _section_phoenix_execution_id(task.result_summary),
                # 仅供改造前的栏目任务回退查询；新任务优先使用精确 execution id。
                "phoenix_owner_id": (
                    f"{profile.agent_run_id}:{task.section}" if profile.agent_run_id else None
                ),
                "unresolved_questions": list(task.unresolved_questions or []),
                "error_code": task.error_code,
                "retry_attempt": _section_retry_attempt(task.result_summary),
            }
            for task in tasks
        ],
    }


def _section_retry_attempt(summary: object) -> int:
    if not isinstance(summary, dict):
        return 0
    value = summary.get("retry_attempt", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _section_phoenix_execution_id(summary: object) -> str | None:
    """仅返回 Phoenix 的关联键；不把 Trace 镜像到业务 API。"""

    if not isinstance(summary, dict):
        return None
    value = summary.get("phoenix_execution_id")
    return value if isinstance(value, str) and value else None
