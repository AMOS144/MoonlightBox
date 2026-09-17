"""人物世界 API：背景读取、建图任务、来源查看和实体归并审核。"""

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import JobService
from moonlightbox.world.client import LightRAGSidecarClient, LightRAGSidecarError
from moonlightbox.world.jobs import (
    WORLD_BUILD_JOB_KIND,
    enqueue_node_investigation,
    enqueue_world_build,
    load_world_messages,
    persist_merge_proposals,
    required_participant,
)
from moonlightbox.world.merges import AliasAgentMessage, generate_merge_candidates
from moonlightbox.world.models import (
    ConversationBundle,
    ConversationBundleMessage,
    EntityMergeProposal,
    PersonWorldProfile,
    WorldGraphVersion,
    WorldPublication,
)
from moonlightbox.world.person_world.jobs import PERSON_WORLD_PROFILE_RECOMPILE_JOB_KIND
from moonlightbox.world.schemas import (
    AliasEvidenceRead,
    EntityMergeProposalCreate,
    EntityMergeProposalRead,
    PersonWorldProfileRead,
    WorldBuildProgressRead,
    WorldBuildRead,
    WorldGraphEdgeRead,
    WorldGraphNodeRead,
    WorldGraphSnapshotRead,
    WorldGraphVersionRead,
    WorldSourceMessageRead,
    WorldSourceRead,
)


def _str_property(properties: dict[str, object], key: str) -> str | None:
    value = properties.get(key)
    return value if isinstance(value, str) else None


def create_world_router(database: Database, settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/projects/{project_id}/world-profile", tags=["world"])

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.get("", response_model=PersonWorldProfileRead)
    def get_profile(project_id: str, session: SessionDependency) -> PersonWorldProfileRead:
        profile, graph = _latest_profile(session, project_id)
        if profile is None or graph is None:
            raise HTTPException(status_code=404, detail="人物背景尚未生成")
        return _profile_read(profile, graph)

    @router.get("/status", response_model=WorldGraphVersionRead)
    def get_status(project_id: str, session: SessionDependency) -> WorldGraphVersionRead:
        graph = session.scalar(
            select(WorldGraphVersion)
            .where(WorldGraphVersion.project_id == project_id)
            .order_by(WorldGraphVersion.created_at.desc(), WorldGraphVersion.id.desc())
        )
        if graph is None:
            raise HTTPException(status_code=404, detail="人物世界尚未构建")
        return WorldGraphVersionRead.model_validate(graph).model_copy(
            update={"build_progress": _active_world_build_progress(session, project_id)}
        )

    @router.get("/graph", response_model=WorldGraphSnapshotRead)
    def get_graph_snapshot(project_id: str, session: SessionDependency) -> WorldGraphSnapshotRead:
        graph = session.scalar(
            select(WorldGraphVersion)
            .where(WorldGraphVersion.project_id == project_id)
            .order_by(WorldGraphVersion.created_at.desc(), WorldGraphVersion.id.desc())
        )
        if graph is None:
            raise HTTPException(status_code=404, detail="人物世界尚未构建")
        sidecar = LightRAGSidecarClient(
            settings.lightrag_sidecar_url,
            settings.lightrag_sidecar_token.get_secret_value(),
            timeout_seconds=settings.lightrag_timeout_seconds,
        )
        try:
            snapshot = sidecar.get_graph(graph.workspace_key)
        except LightRAGSidecarError as error:
            raise HTTPException(status_code=503, detail=error.safe_message) from error
        return WorldGraphSnapshotRead(
            nodes=[
                WorldGraphNodeRead(
                    id=node.id,
                    entity_type=_str_property(node.properties, "entity_type"),
                    description=_str_property(node.properties, "description"),
                )
                for node in snapshot.nodes
            ],
            edges=[
                WorldGraphEdgeRead(
                    source=edge.source,
                    target=edge.target,
                    keywords=_str_property(edge.properties, "keywords"),
                )
                for edge in snapshot.edges
            ],
            truncated=snapshot.is_truncated,
        )

    @router.get("/merge-proposals", response_model=list[EntityMergeProposalRead])
    def list_merge_proposals(
        project_id: str, session: SessionDependency
    ) -> list[EntityMergeProposal]:
        graph = session.scalar(
            select(WorldGraphVersion)
            .where(WorldGraphVersion.project_id == project_id)
            .order_by(WorldGraphVersion.created_at.desc(), WorldGraphVersion.id.desc())
        )
        if graph is None:
            return []
        return session.scalars(
            select(EntityMergeProposal)
            .where(
                EntityMergeProposal.project_id == project_id,
                EntityMergeProposal.graph_version_id == graph.id,
            )
            .order_by(EntityMergeProposal.created_at.desc())
        ).all()

    @router.post("/merge-proposals/generate", response_model=list[EntityMergeProposalRead])
    def generate_merge_proposals(
        project_id: str, session: SessionDependency
    ) -> list[EntityMergeProposal]:
        """从当前图谱生成人工审核提案；只写 pending，不执行合并。"""
        graph = session.scalar(
            select(WorldGraphVersion)
            .where(
                WorldGraphVersion.project_id == project_id,
                WorldGraphVersion.status.in_(["ready", "awaiting_alias_review"]),
            )
            .order_by(WorldGraphVersion.created_at.desc())
        )
        if graph is None:
            raise HTTPException(status_code=404, detail="人物世界尚未构建")
        _reject_published_graph_mutation(session, graph.id)
        if not settings.node_analysis_enabled or settings.node_analysis_api_key is None:
            raise HTTPException(status_code=409, detail="未配置人物节点比较模型")
        messages = load_world_messages(
            session, project_id=project_id, import_ids=graph.source_import_ids
        )
        try:
            subject = required_participant(messages, role="target")
            user = required_participant(messages, role="self")
        except Exception as error:
            raise HTTPException(status_code=409, detail="聊天中缺少目标人物或用户角色") from error
        sidecar = LightRAGSidecarClient(
            settings.lightrag_sidecar_url,
            settings.lightrag_sidecar_token.get_secret_value(),
            timeout_seconds=settings.lightrag_timeout_seconds,
        )
        compiler = NodeAnalysisCloudClient(
            enabled=True,
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
        try:
            # 先调查，成功后再替换待审候选；网络失败不得作废已有审核材料，
            # 也不应在耗时的模型请求期间持有数据库写锁。
            document_by_message = {
                message_id: document_id
                for message_id, document_id in session.execute(
                    select(ConversationBundleMessage.message_id, ConversationBundle.document_id)
                    .join(
                        ConversationBundle,
                        ConversationBundle.id == ConversationBundleMessage.bundle_id,
                    )
                    .where(ConversationBundle.graph_version_id == graph.id)
                ).all()
            }
            message_evidence = [
                AliasAgentMessage(
                    message_id=item.id,
                    document_id=document_by_message.get(item.id),
                    timestamp=item.timestamp.isoformat(),
                    participant=item.participant_name,
                    content=item.content,
                    participant_role=item.participant_role,
                )
                for item in messages
            ]
            candidates = generate_merge_candidates(
                project_id=project_id,
                session=session,
                sidecar=sidecar,
                compiler=compiler,
                workspace=graph.workspace_key,
                subject_name=subject.participant_name,
                user_name=user.participant_name,
                original_messages=message_evidence,
            )
            for old in session.scalars(
                select(EntityMergeProposal).where(
                    EntityMergeProposal.graph_version_id == graph.id,
                    EntityMergeProposal.decision == "pending",
                )
            ):
                old.decision = "reject"
                old.review_note = "重新生成别名候选，旧提案作废。"
                old.reviewed_at = datetime.now(UTC)
            persist_merge_proposals(
                session,
                project_id=project_id,
                graph=graph,
                candidates=candidates,
            )
            pending = (
                session.scalar(
                    select(func.count())
                    .select_from(EntityMergeProposal)
                    .where(
                        EntityMergeProposal.graph_version_id == graph.id,
                        EntityMergeProposal.decision == "pending",
                    )
                )
                or 0
            )
            if pending > 0:
                # 即使当前图版本原本 ready，只要重新生成出了待审提案，
                # 旧 PersonWorldProfile 也必须暂时隐藏，直到本轮全部审核完成。
                graph.status = "awaiting_alias_review"
                graph.completed_at = None
                session.commit()
            elif graph.status in {"awaiting_alias_review", "ready"}:
                graph.status = "ready"
                graph.completed_at = datetime.now(UTC)
                graph.error_code = graph.error_message = None
                session.commit()
                enqueue_node_investigation(JobService(session), project_id=project_id)
            else:
                session.commit()
        finally:
            compiler.close()
            sidecar.close()
        return session.scalars(
            select(EntityMergeProposal)
            .where(EntityMergeProposal.graph_version_id == graph.id)
            .order_by(EntityMergeProposal.created_at.desc())
        ).all()

    @router.post(
        "/merge-proposals",
        response_model=EntityMergeProposalRead,
        status_code=status.HTTP_201_CREATED,
    )
    def create_merge_proposal(
        project_id: str, payload: EntityMergeProposalCreate, session: SessionDependency
    ) -> EntityMergeProposal:
        graph = session.scalar(
            select(WorldGraphVersion)
            .where(WorldGraphVersion.project_id == project_id)
            .order_by(WorldGraphVersion.created_at.desc())
        )
        if graph is None:
            raise HTTPException(status_code=404, detail="人物世界尚未构建")
        _reject_published_graph_mutation(session, graph.id)
        item = EntityMergeProposal(
            project_id=project_id,
            graph_version_id=graph.id,
            source_entities=payload.source_entities,
            target_entity=payload.target_entity,
            reason=payload.reason,
            evidence=payload.evidence,
        )
        session.add(item)
        session.commit()
        session.refresh(item)
        return item

    @router.get("/merge-proposals/{proposal_id}/evidence", response_model=list[AliasEvidenceRead])
    def merge_proposal_evidence(
        project_id: str,
        proposal_id: str,
        session: SessionDependency,
    ) -> list[AliasEvidenceRead]:
        """按 Agent 返回的 message_id 读取审核卡片对应的原文位置。"""
        proposal = session.scalar(
            select(EntityMergeProposal).where(
                EntityMergeProposal.id == proposal_id,
                EntityMergeProposal.project_id == project_id,
            )
        )
        if proposal is None:
            raise HTTPException(status_code=404, detail="归并建议不存在")
        message_ids = {
            str(item.get("message_id"))
            for item in proposal.evidence
            if isinstance(item, dict) and item.get("message_id")
        }
        if not message_ids:
            return []
        rows = session.execute(
            select(
                Message,
                Participant,
                ConversationBundle.document_id,
                ConversationBundleMessage.is_carry_in,
            )
            .join(Participant, Participant.id == Message.participant_id)
            .join(ConversationBundleMessage, ConversationBundleMessage.message_id == Message.id)
            .join(ConversationBundle, ConversationBundle.id == ConversationBundleMessage.bundle_id)
            .where(
                Message.id.in_(message_ids),
                ConversationBundle.graph_version_id == proposal.graph_version_id,
            )
            .order_by(Message.timestamp.asc(), Message.id.asc())
        ).all()
        return [
            AliasEvidenceRead(
                id=message.id,
                timestamp=message.timestamp,
                participant=participant.name,
                role=participant.role,
                kind=message.kind,
                content=message.content,
                is_carry_in=is_carry_in,
                document_id=document_id,
            )
            for message, participant, document_id, is_carry_in in rows
        ]

    @router.post("/merge-proposals/{proposal_id}/review", response_model=EntityMergeProposalRead)
    def review_merge_proposal(
        project_id: str,
        proposal_id: str,
        decision: str,
        session: SessionDependency,
        note: str | None = None,
    ) -> EntityMergeProposal:
        item = session.scalar(
            select(EntityMergeProposal).where(
                EntityMergeProposal.id == proposal_id, EntityMergeProposal.project_id == project_id
            )
        )
        if item is None:
            raise HTTPException(status_code=404, detail="归并建议不存在")
        if decision not in {"approve", "reject", "defer"}:
            raise HTTPException(status_code=422, detail="decision 必须是 approve、reject 或 defer")
        if item.decision != "pending":
            raise HTTPException(status_code=409, detail="该归并建议已经审核")
        _reject_published_graph_mutation(session, item.graph_version_id)
        if decision == "approve":
            graph = session.get(WorldGraphVersion, item.graph_version_id)
            if graph is None:
                raise HTTPException(status_code=404, detail="归并建议对应的图版本不存在")
            client = LightRAGSidecarClient(
                settings.lightrag_sidecar_url,
                settings.lightrag_sidecar_token.get_secret_value(),
                timeout_seconds=settings.lightrag_timeout_seconds,
            )
            try:
                item.merge_result = client.merge_entities(
                    graph.workspace_key,
                    source_entities=item.source_entities,
                    target_entity=item.target_entity,
                )
            finally:
                client.close()
        item.decision = decision
        item.review_note = note
        item.reviewed_at = datetime.now(UTC)
        session.flush()
        remaining = (
            session.scalar(
                select(func.count())
                .select_from(EntityMergeProposal)
                .where(
                    EntityMergeProposal.graph_version_id == item.graph_version_id,
                    EntityMergeProposal.decision == "pending",
                )
            )
            or 0
        )
        if decision in {"approve", "reject"} and remaining == 0:
            graph = session.get(WorldGraphVersion, item.graph_version_id)
            if graph is not None and graph.status in {"awaiting_alias_review", "ready"}:
                graph.status = "ready"
                graph.completed_at = datetime.now(UTC)
                graph.error_code = graph.error_message = None
                session.commit()
                enqueue_node_investigation(JobService(session), project_id=project_id)
            else:
                session.commit()
        else:
            session.commit()
        session.refresh(item)
        return item

    @router.post("/rebuild", response_model=WorldBuildRead, status_code=status.HTTP_202_ACCEPTED)
    def rebuild(
        project_id: str,
        session: SessionDependency,
        force_recompile: bool = False,
    ) -> WorldBuildRead:
        if not settings.lightrag_enabled:
            raise HTTPException(status_code=409, detail="LightRAG 人物世界功能未启用")
        latest_import = session.scalar(
            select(ImportSource)
            .where(ImportSource.project_id == project_id)
            .order_by(ImportSource.confirmed_at.desc(), ImportSource.id.desc())
        )
        if latest_import is None:
            raise HTTPException(status_code=409, detail="请先导入聊天记录")
        try:
            job = enqueue_world_build(
                session,
                settings=settings,
                project_id=project_id,
                trigger_import_id=latest_import.id,
                force_recompile=force_recompile,
                start_alias_review=True,
                request_id=str(uuid4()),
            )
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return WorldBuildRead(job_id=job.id, status=job.status)

    @router.get("/sources/{document_id}", response_model=WorldSourceRead)
    def get_source(
        project_id: str,
        document_id: str,
        session: SessionDependency,
    ) -> WorldSourceRead:
        profile, graph = _latest_profile(session, project_id)
        if profile is None or graph is None:
            raise HTTPException(status_code=404, detail="人物背景尚未生成")
        bundle = session.scalar(
            select(ConversationBundle).where(
                ConversationBundle.project_id == project_id,
                ConversationBundle.graph_version_id == graph.id,
                ConversationBundle.document_id == document_id,
            )
        )
        if bundle is None:
            raise HTTPException(status_code=404, detail="来源文档不存在")
        rows = session.execute(
            select(ConversationBundleMessage, Message, Participant)
            .join(Message, Message.id == ConversationBundleMessage.message_id)
            .join(Participant, Participant.id == Message.participant_id)
            .where(ConversationBundleMessage.bundle_id == bundle.id)
            .order_by(ConversationBundleMessage.ordinal.asc())
        ).all()
        return WorldSourceRead(
            document_id=bundle.document_id,
            started_at=bundle.started_at,
            ended_at=bundle.ended_at,
            messages=[
                WorldSourceMessageRead(
                    id=message.id,
                    timestamp=message.timestamp,
                    participant=participant.name,
                    role=participant.role,
                    kind=message.kind,
                    content=message.content,
                    is_carry_in=mapping.is_carry_in,
                )
                for mapping, message, participant in rows
            ],
        )

    return router


def _latest_profile(
    session: Session,
    project_id: str,
) -> tuple[PersonWorldProfile | None, WorldGraphVersion | None]:
    # 新治理流程使用显式发布指针，确保 Graph 与 Profile 始终成对读取。
    publication = session.scalar(
        select(WorldPublication)
        .where(
            WorldPublication.project_id == project_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash.is_(None),
        )
        .order_by(WorldPublication.published_at.desc(), WorldPublication.id.desc())
    )
    if publication is not None:
        profile = session.get(PersonWorldProfile, publication.profile_id)
        graph = session.get(WorldGraphVersion, publication.graph_version_id)
        return profile, graph
    # 兼容尚未迁移到 WorldPublication 的 legacy ready 图。
    graph = session.scalar(
        select(WorldGraphVersion)
        .where(WorldGraphVersion.project_id == project_id)
        .order_by(WorldGraphVersion.created_at.desc(), WorldGraphVersion.id.desc())
    )
    if graph is None or graph.status != "ready":
        return None, graph
    profile = session.scalar(
        select(PersonWorldProfile)
        .where(PersonWorldProfile.node_boundary_hash.is_(None))
        .where(
            PersonWorldProfile.project_id == project_id,
            PersonWorldProfile.graph_version_id == graph.id,
        )
        .order_by(PersonWorldProfile.created_at.desc(), PersonWorldProfile.id.desc())
    )
    return profile, graph


def _active_world_build_progress(
    session: Session,
    project_id: str,
) -> WorldBuildProgressRead | None:
    """取得该项目正在执行的人物世界任务的最近检查点。

    图版本没有任务外键；按创建时间选择同项目、仍在队列或运行中的世界任务，
    这样不会把已经完成的上一次建图进度显示在当前页面上。
    """
    jobs = session.scalars(
        select(Job)
        .where(
            Job.kind.in_((WORLD_BUILD_JOB_KIND, PERSON_WORLD_PROFILE_RECOMPILE_JOB_KIND)),
            Job.status.in_(("queued", "running")),
        )
        .order_by(Job.created_at.desc(), Job.id.desc())
    )
    job = next(
        (
            item
            for item in jobs
            if isinstance(item.payload, dict) and item.payload.get("project_id") == project_id
        ),
        None,
    )
    if job is None:
        return None

    checkpoint = job.checkpoint if isinstance(job.checkpoint, dict) else {}
    progress = checkpoint.get("progress", job.progress)
    return WorldBuildProgressRead(
        job_id=job.id,
        status=job.status,
        stage=_optional_string(checkpoint.get("stage")),
        progress=float(progress) if isinstance(progress, int | float) else float(job.progress),
        completed_questions=_optional_int(checkpoint.get("completed_questions")),
        question_count=_optional_int(checkpoint.get("question_count")),
        indexed_bundles=_optional_int(checkpoint.get("indexed_bundles")),
        bundle_count=_optional_int(checkpoint.get("bundle_count")),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reject_published_graph_mutation(session: Session, graph_version_id: str) -> None:
    """已发布 workspace 不允许旧别名接口原地写入。"""

    publication = session.scalar(
        select(WorldPublication).where(
            WorldPublication.graph_version_id == graph_version_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash.is_(None),
        )
    )
    if publication is not None:
        raise HTTPException(
            status_code=409,
            detail="已发布人物世界不能原地修改；请使用人物档案纠正流程或新建重构版本",
        )


def _profile_read(
    profile: PersonWorldProfile,
    graph: WorldGraphVersion,
) -> PersonWorldProfileRead:
    return PersonWorldProfileRead.model_validate(
        {
            "id": profile.id,
            "project_id": profile.project_id,
            "subject_person_id": profile.subject_person_id,
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
            "compiler_version": profile.compiler_version,
            "agent_run_id": profile.agent_run_id,
            "generation_summary": profile.generation_summary,
            "profile_v2": profile.profile_v2,
            "profile_v3": profile.profile_v3,
            "profile_schema_version": profile.profile_schema_version,
            "investigation_report": profile.investigation_report,
            "created_at": profile.created_at,
            "graph": WorldGraphVersionRead.model_validate(graph),
        }
    )
