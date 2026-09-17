import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.events.config import config_fingerprint, normalize_config
from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.quoted_sender_preprocess import (
    build_clean_world_messages,
    persist_quoted_sender_aliases,
    preprocess_quoted_senders,
)
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import InvalidJobTransitionError, JobService
from moonlightbox.world.bundles import (
    ConversationBundleDocument,
    WorldMessage,
    build_conversation_bundles,
    source_fingerprint,
)
from moonlightbox.world.client import (
    LightRAGDocument,
    LightRAGMetadata,
    LightRAGSidecarClient,
    LightRAGSidecarError,
)
from moonlightbox.world.compiler import (
    COMPILER_VERSION,
    AgentCompilerClient,
    CompiledWorldProfile,
)
from moonlightbox.world.merges import (
    AliasAgentMessage,
    generate_merge_candidates,
)
from moonlightbox.world.models import (
    ConversationBundle,
    ConversationBundleMessage,
    EntityMergeProposal,
    PersonWorldProfile,
    WorldGraphVersion,
)
from moonlightbox.world.person_world import PersonWorldAgent
from moonlightbox.world.profile_store import persist_profile

WORLD_BUILD_JOB_KIND = "lightrag_world_build_v1"
WORLD_BUNDLE_VERSION = "conversation-bundle-v1"
# 单次 Sidecar 请求的文档数。Bundle 内部并发仍由 Sidecar 保持为 3；
# 缩小请求批次只为避免一批 20 个 Bundle 超过两小时请求超时。
LIGHTRAG_INDEX_BATCH_SIZE = 5


class WorldBuildSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    version: str = WORLD_BUNDLE_VERSION
    enabled: bool
    sidecar_url: str = Field(min_length=1)
    index_revision: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    timeout_seconds: float = Field(gt=0, le=7200)
    bundle_gap_hours: float = Field(gt=0, le=48)
    bundle_max_characters: int = Field(ge=1000, le=100000)
    carry_in_turns: int = Field(ge=0, le=10)
    query_top_k: int = Field(ge=1, le=100)
    query_chunk_top_k: int = Field(ge=1, le=100)
    query_max_total_tokens: int = Field(ge=1000, le=100000)
    person_world_section_concurrency: int = Field(ge=1, le=7)
    compiler_version: str = COMPILER_VERSION
    compiler_endpoint: str = Field(min_length=1)
    compiler_model: str = Field(min_length=1)
    compiler_response_format: str
    compiler_thinking_mode: str
    compiler_max_output_tokens: int = Field(gt=0, le=65536)


def build_world_job_snapshot(settings: Settings) -> dict[str, object]:
    return normalize_config(
        WorldBuildSnapshot(
            enabled=settings.lightrag_enabled,
            sidecar_url=settings.lightrag_sidecar_url,
            index_revision=settings.lightrag_index_revision,
            timeout_seconds=settings.lightrag_timeout_seconds,
            bundle_gap_hours=settings.lightrag_bundle_gap_hours,
            bundle_max_characters=settings.lightrag_bundle_max_characters,
            carry_in_turns=settings.lightrag_carry_in_turns,
            query_top_k=settings.lightrag_query_top_k,
            query_chunk_top_k=settings.lightrag_query_chunk_top_k,
            query_max_total_tokens=settings.lightrag_query_max_total_tokens,
            person_world_section_concurrency=settings.person_world_section_concurrency,
            compiler_endpoint=settings.node_analysis_endpoint,
            compiler_model=settings.node_analysis_model,
            compiler_response_format=settings.node_analysis_response_format,
            compiler_thinking_mode=settings.node_analysis_thinking_mode,
            compiler_max_output_tokens=settings.node_analysis_max_output_tokens,
        ).model_dump(mode="json")
    )


def enqueue_world_build(
    session: Session,
    *,
    settings: Settings,
    project_id: str,
    trigger_import_id: str,
    force_recompile: bool = False,
    compile_profile_only: bool = False,
    start_alias_review: bool = False,
    request_id: str | None = None,
) -> Job:
    messages = load_world_messages(session, project_id=project_id)
    if not messages:
        raise ValueError("项目没有可用于人物世界构建的消息")
    import_ids = list(dict.fromkeys(item.import_id for item in messages))
    source_digest = source_fingerprint(messages)
    snapshot = build_world_job_snapshot(settings)
    identity = normalize_config(
        {
            "project_id": project_id,
            "source_fingerprint": source_digest,
            "world_config": snapshot,
            "force_recompile": force_recompile,
            "compile_profile_only": compile_profile_only,
            "start_alias_review": start_alias_review,
            "request_id": request_id,
        }
    )
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    dedupe_key = f"{WORLD_BUILD_JOB_KIND}:{hashlib.sha256(serialized.encode()).hexdigest()}"
    service = JobService(session)
    job = service.enqueue_unique(
        WORLD_BUILD_JOB_KIND,
        {
            "project_id": project_id,
            "trigger_import_id": trigger_import_id,
            "force_recompile": force_recompile,
            "compile_profile_only": compile_profile_only,
            "start_alias_review": start_alias_review,
            "request_id": request_id,
            "source_import_ids": import_ids,
            "source_fingerprint": source_digest,
            "world_config": snapshot,
        },
        dedupe_key=dedupe_key,
    )
    if job.status not in {"failed", "interrupted"}:
        return job
    try:
        return service.resume(job.id)
    except InvalidJobTransitionError:
        return service.get(job.id)


def create_world_build_handler(
    settings: Settings,
    *,
    lightrag_client: LightRAGSidecarClient | None = None,
    compiler_client: AgentCompilerClient | None = None,
) -> JobHandler:
    def handler(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError("world_job_lease_missing", "人物世界任务缺少 Worker 租约")
        project_id = _required_string(job.payload, "project_id")
        trigger_import_id = _required_string(job.payload, "trigger_import_id")
        source_digest = _required_string(job.payload, "source_fingerprint")
        source_import_ids = _required_string_list(job.payload, "source_import_ids")
        try:
            snapshot = WorldBuildSnapshot.model_validate(job.payload.get("world_config"))
        except ValidationError as error:
            raise JobHandlerError("world_config_invalid", "人物世界任务配置快照无效") from error
        if not snapshot.enabled:
            raise JobHandlerError("world_disabled", "LightRAG 人物世界功能未启用")

        graph = _ensure_graph_version(
            service.session,
            project_id=project_id,
            trigger_import_id=trigger_import_id,
            source_import_ids=source_import_ids,
            source_digest=source_digest,
            snapshot=snapshot,
            version_nonce=(
                _required_string(job.payload, "request_id")
                if (
                    bool(job.payload.get("force_recompile"))
                    or bool(job.payload.get("start_alias_review"))
                )
                and not bool(job.payload.get("compile_profile_only"))
                else None
            ),
        )
        existing_profile = service.session.scalar(
            select(PersonWorldProfile)
            .where(PersonWorldProfile.node_boundary_hash.is_(None))
            .where(PersonWorldProfile.graph_version_id == graph.id)
        )
        # 编译阶段失败后重试时，图谱和全部 Bundle 可能已经成功写入。
        # 记录这个状态，避免为了修复编译器而再次执行耗时的云端建图。
        graph_was_failed = graph.status == "failed"
        force_recompile = bool(job.payload.get("force_recompile", False))
        compile_profile_only = bool(job.payload.get("compile_profile_only", False))
        start_alias_review = bool(job.payload.get("start_alias_review", False))
        if start_alias_review and existing_profile is not None:
            service.session.delete(existing_profile)
            service.session.flush()
        if start_alias_review:
            # “重新生成候选”是一个新的审核轮次。旧的 pending 提案可能来自旧
            # prompt（例如昵称→昵称），必须作废，避免它们继续阻塞审核状态。
            for proposal in service.session.scalars(
                select(EntityMergeProposal).where(
                    EntityMergeProposal.graph_version_id == graph.id,
                    EntityMergeProposal.decision == "pending",
                )
            ):
                proposal.decision = "reject"
                proposal.review_note = "重新生成别名候选，旧提案作废。"
                proposal.reviewed_at = datetime.now(UTC)
        if graph.status == "ready" and not force_recompile and not start_alias_review:
            analysis_job = enqueue_node_investigation(
                service,
                project_id=project_id,
            )
            service.checkpoint(
                job.id,
                {
                    "stage": "world_ready",
                    "progress": 1.0,
                    "analysis_job_id": analysis_job.id,
                },
                token=token,
            )
            return

        owned_lightrag = None
        owned_compiler = None
        sidecar = lightrag_client
        structured_compiler = compiler_client
        try:
            messages = load_world_messages(
                service.session,
                project_id=project_id,
                import_ids=source_import_ids,
            )
            if source_fingerprint(messages) != source_digest:
                raise JobHandlerError(
                    "world_source_changed",
                    "人物世界任务对应的消息集合已变化，请使用最新导入重新构建",
                )
            subject = required_participant(messages, role="target")
            user = required_participant(messages, role="self")
            quoted_sender_observations = preprocess_quoted_senders(messages)
            persist_quoted_sender_aliases(
                service.session,
                project_id=project_id,
                observations=quoted_sender_observations,
            )
            service.session.commit()
            bundle_messages = build_clean_world_messages(messages, quoted_sender_observations)
            from moonlightbox.world.bundles import chronological_source_order

            source_order = chronological_source_order(messages)
            bundles = build_conversation_bundles(
                bundle_messages,
                project_id=project_id,
                session_gap=timedelta(hours=snapshot.bundle_gap_hours),
                max_characters=snapshot.bundle_max_characters,
                carry_in_turns=snapshot.carry_in_turns,
            )
            _persist_bundles(service.session, graph, bundles)
            graph.message_count = len(messages)
            graph.bundle_count = len(bundles)
            graph.status = "compiling_profile" if compile_profile_only else "building"
            graph.error_code = None
            graph.error_message = None
            service.session.commit()
            service.checkpoint(
                job.id,
                {"stage": "bundles_ready", "progress": 0.08, "bundle_count": len(bundles)},
                token=token,
            )

            current_bundle_ids = {item.document_id for item in bundles}
            indexed_bundle_ids = set(
                service.session.scalars(
                    select(ConversationBundle.document_id).where(
                        ConversationBundle.graph_version_id == graph.id,
                        ConversationBundle.indexed.is_(True),
                    )
                )
            )
            # 合并 LightRAG 实体后只需重新查询并编译档案；图谱本身已经存在，
            # 不要因为 force_recompile 再次索引聊天 Bundle。
            reuse_index = (
                graph_was_failed or force_recompile or compile_profile_only or start_alias_review
            ) and current_bundle_ids.issubset(indexed_bundle_ids)

            if reuse_index:
                # 只有数据库明确记录全部 Bundle 已索引时才复用；部分失败
                # 的图仍会走完整索引流程，保证不会牺牲图谱质量。
                service.checkpoint(
                    job.id,
                    {
                        "stage": "indexing_world",
                        "progress": 0.60,
                        "indexed_bundles": len(bundles),
                        "bundle_count": len(bundles),
                        "reused_existing_index": True,
                    },
                    token=token,
                )
            else:
                if sidecar is None:
                    owned_lightrag = LightRAGSidecarClient(
                        snapshot.sidecar_url,
                        settings.lightrag_sidecar_token.get_secret_value(),
                        timeout_seconds=snapshot.timeout_seconds,
                    )
                    sidecar = owned_lightrag
                metadata: LightRAGMetadata | None = None
                for offset in range(0, len(bundles), LIGHTRAG_INDEX_BATCH_SIZE):
                    batch = bundles[offset : offset + LIGHTRAG_INDEX_BATCH_SIZE]
                    metadata = sidecar.index_documents(
                        graph.workspace_key,
                        [
                            LightRAGDocument(
                                id=item.document_id,
                                source=item.source_name,
                                text=item.content,
                                source_version=source_digest,
                                message_spans=item.source_spans(source_order),
                            )
                            for item in batch
                        ],
                    )
                    document_ids = {item.document_id for item in batch}
                    for record in service.session.scalars(
                        select(ConversationBundle).where(
                            ConversationBundle.graph_version_id == graph.id,
                            ConversationBundle.document_id.in_(document_ids),
                        )
                    ):
                        record.indexed = True
                    completed = min(offset + len(batch), len(bundles))
                    service.checkpoint(
                        job.id,
                        {
                            "stage": "indexing_world",
                            "progress": 0.08 + 0.52 * completed / max(1, len(bundles)),
                            "indexed_bundles": completed,
                            "bundle_count": len(bundles),
                        },
                        token=token,
                    )
                if metadata is None:
                    raise JobHandlerError("world_empty", "没有生成可索引的聊天会话窗口")
                _apply_metadata(graph, metadata)
                service.session.commit()

            if sidecar is None:
                owned_lightrag = LightRAGSidecarClient(
                    snapshot.sidecar_url,
                    settings.lightrag_sidecar_token.get_secret_value(),
                    timeout_seconds=snapshot.timeout_seconds,
                )
                sidecar = owned_lightrag

            if structured_compiler is None:
                if (
                    not settings.node_analysis_enabled
                    or settings.node_analysis_api_key is None
                    or not settings.node_analysis_api_key.get_secret_value().strip()
                ):
                    raise JobHandlerError(
                        "world_compiler_unavailable",
                        "人物背景编译模型未配置 API key",
                    )
                owned_compiler = NodeAnalysisCloudClient(
                    enabled=True,
                    endpoint=snapshot.compiler_endpoint,
                    model=snapshot.compiler_model,
                    api_key=settings.node_analysis_api_key,
                    timeout_seconds=settings.world_compiler_timeout_seconds,
                    max_retries=settings.node_analysis_max_retries,
                    backoff_seconds=settings.node_analysis_backoff_seconds,
                    max_backoff_seconds=settings.node_analysis_max_backoff_seconds,
                    max_retry_after_seconds=settings.node_analysis_max_retry_after_seconds,
                    response_format=snapshot.compiler_response_format,  # type: ignore[arg-type]
                    thinking_mode=snapshot.compiler_thinking_mode,  # type: ignore[arg-type]
                    max_output_tokens=snapshot.compiler_max_output_tokens,
                )
                structured_compiler = owned_compiler
            # 首次建图只走到别名审核：不编译人物档案，避免用户在确认前
            # 看到尚未完成实体归并的 PersonWorldProfile。
            if not compile_profile_only:
                try:
                    message_evidence = [
                        AliasAgentMessage(
                            message_id=item.message.id,
                            document_id=bundle.document_id,
                            timestamp=item.message.timestamp.isoformat(),
                            participant=item.message.participant_name,
                            content=item.message.content,
                            participant_role=item.message.participant_role,
                        )
                        for bundle in bundles
                        for item in bundle.messages
                    ]
                    candidates = generate_merge_candidates(
                        project_id=project_id,
                        session=service.session,
                        sidecar=sidecar,
                        compiler=structured_compiler,
                        workspace=graph.workspace_key,
                        subject_name=subject.participant_name,
                        user_name=user.participant_name,
                        original_messages=message_evidence,
                    )
                    with service.session.begin_nested():
                        persist_merge_proposals(
                            service.session,
                            project_id=project_id,
                            graph=graph,
                            candidates=candidates,
                        )
                except LightRAGSidecarError:
                    raise
                except Exception as error:
                    raise JobHandlerError(
                        getattr(error, "code", "alias_resolution_failed"),
                        "别名调查未完成，不能按无候选继续编译；请恢复或重试任务。",
                    ) from error

                pending_count = (
                    service.session.scalar(
                        select(func.count())
                        .select_from(EntityMergeProposal)
                        .where(
                            EntityMergeProposal.graph_version_id == graph.id,
                            EntityMergeProposal.decision == "pending",
                        )
                    )
                    or 0
                )
                if pending_count > 0:
                    graph.status = "awaiting_alias_review"
                    graph.completed_at = None
                    service.session.commit()
                    service.checkpoint(
                        job.id,
                        {
                            "stage": "world_graph_ready",
                            "progress": 1.0,
                            "bundle_count": len(bundles),
                            "pending_merge_proposals": pending_count,
                        },
                        token=token,
                    )
                    return

                # 新项目的主流程在共享图谱可查询后直接调查起点。全量人物背景
                # 不是前置条件；人物背景只在用户批准具体节点后按 node_scope 编译。
                graph.status = "ready"
                graph.completed_at = datetime.now(UTC)
                graph.error_code = None
                graph.error_message = None
                service.session.commit()
                analysis_job = enqueue_node_investigation(service, project_id=project_id)
                service.checkpoint(
                    job.id,
                    {
                        "stage": "world_ready",
                        "progress": 1.0,
                        "bundle_count": len(bundles),
                        "analysis_job_id": analysis_job.id,
                    },
                    token=token,
                )
                return

            def report_agent_progress(stage: str, completed: int, total: int) -> None:
                service.checkpoint(
                    job.id,
                    {
                        "stage": stage,
                        "progress": 0.60 + 0.25 * completed / max(1, total),
                        "completed_questions": completed,
                        "question_count": total,
                    },
                    token=token,
                )

            agent_result = PersonWorldAgent(
                session=service.session,
                graph=graph,
                lightrag=sidecar,
                compiler=structured_compiler,
                subject_name=subject.participant_name,
                user_name=user.participant_name,
                top_k=snapshot.query_top_k,
                chunk_top_k=snapshot.query_chunk_top_k,
                max_total_tokens=snapshot.query_max_total_tokens,
                section_concurrency=snapshot.person_world_section_concurrency,
                progress=report_agent_progress,
            ).run(
                mode="recompile" if compile_profile_only else "initial_compile",
                resume_key=f"job:{job.id}",
            )
            compiled = CompiledWorldProfile(
                draft=agent_result.draft,
                source_message_ids=agent_result.source_message_ids,
                retrieval_manifest=agent_result.retrieval_manifest,
            )
            profile = persist_profile(
                service.session,
                graph=graph,
                subject_person_id=subject.participant_id,
                compiled=compiled,
                agent_run_id=agent_result.run_id,
                generation_summary=agent_result.generation_summary,
                profile_v3=agent_result.profile_v3.model_dump(mode="json"),
                profile_schema_version="v3",
                investigation_report=agent_result.investigation_report.model_dump(mode="json"),
            )
            # Agent 产物必须先由用户审核。此时图谱和 Profile 都只是候选，
            # 不能启动 Runtime 或下游节点分析。
            graph.status = "awaiting_profile_review"
            graph.completed_at = datetime.now(UTC)
            service.session.commit()
            service.checkpoint(
                job.id,
                {
                    "stage": "awaiting_profile_review",
                    "progress": 1.0,
                    "profile_id": profile.id,
                    "agent_run_id": agent_result.run_id,
                    "bundle_count": len(bundles),
                },
                token=token,
            )
        except JobHandlerError as error:
            _mark_graph_failed(service.session, graph, error.code, error.safe_message)
            raise
        except LightRAGSidecarError as error:
            if error.details:
                job.checkpoint = {**(job.checkpoint or {}), 'index_failure': error.details}
            _mark_graph_failed(service.session, graph, error.code, error.safe_message)
            raise JobHandlerError(error.code, error.safe_message) from error
        except Exception as error:
            _mark_graph_failed(
                service.session,
                graph,
                "world_build_failed",
                f"人物世界构建失败：{type(error).__name__}",
            )
            raise JobHandlerError(
                "world_build_failed", f"人物世界构建失败：{type(error).__name__}"
            ) from error
        finally:
            if owned_compiler is not None:
                owned_compiler.close()
            if owned_lightrag is not None:
                owned_lightrag.close()

    return handler


def persist_merge_proposals(
    session: Session,
    *,
    project_id: str,
    graph: WorldGraphVersion,
    candidates: Sequence[object],
) -> None:
    """幂等保存 pending 提案，不自动接受任何提案。"""
    existing = {
        (tuple(sorted(item.source_entities)), item.target_entity)
        for item in session.scalars(
            select(EntityMergeProposal).where(
                EntityMergeProposal.graph_version_id == graph.id,
                # 被“重新生成候选”作废的历史提案允许在新一轮重新出现；
                # pending/approved 才会阻止同一轮重复插入。
                EntityMergeProposal.decision.in_(["pending", "approve"]),
            )
        )
    }
    for candidate in candidates:
        sources = getattr(candidate, "source_entities", [])
        target = getattr(candidate, "target_entity", "")
        key = (tuple(sorted(sources)), target)
        if not sources or not target or key in existing:
            continue
        session.add(
            EntityMergeProposal(
                project_id=project_id,
                graph_version_id=graph.id,
                source_entities=list(sources),
                target_entity=target,
                reason=str(getattr(candidate, "reason", "")),
                evidence=[
                    item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                    for item in list(getattr(candidate, "evidence", []))
                ],
            )
        )
        existing.add(key)
    session.flush()


def load_world_messages(
    session: Session,
    *,
    project_id: str,
    import_ids: list[str] | None = None,
) -> list[WorldMessage]:
    statement = (
        select(Message, Participant)
        .join(Participant, Participant.id == Message.participant_id)
        .where(Message.project_id == project_id)
        .order_by(Message.timestamp.asc(), Message.id.asc())
    )
    if import_ids is not None:
        statement = statement.where(Message.import_id.in_(import_ids))
    return [
        WorldMessage(
            id=message.id,
            import_id=message.import_id,
            participant_id=participant.id,
            participant_name=participant.name,
            participant_role=participant.role,
            timestamp=message.timestamp,
            kind=message.kind,
            content=message.content,
            source_id=message.source_id,
        )
        for message, participant in session.execute(statement).all()
    ]


def required_participant(messages: list[WorldMessage], *, role: str) -> WorldMessage:
    candidates = [item for item in messages if item.participant_role == role]
    if not candidates:
        raise JobHandlerError("world_participant_missing", f"聊天中缺少 {role} 角色")
    return candidates[0]


def _ensure_graph_version(
    session: Session,
    *,
    project_id: str,
    trigger_import_id: str,
    source_import_ids: list[str],
    source_digest: str,
    snapshot: WorldBuildSnapshot,
    version_nonce: str | None = None,
) -> WorldGraphVersion:
    snapshot_fingerprint = config_fingerprint(snapshot.model_dump(mode="json"))
    if version_nonce is not None:
        # 手动重建必须得到新的不可变 workspace。request_id 只用于打破
        # (source, config) 唯一键，不改变真正的模型/切块配置快照。
        snapshot_fingerprint = hashlib.sha256(
            f"{snapshot_fingerprint}:{version_nonce}".encode()
        ).hexdigest()
    existing = session.scalar(
        select(WorldGraphVersion).where(
            WorldGraphVersion.project_id == project_id,
            WorldGraphVersion.source_fingerprint == source_digest,
            WorldGraphVersion.config_fingerprint == snapshot_fingerprint,
        )
    )
    if existing is not None:
        return existing
    graph_id = str(uuid4())
    graph = WorldGraphVersion(
        id=graph_id,
        project_id=project_id,
        trigger_import_id=trigger_import_id,
        workspace_key=f"world_{graph_id.replace('-', '')}",
        status="building",
        source_fingerprint=source_digest,
        config_fingerprint=snapshot_fingerprint,
        source_import_ids=source_import_ids,
        compiler_version=snapshot.compiler_version,
    )
    session.add(graph)
    session.commit()
    return graph


def _persist_bundles(
    session: Session,
    graph: WorldGraphVersion,
    documents: list[ConversationBundleDocument],
) -> None:
    existing = {
        item.document_id
        for item in session.scalars(
            select(ConversationBundle).where(ConversationBundle.graph_version_id == graph.id)
        )
    }
    for document in documents:
        if document.document_id in existing:
            continue
        record = ConversationBundle(
            project_id=graph.project_id,
            graph_version_id=graph.id,
            document_id=document.document_id,
            source_name=document.source_name,
            ordinal=document.ordinal,
            started_at=document.started_at,
            ended_at=document.ended_at,
            content=document.content,
            content_hash=document.content_hash,
            primary_message_count=document.primary_message_count,
            carry_in_message_count=document.carry_in_message_count,
        )
        session.add(record)
        session.flush()
        for ordinal, item in enumerate(document.messages):
            session.add(
                ConversationBundleMessage(
                    bundle_id=record.id,
                    message_id=item.message.id,
                    ordinal=ordinal,
                    is_carry_in=item.is_carry_in,
                )
            )


def _apply_metadata(graph: WorldGraphVersion, metadata: LightRAGMetadata) -> None:
    graph.lightrag_version = metadata.lightrag_version
    graph.embedding_model = metadata.embedding_model
    graph.embedding_dimension = metadata.embedding_dimension
    graph.extraction_model = metadata.extraction_model
    graph.chunking_strategy = metadata.chunking_strategy
    graph.chunk_token_size = metadata.chunk_token_size
    graph.chunk_overlap_token_size = metadata.chunk_overlap_token_size
    graph.entity_prompt_version = metadata.entity_prompt_version


def enqueue_node_investigation(
    service: JobService,
    *,
    project_id: str,
) -> Job:
    """资料发布后进入新的调查工作台，不再启动旧事件评分流水线。"""
    from moonlightbox.node_investigation.store import InvestigationStore

    service.session.commit()
    result = InvestigationStore(service.session.get_bind(), project_id).start("Asia/Shanghai")
    return service.get(result["job_id"])


def _mark_graph_failed(
    session: Session,
    graph: WorldGraphVersion,
    code: str,
    message: str,
) -> None:
    # 这里先确保图谱和人物档案已经持久化，再投递下游事件任务。
    # 之后即使队列或进度记录失败，也不能把已经有效的不可变世界快照标记为失败。
    # 重试世界任务时会走 ready 快速路径，并补投缺失的下游任务。
    if graph.status == "ready":
        return
    graph.status = "failed"
    graph.error_code = code
    graph.error_message = message
    session.commit()


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise JobHandlerError("world_payload_invalid", f"人物世界任务缺少 {key}")
    return value


def _required_string_list(payload: Mapping[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise JobHandlerError("world_payload_invalid", f"人物世界任务缺少 {key}")
    return list(dict.fromkeys(value))
