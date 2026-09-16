"""已批准 Graph Patch 的候选图构建、执行、重编译和验证任务。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select

from moonlightbox.agent_runtime.policy import section_policy
from moonlightbox.agent_runtime.tasks import run_submission_task
from moonlightbox.config import Settings
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.world.client import (
    LightRAGDocument,
    LightRAGMetadata,
    LightRAGSidecarClient,
    LightRAGSidecarError,
)
from moonlightbox.world.compiler import COMPILER_VERSION, CompiledWorldProfile
from moonlightbox.world.jobs import LIGHTRAG_INDEX_BATCH_SIZE, _persist_profile
from moonlightbox.world.models import (
    ConversationBundle,
    ConversationBundleMessage,
    PersonWorldProfile,
    PersonWorldRevisionMessage,
    PersonWorldRevisionSession,
    WorldCorrection,
    WorldGraphChangeSet,
    WorldGraphVersion,
    WorldPublication,
)

from .coordinator_v3 import PersonWorldCoordinatorV3 as PersonWorldCoordinator
from .graph_executor import WorldGraphExecutor
from .schemas import RegressionAssessmentBatch, RegressionValidationBatch

WORLD_GRAPH_PATCH_JOB_KIND = "person_world_graph_patch_v1"
# 已有完整 LightRAG 图谱上的 Profile 重编译不应重新抽取原始聊天。该任务只克隆
# workspace、运行七个栏目 Agent，并把结果停在待审核候选态。
PERSON_WORLD_PROFILE_RECOMPILE_JOB_KIND = "person_world_profile_recompile_v1"


def enqueue_profile_recompile_job(
    service: JobService,
    *,
    settings: Settings,
    base: WorldGraphVersion,
) -> tuple[Job, WorldGraphVersion]:
    """从已完成图谱冻结一个独立候选，供七栏目 Agent 重新调查。

    SQL Bundle 映射在入队时复制，LightRAG workspace 的字节级复制在 Worker 中完成。
    因此候选永远不会向已发布 workspace 写入，也不会重新将聊天文本发送给抽取模型。
    """

    legacy_candidate = base.status == "awaiting_profile_review"
    if legacy_candidate:
        profile = service.session.scalar(
            select(PersonWorldProfile)
            .where(PersonWorldProfile.node_boundary_hash.is_(None))
            .where(PersonWorldProfile.graph_version_id == base.id)
        )
        if profile is None or profile.profile_schema_version not in {"v1", "v2"}:
            raise ValueError("新版候选请先完成审核，不可重复生成")
    if base.status != "ready" and not legacy_candidate:
        raise ValueError("只能基于已完成的人物世界重新调查")
    active = service.session.scalar(
        select(WorldGraphVersion.id).where(
            WorldGraphVersion.project_id == base.project_id,
            WorldGraphVersion.id != base.id,
            WorldGraphVersion.status.in_(
                {"profile_compilation_queued", "compiling_profile", "awaiting_profile_review"}
            ),
        )
    )
    if active is not None:
        raise ValueError("已有一份 PersonWorld 候选正在构建或等待审核")

    if legacy_candidate:
        # 只退役候选的审核入口；保留旧 Profile、Draft 和图谱，新的 v3 使用独立副本。
        # 此变更与新候选入队同事务提交，入队失败时由调用方回滚。
        base.status = "superseded"

    candidate_id = str(uuid4())
    candidate = WorldGraphVersion(
        id=candidate_id,
        project_id=base.project_id,
        trigger_import_id=base.trigger_import_id,
        workspace_key=f"world_{candidate_id.replace('-', '')}",
        status="profile_compilation_queued",
        source_fingerprint=base.source_fingerprint,
        config_fingerprint=hashlib.sha256(
            f"{base.config_fingerprint}:person-world-profile:{candidate_id}".encode()
        ).hexdigest(),
        source_import_ids=list(base.source_import_ids),
        message_count=base.message_count,
        bundle_count=base.bundle_count,
        lightrag_version=base.lightrag_version,
        embedding_model=base.embedding_model,
        embedding_dimension=base.embedding_dimension,
        extraction_model=base.extraction_model,
        chunking_strategy=base.chunking_strategy,
        chunk_token_size=base.chunk_token_size,
        chunk_overlap_token_size=base.chunk_overlap_token_size,
        entity_prompt_version=base.entity_prompt_version,
        compiler_version=COMPILER_VERSION,
        parent_version_id=base.id,
        revision=base.revision + 1,
        correction_head_hash=base.correction_head_hash,
    )
    service.session.add(candidate)
    service.session.flush()
    base_bundles = list(
        service.session.scalars(
            select(ConversationBundle)
            .where(ConversationBundle.graph_version_id == base.id)
            .order_by(ConversationBundle.ordinal.asc())
        )
    )
    clones = _clone_bundles(service, base_bundles, candidate)
    for bundle in clones:
        # Sidecar 复制的是完整的已完成 workspace；候选 Bundle 无需二次 index。
        bundle.indexed = True
    service.session.commit()
    job = service.enqueue_unique(
        PERSON_WORLD_PROFILE_RECOMPILE_JOB_KIND,
        {
            "project_id": base.project_id,
            "base_graph_version_id": base.id,
            "candidate_graph_version_id": candidate.id,
            "sidecar_url": settings.lightrag_sidecar_url,
            "compiler_endpoint": settings.node_analysis_endpoint,
            "compiler_model": settings.node_analysis_model,
        },
        dedupe_key=f"{PERSON_WORLD_PROFILE_RECOMPILE_JOB_KIND}:{candidate.id}",
    )
    return job, candidate


def create_profile_recompile_handler(
    settings: Settings,
    *,
    lightrag_client: LightRAGSidecarClient | None = None,
    compiler_client: NodeAnalysisCloudClient | None = None,
) -> JobHandler:
    """运行已冻结图谱上的七栏目调查，不执行任何图谱抽取或突变。"""

    def handler(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError(
                "person_world_recompile_lease_missing", "人物世界任务缺少 Worker 租约"
            )
        base_id = str(job.payload.get("base_graph_version_id", ""))
        candidate_id = str(job.payload.get("candidate_graph_version_id", ""))
        base = service.session.get(WorldGraphVersion, base_id)
        candidate = service.session.get(WorldGraphVersion, candidate_id)
        if base is None or candidate is None or candidate.parent_version_id != base.id:
            raise JobHandlerError("person_world_recompile_base_missing", "人物世界候选缺少冻结图谱")
        if candidate.status == "awaiting_profile_review":
            service.checkpoint(
                job.id,
                {
                    "stage": "awaiting_profile_review",
                    "progress": 1.0,
                    "candidate_graph_version_id": candidate.id,
                },
                token=token,
            )
            return
        if candidate.status not in {"profile_compilation_queued", "compiling_profile"}:
            raise JobHandlerError(
                "person_world_recompile_invalid_state", "人物世界候选不在可调查状态"
            )

        owned_sidecar = None
        owned_compiler = None
        try:
            sidecar = lightrag_client
            if sidecar is None:
                owned_sidecar = LightRAGSidecarClient(
                    settings.lightrag_sidecar_url,
                    settings.lightrag_sidecar_token.get_secret_value(),
                    timeout_seconds=settings.lightrag_timeout_seconds,
                )
                sidecar = owned_sidecar
            if candidate.status == "profile_compilation_queued":
                metadata = sidecar.clone_workspace(
                    source_workspace=base.workspace_key,
                    workspace=candidate.workspace_key,
                )
                _apply_lightrag_metadata(candidate, metadata)
                candidate.status = "compiling_profile"
                service.session.commit()
                service.checkpoint(
                    job.id,
                    {
                        "stage": "workspace_cloned",
                        "progress": 0.20,
                        "candidate_graph_version_id": candidate.id,
                        "base_graph_version_id": base.id,
                    },
                    token=token,
                )
            else:
                # 进程在 workspace 复制后中断时，候选状态已经是 compiling；不允许
                # 第二次覆盖或复制该目录，直接从已落盘的只读图谱继续调查。
                candidate.status = "compiling_profile"
                service.session.commit()

            compiler = compiler_client
            if compiler is None:
                if not settings.node_analysis_enabled or settings.node_analysis_api_key is None:
                    raise JobHandlerError(
                        "world_compiler_unavailable", "人物世界 Agent 缺少认知模型 API key"
                    )
                owned_compiler = NodeAnalysisCloudClient(
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
                compiler = owned_compiler
            target_name, target_id, user_name = _participant_names(
                service.session, candidate.project_id
            )

            def report(stage: str, completed: int, total: int) -> None:
                service.checkpoint(
                    job.id,
                    {
                        "stage": stage,
                        "progress": 0.20 + 0.75 * completed / max(1, total),
                        "completed_questions": completed,
                        "question_count": total,
                        "candidate_graph_version_id": candidate.id,
                    },
                    token=token,
                )

            result = PersonWorldCoordinator(
                session=service.session,
                graph=candidate,
                lightrag=sidecar,
                compiler=compiler,
                subject_name=target_name,
                user_name=user_name,
                top_k=settings.lightrag_query_top_k,
                chunk_top_k=settings.lightrag_query_chunk_top_k,
                max_total_tokens=settings.lightrag_query_max_total_tokens,
                section_concurrency=settings.person_world_section_concurrency,
                section_runtime_policy=section_policy(settings),
                progress=report,
            ).run(mode="recompile", resume_key=f"job:{job.id}")
            profile = _persist_profile(
                service.session,
                graph=candidate,
                subject_person_id=target_id,
                compiled=CompiledWorldProfile(
                    draft=result.draft,
                    source_message_ids=result.source_message_ids,
                    retrieval_manifest=result.retrieval_manifest,
                ),
                agent_run_id=result.run_id,
                generation_summary=result.generation_summary,
                profile_v3=result.profile_v3.model_dump(mode="json"),
                profile_schema_version="v3",
                investigation_report=result.investigation_report.model_dump(mode="json"),
            )
            candidate.status = "awaiting_profile_review"
            candidate.completed_at = datetime.now(UTC)
            service.session.commit()
            service.checkpoint(
                job.id,
                {
                    "stage": "awaiting_profile_review",
                    "progress": 1.0,
                    "candidate_graph_version_id": candidate.id,
                    "profile_id": profile.id,
                    "agent_run_id": result.run_id,
                },
                token=token,
            )
        except JobHandlerError:
            candidate.status = "failed"
            candidate.error_code = "person_world_recompile_failed"
            candidate.error_message = "人物世界候选调查未完成"
            service.session.commit()
            raise
        except LightRAGSidecarError as error:
            candidate.status = "failed"
            candidate.error_code = error.code
            candidate.error_message = error.safe_message
            service.session.commit()
            raise JobHandlerError(error.code, error.safe_message) from error
        except Exception as error:
            candidate.status = "failed"
            candidate.error_code = "person_world_recompile_failed"
            candidate.error_message = f"{type(error).__name__}"[:200]
            service.session.commit()
            raise JobHandlerError(
                "person_world_recompile_failed", f"人物世界候选调查失败：{type(error).__name__}"
            ) from error
        finally:
            if owned_compiler is not None:
                owned_compiler.close()
            if owned_sidecar is not None:
                owned_sidecar.close()

    return handler


def _apply_lightrag_metadata(candidate: WorldGraphVersion, metadata: LightRAGMetadata) -> None:
    candidate.lightrag_version = metadata.lightrag_version
    candidate.embedding_model = metadata.embedding_model
    candidate.embedding_dimension = metadata.embedding_dimension
    candidate.extraction_model = metadata.extraction_model
    candidate.chunking_strategy = metadata.chunking_strategy
    candidate.chunk_token_size = metadata.chunk_token_size
    candidate.chunk_overlap_token_size = metadata.chunk_overlap_token_size
    candidate.entity_prompt_version = metadata.entity_prompt_version


def enqueue_graph_patch_job(
    service: JobService,
    *,
    settings: Settings,
    change_set: WorldGraphChangeSet,
) -> Job:
    return service.enqueue_unique(
        WORLD_GRAPH_PATCH_JOB_KIND,
        {
            "project_id": change_set.project_id,
            "change_set_id": change_set.id,
            "base_graph_version_id": change_set.base_graph_version_id,
            "payload_hash": change_set.canonical_payload_hash,
            "sidecar_url": settings.lightrag_sidecar_url,
            "compiler_endpoint": settings.node_analysis_endpoint,
            "compiler_model": settings.node_analysis_model,
        },
        dedupe_key=(
            f"{WORLD_GRAPH_PATCH_JOB_KIND}:{change_set.id}:{change_set.canonical_payload_hash}"
        ),
    )


def create_graph_patch_handler(
    settings: Settings,
    *,
    lightrag_client: LightRAGSidecarClient | None = None,
    compiler_client: NodeAnalysisCloudClient | None = None,
) -> JobHandler:
    def handler(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError("world_patch_lease_missing", "图谱修改任务缺少 Worker 租约")
        change_set_id = str(job.payload.get("change_set_id", ""))
        expected_hash = str(job.payload.get("payload_hash", ""))
        change_set = service.session.get(WorldGraphChangeSet, change_set_id)
        if change_set is None:
            raise JobHandlerError("world_patch_missing", "Graph ChangeSet 不存在")
        if change_set.canonical_payload_hash != expected_hash:
            raise JobHandlerError("world_patch_changed", "Graph ChangeSet 已变化，必须重新批准")
        if change_set.status == "publish_ready":
            service.checkpoint(
                job.id,
                {
                    "stage": "publish_ready",
                    "progress": 1.0,
                    "candidate_graph_version_id": change_set.candidate_graph_version_id,
                    "profile_id": dict(change_set.execution_result or {}).get("profile_id"),
                },
                token=token,
            )
            return
        if change_set.status not in {"approved", "applying", "applied"}:
            raise JobHandlerError("world_patch_not_approved", "Graph ChangeSet 尚未批准")
        base = service.session.get(WorldGraphVersion, change_set.base_graph_version_id)
        if base is None:
            raise JobHandlerError("world_patch_base_missing", "基础图版本不存在")
        revision_session = service.session.get(
            PersonWorldRevisionSession, change_set.revision_session_id
        )
        if revision_session is None:
            raise JobHandlerError("world_revision_missing", "纠正会话不存在")
        stale_reason = _stale_base_reason(service, revision_session, base)
        if stale_reason is not None:
            _invalidate_stale_graph_job(
                service,
                revision=revision_session,
                change_set=change_set,
                reason=stale_reason,
            )
            service.session.commit()
            raise JobHandlerError(
                "world_patch_stale_base",
                "人物世界的活动版本已变化；旧 Patch 不能再构建候选图",
            )

        owned_sidecar = None
        owned_compiler = None
        sidecar = lightrag_client
        compiler = compiler_client
        candidate: WorldGraphVersion | None = None
        try:
            if sidecar is None:
                owned_sidecar = LightRAGSidecarClient(
                    settings.lightrag_sidecar_url,
                    settings.lightrag_sidecar_token.get_secret_value(),
                    timeout_seconds=settings.lightrag_timeout_seconds,
                )
                sidecar = owned_sidecar
            if compiler is None:
                if settings.node_analysis_api_key is None:
                    raise JobHandlerError(
                        "world_compiler_unavailable", "人物世界 Agent 缺少认知模型 API key"
                    )
                owned_compiler = NodeAnalysisCloudClient(
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
                compiler = owned_compiler

            candidate = _ensure_candidate(service, change_set, base)
            revision_session.status = "executing"
            service.session.commit()
            base_bundles = list(
                service.session.scalars(
                    select(ConversationBundle)
                    .where(ConversationBundle.graph_version_id == base.id)
                    .order_by(ConversationBundle.ordinal.asc())
                )
            )
            candidate_bundles = _clone_bundles(service, base_bundles, candidate)
            for offset in range(0, len(candidate_bundles), LIGHTRAG_INDEX_BATCH_SIZE):
                batch = candidate_bundles[offset : offset + LIGHTRAG_INDEX_BATCH_SIZE]
                metadata = sidecar.index_documents(
                    candidate.workspace_key,
                    [
                        LightRAGDocument(
                            id=item.document_id, source=item.source_name, text=item.content
                        )
                        for item in batch
                    ],
                )
                for item in batch:
                    item.indexed = True
                candidate.lightrag_version = metadata.lightrag_version
                candidate.embedding_model = metadata.embedding_model
                candidate.embedding_dimension = metadata.embedding_dimension
                candidate.extraction_model = metadata.extraction_model
                candidate.chunking_strategy = metadata.chunking_strategy
                candidate.chunk_token_size = metadata.chunk_token_size
                candidate.chunk_overlap_token_size = metadata.chunk_overlap_token_size
                candidate.entity_prompt_version = metadata.entity_prompt_version
                service.session.commit()
                completed = min(offset + len(batch), len(candidate_bundles))
                service.checkpoint(
                    job.id,
                    {
                        "stage": "candidate_graph_building",
                        "progress": 0.05 + 0.45 * completed / max(1, len(candidate_bundles)),
                        "indexed_bundles": completed,
                        "bundle_count": len(candidate_bundles),
                        "candidate_graph_version_id": candidate.id,
                    },
                    token=token,
                )
            candidate.status = "candidate"
            service.session.commit()

            executor = WorldGraphExecutor(service.session, sidecar)
            replayed: set[str] = set()
            active_corrections = list(
                service.session.scalars(
                    select(WorldCorrection)
                    .where(
                        WorldCorrection.project_id == change_set.project_id,
                        WorldCorrection.status == "active",
                    )
                    .order_by(WorldCorrection.approved_at.asc(), WorldCorrection.id.asc())
                )
            )
            for correction in active_corrections:
                old_change_set = (
                    service.session.get(WorldGraphChangeSet, correction.approved_change_set_id)
                    if correction.approved_change_set_id
                    else None
                )
                if old_change_set is None or old_change_set.id in replayed:
                    continue
                executor.replay(change_set=old_change_set, candidate_graph=candidate)
                replayed.add(old_change_set.id)
            service.checkpoint(
                job.id,
                {
                    "stage": "corrections_replaying",
                    "progress": 0.55,
                    "replayed_change_sets": len(replayed),
                    "candidate_graph_version_id": candidate.id,
                },
                token=token,
            )
            if change_set.status == "applying":
                # Worker 在部分操作完成后重启：持久化操作日志和 Sidecar
                # idempotency_key 会跳过已成功项，因此可从批准态安全续跑。
                change_set.status = "approved"
                candidate.status = "candidate"
                service.session.flush()
            if change_set.status == "approved":
                executor.apply(change_set=change_set, candidate_graph=candidate)
            else:
                # 图操作已全部完成但 Profile 重编译前重启，无需再应用 Patch。
                candidate.status = "validating"
            service.session.commit()
            service.checkpoint(
                job.id,
                {
                    "stage": "graph_patch_applied",
                    "progress": 0.65,
                    "candidate_graph_version_id": candidate.id,
                },
                token=token,
            )

            subject_name, subject_person_id, user_name = _participant_names(
                service.session, candidate.project_id
            )

            def report(stage: str, completed: int, total: int) -> None:
                service.checkpoint(
                    job.id,
                    {
                        "stage": "profile_recompiling",
                        "agent_stage": stage,
                        "progress": 0.65 + 0.20 * completed / max(1, total),
                        "completed_questions": completed,
                        "question_count": total,
                        "candidate_graph_version_id": candidate.id,
                    },
                    token=token,
                )

            from .review.profile_v3 import approved_v3_result

            result = approved_v3_result(
                service.session,
                revision=revision_session,
                change_set=change_set,
                candidate_graph=candidate,
            )
            if result is None:
                if revision_session.scope.get("node_scope"):
                    raise JobHandlerError(
                        "node_revision_contract_missing",
                        "节点纠正缺少已批准的 v3 产物，禁止改走全量重编译",
                    )
                result = PersonWorldCoordinator(
                    session=service.session,
                    graph=candidate,
                    lightrag=sidecar,
                    compiler=compiler,
                    subject_name=subject_name,
                    user_name=user_name,
                    top_k=settings.lightrag_query_top_k,
                    chunk_top_k=settings.lightrag_query_chunk_top_k,
                    max_total_tokens=settings.lightrag_query_max_total_tokens,
                    section_concurrency=settings.person_world_section_concurrency,
                    section_runtime_policy=section_policy(settings),
                    progress=report,
                    correction_change_set_ids={change_set.id},
                ).run(mode="recompile", resume_key=f"job:{job.id}")
            patch_source_ids = [
                value
                for item in change_set.profile_patch
                if isinstance(item, dict)
                for value in item.get("source_message_ids", [])
                if isinstance(value, str)
            ]
            profile = _persist_profile(
                service.session,
                graph=candidate,
                subject_person_id=subject_person_id,
                compiled=CompiledWorldProfile(
                    # 用户批准的 Profile Patch 已作为结构化纠正提供给七栏目 Agent；这里
                    # 只持久化它们重新核验后的 v2 投影。绝不能再按展示文本覆盖旧草稿，
                    # 否则同一人物会同时有两个相互竞争的事实来源。
                    draft=result.draft,
                    source_message_ids=tuple(
                        dict.fromkeys([*result.source_message_ids, *patch_source_ids])
                    ),
                    retrieval_manifest=result.retrieval_manifest,
                ),
                agent_run_id=result.run_id,
                generation_summary=result.generation_summary,
                profile_v3=result.profile_v3.model_dump(mode="json"),
                profile_schema_version="v3",
                investigation_report=result.investigation_report.model_dump(mode="json"),
            )
            validation = _validate_regressions(
                session=service.session,
                compiler=compiler,
                sidecar=sidecar,
                graph=candidate,
                change_set=change_set,
                correction_payloads=[
                    item.corrected_interpretation
                    for item in service.session.scalars(
                        select(WorldCorrection).where(
                            WorldCorrection.approved_change_set_id == change_set.id
                        )
                    )
                ],
            )
            if validation.checks and not all(item.passed for item in validation.checks):
                candidate.status = "failed"
                change_set.status = "validation_failed"
                revision_session.status = "failed"
                change_set.execution_result = {
                    **dict(change_set.execution_result or {}),
                    "profile_id": profile.id,
                    "validation": validation.model_dump(mode="json"),
                }
                service.session.commit()
                raise JobHandlerError("world_patch_validation_failed", "候选图没有通过纠正回归验证")
            candidate.status = "publish_ready"
            candidate.completed_at = datetime.now(UTC)
            change_set.status = "publish_ready"
            change_set.candidate_graph_version_id = candidate.id
            change_set.execution_result = {
                **dict(change_set.execution_result or {}),
                "profile_id": profile.id,
                "agent_run_id": result.run_id,
                "candidate_generation": result.generation_summary,
                "validation": validation.model_dump(mode="json"),
            }
            revision_session.status = "publish_ready"
            service.session.commit()
            service.checkpoint(
                job.id,
                {
                    "stage": "publish_ready",
                    "progress": 1.0,
                    "candidate_graph_version_id": candidate.id,
                    "profile_id": profile.id,
                    "validation": validation.model_dump(mode="json"),
                },
                token=token,
            )
        except JobHandlerError:
            raise
        except LightRAGSidecarError as error:
            if candidate is not None:
                candidate.status = "failed"
            change_set.status = "validation_failed"
            revision_session.status = "failed"
            service.session.commit()
            raise JobHandlerError(error.code, error.safe_message) from error
        except Exception as error:
            if candidate is not None:
                candidate.status = "failed"
            change_set.status = "validation_failed"
            revision_session.status = "failed"
            service.session.commit()
            raise JobHandlerError(
                "world_patch_failed", f"候选图修改失败：{type(error).__name__}"
            ) from error
        finally:
            if owned_compiler is not None:
                owned_compiler.close()
            if owned_sidecar is not None:
                owned_sidecar.close()

    return handler


def _stale_base_reason(
    service: JobService,
    revision: PersonWorldRevisionSession,
    base: WorldGraphVersion,
) -> str | None:
    """候选图 Worker 的最终防线：只比较发布对的稳定 UUID。"""

    publication = service.session.scalar(
        select(WorldPublication)
        .where(
            WorldPublication.project_id == revision.project_id,
            WorldPublication.status == "active",
            WorldPublication.node_boundary_hash
            == (revision.scope.get("node_scope") or {}).get("preview_hash"),
        )
        .order_by(WorldPublication.published_at.desc(), WorldPublication.id.desc())
    )
    if revision.scope.get("base_profile_hash"):
        from .review.profile_v3 import revision_base_is_current

        return (
            None
            if revision_base_is_current(service.session, revision, publication)
            else "v3_profile_or_publication_changed"
        )
    if publication is not None:
        if publication.graph_version_id != revision.base_graph_version_id:
            return "active_graph_changed"
        if publication.profile_id != revision.base_profile_id:
            return "active_profile_changed"
    elif base.status == "superseded":
        return "base_graph_superseded"
    return None


def _invalidate_stale_graph_job(
    service: JobService,
    *,
    revision: PersonWorldRevisionSession,
    change_set: WorldGraphChangeSet,
    reason: str,
) -> None:
    """Worker 发现发布竞争时只作废候选产物，不触碰当前活动图。"""

    now = datetime.now(UTC)
    candidate = (
        service.session.get(WorldGraphVersion, change_set.candidate_graph_version_id)
        if change_set.candidate_graph_version_id
        else None
    )
    if candidate is not None and candidate.status not in {"ready", "superseded"}:
        candidate.status = "rejected"
        candidate.completed_at = now
    change_set.status = "stale"
    for correction in service.session.scalars(
        select(WorldCorrection).where(
            WorldCorrection.approved_change_set_id == change_set.id,
            WorldCorrection.status.in_(["proposed", "approved"]),
        )
    ):
        correction.status = "revoked"
    if revision.status != "stale":
        revision.scope = {
            **dict(revision.scope or {}),
            "stale_base": {
                "reason": reason,
                "base_graph_version_id": revision.base_graph_version_id,
                "base_profile_id": revision.base_profile_id,
                "invalidated_at": now.isoformat(),
            },
        }
        revision.status = "stale"
        revision.pending_turn_id = None
        revision.understanding_payload = None
        revision.understanding_payload_hash = None
        revision.session_revision += 1
        service.session.add(
            PersonWorldRevisionMessage(
                session_id=revision.id,
                role="system",
                kind="base_stale",
                session_revision=revision.session_revision,
                content="活动人物世界版本已变化，已批准的候选图不会应用。请基于最新版本重新开始。",
                payload={
                    "reason": reason,
                    "base_graph_version_id": revision.base_graph_version_id,
                    "base_profile_id": revision.base_profile_id,
                },
            )
        )
    service.session.flush()


def _ensure_candidate(
    service: JobService,
    change_set: WorldGraphChangeSet,
    base: WorldGraphVersion,
) -> WorldGraphVersion:
    if change_set.candidate_graph_version_id:
        existing = service.session.get(WorldGraphVersion, change_set.candidate_graph_version_id)
        if existing is not None:
            return existing
    candidate_id = str(uuid4())
    candidate = WorldGraphVersion(
        id=candidate_id,
        project_id=base.project_id,
        trigger_import_id=base.trigger_import_id,
        workspace_key=f"world_{candidate_id.replace('-', '')}",
        status="building",
        source_fingerprint=base.source_fingerprint,
        config_fingerprint=hashlib.sha256(
            f"{base.config_fingerprint}:{change_set.canonical_payload_hash}".encode()
        ).hexdigest(),
        source_import_ids=list(base.source_import_ids),
        message_count=base.message_count,
        bundle_count=base.bundle_count,
        compiler_version=COMPILER_VERSION,
        parent_version_id=base.id,
        revision=base.revision + 1,
        correction_head_hash=change_set.canonical_payload_hash,
        change_set_id=change_set.id,
    )
    service.session.add(candidate)
    service.session.flush()
    change_set.candidate_graph_version_id = candidate.id
    service.session.commit()
    return candidate


def _clone_bundles(
    service: JobService,
    base_bundles: list[ConversationBundle],
    candidate: WorldGraphVersion,
) -> list[ConversationBundle]:
    existing = {
        item.document_id: item
        for item in service.session.scalars(
            select(ConversationBundle).where(ConversationBundle.graph_version_id == candidate.id)
        )
    }
    output: list[ConversationBundle] = []
    for base in base_bundles:
        clone = existing.get(base.document_id)
        if clone is None:
            clone = ConversationBundle(
                project_id=candidate.project_id,
                graph_version_id=candidate.id,
                document_id=base.document_id,
                source_name=base.source_name,
                ordinal=base.ordinal,
                started_at=base.started_at,
                ended_at=base.ended_at,
                content=base.content,
                content_hash=base.content_hash,
                primary_message_count=base.primary_message_count,
                carry_in_message_count=base.carry_in_message_count,
                indexed=False,
            )
            service.session.add(clone)
            service.session.flush()
            mappings = list(
                service.session.scalars(
                    select(ConversationBundleMessage)
                    .where(ConversationBundleMessage.bundle_id == base.id)
                    .order_by(ConversationBundleMessage.ordinal.asc())
                )
            )
            for mapping in mappings:
                service.session.add(
                    ConversationBundleMessage(
                        bundle_id=clone.id,
                        message_id=mapping.message_id,
                        ordinal=mapping.ordinal,
                        is_carry_in=mapping.is_carry_in,
                    )
                )
        output.append(clone)
    service.session.commit()
    return output


def _participant_names(session: Any, project_id: str) -> tuple[str, str, str]:
    from moonlightbox.imports.models import Message, Participant

    rows = list(
        session.scalars(
            select(Participant)
            .join(Message, Message.participant_id == Participant.id)
            .where(Message.project_id == project_id)
            .distinct()
        )
    )
    target = next((item for item in rows if item.role == "target"), None)
    user = next((item for item in rows if item.role == "self"), None)
    if target is None or user is None:
        raise JobHandlerError("world_participant_missing", "聊天缺少 target 或 self 角色")
    return target.name, target.id, user.name


def _validate_regressions(
    *,
    session=None,
    compiler: NodeAnalysisCloudClient,
    sidecar: LightRAGSidecarClient,
    graph: WorldGraphVersion,
    change_set: WorldGraphChangeSet,
    correction_payloads: list[dict[str, object]],
) -> RegressionValidationBatch:
    queries: list[str] = [
        str(item["query"])
        for item in change_set.regression_queries
        if isinstance(item, dict) and isinstance(item.get("query"), str)
    ]
    if not queries:
        return RegressionValidationBatch(checks=[])
    expected_by_query = {
        str(item["query"]): str(item.get("expected_change", ""))
        for item in change_set.regression_queries
        if isinstance(item, dict) and isinstance(item.get("query"), str)
    }
    from langchain_core.tools import StructuredTool

    from moonlightbox.agent_runtime.contracts import RegisteredTool, ToolContract
    from moonlightbox.agent_runtime.tool_execution import serial

    actual_by_query = {}

    def read_regression_query(query: str):
        """按已批准的回归查询读取候选图谱，不允许变更查询范围。"""
        if query not in expected_by_query:
            from moonlightbox.agent_runtime.tool_errors import ToolInputError

            raise ToolInputError("query 必须原样使用当前批准的回归清单中的查询，不能改写或新增。")
        retrieval = sidecar.query(
            graph.workspace_key, query, mode="mix", top_k=16, chunk_top_k=8, max_total_tokens=6000
        )
        actual = {
            "expected_change": expected_by_query[query],
            "actual_context_excerpt": retrieval.context[:4000],
            "source_references": list(
                dict.fromkeys(ref.file_path for ref in retrieval.references if ref.file_path)
            )[:30],
        }
        actual_by_query[query] = actual
        return {"query": query, "context": retrieval.context, **actual}

    from .tools.graph_query import ReadRegressionQueryArgs

    tool = StructuredTool.from_function(read_regression_query, args_schema=ReadRegressionQueryArgs)
    from pathlib import Path

    def validate_assessment(result, context):
        found = [item.query for item in result.checks]
        if len(found) != len(set(found)) or set(found) != set(queries):
            return "每个输入查询必须恰好评估一次，不能增加、重复或遗漏。"
        if set(actual_by_query) != set(queries):
            return "尚有查询未读取实际结果，请完成对应查询再提交。"
        return None

    assessed = run_submission_task(
        compiler=compiler,
        name="graph_regression",
        system_prompt=Path(__file__)
        .with_name("prompts")
        .joinpath("graph_regression.md")
        .read_text(),
        payload={"approved_corrections": correction_payloads, "queries": expected_by_query},
        result_model=RegressionAssessmentBatch,
        owner_id=f"{change_set.id}:{graph.id}",
        session=session,
        project_id=graph.project_id,
        validator=validate_assessment,
        tools=(
            RegisteredTool(
                tool,
                ToolContract(
                    name=tool.name,
                    timeout_seconds=300,
                    execution=serial("graph_regression", reason="查询后保存核验结果"),
                ),
            ),
        ),
        snapshot_work_state=lambda: dict(actual_by_query),
        restore_work_state=lambda state: actual_by_query.update(state),
    )
    return RegressionValidationBatch(
        checks=[
            {**check.model_dump(mode="json"), **actual_by_query.get(check.query, {})}
            for check in assessed.checks
        ]
    )
