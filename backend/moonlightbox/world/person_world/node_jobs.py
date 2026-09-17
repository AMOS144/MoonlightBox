"""已批准节点的七栏目编译：复用图谱，只保存本次运行的独立审核草稿。"""

from sqlalchemy import select

from moonlightbox.agent_runtime.cancellation import cancellation_scope, job_signal
from moonlightbox.agent_runtime.policy import section_policy
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandlerError
from moonlightbox.world.client import LightRAGSidecarClient
from moonlightbox.world.models import PersonWorldProfileDraft, WorldGraphVersion

from .coordinator_v3 import PersonWorldCoordinatorV3
from .jobs import participant_names
from .node_scope import NodeCompilationScope

NODE_PROFILE_JOB_KIND = "person_world_node_compile_v1"


def enqueue_node_compilation(
    service, *, project_id, graph_id, investigation_id, preview_hash, idempotency_key
):
    graph = service.session.get(WorldGraphVersion, graph_id)
    if graph is None or graph.project_id != project_id:
        raise LookupError("图谱不存在")
    if graph.status not in {"ready", "superseded", "awaiting_profile_review"}:
        raise ValueError("图谱尚未构建完成")
    scope = NodeCompilationScope.from_approved(
        service.session, graph=graph, investigation_id=investigation_id, preview_hash=preview_hash
    )
    payload = {
        "project_id": project_id,
        "graph_version_id": graph.id,
        "investigation_id": investigation_id,
        "preview_hash": preview_hash,
        "node_scope": scope.envelope(),
    }
    # 相同请求可重放，但一个幂等键不能指向另外一个边界或图版本。
    key = f"{NODE_PROFILE_JOB_KIND}:{project_id}:{idempotency_key}"
    old = service.session.scalar(select(Job).where(Job.dedupe_key == key))
    if old is not None and old.payload != payload:
        raise ValueError("幂等键已用于另一份节点编译请求")
    return service.enqueue_unique(NODE_PROFILE_JOB_KIND, payload, dedupe_key=key)


def create_node_compilation_handler(settings, *, compiler_client=None, lightrag_client=None):
    def handler(service, job):
        payload = job.payload
        graph = service.session.get(WorldGraphVersion, payload["graph_version_id"])
        if graph is None or graph.project_id != payload["project_id"]:
            raise JobHandlerError("node_graph_missing", "节点图谱不存在")
        if graph.status not in {"ready", "superseded", "awaiting_profile_review"}:
            raise JobHandlerError("node_graph_not_ready", "节点图谱不再可读")
        try:
            scope = NodeCompilationScope.from_approved(
                service.session,
                graph=graph,
                investigation_id=payload["investigation_id"],
                preview_hash=payload["preview_hash"],
            )
            if scope.envelope() != payload["node_scope"]:
                raise ValueError("冻结编译范围已变化")
        except (ValueError, LookupError) as error:
            raise JobHandlerError("node_scope_stale", str(error)) from error
        if not job.worker_token:
            raise JobHandlerError("node_compile_lease_missing", "编译任务缺少租约")
        owned = []
        try:
            compiler = compiler_client
            if compiler is None:
                if not settings.node_analysis_enabled or settings.node_analysis_api_key is None:
                    raise JobHandlerError("world_compiler_unavailable", "认知模型尚未配置")
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
                owned.append(compiler)
            sidecar = lightrag_client
            if sidecar is None:
                sidecar = LightRAGSidecarClient(
                    settings.lightrag_sidecar_url,
                    settings.lightrag_sidecar_token.get_secret_value(),
                    timeout_seconds=settings.lightrag_timeout_seconds,
                )
                owned.append(sidecar)
            target, target_id, user = participant_names(service.session, graph.project_id)

            def progress(stage, completed, total):
                service.checkpoint(
                    job.id,
                    {
                        "stage": stage,
                        "progress": completed / max(1, total) * 0.95,
                        "completed_sections": completed,
                        "section_count": total,
                        "agent_run_id": getattr(coordinator, "_v3_run_id", None),
                    },
                    token=job.worker_token,
                )

            # 普通重启沿用 job 身份与七栏检查点，不复制图，也不改写全量 Profile。
            with cancellation_scope(
                job_signal(service.session.get_bind(), job.id, job.worker_token)
            ):
                coordinator = PersonWorldCoordinatorV3(
                    session=service.session,
                    graph=graph,
                    lightrag=sidecar,
                    compiler=compiler,
                    subject_name=target,
                    target_participant_id=target_id,
                    user_name=user,
                    timezone=scope.boundary["timezone"],
                    node_scope=scope,
                    top_k=settings.lightrag_query_top_k,
                    chunk_top_k=settings.lightrag_query_chunk_top_k,
                    max_total_tokens=settings.lightrag_query_max_total_tokens,
                    section_concurrency=settings.person_world_section_concurrency,
                    section_runtime_policy=section_policy(settings),
                    progress=progress,
                )
                result = coordinator.run(mode="node_compile", resume_key=f"job:{job.id}")
            draft = service.session.scalar(
                select(PersonWorldProfileDraft).where(
                    PersonWorldProfileDraft.agent_run_id == result.run_id
                )
            )
            if draft is None:
                raise JobHandlerError("node_draft_missing", "节点编译没有保存审核草稿")
            service.checkpoint(
                job.id,
                {
                    "stage": "awaiting_review",
                    "progress": 1.0,
                    "agent_run_id": result.run_id,
                    "profile_draft_id": draft.id,
                    "failed_sections": result.generation_summary.get("failed_sections", []),
                },
                token=job.worker_token,
            )
        finally:
            for client in reversed(owned):
                client.close()

    return handler
