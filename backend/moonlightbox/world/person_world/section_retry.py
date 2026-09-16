"""PersonWorld 失败栏目独立重试。

重试不是重新运行七个栏目，也不会触碰已发布的 Graph/Profile。它只允许在候选 Profile
审核阶段，对 ``cloud_error`` 或 ``schema_error`` 的一个栏目重新调查；其余栏目继续复用
已经通过契约的结构化结果与 Claim 身份。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from moonlightbox.config import Settings
from moonlightbox.events.cloud_client import (
    NodeAnalysisCloudClient,
    NodeAnalysisCloudError,
)
from moonlightbox.imports.models import Participant
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.world.client import LightRAGSidecarClient, LightRAGSidecarError
from moonlightbox.world.models import (
    PersonWorldAgentRun,
    PersonWorldProfile,
    PersonWorldProfileDraft,
    PersonWorldSectionTask,
    WorldGraphVersion,
)

PERSON_WORLD_SECTION_RETRY_JOB_KIND = "person_world_section_retry_v1"
_RETRYABLE_SECTION_STATES = frozenset({"cloud_error", "schema_error", "budget_exhausted", "failed"})


class SectionRetryStateError(ValueError):
    """栏目不满足独立重试的稳定前置条件。"""


def enqueue_section_retry_job(
    service: JobService,
    *,
    settings: Settings,
    run: PersonWorldAgentRun,
    task: PersonWorldSectionTask,
    idempotency_key: str,
) -> Job:
    """请求一次栏目的可恢复重试，重复 HTTP 请求返回同一 Job。

    ``idempotency_key`` 是用户写操作的一部分，而不是拿自然语言或栏目文案做去重。
    同一次网络重放既不会第二次递增 retry_attempt，也不会启动第二个模型调用。
    """

    if not idempotency_key.strip():
        raise SectionRetryStateError("栏目重试缺少幂等键")
    if task.agent_run_id != run.id:
        raise SectionRetryStateError("栏目任务不属于指定调查运行")
    draft = service.session.scalar(
        select(PersonWorldProfileDraft).where(PersonWorldProfileDraft.agent_run_id == run.id)
    )
    if draft is None or draft.profile_schema_version != "v3":
        raise SectionRetryStateError("旧版画像已退役，请重新生成 v3；历史数据仍可查看")
    # 先识别相同请求的网络重放；首个请求已把栏目从错误态切到 pending 后，
    # 不能反过来因状态变化否定幂等语义。
    dedupe_key = f"{PERSON_WORLD_SECTION_RETRY_JOB_KIND}:{task.id}:{idempotency_key}"
    existing = service.session.scalar(select(Job).where(Job.dedupe_key == dedupe_key))
    if existing is not None:
        return existing
    if task.status not in _RETRYABLE_SECTION_STATES:
        raise SectionRetryStateError("只有失败或运行预算耗尽的栏目可以单独重试")
    if run.status != "awaiting_review":
        raise SectionRetryStateError("当前调查运行不处于可重试的候选审核阶段")

    graph = service.session.get(WorldGraphVersion, run.graph_version_id)
    if graph is None:
        raise SectionRetryStateError("节点图谱不存在")
    is_node = run.mode == "node_compile"
    if is_node:
        from .node_scope import inherited_node_scope

        try:
            if (
                inherited_node_scope(service.session, graph, run.state, draft.generation_summary)
                is None
            ):
                raise ValueError("节点调查缺少冻结范围")
        except ValueError as error:
            raise SectionRetryStateError(str(error)) from error
    if (
        graph is None
        or graph.status
        not in (
            {"ready", "superseded", "awaiting_profile_review"}
            if is_node
            else {"awaiting_profile_review"}
        )
        or draft.status != "awaiting_review"
    ):
        raise SectionRetryStateError("候选人物世界已不再等待审核，不能修改旧栏目")
    previous_summary = dict(task.result_summary or {})
    previous_status = task.status
    attempt = _retry_attempt(previous_summary) + 1
    task.status = "pending"
    task.completed_at = None
    task.error_code = None
    task.error_diagnostic = {
        "retry_requested": True,
        "previous_status": previous_summary.get("last_terminal_status", previous_status),
        "retry_attempt": attempt,
    }
    task.result_summary = {
        **previous_summary,
        "retry_attempt": attempt,
        "last_terminal_status": "pending",
    }
    service.session.flush()
    job = service.enqueue_unique(
        PERSON_WORLD_SECTION_RETRY_JOB_KIND,
        {
            "project_id": run.project_id,
            "agent_run_id": run.id,
            "section_task_id": task.id,
            "section": task.section,
            "retry_attempt": attempt,
            "graph_version_id": graph.id,
            "graph_source_fingerprint": graph.source_fingerprint,
            "compiler_endpoint": settings.node_analysis_endpoint,
            "compiler_model": settings.node_analysis_model,
            "lightrag_sidecar_url": settings.lightrag_sidecar_url,
        },
        dedupe_key=dedupe_key,
        commit=False,
    )
    task.result_summary = {**dict(task.result_summary or {}), "retry_job_id": job.id}
    service.session.commit()
    return job


def create_section_retry_handler(
    settings: Settings,
    *,
    lightrag_client: LightRAGSidecarClient | None = None,
    compiler_client: NodeAnalysisCloudClient | None = None,
) -> JobHandler:
    """创建只重跑一个失败 Section Agent 的后台处理器。"""

    def handler(service: JobService, job: Job) -> None:
        retry_started = False
        token = job.worker_token
        if token is None:
            raise JobHandlerError("section_retry_lease_missing", "栏目重试任务缺少 Worker 租约")
        run_id = _payload_string(job.payload, "agent_run_id")
        task_id = _payload_string(job.payload, "section_task_id")
        section = _payload_string(job.payload, "section")
        retry_attempt = _payload_positive_int(job.payload, "retry_attempt")
        graph_id = _payload_string(job.payload, "graph_version_id")
        source_fingerprint = _payload_string(job.payload, "graph_source_fingerprint")
        run = service.session.get(PersonWorldAgentRun, run_id)
        task = service.session.get(PersonWorldSectionTask, task_id)
        graph = service.session.get(WorldGraphVersion, graph_id)
        if run is None or task is None or graph is None:
            raise JobHandlerError(
                "section_retry_missing",
                "栏目重试所需的调查运行、任务或图版本不存在",
            )
        if task.agent_run_id != run.id or task.section != section:
            raise JobHandlerError("section_retry_payload_invalid", "栏目重试任务参数与记录不匹配")
        if graph.id != run.graph_version_id or graph.source_fingerprint != source_fingerprint:
            raise JobHandlerError("section_retry_stale_graph", "栏目重试的图版本已变化")
        is_node = run.mode == "node_compile"
        if (
            graph.status
            not in (
                {"ready", "superseded", "awaiting_profile_review"}
                if is_node
                else {"awaiting_profile_review"}
            )
            or run.status != "awaiting_review"
        ):
            raise JobHandlerError("section_retry_stale_graph", "候选人物世界已不再处于可重试阶段")
        if _retry_attempt(task.result_summary) != retry_attempt:
            service.checkpoint(
                job.id,
                {"stage": "stale", "progress": 1.0, "section": section},
                token=token,
            )
            return
        if task.status not in {"pending", "researching"}:
            service.checkpoint(
                job.id,
                {"stage": "already_terminal", "progress": 1.0, "section": section},
                token=token,
            )
            return

        draft = service.session.scalar(
            select(PersonWorldProfileDraft).where(PersonWorldProfileDraft.agent_run_id == run.id)
        )
        if is_node:
            profile = draft  # 节点候选只更新自己的草稿，不碰共享图谱上的全量 Profile。
        else:
            profile = service.session.scalar(
                select(PersonWorldProfile)
                .where(PersonWorldProfile.node_boundary_hash.is_(None))
                .where(PersonWorldProfile.graph_version_id == graph.id)
            )
        if profile is None or draft is None or draft.status != "awaiting_review":
            raise JobHandlerError("section_retry_profile_missing", "候选 Profile 草稿不存在")
        # 即使是退役前已经入队的任务，也必须在创建模型客户端之前拒绝。
        if profile.profile_schema_version != "v3" or draft.profile_schema_version != "v3":
            raise JobHandlerError(
                "legacy_profile_retired", "旧版画像已退役，请重新生成 v3；历史数据仍可查看"
            )
        if profile.agent_run_id != run.id:
            raise JobHandlerError(
                "section_retry_profile_mismatch",
                "候选 Profile 不属于指定调查运行",
            )

        target, user = _participants(service, project_id=graph.project_id)
        task.status = "researching"
        task.started_at = datetime.now(UTC)
        task.completed_at = None
        task.error_code = None
        task.error_diagnostic = {"retry_attempt": retry_attempt}
        task.result_summary = {
            **dict(task.result_summary or {}),
            "retry_attempt": retry_attempt,
            "last_terminal_status": "researching",
        }
        run.trace = [
            *list(run.trace or []),
            {
                "stage": "section_retry_started",
                "section": section,
                "retry_attempt": retry_attempt,
                "job_id": job.id,
            },
        ]
        service.session.commit()
        retry_started = True
        service.checkpoint(
            job.id,
            {"stage": "researching", "progress": 0.10, "section": section},
            token=token,
        )

        owned_sidecar: LightRAGSidecarClient | None = None
        owned_compiler: NodeAnalysisCloudClient | None = None
        sidecar = lightrag_client
        compiler = compiler_client
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
                        "section_retry_compiler_unavailable",
                        "栏目重试缺少认知模型 API key",
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
            from .section_retry_v3 import retry_v3

            retry_v3(
                service,
                resume_key=f"job:{job.id}",
                run=run,
                task=task,
                graph=graph,
                profile=profile,
                draft=draft,
                target=target,
                user=user,
                compiler=compiler,
                sidecar=sidecar,
                settings=settings,
            )
            service.checkpoint(
                job.id,
                {
                    "stage": "completed",
                    "progress": 1.0,
                    "section": section,
                    "section_status": task.status,
                },
                token=token,
            )
        except (NodeAnalysisCloudError, LightRAGSidecarError) as error:
            _record_retry_failure(task=task, run=run, error=error, retry_attempt=retry_attempt)
            service.session.commit()
            if isinstance(error, LightRAGSidecarError):
                raise JobHandlerError(error.code, error.safe_message) from error
            raise JobHandlerError("section_retry_failed", "栏目重试未能完成") from error
        except JobHandlerError as error:
            # 例如当前环境缺少模型 key。任务已经进入 researching 后，同样必须回到
            # 可见、可再次重试的错误终态，不能永久停在执行中。
            if retry_started and error.code != "section_retry_stale_graph":
                _record_retry_failure(task=task, run=run, error=error, retry_attempt=retry_attempt)
                service.session.commit()
            raise
        except Exception as error:
            _record_retry_failure(task=task, run=run, error=error, retry_attempt=retry_attempt)
            service.session.commit()
            raise JobHandlerError("section_retry_failed", "栏目重试未能完成") from error
        finally:
            if owned_compiler is not None:
                owned_compiler.close()
            if owned_sidecar is not None:
                owned_sidecar.close()

    return handler


def _record_retry_failure(
    *,
    task: PersonWorldSectionTask,
    run: PersonWorldAgentRun,
    error: Exception,
    retry_attempt: int,
) -> None:
    """失败仍是错误终态，绝不能降级成“没有调查到证据”。"""

    if isinstance(error, NodeAnalysisCloudError):
        status = "cloud_error"
        code = str(error.code)
        diagnostic = {
            "attempts": error.attempts,
            "retryable": error.retryable,
            "diagnostic": dict(error.diagnostic),
        }
    elif isinstance(error, LightRAGSidecarError):
        status = "cloud_error"
        code = error.code
        diagnostic = {"safe_message": error.safe_message}
    else:
        status = "cloud_error"
        code = "section_retry_failed"
        diagnostic = {"error_type": type(error).__name__}
    task.status = status
    task.error_code = code
    task.error_diagnostic = {**diagnostic, "retry_attempt": retry_attempt}
    task.completed_at = datetime.now(UTC)
    task.result_summary = {
        **dict(task.result_summary or {}),
        "retry_attempt": retry_attempt,
        "last_terminal_status": status,
    }
    run.trace = [
        *list(run.trace or []),
        {
            "stage": "section_retry_failed",
            "section": task.section,
            "retry_attempt": retry_attempt,
            "error_code": code,
            "diagnostic": diagnostic,
        },
    ]


def _participants(service: JobService, *, project_id: str) -> tuple[Participant, Participant]:
    target = service.session.scalar(
        select(Participant)
        .where(Participant.project_id == project_id, Participant.role == "target")
        .order_by(Participant.id.asc())
    )
    user = service.session.scalar(
        select(Participant)
        .where(Participant.project_id == project_id, Participant.role == "self")
        .order_by(Participant.id.asc())
    )
    if target is None or user is None:
        raise JobHandlerError("section_retry_participant_missing", "聊天缺少 target 或 self 参与者")
    return target, user


def _retry_attempt(summary: object) -> int:
    if not isinstance(summary, dict):
        return 0
    value = summary.get("retry_attempt", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _payload_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise JobHandlerError("section_retry_payload_invalid", "栏目重试任务参数无效")
    return value


def _payload_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise JobHandlerError("section_retry_payload_invalid", "栏目重试任务参数无效")
    return value
