"""Revision Agent 回合的可恢复后台任务。

HTTP 只记录用户输入和排队意图；实际 LangGraph 调查与模型调用在 Worker 中完成。这样
重启、租约到期或网络失败不会把一次半完成的 Agent 回合伪装成用户已经收到的回答。
"""

from __future__ import annotations

from typing import Any

from moonlightbox.config import Settings
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient, NodeAnalysisCloudError
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.world.client import LightRAGSidecarClient
from moonlightbox.world.models import PersonWorldRevisionSession

from .service import PersonWorldReviewService, RevisionBaseStaleError, RevisionStateError

REVISION_TURN_JOB_KIND = "person_world_revision_turn_v1"


def enqueue_revision_turn_job(
    service: JobService,
    *,
    settings: Settings,
    revision: PersonWorldRevisionSession,
) -> Job:
    """为一个不可变的用户/范围版本排队恰好一个 Agent 回合。"""

    return service.enqueue_unique(
        REVISION_TURN_JOB_KIND,
        {
            "project_id": revision.project_id,
            "revision_session_id": revision.id,
            "input_session_revision": revision.session_revision,
            # 仅记录可审计的非敏感配置标识；密钥始终由 Worker 自己的 Settings 读取。
            "compiler_endpoint": settings.node_analysis_endpoint,
            "compiler_model": settings.node_analysis_model,
            "lightrag_sidecar_url": settings.lightrag_sidecar_url,
        },
        dedupe_key=(f"{REVISION_TURN_JOB_KIND}:{revision.id}:{revision.session_revision}"),
    )


def create_revision_turn_handler(
    settings: Settings,
    *,
    lightrag_client: LightRAGSidecarClient | None = None,
    compiler_client: NodeAnalysisCloudClient | None = None,
) -> JobHandler:
    """构建 Worker 使用的回合处理器；客户端可注入以便做无网络冒烟测试。"""

    def handler(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError("revision_turn_lease_missing", "纠正 Agent 任务缺少 Worker 租约")
        revision_id = _required_string(job.payload, "revision_session_id")
        expected_revision = _required_int(job.payload, "input_session_revision")
        revision = service.session.get(PersonWorldRevisionSession, revision_id)
        if revision is None:
            raise JobHandlerError("revision_turn_missing", "纠正会话不存在")
        if not _owns_or_may_start(revision, job_id=job.id, expected_revision=expected_revision):
            service.checkpoint(
                job.id,
                {"stage": "stale", "progress": 1.0, "session_id": revision.id},
                token=token,
            )
            return

        scope = dict(revision.scope or {})
        scope["active_agent_job_id"] = job.id
        scope["agent_stage"] = "context_assembling"
        scope["agent_stage_progress"] = _STAGE_PROGRESS["context_assembling"]
        revision.scope = scope
        revision.status = "agent_running"
        revision.session_revision += 1
        service.session.commit()

        def report_stage(stage: str) -> None:
            """同步 Job checkpoint 与 Session SSE 状态，不存储工具原文。"""

            if not service.is_running_with_token(job.id, token):
                raise JobHandlerError(
                    "revision_turn_cancelled",
                    "纠正 Agent 回合已取消或租约失效",
                )
            current = service.session.get(PersonWorldRevisionSession, revision_id)
            if current is None or current.status != "agent_running":
                raise JobHandlerError(
                    "revision_turn_cancelled",
                    "纠正会话已取消或不再允许写入",
                )
            current_scope = dict(current.scope or {})
            if current_scope.get("active_agent_job_id") != job.id:
                raise JobHandlerError(
                    "revision_turn_cancelled",
                    "纠正 Agent 回合已被新的任务替代",
                )
            progress = _STAGE_PROGRESS[stage]
            changed = (
                current_scope.get("agent_stage") != stage
                or current_scope.get("agent_stage_progress") != progress
            )
            if changed:
                current.scope = {
                    **current_scope,
                    "agent_stage": stage,
                    "agent_stage_progress": progress,
                }
                # session_revision 同时是 SSE 的事件游标；阶段变化也必须推送给已打开的
                # 工作区，且 Agent 运行态不允许用户提交与之竞争的写入。
                current.session_revision += 1
                service.session.commit()
            service.checkpoint(
                job.id,
                {
                    "stage": stage,
                    "progress": progress,
                    "session_id": revision_id,
                    "input_session_revision": expected_revision,
                    "session_revision": current.session_revision,
                },
                token=token,
            )

        report_stage("context_assembling")

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
                        "revision_compiler_unavailable", "人物世界 Agent 缺少认知模型 API key"
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

            PersonWorldReviewService(
                service.session,
                compiler=compiler,
                lightrag=sidecar,
            ).advance(revision, on_stage=report_stage)
            # Agent 已写入等待用户或共同理解状态后才允许任务成功。若 Worker 被取消，
            # JobService 的 CAS 会失败，避免把迟到的模型结果作为有效回合提交。
            if not service.is_running_with_token(job.id, token):
                raise JobHandlerError(
                    "revision_turn_lease_lost", "纠正 Agent 回合已被取消或租约失效"
                )
            scope = dict(revision.scope or {})
            scope.pop("active_agent_job_id", None)
            scope["agent_stage"] = "completed"
            scope["agent_stage_progress"] = 1.0
            revision.scope = scope
            service.session.commit()
            service.checkpoint(
                job.id,
                {
                    "stage": "completed",
                    "progress": 1.0,
                    "session_id": revision.id,
                    "result_status": revision.status,
                    "session_revision": revision.session_revision,
                },
                token=token,
            )
        except JobHandlerError:
            _mark_agent_failure(service, revision)
            raise
        except NodeAnalysisCloudError as error:
            _mark_agent_failure(service, revision, error_code=error.code)
            raise JobHandlerError(
                "revision_agent_cloud_error", "纠正 Agent 调用认知模型失败"
            ) from error
        except RevisionBaseStaleError as error:
            # ``advance`` 已用稳定发布指针作废旧 Scope/Patch；不要把它再覆盖成
            # agent_failed，否则用户会被错误提示去重试一个已经不再有效的基线。
            service.session.commit()
            raise JobHandlerError(
                "revision_agent_stale_base",
                "人物世界活动版本已变化，请基于最新版本重新开始纠正",
            ) from error
        except (RevisionStateError, ValueError, TypeError) as error:
            _mark_agent_failure(service, revision, error_code="revision_agent_protocol_error")
            raise JobHandlerError(
                "revision_agent_protocol_error", "纠正 Agent 未能形成有效回合"
            ) from error
        except Exception as error:
            _mark_agent_failure(service, revision, error_code="revision_agent_failed")
            raise JobHandlerError("revision_agent_failed", "纠正 Agent 执行失败") from error
        finally:
            if owned_compiler is not None:
                owned_compiler.close()
            if owned_sidecar is not None:
                owned_sidecar.close()

    return handler


def _owns_or_may_start(
    revision: PersonWorldRevisionSession,
    *,
    job_id: str,
    expected_revision: int,
) -> bool:
    """只允许当前输入版本或同一任务的中断重试继续执行。"""

    if revision.status == "agent_queued" and revision.session_revision == expected_revision:
        return True
    return (
        revision.status == "agent_running"
        and isinstance(revision.scope, dict)
        and revision.scope.get("active_agent_job_id") == job_id
    )


def _mark_agent_failure(
    service: JobService,
    revision: PersonWorldRevisionSession,
    *,
    error_code: str = "revision_agent_failed",
) -> None:
    """将失败显式呈现给会话，而不把半个回合留在 ``agent_running``。"""

    if revision.status != "agent_running":
        return
    scope = dict(revision.scope or {})
    scope.pop("active_agent_job_id", None)
    scope["agent_stage"] = "failed"
    scope["agent_stage_progress"] = 1.0
    scope["agent_error_code"] = str(error_code)
    revision.scope = scope
    revision.status = "agent_failed"
    revision.session_revision += 1
    service.session.commit()


_STAGE_PROGRESS: dict[str, float] = {
    "context_assembling": 0.15,
    "agent_deliberating": 0.40,
    "tool_running": 0.65,
    "turn_validating": 0.85,
}


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise JobHandlerError("revision_turn_payload_invalid", "纠正 Agent 任务参数无效")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise JobHandlerError("revision_turn_payload_invalid", "纠正 Agent 任务参数无效")
    return value
