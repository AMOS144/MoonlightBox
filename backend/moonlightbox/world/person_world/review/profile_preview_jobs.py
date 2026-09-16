"""已确认理解后的模块/跨栏补全后台任务。HTTP 不等待多轮云端推理。"""

from sqlalchemy.orm import Session

from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.jobs.registry import JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.world.client import LightRAGSidecarClient
from moonlightbox.world.models import (
    PersonWorldProfile,
    PersonWorldRevisionSession,
    WorldGraphVersion,
)

from .profile_v3 import create_v3_change_set
from .service import PersonWorldReviewService

PROFILE_PREVIEW_JOB_KIND = "person_world_profile_preview_v3"


def enqueue_profile_preview(service, *, revision):
    return service.enqueue_unique(
        PROFILE_PREVIEW_JOB_KIND,
        {
            "project_id": revision.project_id,
            "revision_session_id": revision.id,
            "input_revision": revision.input_revision,
            "understanding_hash": revision.understanding_payload_hash,
            "attempt": revision.scope["profile_preview_attempt"],
        },
        dedupe_key=f"{PROFILE_PREVIEW_JOB_KIND}:{revision.id}:{revision.input_revision}:{revision.scope['profile_preview_attempt']}",
        commit=False,
    )


def create_profile_preview_handler(settings, *, compiler_client=None, lightrag_client=None):
    def handler(service, job):
        revision = service.session.get(
            PersonWorldRevisionSession, job.payload["revision_session_id"]
        )
        expected = job.payload["input_revision"]
        expected_hash = job.payload["understanding_hash"]
        expected_attempt = job.payload["attempt"]
        database_bind = service.session.get_bind()

        def owns():
            service.session.refresh(revision)
            return (
                revision.status == "profile_compiling"
                and revision.input_revision == expected
                and revision.understanding_payload_hash == expected_hash
                and revision.scope.get("profile_preview_attempt") == expected_attempt
                and service.is_running_with_token(job.id, job.worker_token)
            )

        if revision is None or not owns():
            return
        compiler = compiler_client
        sidecar = lightrag_client
        try:
            if compiler is None:
                compiler = NodeAnalysisCloudClient(
                    enabled=settings.node_analysis_enabled,
                    endpoint=settings.node_analysis_endpoint,
                    model=settings.node_analysis_model,
                    api_key=settings.node_analysis_api_key,
                    response_format=settings.node_analysis_response_format,
                    thinking_mode=settings.node_analysis_thinking_mode,
                    max_retries=settings.node_analysis_max_retries,
                    timeout_seconds=settings.world_compiler_timeout_seconds,
                    max_output_tokens=settings.node_analysis_max_output_tokens,
                )
            if sidecar is None:
                sidecar = LightRAGSidecarClient(
                    settings.lightrag_sidecar_url,
                    settings.lightrag_sidecar_token.get_secret_value(),
                    timeout_seconds=settings.lightrag_timeout_seconds,
                )
            from moonlightbox.agent_runtime.policy import section_policy

            review = PersonWorldReviewService(service.session, compiler=compiler, lightrag=sidecar)
            review.section_runtime_policy = section_policy(settings)
            review._ensure_base_is_current(revision)
            revision_id, job_id, worker_token = revision.id, job.id, job.worker_token

            def current_input():
                # Harness 可在工具执行线程中检查；不跨线程共享 ORM Session。
                with Session(bind=database_bind) as check:
                    current = check.get(PersonWorldRevisionSession, revision_id)
                    valid = (
                        current is not None
                        and current.status == "profile_compiling"
                        and current.input_revision == expected
                        and current.understanding_payload_hash == expected_hash
                        and current.scope.get("profile_preview_attempt") == expected_attempt
                        and JobService(check).is_running_with_token(job_id, worker_token)
                    )
                    return expected if valid else None

            review.profile_preview_current_input = current_input
            steps = 0

            def progress(section):
                nonlocal steps
                if not owns():
                    raise ValueError("已取消、输入改变或 Worker 租约失效")
                steps += 1
                stage = (
                    "module_revising" if section == "life_context" else "related_sections_revising"
                )
                revision.scope = {
                    **revision.scope,
                    "agent_stage": stage,
                    "active_agent_job_id": job.id,
                    "agent_stage_progress": min(0.9, steps / 8),
                }
                revision.session_revision += 1
                service.session.commit()
                service.checkpoint(
                    job.id,
                    {"stage": stage, "section": section, "progress": min(0.9, steps / 8)},
                    token=job.worker_token,
                )

            review.profile_preview_progress = progress
            profile = service.session.get(PersonWorldProfile, revision.base_profile_id)
            graph = service.session.get(WorldGraphVersion, revision.base_graph_version_id)
            if profile is None or graph is None:
                raise ValueError("确认时的基础画像或图版本不存在")
            change_set = create_v3_change_set(review, revision, profile, graph)
            revision.scope = {
                **revision.scope,
                "agent_stage": "profile_diff_ready",
                "agent_stage_progress": 1.0,
                "active_agent_job_id": None,
            }
            service.session.commit()
            service.checkpoint(
                job.id,
                {"stage": "profile_diff_ready", "change_set_id": change_set.id, "progress": 1.0},
                token=job.worker_token,
            )
        except Exception as error:
            service.session.rollback()
            if owns():
                revision.status = "understanding_ready"
                revision.scope = {
                    **revision.scope,
                    "profile_preview_error": (
                        "合并候选生成失败，请查看 Phoenix 后重试；尚未修改画像或图谱。"
                    ),
                    "agent_stage": "profile_preview_failed",
                    "active_agent_job_id": None,
                }
                revision.session_revision += 1
                service.session.commit()
            raise JobHandlerError("profile_preview_failed", "人物画像合并候选未完成") from error
        finally:
            if compiler is not None and compiler_client is None:
                compiler.close()
            if sidecar is not None and lightrag_client is None:
                sidecar.close()

    return handler
