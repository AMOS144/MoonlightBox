import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Literal

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.cloud_client import (
    NodeAnalysisCloudClient,
    NodeAnalysisCloudError,
)
from moonlightbox.events.config import config_fingerprint, normalize_config
from moonlightbox.events.linux_summary import LinuxEventNarrativeSummarizer
from moonlightbox.events.pipeline import (
    EventV2Pipeline,
    PipelineCancelledError,
    PipelineConfig,
)
from moonlightbox.events.reviewer import NodeAnalysisClient, TwoStageEventReviewer
from moonlightbox.events.v3_pipeline import (
    EventV3Pipeline,
    V3NarrativeSummarizer,
    V3PipelineConfig,
)
from moonlightbox.events.v3_reviewer import DualChannelEventReviewer
from moonlightbox.imports.analysis import analyze_import
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobLeaseLostError, JobService

ANALYSIS_JOB_KIND = "event_analysis_v2"
ANALYSIS_VERSION = "hybrid-v2"
LEGACY_ANALYSIS_JOB_KIND = "analyze_import"
ANALYSIS_PROMPT_VERSION = "event-analysis-v2.2"
V3_ANALYSIS_JOB_KIND = "event_analysis_v3"
V3_ANALYSIS_VERSION = "hybrid-v3"
V3_RELATIONSHIP_PROMPT_VERSION = "event-analysis-v3-relationship-1"
V3_SHARED_EXPERIENCE_PROMPT_VERSION = "event-analysis-v3-shared-experience-1"
V3_GLOBAL_SELECTION_PROMPT_VERSION = "event-analysis-v3-global-selection-1"
JOB_LEASE_DURATION = timedelta(minutes=2)
V3_SCORE_WEIGHTS = {
    "event_significance": 0.25,
    "relationship_impact": 0.20,
    "evidence_quality": 0.25,
    "persistence": 0.10,
    "type_support": 0.10,
    "model_confidence": 0.10,
}


class AnalysisPipelineSnapshot(BaseModel):
    """可持久化且不含凭据的流水线配置。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    session_gap_seconds: float = Field(gt=0)
    character_budget: int = Field(gt=0)
    overlap_messages: int = Field(ge=0)
    persistence_session_limit: int = Field(ge=0, le=3)
    persistence_character_budget: int = Field(gt=0)
    evidence_alignment_threshold: float = Field(ge=0, le=1)
    acceptance_threshold: float = Field(ge=0, le=1)
    maximum_nodes: int = Field(ge=0)


class AnalysisJobSnapshot(BaseModel):
    """Job 与 AnalysisRun 共享的不可变分析配置快照。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    analysis_version: str
    prompt_version: str
    enabled: bool
    endpoint: str
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0, le=300)
    max_retries: int = Field(ge=0, le=10)
    backoff_seconds: float = Field(gt=0, le=60)
    max_backoff_seconds: float = Field(gt=0, le=300)
    max_retry_after_seconds: float = Field(gt=0, le=86400)
    response_format: Literal["json_schema", "json_object"]
    thinking_mode: Literal["default", "disabled"]
    max_output_tokens: int = Field(gt=0, le=65536)
    pipeline: AnalysisPipelineSnapshot

    @field_validator("analysis_version")
    @classmethod
    def validate_analysis_version(cls, value: str) -> str:
        if value != ANALYSIS_VERSION:
            raise ValueError("未知 analysis_version")
        return value

    @field_validator("prompt_version")
    @classmethod
    def validate_prompt_version(cls, value: str) -> str:
        if value != ANALYSIS_PROMPT_VERSION:
            raise ValueError("未知 prompt_version")
        return value

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        normalized = value.strip()
        try:
            parsed = httpx.URL(normalized)
        except httpx.InvalidURL:
            raise ValueError("endpoint 必须是有效的 http(s) URL") from None
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("endpoint 必须是有效的 http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("endpoint 不允许包含用户凭据")
        return normalized

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model 不能为空")
        return normalized


class V3AnalysisPipelineSnapshot(BaseModel):
    """V3 双通道与六维评分配置快照。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    session_gap_seconds: float = Field(gt=0)
    character_budget: int = Field(gt=0)
    overlap_messages: int = Field(ge=0)
    persistence_session_limit: int = Field(ge=0, le=3)
    acceptance_threshold: float = Field(ge=0, le=1)
    maximum_nodes: int = Field(ge=0)
    relationship_prompt_version: str
    shared_experience_prompt_version: str
    global_selection_prompt_version: str
    weights: dict[str, float]

    @model_validator(mode="after")
    def validate_v3_pipeline_contract(self) -> "V3AnalysisPipelineSnapshot":
        if self.relationship_prompt_version != V3_RELATIONSHIP_PROMPT_VERSION:
            raise ValueError("未知 relationship prompt_version")
        if self.shared_experience_prompt_version != V3_SHARED_EXPERIENCE_PROMPT_VERSION:
            raise ValueError("未知 shared_experience prompt_version")
        if self.global_selection_prompt_version != V3_GLOBAL_SELECTION_PROMPT_VERSION:
            raise ValueError("未知 global_selection prompt_version")
        if self.weights != V3_SCORE_WEIGHTS:
            raise ValueError("V3 评分权重与固定设计不一致")
        return self


class V3AnalysisJobSnapshot(BaseModel):
    """不含凭据的 V3 Job 不可变配置快照。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    analysis_version: str
    enabled: bool
    endpoint: str
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(gt=0, le=300)
    max_retries: int = Field(ge=0, le=10)
    backoff_seconds: float = Field(gt=0, le=60)
    max_backoff_seconds: float = Field(gt=0, le=300)
    max_retry_after_seconds: float = Field(gt=0, le=86400)
    response_format: Literal["json_schema", "json_object"]
    thinking_mode: Literal["default", "disabled"]
    max_output_tokens: int = Field(gt=0, le=65536)
    pipeline: V3AnalysisPipelineSnapshot

    @field_validator("analysis_version")
    @classmethod
    def validate_analysis_version(cls, value: str) -> str:
        if value != V3_ANALYSIS_VERSION:
            raise ValueError("未知 V3 analysis_version")
        return value

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        normalized = value.strip()
        try:
            parsed = httpx.URL(normalized)
        except httpx.InvalidURL:
            raise ValueError("endpoint 必须是有效的 http(s) URL") from None
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ValueError("endpoint 必须是有效的 http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("endpoint 不允许包含用户凭据")
        return normalized

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("model 不能为空")
        return normalized


def build_analysis_job_snapshot(settings: Settings) -> dict[str, object]:
    """在 API 入队时冻结完整、无凭据的 V2 配置。"""

    snapshot = AnalysisJobSnapshot(
        analysis_version=ANALYSIS_VERSION,
        prompt_version=ANALYSIS_PROMPT_VERSION,
        enabled=settings.node_analysis_enabled,
        endpoint=settings.node_analysis_endpoint,
        model=settings.node_analysis_model,
        timeout_seconds=settings.node_analysis_timeout_seconds,
        max_retries=settings.node_analysis_max_retries,
        backoff_seconds=settings.node_analysis_backoff_seconds,
        max_backoff_seconds=settings.node_analysis_max_backoff_seconds,
        max_retry_after_seconds=settings.node_analysis_max_retry_after_seconds,
        response_format=settings.node_analysis_response_format,
        thinking_mode=settings.node_analysis_thinking_mode,
        max_output_tokens=settings.node_analysis_max_output_tokens,
        pipeline=AnalysisPipelineSnapshot.model_validate(PipelineConfig().snapshot()),
    )
    return normalize_config(snapshot.model_dump(mode="json"))


def build_v3_analysis_job_snapshot(settings: Settings) -> dict[str, object]:
    """在 API 入队时冻结完整、无凭据的 V3 配置。"""

    pipeline = V3PipelineConfig().snapshot()
    pipeline.update(
        {
            "relationship_prompt_version": V3_RELATIONSHIP_PROMPT_VERSION,
            "shared_experience_prompt_version": (V3_SHARED_EXPERIENCE_PROMPT_VERSION),
            "global_selection_prompt_version": V3_GLOBAL_SELECTION_PROMPT_VERSION,
        }
    )
    snapshot = V3AnalysisJobSnapshot(
        analysis_version=V3_ANALYSIS_VERSION,
        enabled=settings.node_analysis_enabled,
        endpoint=settings.node_analysis_endpoint,
        model=settings.node_analysis_model,
        timeout_seconds=settings.node_analysis_timeout_seconds,
        max_retries=settings.node_analysis_max_retries,
        backoff_seconds=settings.node_analysis_backoff_seconds,
        max_backoff_seconds=settings.node_analysis_max_backoff_seconds,
        max_retry_after_seconds=settings.node_analysis_max_retry_after_seconds,
        response_format=settings.node_analysis_response_format,
        thinking_mode=settings.node_analysis_thinking_mode,
        max_output_tokens=settings.node_analysis_max_output_tokens,
        pipeline=V3AnalysisPipelineSnapshot.model_validate(pipeline),
    )
    return normalize_config(snapshot.model_dump(mode="json"))


def analysis_dedupe_key(
    import_id: str,
    snapshot: Mapping[str, object],
) -> str:
    """根据 import_id 与完整规范化快照生成稳定幂等键。"""

    identity = normalize_config(
        {
            "import_id": import_id,
            "analysis_config": normalize_config(snapshot),
        }
    )
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return f"{ANALYSIS_JOB_KIND}:{hashlib.sha256(serialized.encode()).hexdigest()}"


def v3_analysis_dedupe_key(
    import_id: str,
    snapshot: Mapping[str, object],
) -> str:
    """根据导入与完整 V3 快照生成稳定幂等键。"""

    identity = normalize_config(
        {
            "import_id": import_id,
            "analysis_config": normalize_config(snapshot),
        }
    )
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    return f"{V3_ANALYSIS_JOB_KIND}:{digest}"


def create_node_analysis_cloud_client(
    settings: Settings,
    snapshot: AnalysisJobSnapshot | V3AnalysisJobSnapshot,
) -> NodeAnalysisCloudClient:
    """根据运行配置创建任务 4 云端客户端。"""

    return NodeAnalysisCloudClient(
        enabled=snapshot.enabled,
        endpoint=snapshot.endpoint,
        model=snapshot.model,
        api_key=settings.node_analysis_api_key,
        timeout_seconds=snapshot.timeout_seconds,
        max_retries=snapshot.max_retries,
        backoff_seconds=snapshot.backoff_seconds,
        max_backoff_seconds=snapshot.max_backoff_seconds,
        max_retry_after_seconds=snapshot.max_retry_after_seconds,
        response_format=snapshot.response_format,
        thinking_mode=snapshot.thinking_mode,
        max_output_tokens=snapshot.max_output_tokens,
    )


def create_event_analysis_v2_handler(
    settings: Settings,
    *,
    cloud_client: NodeAnalysisClient | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> JobHandler:
    """创建从 Job payload 驱动 V2 流水线的 Registry handler。"""

    def handler(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError("job_lease_missing", "分析任务缺少 Worker 租约")
        snapshot = _validated_job_snapshot(job)
        owned_client = (
            create_node_analysis_cloud_client(settings, snapshot) if cloud_client is None else None
        )
        client = cloud_client or owned_client
        if client is None:
            raise JobHandlerError("event_analysis_configuration", "节点分析客户端配置失败")
        try:
            project_id = _required_payload_value(job, "project_id")
            import_id = _required_payload_value(job, "import_id")
            pipeline_config = PipelineConfig.from_snapshot(
                snapshot.pipeline.model_dump(mode="python")
            )

            def should_cancel() -> bool:
                if should_stop is not None and should_stop():
                    return True
                active = service.heartbeat(
                    job.id,
                    token=token,
                    lease_duration=JOB_LEASE_DURATION,
                )
                # cancelling 仍需续租直至退出，但不能继续分析或发布。
                return active is None or active.status != "running"

            def report_progress(checkpoint: dict[str, object]) -> None:
                if should_cancel():
                    raise PipelineCancelledError("Job 已取消或租约已被接管")
                service.checkpoint(job.id, checkpoint, token=token)

            pipeline = EventV2Pipeline(
                service.session,
                TwoStageEventReviewer(
                    client,
                    persistence_character_budget=(pipeline_config.persistence_character_budget),
                    persistence_session_limit=pipeline_config.persistence_session_limit,
                ),
                config=pipeline_config,
                worker_id=f"job-{job.id}",
                progress_callback=report_progress,
                should_cancel=should_cancel,
                job_id=job.id,
                job_worker_token=token,
            )
            pipeline.run(
                project_id=project_id,
                import_id=import_id,
                prompt_version=snapshot.prompt_version,
                model=snapshot.model,
                audit_config=snapshot.model_dump(mode="json"),
            )
        except PipelineCancelledError:
            if should_stop is not None and should_stop():
                service.interrupt(
                    job.id,
                    "Worker 正在退出",
                    token=token,
                )
            return
        except JobLeaseLostError:
            return
        except JobHandlerError:
            raise
        except Exception as error:
            raise _safe_job_error(error) from error
        finally:
            if owned_client is not None:
                owned_client.close()

    return handler


def create_event_analysis_v3_handler(
    settings: Settings,
    *,
    cloud_client: NodeAnalysisClient | None = None,
    narrative_summarizer: V3NarrativeSummarizer | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> JobHandler:
    """创建从 Job payload 驱动 V3 双通道流水线的 handler。"""

    def handler(service: JobService, job: Job) -> None:
        token = job.worker_token
        if token is None:
            raise JobHandlerError("job_lease_missing", "分析任务缺少 Worker 租约")
        snapshot = _validated_v3_job_snapshot(job)
        owned_client = (
            create_node_analysis_cloud_client(settings, snapshot) if cloud_client is None else None
        )
        client = cloud_client or owned_client
        if client is None:
            raise JobHandlerError("event_analysis_configuration", "节点分析客户端配置失败")
        try:
            project_id = _required_payload_value(job, "project_id")
            import_id = _required_payload_value(job, "import_id")
            pipeline_config = V3PipelineConfig.from_snapshot(
                snapshot.pipeline.model_dump(mode="python")
            )

            def should_cancel() -> bool:
                if should_stop is not None and should_stop():
                    return True
                active = service.heartbeat(
                    job.id,
                    token=token,
                    lease_duration=JOB_LEASE_DURATION,
                )
                # cancelling 仍需续租直至退出，但不能继续分析或发布。
                return active is None or active.status != "running"

            def report_progress(checkpoint: dict[str, object]) -> None:
                if should_cancel():
                    raise PipelineCancelledError("Job 已取消或租约已被接管")
                service.checkpoint(job.id, checkpoint, token=token)

            reviewer = DualChannelEventReviewer(client)
            summarizer = narrative_summarizer or LinuxEventNarrativeSummarizer(
                settings.training_base_model,
                device=settings.persona_device,
                load_in_4bit=settings.persona_load_in_4bit,
            )
            pipeline = EventV3Pipeline(
                service.session,
                reviewer,
                global_selector=reviewer,
                narrative_summarizer=summarizer,
                config=pipeline_config,
                worker_id=f"job-{job.id}",
                progress_callback=report_progress,
                should_cancel=should_cancel,
                job_id=job.id,
                job_worker_token=token,
            )
            pipeline.run(
                project_id=project_id,
                import_id=import_id,
                relationship_prompt_version=(snapshot.pipeline.relationship_prompt_version),
                shared_experience_prompt_version=(
                    snapshot.pipeline.shared_experience_prompt_version
                ),
                global_selection_prompt_version=(snapshot.pipeline.global_selection_prompt_version),
                model=snapshot.model,
                audit_config=snapshot.model_dump(mode="json"),
            )
        except PipelineCancelledError:
            if should_stop is not None and should_stop():
                service.interrupt(
                    job.id,
                    "Worker 正在退出",
                    token=token,
                )
            return
        except JobLeaseLostError:
            return
        except JobHandlerError:
            raise
        except Exception as error:
            raise _safe_job_error(error) from error
        finally:
            if owned_client is not None:
                owned_client.close()

    return handler


def _validated_job_snapshot(job: Job) -> AnalysisJobSnapshot:
    raw_snapshot = job.payload.get("analysis_config")
    fingerprint = job.payload.get("config_fingerprint")
    try:
        if not isinstance(raw_snapshot, Mapping) or not isinstance(fingerprint, str):
            raise ValueError("分析配置快照缺失")
        snapshot = AnalysisJobSnapshot.model_validate(raw_snapshot)
        if config_fingerprint(snapshot.model_dump(mode="json")) != fingerprint:
            raise ValueError("分析配置指纹不匹配")
        return snapshot
    except (ValidationError, ValueError) as error:
        raise JobHandlerError(
            "invalid_analysis_config",
            "分析任务配置无效或已被篡改",
        ) from error


def _validated_v3_job_snapshot(job: Job) -> V3AnalysisJobSnapshot:
    raw_snapshot = job.payload.get("analysis_config")
    fingerprint = job.payload.get("config_fingerprint")
    try:
        if not isinstance(raw_snapshot, Mapping) or not isinstance(fingerprint, str):
            raise ValueError("V3 分析配置快照缺失")
        snapshot = V3AnalysisJobSnapshot.model_validate(raw_snapshot)
        if config_fingerprint(snapshot.model_dump(mode="json")) != fingerprint:
            raise ValueError("V3 分析配置指纹不匹配")
        return snapshot
    except (ValidationError, ValueError) as error:
        raise JobHandlerError(
            "invalid_analysis_config",
            "V3 分析任务配置无效或已被篡改",
        ) from error


def _required_payload_value(job: Job, key: str) -> str:
    value = job.payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JobHandlerError(
            "invalid_job_payload",
            f"分析任务缺少有效字段：{key}",
        )
    return value


def _safe_job_error(error: Exception) -> JobHandlerError:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, NodeAnalysisCloudError):
            return JobHandlerError(
                f"node_analysis_{current.code.value}",
                str(current),
            )
        current = current.__cause__
    return JobHandlerError("event_analysis_failed", "事件分析执行失败")


def run_import_analysis_job(database: Database, job_id: str) -> None:
    """兼容黄金基线的旧 heuristic 执行入口，新请求不再调用。"""

    with Session(database.engine) as session:
        service = JobService(session)
        job = service.get(job_id)
        if job.status != "queued":
            return
        running = service.start(job_id)
        token = running.worker_token
        if token is None:
            return
        try:
            service.checkpoint(
                job_id,
                {"stage": "loading_messages", "progress": 0.1},
                token=token,
            )
            result = analyze_import(
                session,
                project_id=str(running.payload["project_id"]),
                import_id=str(running.payload["import_id"]),
            )
            service.checkpoint(
                job_id,
                {
                    "stage": "events_created",
                    "event_count": result.event_count,
                    "progress": 0.95,
                },
                token=token,
            )
            service.succeed(job_id, token=token)
        except Exception as error:
            service.fail(job_id, "analysis_failed", str(error), token=token)
