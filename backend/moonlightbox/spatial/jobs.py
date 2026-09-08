import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.embeddings import LocalChineseEmbedder
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import InvalidJobTransitionError, JobService
from moonlightbox.spatial.bundle_analysis import (
    BundleAnalyzer,
    CloudBundleAnalyzer,
    LocalLinuxBundleAnalyzer,
    StructuredLocationBundleAnalyzer,
)
from moonlightbox.spatial.config import SpatialPipelineConfig
from moonlightbox.spatial.map_provider import (
    AmapWebServiceProvider,
    MapProvider,
    NullMapProvider,
)
from moonlightbox.spatial.pipeline import SpatialPipeline
from moonlightbox.spatial.resolution import PlaceResolutionService
from moonlightbox.spatial.retrieval import SpatialEpisodeIndex

SPATIAL_ANALYSIS_JOB_KIND = "spatial_analysis_v1"


class SpatialJobSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: str = "spatial-analysis-v1"
    enabled: bool
    backend: str = Field(min_length=1)
    map_provider: str
    persist_provider_enrichment: bool
    pipeline: dict[str, object]


def build_spatial_job_snapshot(settings: Settings) -> dict[str, object]:
    config = SpatialPipelineConfig()
    has_map_key = bool(
        settings.spatial_amap_web_key and settings.spatial_amap_web_key.get_secret_value().strip()
    )
    return SpatialJobSnapshot(
        enabled=settings.spatial_analysis_enabled,
        backend=settings.spatial_analysis_backend,
        map_provider="amap" if has_map_key else "none",
        persist_provider_enrichment=settings.spatial_provider_persistence_allowed,
        pipeline=config.snapshot(),
    ).model_dump(mode="json")


def spatial_analysis_dedupe_key(
    import_id: str,
    snapshot: Mapping[str, object],
) -> str:
    identity = {"import_id": import_id, "spatial_config": snapshot}
    serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{SPATIAL_ANALYSIS_JOB_KIND}:{hashlib.sha256(serialized.encode()).hexdigest()}"


def enqueue_spatial_analysis(
    session: Session,
    *,
    settings: Settings,
    project_id: str,
    import_id: str,
) -> Job:
    service = JobService(session)
    snapshot = build_spatial_job_snapshot(settings)
    job = service.enqueue_unique(
        SPATIAL_ANALYSIS_JOB_KIND,
        {
            "project_id": project_id,
            "import_id": import_id,
            "spatial_config": snapshot,
        },
        dedupe_key=spatial_analysis_dedupe_key(import_id, snapshot),
    )
    if job.status not in {"failed", "interrupted"}:
        return job
    try:
        return service.resume(job.id)
    except InvalidJobTransitionError:
        return service.get(job.id)


def create_spatial_analysis_handler(settings: Settings) -> JobHandler:
    def handler(service: JobService, job: Job) -> None:
        project_id = job.payload.get("project_id")
        import_id = job.payload.get("import_id")
        if not isinstance(project_id, str) or not isinstance(import_id, str):
            raise JobHandlerError("spatial_payload_invalid", "空间分析任务缺少项目或导入 ID")
        snapshot_value = job.payload.get("spatial_config")
        try:
            snapshot = SpatialJobSnapshot.model_validate(snapshot_value)
        except ValueError as error:
            raise JobHandlerError("spatial_config_invalid", "空间分析配置快照无效") from error
        if not snapshot.enabled:
            return
        analyzer, owned_cloud = _create_analyzer(settings, snapshot.backend)
        provider = _create_provider(settings)
        runtime_config = replace(
            SpatialPipelineConfig.from_snapshot(snapshot.pipeline),
            analyzer_version=f"spatial-bundle-v1:{_analyzer_name(analyzer)}",
            resolver_version=f"spatial-resolver-v1:{provider.name}",
        )
        embedding_root = settings.model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
        if not embedding_root.is_dir():
            raise JobHandlerError(
                "spatial_embedding_unavailable",
                "缺少本地中文向量模型，无法执行 Episode 空间召回",
            )
        try:
            embedder = LocalChineseEmbedder(embedding_root)
            index = SpatialEpisodeIndex(
                chroma_dir=str(settings.chroma_dir / "spatial"),
                project_id=project_id,
                import_id=import_id,
                embedder=embedder,
            )
            pipeline = SpatialPipeline(
                config=runtime_config,
                index=index,
                analyzer=analyzer,
                resolver=PlaceResolutionService(
                    provider=provider,
                    config=runtime_config,
                    persist_provider_enrichment=snapshot.persist_provider_enrichment,
                ),
            )
            pipeline.run(
                service.session,
                project_id=project_id,
                import_id=import_id,
            )
        except JobHandlerError:
            raise
        except Exception as error:
            raise JobHandlerError(
                "spatial_analysis_failed",
                f"空间分析失败：{type(error).__name__}",
            ) from error
        finally:
            if owned_cloud is not None:
                owned_cloud.close()
            provider.close()

    return handler


def _create_analyzer(
    settings: Settings,
    backend: str,
) -> tuple[BundleAnalyzer, NodeAnalysisCloudClient | None]:
    selected = backend
    if selected == "auto":
        if _local_linux_available(settings):
            selected = "local"
        elif settings.node_analysis_enabled and _cloud_key_available(settings):
            selected = "cloud"
        else:
            selected = "structured_only"
    if selected == "local":
        if not _local_linux_available(settings):
            raise JobHandlerError(
                "spatial_local_model_unavailable",
                "本机缺少 Linux 推理依赖或基础模型不可加载",
            )
        return LocalLinuxBundleAnalyzer(
            settings.training_base_model,
            device=settings.persona_device,
            load_in_4bit=settings.persona_load_in_4bit,
        ), None
    if selected == "cloud":
        if not settings.node_analysis_enabled or not _cloud_key_available(settings):
            raise JobHandlerError("spatial_cloud_unavailable", "空间云分析未配置 API key")
        client = NodeAnalysisCloudClient(
            enabled=True,
            endpoint=settings.node_analysis_endpoint,
            model=settings.node_analysis_model,
            api_key=settings.node_analysis_api_key,
            timeout_seconds=settings.node_analysis_timeout_seconds,
            max_retries=settings.node_analysis_max_retries,
            backoff_seconds=settings.node_analysis_backoff_seconds,
            max_backoff_seconds=settings.node_analysis_max_backoff_seconds,
            max_retry_after_seconds=settings.node_analysis_max_retry_after_seconds,
            response_format=settings.node_analysis_response_format,
            thinking_mode=settings.node_analysis_thinking_mode,
            max_output_tokens=settings.node_analysis_max_output_tokens,
        )
        return CloudBundleAnalyzer(client), client
    if selected == "structured_only":
        return StructuredLocationBundleAnalyzer(), None
    raise JobHandlerError("spatial_backend_invalid", "未知空间分析后端")


def _create_provider(settings: Settings) -> MapProvider:
    if (
        settings.spatial_amap_web_key is None
        or not settings.spatial_amap_web_key.get_secret_value().strip()
    ):
        return NullMapProvider()
    return AmapWebServiceProvider(
        settings.spatial_amap_web_key.get_secret_value(),
        base_url=settings.spatial_amap_base_url,
        timeout_seconds=settings.spatial_map_timeout_seconds,
    )


def _analyzer_name(analyzer: BundleAnalyzer) -> str:
    return type(analyzer).__name__


def _local_linux_available(settings: Settings) -> bool:
    """模型可以是已下载目录，也可以是 Hugging Face 仓库名。"""

    import importlib.util

    return all(
        importlib.util.find_spec(package) is not None
        for package in ("torch", "transformers", "peft")
    ) and bool(settings.training_base_model.strip())


def _cloud_key_available(settings: Settings) -> bool:
    return bool(
        settings.node_analysis_api_key and settings.node_analysis_api_key.get_secret_value().strip()
    )
