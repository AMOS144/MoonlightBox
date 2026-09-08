import importlib.util
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from moonlightbox.agent.activation_router import create_subject_agent_activation_router
from moonlightbox.agent.models import (
    AgentGoal,
    AgentIntention,
    AgentWakeup,
    CognitiveCycle,
    MentalStateVersion,
    PerceptionEvent,
    PrivateCognitionNote,
    SubjectAgentAcceptanceReport,
)
from moonlightbox.agent.router import create_cognition_router
from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.continuity_router import create_continuity_router
from moonlightbox.branches.embeddings import LocalChineseEmbedder
from moonlightbox.branches.generation import BranchGenerator
from moonlightbox.branches.linux_generation import DatabaseLinuxGenerator
from moonlightbox.branches.memory_jobs import ProjectMemoryRepository
from moonlightbox.branches.preparation_router import create_branch_preparation_router
from moonlightbox.branches.reviewer import DeepSeekReplyReviewer
from moonlightbox.branches.router import create_branches_router
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.evaluation.models import (
    HumanBlindCase,
    HumanBlindRating,
    HumanBlindStudy,
)
from moonlightbox.evaluation.router import create_evaluation_router
from moonlightbox.events.router import create_events_router
from moonlightbox.imports.router import create_imports_router
from moonlightbox.jobs.router import create_jobs_router
from moonlightbox.media.router import create_media_router
from moonlightbox.projects.router import create_projects_router
from moonlightbox.runtime_v1.db_models import (
    RuntimeClockRow,
    RuntimeDayPlanRow,
    RuntimeEventRow,
    RuntimeLifeEventRow,
    RuntimeLifeStateRow,
    RuntimeMemoryRow,
    RuntimeSnapshotRow,
    RuntimeWakeupRow,
)
from moonlightbox.runtime_v1.router import create_runtime_router
from moonlightbox.spatial.router import create_spatial_router
from moonlightbox.training.confirmation_router import (
    create_training_confirmation_router,
)
from moonlightbox.training.router import create_models_router
from moonlightbox.world.models import (
    ConversationBundle,
    ConversationBundleMessage,
    EntityMergeProposal,
    PersonWorldProfile,
    WorldGraphVersion,
)
from moonlightbox.world.router import create_world_router

_COGNITIVE_MODELS = (
    PerceptionEvent,
    CognitiveCycle,
    PrivateCognitionNote,
    MentalStateVersion,
    AgentGoal,
    AgentIntention,
    AgentWakeup,
    SubjectAgentAcceptanceReport,
    HumanBlindStudy,
    HumanBlindCase,
    HumanBlindRating,
    WorldGraphVersion,
    ConversationBundle,
    ConversationBundleMessage,
    PersonWorldProfile,
    EntityMergeProposal,
    RuntimeClockRow,
    RuntimeDayPlanRow,
    RuntimeEventRow,
    RuntimeLifeEventRow,
    RuntimeLifeStateRow,
    RuntimeMemoryRow,
    RuntimeSnapshotRow,
    RuntimeWakeupRow,
)


def is_linux_inference_available() -> bool:
    """只检查 Linux 推理依赖是否存在，API 进程不预加载模型。"""

    return all(
        importlib.util.find_spec(package) is not None
        for package in ("torch", "transformers", "peft")
    )


def create_app(
    settings: Settings | None = None,
    generator: BranchGenerator | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings()
    database = Database(resolved_settings.database_url)
    resolved_generator = generator or DatabaseLinuxGenerator(
        database,
        device=resolved_settings.persona_device,
        load_in_4bit=resolved_settings.persona_load_in_4bit,
    )
    embedding_root = resolved_settings.model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
    embedder = LocalChineseEmbedder(embedding_root) if embedding_root.is_dir() else None
    memory_repository = (
        ProjectMemoryRepository(str(resolved_settings.chroma_dir), embedder)
        if embedder is not None
        else None
    )
    continuity_repository = (
        BranchContinuityRepository(str(resolved_settings.chroma_dir), embedder)
        if embedder is not None
        else None
    )
    reviewer = None
    if (
        generator is None
        and resolved_settings.reply_review_enabled
        and resolved_settings.node_analysis_api_key is not None
    ):
        reviewer = DeepSeekReplyReviewer(
            endpoint=resolved_settings.node_analysis_endpoint,
            model=resolved_settings.node_analysis_model,
            api_key=resolved_settings.node_analysis_api_key.get_secret_value(),
            timeout_seconds=resolved_settings.node_analysis_timeout_seconds,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        resolved_settings.ensure_directories()
        if resolved_settings.auto_create_schema:
            database.create_schema()
        yield
        database.close()

    app = FastAPI(title="月光宝盒", lifespan=lifespan)
    app.include_router(create_projects_router(database, resolved_settings))
    app.include_router(create_jobs_router(database))
    app.include_router(
        create_imports_router(
            resolved_settings.data_dir,
            database,
            resolved_settings,
        )
    )
    app.include_router(create_events_router(database))
    app.include_router(create_training_confirmation_router(database, resolved_settings))
    app.include_router(create_models_router(database))
    app.include_router(create_media_router(resolved_settings.data_dir, database))
    app.include_router(
        create_branches_router(
            database,
            resolved_generator,
            reviewer,
            memory_repository,
            continuity_repository,
        )
    )
    app.include_router(create_continuity_router(database, continuity_repository))
    app.include_router(create_branch_preparation_router(database))
    app.include_router(create_cognition_router(database))
    app.include_router(create_subject_agent_activation_router(database))
    app.include_router(create_evaluation_router(resolved_settings, database))
    app.include_router(create_spatial_router(database, resolved_settings))
    app.include_router(create_world_router(database, resolved_settings))
    app.include_router(create_runtime_router(database, resolved_settings))

    @app.get("/api/health")
    def health() -> dict[str, str | bool]:
        database_ok = database.check_health()
        return {
            "status": "ok" if database_ok else "degraded",
            "database": "ok" if database_ok else "error",
            "linux_inference_available": is_linux_inference_available(),
        }

    return app


app = create_app()
