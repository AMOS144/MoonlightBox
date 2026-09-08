import importlib.util
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

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

_RUNTIME_METADATA_MODELS = (
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
) -> FastAPI:
    resolved_settings = settings or Settings()
    database = Database(resolved_settings.database_url)

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
