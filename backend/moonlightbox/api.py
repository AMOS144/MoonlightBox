import importlib.util
import platform
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.projects.router import create_projects_router


def is_mlx_available() -> bool:
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
    )


def create_app(settings: Settings | None = None) -> FastAPI:
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
    app.include_router(create_projects_router(database))

    @app.get("/api/health")
    def health() -> dict[str, str | bool]:
        database_ok = database.check_health()
        return {
            "status": "ok" if database_ok else "degraded",
            "database": "ok" if database_ok else "error",
            "mlx_available": is_mlx_available(),
        }

    return app


app = create_app()
