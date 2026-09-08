from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.config import config_fingerprint
from moonlightbox.imports.analysis_job import (
    V3_ANALYSIS_JOB_KIND,
    build_v3_analysis_job_snapshot,
    v3_analysis_dedupe_key,
)
from moonlightbox.imports.schemas import (
    ImportConfirm,
    ImportConfirmRead,
    ImportPreviewRead,
)
from moonlightbox.imports.service import (
    ImportService,
    InvalidMediaArchiveError,
    UnsupportedImportFormatError,
)
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.service import InvalidJobTransitionError, JobService
from moonlightbox.spatial.jobs import (
    enqueue_spatial_analysis,
)
from moonlightbox.world.jobs import enqueue_world_build


def create_imports_router(
    data_dir: Path,
    database: Database,
    settings: Settings,
) -> APIRouter:
    router = APIRouter(prefix="/api/projects/{project_id}/imports", tags=["imports"])
    service = ImportService(data_dir)

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.post(
        "/preview",
        response_model=ImportPreviewRead,
        status_code=status.HTTP_201_CREATED,
    )
    async def preview_import(
        project_id: str,
        file: Annotated[UploadFile, File()],
        media_files: Annotated[list[UploadFile] | None, File()] = None,
        media_paths: Annotated[list[str] | None, Form()] = None,
    ) -> ImportPreviewRead:
        try:
            uploaded_media = media_files or []
            uploaded_paths = media_paths or [
                item.filename or f"media-{index}" for index, item in enumerate(uploaded_media)
            ]
            if len(uploaded_media) != len(uploaded_paths):
                raise HTTPException(status_code=422, detail="媒体路径数量不匹配")
            return service.preview(
                project_id,
                file.filename or "chat",
                await file.read(),
                [
                    (path, await upload.read())
                    for path, upload in zip(
                        uploaded_paths,
                        uploaded_media,
                        strict=True,
                    )
                ],
            )
        except UnsupportedImportFormatError as error:
            raise HTTPException(status_code=415, detail="不支持的聊天记录格式") from error
        except InvalidMediaArchiveError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.post(
        "/{preview_id}/confirm",
        response_model=ImportConfirmRead,
        status_code=status.HTTP_201_CREATED,
    )
    def confirm_import(
        project_id: str,
        preview_id: str,
        payload: ImportConfirm,
        response: Response,
        session: SessionDependency,
    ) -> ImportConfirmRead:
        try:
            result = service.confirm(session, project_id, preview_id, payload)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="导入预览不存在") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not result.created:
            response.status_code = status.HTTP_200_OK
        world_job = (
            enqueue_world_build(
                session,
                settings=settings,
                project_id=project_id,
                trigger_import_id=result.import_id,
            )
            if settings.lightrag_enabled
            else None
        )
        job = (
            None
            if world_job is not None
            else _ensure_analysis_job(
                session,
                project_id,
                result.import_id,
                settings,
            )
        )
        spatial_job = (
            _ensure_spatial_job(
                session,
                project_id,
                result.import_id,
                settings,
            )
            if settings.spatial_analysis_enabled
            else None
        )
        return result.model_copy(
            update={
                "analysis_job_id": job.id if job is not None else None,
                "world_job_id": world_job.id if world_job is not None else None,
                "spatial_job_id": spatial_job.id if spatial_job is not None else None,
            }
        )

    return router


def _ensure_analysis_job(
    session: Session,
    project_id: str,
    import_id: str,
    settings: Settings,
) -> Job:
    service = JobService(session)
    snapshot = build_v3_analysis_job_snapshot(settings)
    job = service.enqueue_unique(
        V3_ANALYSIS_JOB_KIND,
        {
            "project_id": project_id,
            "import_id": import_id,
            "analysis_config": snapshot,
            "config_fingerprint": config_fingerprint(snapshot),
        },
        dedupe_key=v3_analysis_dedupe_key(import_id, snapshot),
    )
    if job.status not in {"failed", "interrupted"}:
        return job
    try:
        return service.resume(job.id)
    except InvalidJobTransitionError:
        return service.get(job.id)


def _ensure_spatial_job(
    session: Session,
    project_id: str,
    import_id: str,
    settings: Settings,
) -> Job:
    return enqueue_spatial_analysis(
        session,
        settings=settings,
        project_id=project_id,
        import_id=import_id,
    )
