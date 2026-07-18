from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status
from sqlalchemy.orm import Session

from moonlightbox.db import Database
from moonlightbox.imports.schemas import (
    ImportConfirm,
    ImportConfirmRead,
    ImportPreviewRead,
)
from moonlightbox.imports.service import ImportService, UnsupportedImportFormatError


def create_imports_router(data_dir: Path, database: Database) -> APIRouter:
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
    ) -> ImportPreviewRead:
        try:
            return service.preview(
                project_id,
                file.filename or "chat",
                await file.read(),
            )
        except UnsupportedImportFormatError as error:
            raise HTTPException(status_code=415, detail="不支持的聊天记录格式") from error

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
        return result

    return router
