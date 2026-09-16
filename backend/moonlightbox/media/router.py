from collections.abc import Iterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.models import Participant
from moonlightbox.media.models import MediaAsset, MediaSemanticAnnotation
from moonlightbox.media.service import MediaAnnotationService, MediaStore


class MediaAnnotationWrite(BaseModel):
    summary: str = ""
    transcript: str = ""
    ocr_text: str = ""
    safety_tags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    status: Literal["pending", "succeeded", "failed", "needs_review"] = "succeeded"
    reuse_decision: Literal["pending", "approved", "blocked"] = "pending"
    failure_code: str | None = None


def _annotation_payload(annotation: MediaSemanticAnnotation) -> dict[str, object]:
    return {
        "id": annotation.id,
        "project_id": annotation.project_id,
        "asset_id": annotation.asset_id,
        "modality": annotation.modality,
        "status": annotation.status,
        "summary": annotation.summary,
        "transcript": annotation.transcript,
        "ocr_text": annotation.ocr_text,
        "safety_tags": annotation.safety_tags,
        "source_model": annotation.source_model,
        "source_version": annotation.source_version,
        "confidence": annotation.confidence,
        "reusable": annotation.reusable,
        "reuse_decision": annotation.reuse_decision,
        "failure_code": annotation.failure_code,
        "review_source": annotation.review_source,
        "reviewed_at": annotation.reviewed_at,
        "updated_at": annotation.updated_at,
    }


def create_media_router(settings: Settings, database: Database) -> APIRouter:
    router = APIRouter(prefix="/api/projects/{project_id}/media", tags=["media"])
    store = MediaStore(
        settings.data_dir,
        read_only_source_dir=settings.media_read_only_source_dir,
    )

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

    @router.get("/avatars/list")
    def list_avatars(
        project_id: str,
        session: SessionDependency,
    ) -> list[dict[str, str | None]]:
        participants = session.scalars(
            select(Participant).where(
                Participant.project_id == project_id,
                Participant.role.in_(("self", "target")),
            )
        )
        return [
            {
                "role": participant.role,
                "name": participant.name,
                "asset_id": participant.avatar_asset_id,
            }
            for participant in participants
        ]

    @router.get("/annotations/list")
    def list_annotations(
        project_id: str,
        session: SessionDependency,
        modality: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, object]]:
        return [
            _annotation_payload(annotation)
            for annotation in MediaAnnotationService(session).list_for_project(
                project_id,
                modality=modality,
                status=status,
            )
        ]

    @router.get("/annotations/summary")
    def annotation_summary(
        project_id: str,
        session: SessionDependency,
    ) -> dict[str, int]:
        return MediaAnnotationService(session).summary(project_id)

    @router.put("/{asset_id}/annotation")
    def write_annotation(
        project_id: str,
        asset_id: str,
        payload: MediaAnnotationWrite,
        session: SessionDependency,
    ) -> dict[str, object]:
        existing = session.scalar(
            select(MediaSemanticAnnotation).where(
                MediaSemanticAnnotation.asset_id == asset_id,
                MediaSemanticAnnotation.project_id == project_id,
            )
        )
        try:
            annotation = MediaAnnotationService(session).annotate(
                project_id,
                asset_id,
                summary=payload.summary,
                transcript=payload.transcript,
                ocr_text=payload.ocr_text,
                safety_tags=payload.safety_tags,
                source_model=(
                    existing.source_model if existing is not None else "manual-review"
                ),
                source_version=(
                    existing.source_version if existing is not None else "v1"
                ),
                confidence=payload.confidence,
                status=payload.status,
                reuse_decision=payload.reuse_decision,
                failure_code=payload.failure_code,
                review_source="manual-ui",
            )
        except LookupError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _annotation_payload(annotation)

    @router.get("/{asset_id}")
    def read_media(
        project_id: str,
        asset_id: str,
        session: SessionDependency,
    ) -> FileResponse:
        asset = session.get(MediaAsset, asset_id)
        if asset is None or asset.project_id != project_id:
            raise HTTPException(status_code=404, detail="媒体不存在")
        path = store.path_for(asset)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="媒体文件不存在")
        return FileResponse(path, media_type=asset.mime_type)

    return router
