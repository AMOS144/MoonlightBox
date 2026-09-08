import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, status

from moonlightbox_lightrag_sidecar.config import SidecarSettings
from moonlightbox_lightrag_sidecar.core import CoreLightRAGRegistry
from moonlightbox_lightrag_sidecar.schemas import (
    DocumentBatchRequest,
    DocumentBatchResponse,
    DocumentInput,
    QueryRequest,
    QueryResponse,
    SidecarMetadata,
    EntityListResponse,
    EntityMergeRequest,
    EntityMergeResponse,
    EntityRead,
)

WORKSPACE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class LightRAGRegistry(Protocol):
    @property
    def metadata(self) -> SidecarMetadata: ...

    async def index(self, workspace: str, documents: list[DocumentInput]) -> str | None: ...

    async def query(self, workspace: str, request: QueryRequest) -> QueryResponse: ...

    async def entities(self, workspace: str) -> EntityListResponse: ...

    async def merge_entities(self, workspace: str, request: EntityMergeRequest) -> EntityMergeResponse: ...

    async def close(self) -> None: ...


def create_app(
    settings: SidecarSettings | None = None,
    registry: LightRAGRegistry | None = None,
) -> FastAPI:
    resolved_settings = settings or SidecarSettings()
    resolved_registry = registry or CoreLightRAGRegistry(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        resolved_settings.ensure_storage_dir()
        yield
        await resolved_registry.close()

    app = FastAPI(title="MoonlightBox LightRAG Sidecar", lifespan=lifespan)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        expected = resolved_settings.api_token.get_secret_value()
        if authorization != f"Bearer {expected}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid sidecar token",
            )

    def validate_workspace(workspace: str) -> str:
        if not WORKSPACE_PATTERN.fullmatch(workspace):
            raise HTTPException(status_code=422, detail="invalid workspace")
        return workspace

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "moonlightbox-lightrag-sidecar"}

    @app.post(
        "/v1/workspaces/{workspace}/documents:batch",
        response_model=DocumentBatchResponse,
        dependencies=[Depends(authenticate)],
    )
    async def index_documents(
        workspace: str,
        payload: DocumentBatchRequest,
    ) -> DocumentBatchResponse:
        resolved_workspace = validate_workspace(workspace)
        try:
            track_id = await resolved_registry.index(resolved_workspace, payload.documents)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return DocumentBatchResponse(
            indexed_document_ids=[item.id for item in payload.documents],
            track_id=track_id,
            metadata=resolved_registry.metadata,
        )

    @app.post(
        "/v1/workspaces/{workspace}/query",
        response_model=QueryResponse,
        dependencies=[Depends(authenticate)],
    )
    async def query_workspace(workspace: str, payload: QueryRequest) -> QueryResponse:
        resolved_workspace = validate_workspace(workspace)
        try:
            return await resolved_registry.query(resolved_workspace, payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get(
        "/v1/workspaces/{workspace}/entities",
        response_model=EntityListResponse,
        dependencies=[Depends(authenticate)],
    )
    async def list_entities(workspace: str) -> EntityListResponse:
        resolved_workspace = validate_workspace(workspace)
        try:
            return await resolved_registry.entities(resolved_workspace)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post(
        "/v1/workspaces/{workspace}/entities:merge",
        response_model=EntityMergeResponse,
        dependencies=[Depends(authenticate)],
    )
    async def merge_entities(workspace: str, payload: EntityMergeRequest) -> EntityMergeResponse:
        resolved_workspace = validate_workspace(workspace)
        if payload.target_entity in payload.source_entities:
            raise HTTPException(status_code=422, detail="target_entity 不能同时出现在 source_entities")
        try:
            return await resolved_registry.merge_entities(resolved_workspace, payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


app = create_app()
