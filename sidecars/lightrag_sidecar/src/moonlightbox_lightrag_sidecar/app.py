import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, status

from moonlightbox_lightrag_sidecar.config import SidecarSettings
from moonlightbox_lightrag_sidecar.core import CoreLightRAGRegistry, IndexInProgress
from moonlightbox_lightrag_sidecar.index_errors import IndexFailure, index_failure
from moonlightbox_lightrag_sidecar.schemas import (
    DocumentBatchRequest,
    DocumentBatchResponse,
    DocumentInput,
    DocumentStatusRequest,
    EntityDeleteRequest,
    EntityListResponse,
    EntityMergeRequest,
    EntityMergeResponse,
    EntityMutationRequest,
    EntityRead,
    GraphMutationResponse,
    GraphResponse,
    QueryRequest,
    QueryResponse,
    RelationDeleteRequest,
    RelationMutationRequest,
    RuntimeConfigResponse,
    RuntimeConfigUpdate,
    RuntimeConnectionTest,
    SidecarMetadata,
    WorkspaceCloneRequest,
    WorkspaceCloneResponse,
)
from moonlightbox_lightrag_sidecar.source_index import TemporalIndexError

WORKSPACE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class LightRAGRegistry(Protocol):
    @property
    def metadata(self) -> SidecarMetadata: ...

    @property
    def runtime_config(self) -> RuntimeConfigResponse: ...

    async def index(self, workspace: str, documents: list[DocumentInput]) -> str | None: ...

    async def index_status(self, workspace: str, document_ids: list[str]) -> dict: ...

    async def clone_workspace(self, source_workspace: str, workspace: str) -> None: ...

    async def query(self, workspace: str, request: QueryRequest) -> QueryResponse: ...

    async def entities(self, workspace: str) -> EntityListResponse: ...

    async def graph(self, workspace: str) -> GraphResponse: ...

    async def merge_entities(
        self, workspace: str, request: EntityMergeRequest
    ) -> EntityMergeResponse: ...

    async def entity(self, workspace: str, entity_name: str) -> EntityRead: ...

    async def relation(
        self, workspace: str, source_entity: str, target_entity: str
    ) -> GraphMutationResponse: ...

    async def create_entity(
        self, workspace: str, request: EntityMutationRequest
    ) -> GraphMutationResponse: ...

    async def update_entity(
        self, workspace: str, request: EntityMutationRequest
    ) -> GraphMutationResponse: ...

    async def delete_entity(
        self, workspace: str, request: EntityDeleteRequest
    ) -> GraphMutationResponse: ...

    async def create_relation(
        self, workspace: str, request: RelationMutationRequest
    ) -> GraphMutationResponse: ...

    async def update_relation(
        self, workspace: str, request: RelationMutationRequest
    ) -> GraphMutationResponse: ...

    async def delete_relation(
        self, workspace: str, request: RelationDeleteRequest
    ) -> GraphMutationResponse: ...

    async def close(self) -> None: ...

    async def reconfigure(self, update: RuntimeConfigUpdate) -> RuntimeConfigResponse: ...

    async def test_connection(self, request: RuntimeConnectionTest) -> None: ...


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

    @app.put(
        "/v1/config",
        response_model=RuntimeConfigResponse,
        dependencies=[Depends(authenticate)],
    )
    async def update_runtime_config(payload: RuntimeConfigUpdate) -> RuntimeConfigResponse:
        try:
            return await resolved_registry.reconfigure(payload)
        except (ValueError, RuntimeError) as error:
            failure = index_failure(error)
            raise HTTPException(status_code=failure.status_code, detail=failure.detail) from error

    @app.get(
        "/v1/config",
        response_model=RuntimeConfigResponse,
        dependencies=[Depends(authenticate)],
    )
    async def read_runtime_config() -> RuntimeConfigResponse:
        return resolved_registry.runtime_config

    @app.post("/v1/config:test", dependencies=[Depends(authenticate)])
    async def test_runtime_config(payload: RuntimeConnectionTest) -> dict[str, object]:
        try:
            await resolved_registry.test_connection(payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return {"ok": True, "message": "连接成功"}

    @app.post(
        "/v1/workspaces/{workspace}:clone",
        response_model=WorkspaceCloneResponse,
        dependencies=[Depends(authenticate)],
    )
    async def clone_workspace(
        workspace: str,
        payload: WorkspaceCloneRequest,
    ) -> WorkspaceCloneResponse:
        resolved_workspace = validate_workspace(workspace)
        source_workspace = validate_workspace(payload.source_workspace)
        if source_workspace == resolved_workspace:
            raise HTTPException(status_code=422, detail="候选 workspace 不能与来源相同")
        try:
            await resolved_registry.clone_workspace(source_workspace, resolved_workspace)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return WorkspaceCloneResponse(
            source_workspace=source_workspace,
            workspace=resolved_workspace,
            metadata=resolved_registry.metadata,
        )

    @app.post("/v1/workspaces/{workspace}/documents:status", dependencies=[Depends(authenticate)])
    async def index_status(workspace: str, payload: DocumentStatusRequest):
        try:
            return await resolved_registry.index_status(
                validate_workspace(workspace), payload.document_ids
            )
        except Exception as error:
            failure = index_failure(error)
            raise HTTPException(status_code=failure.status_code, detail=failure.detail) from error

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
        except IndexInProgress as error:
            raise HTTPException(
                status_code=409, detail={"code": "lightrag_index_running", "message": str(error)}
            ) from error
        except TemporalIndexError as error:
            raise HTTPException(status_code=409, detail={"code": str(error)}) from error
        except IndexFailure as error:
            raise HTTPException(status_code=error.status_code, detail=error.detail) from error
        except Exception as error:
            failure = index_failure(error)
            # 只从本次请求的文档标识定位来源，不返回供应商原始错误正文。
            chunk_id = failure.detail["chunk_id"]
            matches = [
                item.id for item in payload.documents
                if isinstance(chunk_id, str) and chunk_id.startswith(f"{item.id}-chunk-")
            ]
            if len(matches) == 1:
                failure.detail["document_id"] = matches[0]
            elif len(payload.documents) == 1:
                failure.detail["document_id"] = payload.documents[0].id
            raise HTTPException(status_code=failure.status_code, detail=failure.detail) from error
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
        except TemporalIndexError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": error.code,
                    "message": (
                        "时间来源索引或查询契约不匹配；请检查来源索引与冻结图版本，"
                        "不能解释为没有相关资料。"
                    ),
                    "retryable": False,
                },
            ) from error
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

    @app.get(
        "/v1/workspaces/{workspace}/graph",
        response_model=GraphResponse,
        dependencies=[Depends(authenticate)],
    )
    async def get_graph(workspace: str) -> GraphResponse:
        resolved_workspace = validate_workspace(workspace)
        try:
            return await resolved_registry.graph(resolved_workspace)
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
            raise HTTPException(
                status_code=422, detail="target_entity 不能同时出现在 source_entities"
            )
        try:
            return await resolved_registry.merge_entities(resolved_workspace, payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get(
        "/v1/workspaces/{workspace}/entities/{entity_name}",
        response_model=EntityRead,
        dependencies=[Depends(authenticate)],
    )
    async def get_entity(workspace: str, entity_name: str) -> EntityRead:
        try:
            return await resolved_registry.entity(validate_workspace(workspace), entity_name)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get(
        "/v1/workspaces/{workspace}/relations/{source_entity}/{target_entity}",
        response_model=GraphMutationResponse,
        dependencies=[Depends(authenticate)],
    )
    async def get_relation(
        workspace: str, source_entity: str, target_entity: str
    ) -> GraphMutationResponse:
        try:
            return await resolved_registry.relation(
                validate_workspace(workspace), source_entity, target_entity
            )
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post(
        "/v1/workspaces/{workspace}/entities",
        response_model=GraphMutationResponse,
        dependencies=[Depends(authenticate)],
    )
    async def create_entity(
        workspace: str, payload: EntityMutationRequest
    ) -> GraphMutationResponse:
        try:
            return await resolved_registry.create_entity(validate_workspace(workspace), payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.patch(
        "/v1/workspaces/{workspace}/entities",
        response_model=GraphMutationResponse,
        dependencies=[Depends(authenticate)],
    )
    async def update_entity(
        workspace: str, payload: EntityMutationRequest
    ) -> GraphMutationResponse:
        try:
            return await resolved_registry.update_entity(validate_workspace(workspace), payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post(
        "/v1/workspaces/{workspace}/entities:delete",
        response_model=GraphMutationResponse,
        dependencies=[Depends(authenticate)],
    )
    async def delete_entity(workspace: str, payload: EntityDeleteRequest) -> GraphMutationResponse:
        try:
            return await resolved_registry.delete_entity(validate_workspace(workspace), payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post(
        "/v1/workspaces/{workspace}/relations",
        response_model=GraphMutationResponse,
        dependencies=[Depends(authenticate)],
    )
    async def create_relation(
        workspace: str, payload: RelationMutationRequest
    ) -> GraphMutationResponse:
        try:
            return await resolved_registry.create_relation(validate_workspace(workspace), payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.patch(
        "/v1/workspaces/{workspace}/relations",
        response_model=GraphMutationResponse,
        dependencies=[Depends(authenticate)],
    )
    async def update_relation(
        workspace: str, payload: RelationMutationRequest
    ) -> GraphMutationResponse:
        try:
            return await resolved_registry.update_relation(validate_workspace(workspace), payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post(
        "/v1/workspaces/{workspace}/relations:delete",
        response_model=GraphMutationResponse,
        dependencies=[Depends(authenticate)],
    )
    async def delete_relation(
        workspace: str, payload: RelationDeleteRequest
    ) -> GraphMutationResponse:
        try:
            return await resolved_registry.delete_relation(validate_workspace(workspace), payload)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return app


app = create_app()
