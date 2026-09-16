"""单一常驻人格推理进程的 FastAPI 入口。"""

from __future__ import annotations

import hmac
import logging
from collections.abc import AsyncIterator
from concurrent.futures import CancelledError, TimeoutError
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from moonlightbox.agent.inference import (
    DuplicateInferenceRequestError,
    InferencePriority,
    PersonaInferenceBackend,
    PersonaInferenceRequest,
    PersonaInferenceScheduler,
)
from moonlightbox.agent.linux_inference import DatabaseLinuxRuntimeInferenceBackend
from moonlightbox.config import Settings
from moonlightbox.db import Database

logger = logging.getLogger(__name__)


class _InferenceRequestPayload(BaseModel):
    request_id: str
    request_type: Literal[
        "runtime_director",
        "runtime_actor",
        "runtime_token_count",
    ]
    priority: InferencePriority
    deadline: datetime
    model_version_id: str
    payload: dict[str, object]

    def to_domain(self) -> PersonaInferenceRequest:
        return PersonaInferenceRequest(
            request_id=self.request_id,
            request_type=self.request_type,
            priority=self.priority,
            deadline=self.deadline,
            model_version_id=self.model_version_id,
            payload=self.payload,
        )


def create_persona_runtime_app(
    settings: Settings | None = None,
    *,
    backend: PersonaInferenceBackend | None = None,
) -> FastAPI:
    """创建独立人格推理应用，不把请求或 Session 交给工作线程。"""

    resolved_settings = settings or Settings()
    state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        database: Database | None = None
        resolved_backend = backend
        if resolved_backend is None:
            database = Database(resolved_settings.database_url)
            # API 与 Worker 不直接加载权重；CUDA 与 PEFT adapter 仅驻留在此服务。
            resolved_backend = DatabaseLinuxRuntimeInferenceBackend(
                database,
                device=resolved_settings.persona_device,
                load_in_4bit=resolved_settings.persona_load_in_4bit,
            )
        scheduler = PersonaInferenceScheduler(resolved_backend)
        state["scheduler"] = scheduler
        state["backend"] = resolved_backend
        state["database"] = database
        try:
            yield
        finally:
            scheduler.shutdown()
            close = getattr(resolved_backend, "close", None)
            if callable(close):
                close()
            if database is not None:
                database.close()
            state.clear()

    app = FastAPI(title="月光宝盒人格推理", lifespan=lifespan)

    def authenticate(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        expected = resolved_settings.persona_inference_token.get_secret_value()
        supplied = ""
        if authorization and authorization.startswith("Bearer "):
            supplied = authorization.removeprefix("Bearer ").strip()
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="认证失败")

    def execute(
        payload: _InferenceRequestPayload,
        expected_type: Literal[
            "runtime_director",
            "runtime_actor",
            "runtime_token_count",
        ],
    ) -> dict[str, object]:
        if payload.request_type != expected_type:
            raise HTTPException(status_code=422, detail="请求类型与端点不匹配")
        scheduler = state.get("scheduler")
        if not isinstance(scheduler, PersonaInferenceScheduler):
            raise HTTPException(
                status_code=503,
                detail={"code": "runtime_unavailable", "message": "人格推理服务未启动"},
            )
        try:
            future = scheduler.submit(payload.to_domain())
            result = future.result(timeout=resolved_settings.persona_inference_timeout_seconds)
        except DuplicateInferenceRequestError:
            raise HTTPException(
                status_code=409,
                detail={"code": "duplicate_request", "message": "请求已提交"},
            ) from None
        except (CancelledError, TimeoutError):
            raise HTTPException(
                status_code=408,
                detail={"code": "request_expired", "message": "人格推理请求已过期"},
            ) from None
        except Exception:
            logger.exception(
                "人格推理失败：request_id=%s request_type=%s",
                payload.request_id,
                payload.request_type,
            )
            raise HTTPException(
                status_code=503,
                detail={"code": "inference_failed", "message": "人格推理失败"},
            ) from None
        return {
            "request_id": result.request_id,
            "request_type": result.request_type,
            "output": dict(result.output),
        }

    authentication = Depends(authenticate)

    @app.post("/v1/inference/runtime-director", dependencies=[authentication])
    def runtime_director(payload: _InferenceRequestPayload) -> dict[str, object]:
        """供 Runtime Director 使用的结构化决策入口。

        Director 与 PersonaActor 经同一个 Linux 人格服务串行执行，API/Worker
        不直接加载模型，以避免多个进程争抢有限显存。
        """

        return execute(payload, "runtime_director")

    @app.post("/v1/inference/runtime-actor", dependencies=[authentication])
    def runtime_actor(payload: _InferenceRequestPayload) -> dict[str, object]:
        """Runtime PersonaActor 的原始结构化表达入口，体验阶段仅使用基础模型。"""

        return execute(payload, "runtime_actor")

    @app.post("/v1/inference/runtime-token-count", dependencies=[authentication])
    def runtime_token_count(payload: _InferenceRequestPayload) -> dict[str, object]:
        """用当前 Director 基座的真实 tokenizer 计算 Runtime 上下文预算。

        请求进入与推理相同的串行调度器，因此不会和 LoRA 加载竞争同一组模型状态。
        """

        return execute(payload, "runtime_token_count")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok" if "scheduler" in state else "starting"}

    return app


app = create_persona_runtime_app()
