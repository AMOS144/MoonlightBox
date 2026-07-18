from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from moonlightbox.config import Settings
from moonlightbox.evaluation.cloud import (
    CloudEvaluationCase,
    CloudEvaluationError,
    CloudEvaluationScore,
    CloudEvaluator,
)


class CloudEvaluationRequest(BaseModel):
    prompt: str
    reference_answer: str
    candidate_answer: str


def create_evaluation_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

    @router.post("/cloud", response_model=CloudEvaluationScore)
    def evaluate_with_cloud(payload: CloudEvaluationRequest) -> object:
        if not settings.cloud_evaluation_enabled:
            raise HTTPException(status_code=403, detail="云端评测已关闭")
        if not settings.cloud_evaluation_api_key:
            raise HTTPException(status_code=503, detail="尚未配置云端评测密钥")
        evaluator = CloudEvaluator(
            endpoint=settings.cloud_evaluation_endpoint,
            api_key=settings.cloud_evaluation_api_key,
            model=settings.cloud_evaluation_model,
        )
        try:
            return evaluator.evaluate(CloudEvaluationCase(**payload.model_dump()))
        except CloudEvaluationError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    return router
