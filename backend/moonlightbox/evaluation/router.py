from collections.abc import Iterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.evaluation.blind_service import (
    BlindStudyScopeError,
    BlindStudyStateError,
    HumanBlindStudyService,
)
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


class HumanBlindCaseCreate(BaseModel):
    context: list[str] = Field(min_length=1)
    human_reply: str = Field(min_length=1)
    candidate_reply: str = Field(min_length=1)


class HumanBlindStudyCreate(BaseModel):
    cases: list[HumanBlindCaseCreate] = Field(min_length=1)
    minimum_ratings: int = Field(default=20, ge=20)
    minimum_preference: float = Field(default=0.45, ge=0, le=1)
    seed: int = 0


class HumanBlindRatingCreate(BaseModel):
    case_id: str = Field(min_length=1)
    rater_key: str = Field(min_length=1, max_length=128)
    choice: Literal["a", "b", "tie"]


class HumanBlindStudyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    model_version_id: str
    status: str
    minimum_ratings: int
    minimum_preference: float
    valid_rating_count: int
    candidate_preference_rate: float
    report: dict[str, object]


def create_evaluation_router(settings: Settings, database: Database) -> APIRouter:
    router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

    def get_session() -> Iterator[Session]:
        yield from database.session()

    SessionDependency = Annotated[Session, Depends(get_session)]

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

    @router.post(
        "/projects/{project_id}/models/{model_version_id}/blind-studies",
        response_model=HumanBlindStudyRead,
        status_code=status.HTTP_201_CREATED,
    )
    def create_blind_study(
        project_id: str,
        model_version_id: str,
        payload: HumanBlindStudyCreate,
        session: SessionDependency,
    ) -> object:
        try:
            return HumanBlindStudyService(session).create(
                project_id=project_id,
                model_version_id=model_version_id,
                cases=[case.model_dump() for case in payload.cases],
                minimum_ratings=payload.minimum_ratings,
                minimum_preference=payload.minimum_preference,
                seed=payload.seed,
            )
        except BlindStudyScopeError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except BlindStudyStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/blind-studies/{study_id}/cases")
    def get_blind_cases(study_id: str, session: SessionDependency) -> object:
        try:
            return HumanBlindStudyService(session).public_cases(study_id)
        except BlindStudyScopeError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except BlindStudyStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.get(
        "/projects/{project_id}/blind-studies",
        response_model=list[HumanBlindStudyRead],
    )
    def list_blind_studies(
        project_id: str,
        session: SessionDependency,
    ) -> object:
        return HumanBlindStudyService(session).list_for_project(project_id)

    @router.post(
        "/blind-studies/{study_id}/ratings",
        status_code=status.HTTP_201_CREATED,
    )
    def rate_blind_case(
        study_id: str,
        payload: HumanBlindRatingCreate,
        session: SessionDependency,
    ) -> object:
        try:
            rating = HumanBlindStudyService(session).rate(
                study_id=study_id,
                case_id=payload.case_id,
                rater_key=payload.rater_key,
                choice=payload.choice,
            )
            return {"id": rating.id, "accepted": True}
        except BlindStudyScopeError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except BlindStudyStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @router.post(
        "/projects/{project_id}/models/{model_version_id}/blind-studies/{study_id}/finalize",
        response_model=HumanBlindStudyRead,
    )
    def finalize_blind_study(
        project_id: str,
        model_version_id: str,
        study_id: str,
        session: SessionDependency,
    ) -> object:
        try:
            return HumanBlindStudyService(session).finalize(
                project_id=project_id,
                model_version_id=model_version_id,
                study_id=study_id,
            )
        except BlindStudyScopeError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except BlindStudyStateError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return router
