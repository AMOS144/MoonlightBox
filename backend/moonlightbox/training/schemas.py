from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ModelVersionCreate(BaseModel):
    base_model: str
    adapter_path: str
    dataset_hash: str
    metrics: dict[str, float]


class ModelVersionRead(ModelVersionCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    status: str
    recommended: bool
    active: bool
    created_at: datetime


class EventRevisionReference(BaseModel):
    event_id: str
    revision_number: int


class ConfirmAndTrainRequest(BaseModel):
    analysis_run_id: str
    event_revisions: list[EventRevisionReference]


class ConfirmAndTrainRead(BaseModel):
    confirmation_id: str
    training_job_id: str
    status: str
