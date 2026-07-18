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
    created_at: datetime
