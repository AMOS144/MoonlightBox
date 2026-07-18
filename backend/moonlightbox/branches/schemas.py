from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class BranchCreate(BaseModel):
    origin_event_id: str
    model_version_id: str
    title: str = Field(min_length=1)
    origin_time: datetime
    state_snapshot: dict[str, object]


class BranchRead(BranchCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    created_at: datetime


class BranchMessageCreate(BaseModel):
    content: str = Field(min_length=1)


class BranchMessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    branch_id: str
    sequence: int
    role: str
    content: str
    generation_metadata: dict[str, object]
    created_at: datetime
