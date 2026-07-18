from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ReviewedEvent(BaseModel):
    type: str
    start_message_id: str
    end_message_id: str
    before_state: str
    after_state: str
    emotion_labels: list[str]
    topic: str
    conflict_level: int = Field(ge=0, le=5)
    importance: float = Field(ge=0.0, le=1.0)
    reason: str
    evidence_ids: list[str]


class EventNodeRead(ReviewedEvent):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    status: str
    created_at: datetime


class EventRevisionRequest(BaseModel):
    changes: dict[str, object]
    reason: str = Field(min_length=1)
