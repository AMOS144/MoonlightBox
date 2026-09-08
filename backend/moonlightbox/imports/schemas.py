from datetime import datetime

from pydantic import BaseModel, Field


class MessageSample(BaseModel):
    timestamp: datetime
    sender: str
    kind: str
    content: str


class ImportErrorRead(BaseModel):
    line: int
    code: str
    message: str


class ImportPreviewRead(BaseModel):
    id: str
    message_count: int
    participants: list[str]
    time_range: tuple[datetime, datetime] | None
    kind_counts: dict[str, int]
    sample_messages: list[MessageSample]
    errors: list[ImportErrorRead]
    media_file_count: int = 0
    linked_sticker_count: int = 0
    unlinked_sticker_count: int = 0
    deduplicated_asset_count: int = 0
    avatar_status: dict[str, bool] = Field(default_factory=dict)
    failure_reasons: dict[str, int] = Field(default_factory=dict)


class ImportConfirm(BaseModel):
    self_participant: str
    target_participant: str


class ImportConfirmRead(BaseModel):
    import_id: str
    message_count: int
    created: bool
    analysis_job_id: str | None = None
    world_job_id: str | None = None
    spatial_job_id: str | None = None
