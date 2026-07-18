from datetime import datetime

from pydantic import BaseModel


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


class ImportConfirm(BaseModel):
    self_participant: str
    target_participant: str


class ImportConfirmRead(BaseModel):
    import_id: str
    message_count: int
    created: bool
