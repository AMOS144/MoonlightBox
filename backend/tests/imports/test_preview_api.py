from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.config import Settings
from moonlightbox.db import Database
from sqlalchemy import func, select
from sqlalchemy.orm import Session


def test_preview_reports_chat_without_confirming_messages(client: TestClient) -> None:
    project = client.post("/api/projects", json={"name": "导入测试"}).json()
    source = Path("tests/fixtures/wxecho_sample.csv")

    with source.open("rb") as upload:
        response = client.post(
            f"/api/projects/{project['id']}/imports/preview",
            files={"file": ("chat.csv", upload, "text/csv")},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["message_count"] == 2
    assert body["participants"] == ["甲", "乙"]
    assert body["kind_counts"] == {"text": 2}
    assert len(body["sample_messages"]) == 2
    assert body["errors"] == []


def test_confirm_import_is_idempotent(client: TestClient, settings: Settings) -> None:
    from moonlightbox.imports.models import Message

    project = client.post("/api/projects", json={"name": "确认测试"}).json()
    source = Path("tests/fixtures/wxecho_sample.csv")
    with source.open("rb") as upload:
        preview = client.post(
            f"/api/projects/{project['id']}/imports/preview",
            files={"file": ("chat.csv", upload, "text/csv")},
        ).json()

    payload = {"self_participant": "乙", "target_participant": "甲"}
    first = client.post(
        f"/api/projects/{project['id']}/imports/{preview['id']}/confirm",
        json=payload,
    )
    second = client.post(
        f"/api/projects/{project['id']}/imports/{preview['id']}/confirm",
        json=payload,
    )
    with source.open("rb") as upload:
        repeated_preview = client.post(
            f"/api/projects/{project['id']}/imports/preview",
            files={"file": ("chat.csv", upload, "text/csv")},
        ).json()
    repeated = client.post(
        f"/api/projects/{project['id']}/imports/{repeated_preview['id']}/confirm",
        json=payload,
    )

    database = Database(settings.database_url)
    with Session(database.engine) as session:
        message_count = session.scalar(select(func.count(Message.id)))

    assert first.status_code == 201
    assert second.status_code == 200
    assert repeated.status_code == 200
    assert first.json()["message_count"] == 2
    assert second.json()["import_id"] == first.json()["import_id"]
    assert repeated.json()["import_id"] == first.json()["import_id"]
    assert message_count == 2
