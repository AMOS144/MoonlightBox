from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.config import Settings
from moonlightbox.db import Database
from sqlalchemy import func, select
from sqlalchemy.orm import Session


def test_system_sender_is_not_a_selectable_person(client: TestClient):
    project = client.post('/api/projects', json={'name': '系统通知过滤'}).json()
    root = f"/api/projects/{project['id']}/imports"
    content = ('时间,发送者,类型,内容\n'
               '2026-01-01 20:00:00,甲,文本,你好\n'
               '2026-01-01 20:00:01,乙,文本,你好\n'
               '2026-01-01 20:00:02,系统,系统,撤回了一条消息\n').encode()
    preview = client.post(root + '/preview', files={'file': ('chat.csv', content, 'text/csv')}).json()
    assert preview['participants'] == ['甲', '乙']
    assert preview['message_count'] == 3
    result = client.post(root + f"/{preview['id']}/confirm", json={'self_participant': '甲', 'target_participant': '系统'})
    assert result.status_code in {400, 422}


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
