from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.media.annotation_pipeline import (
    SemanticResult,
    _bounded_confidence,
    _json_object,
    annotate_project_media,
)
from moonlightbox.media.models import MediaAsset
from moonlightbox.media.service import MediaAnnotationService, MediaStore
from moonlightbox.projects.models import Project
from sqlalchemy.orm import Session


class FakeTranscriber:
    model_id = "fake-whisper"
    source_version = "1"

    def transcribe(self, _path: Path) -> SemanticResult:
        return SemanticResult(transcript="我刚到家，你呢", confidence=0.91)


class FakeVisualAnnotator:
    model_id = "fake-vlm"
    source_version = "1"

    def annotate(self, _path: Path) -> SemanticResult:
        return SemanticResult(summary="桌上的一杯咖啡", confidence=0.88)


def test_visual_output_parser_accepts_bounded_numeric_strings_and_extra_text() -> None:
    payload = _json_object(
        'result: {"summary":"桌上的咖啡","confidence":"0.82"}\n处理完成'
    )

    assert payload["summary"] == "桌上的咖啡"
    assert _bounded_confidence(payload["confidence"], maximum=0.9) == 0.82
    assert _bounded_confidence("nan", maximum=0.9) == 0


def test_visual_output_parser_accepts_safe_python_dict_fallback() -> None:
    assert _json_object("{'summary': '一只猫', 'confidence': 0.8}") == {
        "summary": "一只猫",
        "confidence": 0.8,
    }


def _link_target_assets(
    session: Session,
    project_id: str,
    assets: list[tuple[MediaAsset, str]],
) -> None:
    session.add_all(
        [
            ImportSource(
                id=f"import-{project_id}",
                project_id=project_id,
                preview_id=f"preview-{project_id}",
                source_path="history.json",
                message_count=len(assets),
                confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            Participant(
                id=f"target-{project_id}",
                project_id=project_id,
                name="她",
                role="target",
            ),
        ]
    )
    session.commit()
    session.add_all(
        [
            Message(
                id=f"message-{project_id}-{index}",
                project_id=project_id,
                import_id=f"import-{project_id}",
                participant_id=f"target-{project_id}",
                source_id=str(index),
                timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
                kind=kind,
                content=f"[{kind}]",
                raw={},
                media_asset_id=asset.id,
            )
            for index, (asset, kind) in enumerate(assets)
        ]
    )
    session.commit()


def test_media_store_deduplicates_within_project(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'media.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="媒体"))
        session.commit()
        store = MediaStore(tmp_path / "data")
        first = store.save(session, "project-1", kind="sticker", filename="a.gif", content=b"same")
        second = store.save(
            session, "project-1", kind="sticker", filename="../b.gif", content=b"same"
        )

        assert first.id == second.id
        assert session.query(MediaAsset).count() == 1
        assert ".." not in first.relative_path


def test_media_annotation_fails_closed_before_audio_can_be_reused(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'annotations.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="媒体语义"))
        session.commit()
        session.add(
            MediaAsset(
                id="audio-1",
                project_id="project-1",
                kind="audio",
                sha256="a" * 64,
                relative_path="projects/project-1/media/audio.wav",
                mime_type="audio/wav",
            )
        )
        session.commit()
        service = MediaAnnotationService(session)

        incomplete = service.annotate(
            "project-1",
            "audio-1",
            source_model="whisper-test",
            source_version="v1",
            confidence=0.9,
            reuse_decision="approved",
        )

        assert incomplete.status == "needs_review"
        assert incomplete.reuse_decision == "pending"
        assert incomplete.reusable is False

        approved = service.annotate(
            "project-1",
            "audio-1",
            transcript="我刚到家，你呢",
            source_model="whisper-test",
            source_version="v1",
            confidence=0.91,
            reuse_decision="approved",
        )

        assert approved.status == "succeeded"
        assert approved.reusable is True
        assert service.reusable_assets("project-1", "audio") == [approved]


def test_media_annotation_blocks_sensitive_or_low_confidence_assets(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'blocked.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="媒体语义"))
        session.commit()
        session.add_all(
            [
                MediaAsset(
                    id="image-sensitive",
                    project_id="project-1",
                    kind="image",
                    sha256="b" * 64,
                    relative_path="projects/project-1/media/sensitive.jpg",
                    mime_type="image/jpeg",
                ),
                MediaAsset(
                    id="image-uncertain",
                    project_id="project-1",
                    kind="image",
                    sha256="c" * 64,
                    relative_path="projects/project-1/media/uncertain.jpg",
                    mime_type="image/jpeg",
                ),
            ]
        )
        session.commit()
        service = MediaAnnotationService(session)

        sensitive = service.annotate(
            "project-1",
            "image-sensitive",
            summary="一张身份证件照片",
            safety_tags=["Identity_Document"],
            source_model="vlm-test",
            source_version="v1",
            confidence=0.98,
            reuse_decision="approved",
        )
        uncertain = service.annotate(
            "project-1",
            "image-uncertain",
            summary="室内的一张模糊照片",
            source_model="vlm-test",
            source_version="v1",
            confidence=0.6,
            reuse_decision="approved",
        )

        assert sensitive.reusable is False
        assert uncertain.reusable is False
        assert service.reusable_assets("project-1", "image") == []


def test_media_annotation_api_is_project_scoped_and_auditable(
    client: TestClient,
    settings: Settings,
) -> None:
    project_id = client.post("/api/projects", json={"name": "媒体 API"}).json()["id"]
    other_id = client.post("/api/projects", json={"name": "其他项目"}).json()["id"]
    database = Database(settings.database_url)
    with Session(database.engine) as session:
        asset = MediaAsset(
                id="image-1",
                project_id=project_id,
                kind="image",
                sha256="d" * 64,
                relative_path=f"projects/{project_id}/media/image.jpg",
                mime_type="image/jpeg",
            )
        session.add(asset)
        session.commit()
        _link_target_assets(session, project_id, [(asset, "image")])
    database.close()

    written = client.put(
        f"/api/projects/{project_id}/media/image-1/annotation",
        json={
            "summary": "桌上的一杯咖啡",
            "safety_tags": [],
            "confidence": 0.93,
            "reuse_decision": "approved",
        },
    )

    assert written.status_code == 200
    assert written.json()["source_model"] == "manual-review"
    assert written.json()["review_source"] == "manual-ui"
    assert written.json()["reviewed_at"] is not None
    assert written.json()["reusable"] is True
    assert client.get(f"/api/projects/{project_id}/media/annotations/list").json()[
        0
    ]["asset_id"] == "image-1"
    assert client.get(
        f"/api/projects/{project_id}/media/annotations/summary"
    ).json() == {
        "eligible": 1,
        "annotated": 1,
        "succeeded": 1,
        "failed": 0,
        "needs_review": 0,
        "approved": 1,
        "blocked": 0,
    }
    assert (
        client.put(
            f"/api/projects/{other_id}/media/image-1/annotation",
            json={"confidence": 0.9},
        ).status_code
        == 404
    )


def test_local_annotation_pipeline_understands_audio_and_images_but_skips_stickers(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'pipeline.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="媒体批处理"))
        session.commit()
        store = MediaStore(tmp_path / "data")
        audio = store.save(
            session,
            "project-1",
            kind="audio",
            filename="voice.wav",
            content=b"voice",
        )
        image = store.save(
            session,
            "project-1",
            kind="image",
            filename="coffee.jpg",
            content=b"image",
        )
        sticker = store.save(
            session,
            "project-1",
            kind="sticker",
            filename="sticker.gif",
            content=b"sticker",
        )
        session.commit()
        _link_target_assets(
            session,
            "project-1",
            [(audio, "audio"), (image, "image"), (sticker, "sticker")],
        )

        result = annotate_project_media(
            session,
            project_id="project-1",
            store=store,
            audio_transcriber=FakeTranscriber(),
            visual_annotator=FakeVisualAnnotator(),
        )

        annotations = MediaAnnotationService(session).list_for_project("project-1")
        by_asset = {item.asset_id: item for item in annotations}
        assert result["succeeded"] == 2
        assert result["skipped"] == 1
        assert by_asset[audio.id].transcript == "我刚到家，你呢"
        assert by_asset[image.id].summary == "桌上的一杯咖啡"
        assert sticker.id not in by_asset
        assert all(item.reuse_decision == "pending" for item in annotations)
        assert all(item.reusable is False for item in annotations)


def test_local_annotation_pipeline_also_understands_incoming_self_media(
    tmp_path: Path,
) -> None:
    database = Database(f"sqlite:///{tmp_path / 'incoming-media.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="双边媒体"))
        session.commit()
        store = MediaStore(tmp_path / "data")
        audio = store.save(
            session,
            "project-1",
            kind="audio",
            filename="incoming.wav",
            content=b"voice",
        )
        session.add_all(
            [
                ImportSource(
                    id="import-project-1",
                    project_id="project-1",
                    preview_id="preview-project-1",
                    source_path="history.json",
                    message_count=1,
                    confirmed_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
                Participant(
                    id="self-project-1",
                    project_id="project-1",
                    name="我",
                    role="self",
                ),
            ]
        )
        session.commit()
        session.add(
            Message(
                id="incoming-audio",
                project_id="project-1",
                import_id="import-project-1",
                participant_id="self-project-1",
                source_id="incoming-audio",
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                kind="audio",
                content="[语音]",
                raw={},
                media_asset_id=audio.id,
            )
        )
        session.commit()

        result = annotate_project_media(
            session,
            project_id="project-1",
            store=store,
            audio_transcriber=FakeTranscriber(),
        )

        annotation = MediaAnnotationService(session).list_for_project("project-1")[0]
        assert result["succeeded"] == 1
        assert annotation.asset_id == audio.id
        assert annotation.transcript == "我刚到家，你呢"
        assert annotation.reuse_decision == "pending"
        assert annotation.reusable is False
    database.close()
