from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from moonlightbox.branches.baseline_models import BranchBaselineManifest
from moonlightbox.branches.models import Branch
from moonlightbox.db import Base, Database
from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.media.models import MediaAsset
from moonlightbox.media.selection import select_reusable_media
from moonlightbox.media.service import MediaAnnotationService
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class SameVectorEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture
def database(tmp_path: Path) -> Database:
    import moonlightbox.api as api

    assert api.app is not None
    database = Database(f"sqlite:///{tmp_path / 'media-selection.db'}")
    Base.metadata.create_all(database.engine)
    return database


def test_media_selection_obeys_target_and_frozen_branch_boundary(
    database: Database,
) -> None:
    boundary_time = datetime(2026, 1, 2, tzinfo=UTC)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="媒体选择"))
        session.commit()
        session.add_all(
            [
                ImportSource(
                    id="import-1",
                    project_id="project-1",
                    preview_id="preview-1",
                    source_path="history.json",
                    message_count=3,
                    confirmed_at=boundary_time,
                ),
                Participant(
                    id="target-1",
                    project_id="project-1",
                    name="她",
                    role="target",
                ),
                ModelVersion(
                    id="model-1",
                    project_id="project-1",
                    base_model="test",
                    adapter_path="test",
                    dataset_hash="hash",
                    metrics={},
                ),
                EventNode(
                    id="event-1",
                    project_id="project-1",
                    type="origin",
                    start_message_id="boundary",
                    end_message_id="boundary",
                    emotion_labels=[],
                    topic="",
                    conflict_level=0,
                    importance=0,
                    reason="",
                    evidence_ids=[],
                ),
            ]
        )
        session.commit()
        session.add_all(
            [
                MediaAsset(
                    id="past-image",
                    project_id="project-1",
                    kind="image",
                    sha256="e" * 64,
                    relative_path="projects/project-1/media/past.jpg",
                    mime_type="image/jpeg",
                ),
                MediaAsset(
                    id="future-image",
                    project_id="project-1",
                    kind="image",
                    sha256="f" * 64,
                    relative_path="projects/project-1/media/future.jpg",
                    mime_type="image/jpeg",
                ),
            ]
        )
        session.commit()
        session.add_all(
            [
                Message(
                    id="past-message",
                    project_id="project-1",
                    import_id="import-1",
                    participant_id="target-1",
                    source_id="001",
                    timestamp=boundary_time - timedelta(hours=1),
                    kind="image",
                    content="[图片]",
                    raw={},
                    media_asset_id="past-image",
                ),
                Message(
                    id="boundary-message",
                    project_id="project-1",
                    import_id="import-1",
                    participant_id="target-1",
                    source_id="002",
                    timestamp=boundary_time,
                    kind="text",
                    content="边界",
                    raw={},
                ),
                Message(
                    id="future-message",
                    project_id="project-1",
                    import_id="import-1",
                    participant_id="target-1",
                    source_id="003",
                    timestamp=boundary_time + timedelta(hours=1),
                    kind="image",
                    content="[图片]",
                    raw={},
                    media_asset_id="future-image",
                ),
            ]
        )
        session.commit()
        branch = Branch(
            id="branch-1",
            project_id="project-1",
            origin_event_id="event-1",
            model_version_id="model-1",
            title="媒体选择",
            origin_time=boundary_time,
            state_snapshot={},
        )
        session.add(branch)
        session.commit()
        session.add(
            BranchBaselineManifest(
                id="manifest-1",
                branch_id="branch-1",
                project_id="project-1",
                import_id="import-1",
                origin_event_id="event-1",
                boundary_message_id="boundary-message",
                boundary_timestamp=boundary_time,
                boundary_source_id="002",
                boundary_inclusive=True,
                message_count=2,
                event_snapshot_count=0,
                message_digest="digest",
                event_digest="digest",
                index_fingerprint="fingerprint",
                recent_tail_message_ids=[],
                protocol_version="test",
                validated_at=boundary_time,
            )
        )
        session.commit()
        annotations = MediaAnnotationService(session)
        annotations.annotate(
            "project-1",
            "past-image",
            summary="桌上的一杯咖啡",
            source_model="vlm-test",
            source_version="v1",
            confidence=0.9,
            reuse_decision="approved",
        )
        annotations.annotate(
            "project-1",
            "future-image",
            summary="桌上的一杯咖啡",
            source_model="vlm-test",
            source_version="v1",
            confidence=0.99,
            reuse_decision="approved",
        )

        selected = select_reusable_media(
            session,
            branch=branch,
            modality="image",
            query="给我看看咖啡",
            embedder=SameVectorEmbedder(),
        )

        assert selected is not None
        assert selected.asset_id == "past-image"
