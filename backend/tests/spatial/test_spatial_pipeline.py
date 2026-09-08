from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from moonlightbox.api import create_app
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.spatial.bundle_analysis import (
    BundleAnalysisResult,
    BundlePlaceMention,
    parse_bundle_analysis,
)
from moonlightbox.spatial.bundles import build_bundles
from moonlightbox.spatial.config import SpatialPipelineConfig
from moonlightbox.spatial.episodes import EpisodeMessage, build_episode_manifests
from moonlightbox.spatial.map_provider import Coordinate, NullMapProvider
from moonlightbox.spatial.models import (
    PlaceEntity,
    PlaceGraphNode,
    PlaceGraphSnapshot,
    SpatialAnalysisRun,
)
from moonlightbox.spatial.pipeline import SpatialPipeline
from moonlightbox.spatial.resolution import (
    PlaceResolutionService,
    ResolutionCandidate,
    _deterministic_provider_candidate,
)
from moonlightbox.spatial.retrieval import EpisodeHit
from sqlalchemy import select
from sqlalchemy.orm import Session


def _message(identifier: str, timestamp: datetime, content: str = "回家了") -> EpisodeMessage:
    return EpisodeMessage(
        id=identifier,
        timestamp=timestamp,
        participant_id="target",
        participant_name="妈妈",
        participant_role="target",
        kind="text",
        content=content,
        raw={},
        media_asset_id=None,
    )


def test_episode_split_uses_time_and_budget_but_not_midnight() -> None:
    messages = [
        _message("m1", datetime(2026, 1, 1, 23, 58, tzinfo=UTC)),
        _message("m2", datetime(2026, 1, 2, 0, 2, tzinfo=UTC)),
        _message("m3", datetime(2026, 1, 2, 3, 0, tzinfo=UTC)),
    ]

    episodes = build_episode_manifests(
        messages,
        project_id="project",
        import_id="import",
        segmentation_version="v1",
        episode_gap=timedelta(minutes=90),
        character_budget=8_000,
        message_limit=100,
    )

    assert [episode.message_ids for episode in episodes] == [("m1", "m2"), ("m3",)]
    repeated = build_episode_manifests(
        messages,
        project_id="project",
        import_id="import",
        segmentation_version="v1",
        episode_gap=timedelta(minutes=90),
        character_budget=8_000,
        message_limit=100,
    )
    assert [episode.id for episode in repeated] == [episode.id for episode in episodes]


def test_bundle_expands_one_neighbor_and_merges_overlap() -> None:
    messages = [
        _message(f"m{index}", datetime(2026, 1, 1, index, tzinfo=UTC))
        for index in range(4)
    ]
    episodes = build_episode_manifests(
        messages,
        project_id="project",
        import_id="import",
        segmentation_version="v1",
        episode_gap=timedelta(minutes=10),
        character_budget=8_000,
        message_limit=100,
    )
    hits = [
        EpisodeHit(episode_id=episodes[1].id, time_partition="2026-01"),
        EpisodeHit(episode_id=episodes[2].id, time_partition="2026-01"),
    ]

    bundles = build_bundles(
        episodes,
        hits,
        neighbor_gap=timedelta(hours=2),
        character_budget=20_000,
    )

    assert len(bundles) == 1
    assert bundles[0].episode_ids == tuple(episode.id for episode in episodes)


def test_bundle_protocol_rejects_invalid_span_contract() -> None:
    value = {
        "mentions": [
            {
                "message_id": "m1",
                "span_start": 0,
                "span_end": None,
                "raw_text": "公司",
                "mention_type": "personal_anchor",
                "subject": "target",
                "relation": "at",
                "movement_phase": "stationary",
                "assertion_mode": "reported",
                "time_start": "2026-01-01T08:00:00Z",
                "time_end": None,
                "time_precision": "message",
                "evidence_message_ids": ["m1"],
                "confidence": 0.9,
            }
        ]
    }

    try:
        parse_bundle_analysis(value)
    except ValueError as error:
        assert str(error) == "空间分析输出不符合协议"
    else:
        raise AssertionError("无效 span 协议不应被接受")


def test_provider_candidate_requires_named_poi_cluster_without_city_context() -> None:
    mention = SimpleNamespace(mention_type="named_poi", raw_text="正大广场的摩天轮")
    candidates = [
        ResolutionCandidate(
            key="provider:1",
            source="map_provider",
            canonical_name="合肥正大广场摩天轮乐园",
            place_type="风景名胜",
            coordinate=Coordinate(117.22840, 31.78144, "GCJ-02"),
            features={"name_similarity": 0.74, "geographic_context": 0.0},
        ),
        ResolutionCandidate(
            key="provider:2",
            source="map_provider",
            canonical_name="合肥正大广场",
            place_type="购物服务",
            coordinate=Coordinate(117.22891, 31.78135, "GCJ-02"),
            features={"name_similarity": 0.57, "geographic_context": 0.0},
        ),
    ]

    selected = _deterministic_provider_candidate(mention, candidates)  # type: ignore[arg-type]

    assert selected is candidates[0]


def test_provider_candidate_rejects_generic_default_city_results() -> None:
    mention = SimpleNamespace(mention_type="named_poi", raw_text="医院")
    candidates = [
        ResolutionCandidate(
            key="provider:1",
            source="map_provider",
            canonical_name="北京协和医院西单院区",
            place_type="医疗保健服务",
            coordinate=Coordinate(116.36730, 39.91262, "GCJ-02"),
            features={"name_similarity": 0.25, "geographic_context": 0.0},
        ),
        ResolutionCandidate(
            key="provider:2",
            source="map_provider",
            canonical_name="首都医科大学附属北京儿童医院",
            place_type="医疗保健服务",
            coordinate=Coordinate(116.35437, 39.91250, "GCJ-02"),
            features={"name_similarity": 0.20, "geographic_context": 0.0},
        ),
    ]

    selected = _deterministic_provider_candidate(mention, candidates)  # type: ignore[arg-type]

    assert selected is None


class _FakeIndex:
    def replace(self, episodes: list[object]) -> None:
        self.episodes = episodes

    def retrieve(
        self,
        episodes: list[object],
        queries: list[object],
        *,
        top_k_per_partition: int,
        minimum_similarity: float,
    ) -> list[EpisodeHit]:
        del queries, top_k_per_partition, minimum_similarity
        return [
            EpisodeHit(
                episode_id=episode.id,
                time_partition=episode.time_partition,
                max_similarity=0.95,
            )
            for episode in episodes
        ]


class _FakeAnalyzer:
    def analyze(
        self,
        *,
        messages: list[EpisodeMessage],
        target_person_id: str,
        run_id: str,
        bundle_id: str,
    ) -> BundleAnalysisResult:
        del run_id, bundle_id
        mentions = []
        for message in messages:
            raw_text = "家" if "家" in message.content else "公司"
            mentions.append(
                BundlePlaceMention(
                    message_id=message.id,
                    span_start=message.content.index(raw_text),
                    span_end=message.content.index(raw_text) + len(raw_text),
                    raw_text=raw_text,
                    mention_type="personal_anchor",
                    subject=target_person_id,
                    relation="at",
                    movement_phase="stationary",
                    assertion_mode="reported",
                    time_start=message.timestamp,
                    time_precision="message",
                    evidence_message_ids=[message.id],
                    confidence=0.95,
                )
            )
        return BundleAnalysisResult(
            mentions=tuple(mentions),
            method="fake_test_analyzer",
            version="test-v1",
        )


def test_pipeline_builds_personal_graph_and_heatmap_projection(
    client: TestClient,
    settings: Settings,
    tmp_path: Path,
) -> None:
    project = client.post("/api/projects", json={"name": "地点图"}).json()
    database = Database(settings.database_url)
    with Session(database.engine) as session:
        target = Participant(project_id=project["id"], name="妈妈", role="target")
        self_person = Participant(project_id=project["id"], name="我", role="self")
        session.add_all([target, self_person])
        session.flush()
        imported = ImportSource(
            project_id=project["id"],
            preview_id="preview-spatial",
            source_path=str(tmp_path / "chat.csv"),
            message_count=2,
            confirmed_at=datetime.now(UTC),
        )
        session.add(imported)
        session.flush()
        session.add_all(
            [
                Message(
                    project_id=project["id"],
                    import_id=imported.id,
                    participant_id=target.id,
                    source_id="m1",
                    timestamp=datetime(2026, 1, 1, 8, tzinfo=UTC),
                    kind="text",
                    content="我到家了",
                    raw={},
                ),
                Message(
                    project_id=project["id"],
                    import_id=imported.id,
                    participant_id=target.id,
                    source_id="m2",
                    timestamp=datetime(2026, 1, 1, 9, tzinfo=UTC),
                    kind="text",
                    content="我在公司",
                    raw={},
                ),
            ]
        )
        session.commit()
        config = SpatialPipelineConfig()
        pipeline = SpatialPipeline(
            config=config,
            index=_FakeIndex(),  # type: ignore[arg-type]
            analyzer=_FakeAnalyzer(),
            resolver=PlaceResolutionService(
                provider=NullMapProvider(),
                config=config,
                persist_provider_enrichment=False,
            ),
        )

        run = pipeline.run(
            session,
            project_id=project["id"],
            import_id=imported.id,
        )

        assert run.status == "completed"
        assert session.scalar(select(SpatialAnalysisRun).where(SpatialAnalysisRun.id == run.id))
        snapshot = session.scalar(
            select(PlaceGraphSnapshot).where(PlaceGraphSnapshot.run_id == run.id)
        )
        assert snapshot is not None
        nodes = list(
            session.scalars(
                select(PlaceGraphNode).where(PlaceGraphNode.snapshot_id == snapshot.id)
            )
        )
        assert len(nodes) == 2
        assert abs(sum(node.activity_weight for node in nodes) - 1.0) < 1e-9
        places = list(
            session.scalars(
                select(PlaceEntity).where(PlaceEntity.project_id == project["id"])
            )
        )
        assert {place.canonical_name for place in places} == {"家", "公司"}

    response = client.get(f"/api/projects/{project['id']}/place-heatmap")
    assert response.status_code == 200
    heatmap = response.json()
    assert heatmap["type"] == "FeatureCollection"
    assert heatmap["features"] == []
    assert {place["display_name"] for place in heatmap["unlocated_places"]} == {"家", "公司"}
    weekend = client.get(
        f"/api/projects/{project['id']}/place-heatmap?weekdays=5,6"
    ).json()
    assert weekend["features"] == []
    assert weekend["unlocated_places"] == []
    database.close()


def test_enabled_spatial_analysis_is_enqueued_with_import(tmp_path: Path) -> None:
    settings = Settings.model_construct(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'enabled.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
        spatial_analysis_enabled=True,
    )
    with TestClient(create_app(settings)) as enabled_client:
        project = enabled_client.post("/api/projects", json={"name": "空间任务"}).json()
        preview = enabled_client.post(
            f"/api/projects/{project['id']}/imports/preview",
            files={
                "file": (
                    "chat.csv",
                    (
                        "时间,发送者,类型,内容\n"
                        "2026-01-01 08:00:00,甲,文本,到家了\n"
                        "2026-01-01 08:01:00,乙,文本,好\n"
                    ).encode(),
                    "text/csv",
                )
            },
        ).json()
        confirmed = enabled_client.post(
            f"/api/projects/{project['id']}/imports/{preview['id']}/confirm",
            json={"self_participant": "乙", "target_participant": "甲"},
        ).json()

        assert confirmed["spatial_job_id"]
        job = enabled_client.get(f"/api/jobs/{confirmed['spatial_job_id']}").json()
        assert job["kind"] == "spatial_analysis_v1"
        assert "api_key" not in str(job["payload"]).lower()
        regenerated = enabled_client.post(
            f"/api/projects/{project['id']}/spatial-analysis"
        )
        assert regenerated.status_code == 202
        assert regenerated.json()["job_ids"] == [confirmed["spatial_job_id"]]
