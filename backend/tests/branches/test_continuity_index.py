from datetime import UTC, datetime
from pathlib import Path

from moonlightbox.branches.continuity_models import (
    BranchMemoryEpisode,
    BranchMemoryItem,
)
from moonlightbox.branches.models import Branch
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


class TinyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                1.0 if "海边" in text else 0.0,
                1.0 if "分支一" in text else 0.0,
                1.0 if "分支二" in text else 0.0,
                0.1,
            ]
            for text in texts
        ]


def _semantic_item(
    item_id: str,
    *,
    root: str,
    episode_id: str,
    stance: str = "support",
    confidence: float = 0.8,
) -> BranchMemoryItem:
    now = datetime.now(UTC)
    return BranchMemoryItem(
        id=item_id,
        branch_id="branch-1",
        kind="fact",
        content="用户按约定回来",
        subject="user",
        predicate="遵守约定",
        object="是",
        confidence=confidence,
        importance=8,
        valid_from=now,
        source_episode_ids=[episode_id],
        source_item_ids=[],
        lineage_hash=f"lineage-{item_id}",
        review_status="approved",
        verification_status="verified_interaction",
        stance=stance,
        root_episode_hashes=[root],
    )


def _candidate(item: BranchMemoryItem) -> tuple[object, ...]:
    return (
        item.id,
        item.kind,
        item.content,
        0.9,
        item.importance,
        item.valid_from,
        item.confidence,
        1.0,
    )


def test_retrieval_score_prioritizes_authority_over_small_similarity_gain() -> None:
    from moonlightbox.branches.continuity_index import _combined_score

    now = datetime.now(UTC)
    verified = _combined_score(
        semantic_similarity=0.8,
        importance=8,
        happened_at=now,
        confidence=0.9,
        authority=1.0,
        now=now,
    )
    unverified = _combined_score(
        semantic_similarity=0.95,
        importance=8,
        happened_at=now,
        confidence=0.9,
        authority=0.25,
        now=now,
    )

    assert verified > unverified


def test_semantic_consolidation_deduplicates_roots_and_preserves_independent_support() -> None:
    from typing import cast

    from moonlightbox.branches.continuity_index import (
        CandidateRow,
        _consolidate_semantic_candidates,
    )

    first = _semantic_item("first", root="same-root", episode_id="episode-1")
    duplicate = _semantic_item("duplicate", root="same-root", episode_id="episode-copy")
    independent = _semantic_item("independent", root="other-root", episode_id="episode-2")
    items = {item.id: item for item in (first, duplicate, independent)}

    result = _consolidate_semantic_candidates(
        [cast(CandidateRow, _candidate(item)) for item in items.values()],
        items=items,
        now=datetime.now(UTC),
    )

    assert len(result) == 1
    assert set(result[0][1]) == {"episode-1", "episode-2"}
    assert result[0][0][6] == 0.96


def test_semantic_opposition_suppresses_disproved_positive_claim() -> None:
    from typing import cast

    from moonlightbox.branches.continuity_index import (
        CandidateRow,
        _consolidate_semantic_candidates,
    )

    support = _semantic_item(
        "support",
        root="support-root",
        episode_id="episode-1",
        confidence=0.7,
    )
    opposition = _semantic_item(
        "opposition",
        root="opposition-root",
        episode_id="episode-2",
        stance="oppose",
        confidence=0.95,
    )
    items = {item.id: item for item in (support, opposition)}

    result = _consolidate_semantic_candidates(
        [cast(CandidateRow, _candidate(item)) for item in items.values()],
        items=items,
        now=datetime.now(UTC),
    )

    assert result == []


def test_generic_extractor_triples_do_not_merge_different_experiences() -> None:
    from typing import cast

    from moonlightbox.branches.continuity_index import (
        CandidateRow,
        _consolidate_semantic_candidates,
    )

    first = _semantic_item("first-message", root="root-1", episode_id="episode-1")
    first.kind = "experience"
    first.predicate = "发送"
    first.object = "消息"
    first.content = "用户说臭"
    second = _semantic_item("second-message", root="root-2", episode_id="episode-2")
    second.kind = "experience"
    second.predicate = "发送"
    second.object = "消息"
    second.content = "用户说明天见"
    items = {item.id: item for item in (first, second)}

    result = _consolidate_semantic_candidates(
        [cast(CandidateRow, _candidate(item)) for item in items.values()],
        items=items,
        now=datetime.now(UTC),
    )

    assert len(result) == 2


def test_continuity_retrieval_is_strictly_isolated_by_branch(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.continuity_index import BranchContinuityRepository

    database = Database(f"sqlite:///{tmp_path / 'index.db'}")
    Project.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="隔离索引"))
        session.flush()
        session.add(
            EventNode(
                id="event-1",
                project_id="project-1",
                type="travel",
                title="旅行",
                summary="一次旅行",
                start_message_id="m1",
                end_message_id="m2",
                emotion_labels=[],
                topic="旅行",
                conflict_level=0,
                importance=0.8,
                reason="起点",
                evidence_ids=["m1"],
            )
        )
        session.add(
            ModelVersion(
                id="model-1",
                project_id="project-1",
                base_model="qwen",
                adapter_path="/tmp/adapter",
                dataset_hash="hash",
                metrics={},
            )
        )
        session.flush()
        branches = [
            Branch(
                id=f"branch-{index}",
                project_id="project-1",
                origin_event_id="event-1",
                model_version_id="model-1",
                title=f"分支{label}",
                origin_time=now,
                state_snapshot={},
            )
            for index, label in ((1, "一"), (2, "二"))
        ]
        session.add_all(branches)
        session.flush()
        session.add_all(
            [
                BranchMemoryEpisode(
                    id="episode-1",
                    branch_id="branch-1",
                    user_turn_id="user-1",
                    assistant_turn_id="assistant-1",
                    user_content="还记得海边吗",
                    assistant_bubbles=[
                        {"type": "text", "content": "分支一选择一起看日落"}
                    ],
                    model_version_id="model-1",
                    episode_hash="hash-1",
                    importance=8,
                    processing_status="processed",
                    started_at=now,
                    ended_at=now,
                ),
                BranchMemoryEpisode(
                    id="episode-2",
                    branch_id="branch-2",
                    user_turn_id="user-2",
                    assistant_turn_id="assistant-2",
                    user_content="还记得海边吗",
                    assistant_bubbles=[
                        {"type": "text", "content": "分支二选择独自离开"}
                    ],
                    model_version_id="model-1",
                    episode_hash="hash-2",
                    importance=8,
                    processing_status="processed",
                    started_at=now,
                    ended_at=now,
                ),
            ]
        )
        session.add(
            BranchMemoryItem(
                id="asserted-fact",
                branch_id="branch-1",
                kind="fact",
                content="用户声称海边那天见过一个朋友",
                subject="user",
                predicate="见过",
                object="朋友",
                confidence=0.6,
                importance=5,
                valid_from=now,
                source_episode_ids=["episode-1"],
                source_item_ids=[],
                lineage_hash="asserted-fact-lineage",
                review_status="approved",
                verification_status="asserted_by_user",
            )
        )
        session.add(
            BranchMemoryItem(
                id="inactive-belief",
                branch_id="branch-1",
                kind="belief",
                content="我已经认定用户永远不会回来",
                subject="digital_human",
                predicate="用户会回来",
                object="否",
                confidence=0.99,
                importance=10,
                valid_from=now,
                source_episode_ids=["episode-1"],
                source_item_ids=[],
                lineage_hash="inactive-belief-lineage",
                review_status="approved",
                verification_status="inferred",
            )
        )
        session.commit()
        repository = BranchContinuityRepository(
            str(tmp_path / "chroma"),
            TinyEmbedder(),
        )

        first = repository.retrieve(session, branches[0], "海边", now=now)
        second = repository.retrieve(session, branches[1], "海边", now=now)

        assert len(first) >= 2
        assert any("分支一" in item.content for item in first)
        assert all("分支二" not in item.content for item in first)
        assert all("永远不会回来" not in item.content for item in first)
        asserted = next(item for item in first if "用户声称海边" in item.content)
        assert "未验证陈述，不得作为客观事实" in asserted.content
        assert len(second) == 1
        assert "分支二" in second[0].content
        assert "分支一" not in second[0].content
    database.close()
