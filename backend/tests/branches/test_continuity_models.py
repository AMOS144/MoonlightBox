from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from moonlightbox.branches.models import Branch
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


def test_branch_memory_schema_preserves_lineage_and_versions(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.continuity_models import (
        BranchMemoryEpisode,
        BranchMemoryItem,
        BranchStateVersion,
        IdentityKernel,
    )

    database = Database(f"sqlite:///{tmp_path / 'continuity.db'}")
    Project.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="持续记忆"))
        session.flush()
        session.add(
            EventNode(
                id="event-1",
                project_id="project-1",
                type="relationship",
                title="重新联系",
                summary="重新建立交流",
                start_message_id="m1",
                end_message_id="m2",
                emotion_labels=[],
                topic="关系",
                conflict_level=0,
                importance=0.8,
                reason="分支起点",
                evidence_ids=["m1", "m2"],
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
        session.add(
            Branch(
                id="branch-1",
                project_id="project-1",
                origin_event_id="event-1",
                model_version_id="model-1",
                title="分支",
                origin_time=now,
                state_snapshot={},
            )
        )
        session.flush()
        kernel = IdentityKernel(
            id="kernel-1",
            project_id="project-1",
            model_version_id="model-1",
            schema_version="v1",
            content={"persona": "稳定"},
            evidence_message_ids=["m1"],
            content_hash="kernel-hash",
            locked_at=now,
        )
        episode = BranchMemoryEpisode(
            id="episode-1",
            branch_id="branch-1",
            user_turn_id="user-turn",
            assistant_turn_id="assistant-turn",
            user_content="你最近怎么样",
            assistant_bubbles=[{"type": "text", "content": "还好"}],
            model_version_id="model-1",
            episode_hash="episode-hash",
            importance=4.0,
            processing_status="pending",
            started_at=now,
            ended_at=now,
        )
        item = BranchMemoryItem(
            id="item-1",
            branch_id="branch-1",
            kind="self_narrative",
            content="我觉得自己还算平静",
            subject="digital_human",
            predicate="情绪",
            object="平静",
            confidence=0.6,
            importance=4.0,
            valid_from=now,
            source_episode_ids=["episode-1"],
            source_item_ids=[],
            lineage_hash="lineage-1",
            review_status="approved",
        )
        state = BranchStateVersion(
            id="state-1",
            branch_id="branch-1",
            version=1,
            persona_state={},
            relationship_state={"trust": 50},
            user_model={},
            emotional_tendency={"calm": 0.5},
            active_belief_ids=["item-1"],
            reason="初始状态",
            source_episode_ids=["episode-1"],
        )
        session.add_all([kernel, episode, item, state])
        session.commit()

        loaded = session.get(BranchMemoryItem, "item-1")

    assert loaded is not None
    assert loaded.source_episode_ids == ["episode-1"]
    assert loaded.kind == "self_narrative"
    database.close()


def test_branch_episode_turn_pair_is_unique(tmp_path: Path) -> None:
    from moonlightbox.branches.continuity_models import BranchMemoryEpisode

    database = Database(f"sqlite:///{tmp_path / 'unique.db'}")
    Project.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add_all(
            [
                BranchMemoryEpisode(
                    branch_id="branch-1",
                    user_turn_id="user-turn",
                    assistant_turn_id="assistant-turn",
                    user_content="第一条",
                    assistant_bubbles=[],
                    model_version_id="model-1",
                    episode_hash="hash-1",
                    importance=1,
                    processing_status="pending",
                    started_at=now,
                    ended_at=now,
                ),
                BranchMemoryEpisode(
                    branch_id="branch-1",
                    user_turn_id="user-turn",
                    assistant_turn_id="assistant-turn",
                    user_content="重复",
                    assistant_bubbles=[],
                    model_version_id="model-1",
                    episode_hash="hash-2",
                    importance=1,
                    processing_status="pending",
                    started_at=now,
                    ended_at=now,
                ),
            ]
        )
        with pytest.raises(IntegrityError):
            session.commit()
    database.close()


def test_0017_continual_memory_migration_is_reversible(tmp_path: Path) -> None:
    database_url = f"sqlite:///{tmp_path / 'migration.db'}"
    config = Config("backend/alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "0017_add_branch_continual_memory")
    tables = set(inspect(create_engine(database_url)).get_table_names())
    assert {
        "identity_kernels",
        "branch_memory_episodes",
        "branch_memory_items",
        "branch_belief_evidence",
        "branch_state_versions",
        "branch_reflection_runs",
    }.issubset(tables)

    command.downgrade(config, "0016_add_branch_upgrade_fields")
    downgraded = set(inspect(create_engine(database_url)).get_table_names())
    assert "branch_memory_episodes" not in downgraded

    command.upgrade(config, "0017_add_branch_continual_memory")
    upgraded = set(inspect(create_engine(database_url)).get_table_names())
    assert "branch_memory_episodes" in upgraded
