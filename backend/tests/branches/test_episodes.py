from datetime import UTC, datetime
from pathlib import Path

from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.jobs.models import Job
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


def test_successful_reply_creates_one_idempotent_episode(tmp_path: Path) -> None:
    from moonlightbox.branches.continuity_models import BranchMemoryEpisode
    from moonlightbox.branches.episodes import EpisodeService

    database = Database(f"sqlite:///{tmp_path / 'episode.db'}")
    Project.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="Episode"))
        session.flush()
        session.add(
            EventNode(
                id="event-1",
                project_id="project-1",
                type="relationship",
                title="起点",
                summary="重新交流",
                start_message_id="m1",
                end_message_id="m2",
                emotion_labels=[],
                topic="关系",
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
        branch = Branch(
            id="branch-1",
            project_id="project-1",
            origin_event_id="event-1",
            model_version_id="model-1",
            title="分支",
            origin_time=now,
            state_snapshot={},
        )
        user = BranchMessage(
            id="user-message",
            branch_id="branch-1",
            sequence=0,
            role="user",
            content="你最近怎么样",
            turn_id="user-turn",
            bubble_index=0,
        )
        assistants = [
            BranchMessage(
                id="assistant-message-1",
                branch_id="branch-1",
                sequence=1,
                role="assistant",
                content="还好",
                type="text",
                turn_id="assistant-turn",
                bubble_index=0,
            ),
            BranchMessage(
                id="assistant-message-2",
                branch_id="branch-1",
                sequence=2,
                role="assistant",
                content="就是有点累",
                type="text",
                turn_id="assistant-turn",
                bubble_index=1,
            ),
        ]
        session.add_all([branch, user, *assistants])
        session.flush()
        service = EpisodeService(session)

        first, first_job = service.record_turn(branch, user, assistants)
        second, second_job = service.record_turn(branch, user, assistants)
        session.commit()

        assert first.id == second.id
        assert first_job.id == second_job.id
        assert first.assistant_bubbles == [
            {
                "type": "text",
                "content": "还好",
                "asset_id": None,
                "bubble_index": 0,
                "delay_ms": 0,
            },
            {
                "type": "text",
                "content": "就是有点累",
                "asset_id": None,
                "bubble_index": 1,
                "delay_ms": 0,
            },
        ]
        assert first.user_messages[0] | {"created_at": None} == {
            "message_id": "user-message",
            "turn_id": "user-turn",
            "content": "你最近怎么样",
            "sequence": 0,
            "created_at": None,
        }
        assert first.user_messages[0]["created_at"]
        assert session.query(BranchMemoryEpisode).count() == 1
        assert session.query(Job).count() == 1
        assert first_job.dedupe_key == f"branch-memory:{first.id}"
    database.close()


def test_episode_preserves_all_user_contributions_as_causal_evidence(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.continuity_models import BranchMemoryEpisode
    from moonlightbox.branches.episodes import EpisodeService

    database = Database(f"sqlite:///{tmp_path / 'multi-message-episode.db'}")
    Project.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="Episode"))
        session.flush()
        session.add(
            EventNode(
                id="event-1", project_id="project-1", type="relationship",
                title="起点", summary="重新交流", start_message_id="m1",
                end_message_id="m2", emotion_labels=[], topic="关系",
                conflict_level=0, importance=0.8, reason="起点", evidence_ids=["m1"],
            )
        )
        session.add(
            ModelVersion(
                id="model-1", project_id="project-1", base_model="qwen",
                adapter_path="/tmp/adapter", dataset_hash="hash", metrics={},
            )
        )
        session.flush()
        branch = Branch(
            id="branch-1", project_id="project-1", origin_event_id="event-1",
            model_version_id="model-1", title="分支", origin_time=now,
            state_snapshot={},
        )
        users = [
            BranchMessage(
                id="user-1", branch_id=branch.id, sequence=0, role="user",
                content="我周末去上海", turn_id="turn-1", bubble_index=0,
            ),
            BranchMessage(
                id="user-2", branch_id=branch.id, sequence=1, role="user",
                content="你要一起吗", turn_id="turn-2", bubble_index=0,
            ),
        ]
        assistant = BranchMessage(
            id="assistant-1", branch_id=branch.id, sequence=2, role="assistant",
            content="好呀", turn_id="assistant-turn", bubble_index=0,
        )
        session.add_all([branch, *users, assistant])
        session.flush()

        episode, _ = EpisodeService(session).record_turn(branch, users, [assistant])
        session.commit()

        assert episode.user_turn_id == "turn-2"
        assert episode.user_content == "我周末去上海\n你要一起吗"
        assert [item["message_id"] for item in episode.user_messages] == [
            "user-1", "user-2"
        ]
        assert session.query(BranchMemoryEpisode).count() == 1
    database.close()
