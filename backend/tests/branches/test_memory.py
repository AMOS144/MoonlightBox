from datetime import datetime, timedelta
from pathlib import Path

from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session


def test_memory_packet_never_includes_messages_or_events_after_origin(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.memory import build_memory_packet

    database = Database(f"sqlite:///{tmp_path / 'memory.db'}")
    Project.metadata.create_all(database.engine)
    origin = datetime(2026, 1, 10)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="记忆测试"))
        session.flush()
        session.add(Participant(id="person-1", project_id="project-1", name="她", role="target"))
        session.add(
            ImportSource(
                id="import-1",
                project_id="project-1",
                preview_id="preview-1",
                source_path="/tmp/test.csv",
                message_count=2,
                confirmed_at=origin,
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
        session.add_all(
            [
                *[
                    Message(
                        project_id="project-1",
                        import_id="import-1",
                        participant_id="person-1",
                        source_id=f"old-{index}",
                        timestamp=origin - timedelta(days=8 - index),
                        kind="text",
                        content=f"很早以前{index}",
                        raw={},
                    )
                    for index in range(6)
                ],
                Message(
                    project_id="project-1",
                    import_id="import-1",
                    participant_id="person-1",
                    source_id="before",
                    timestamp=origin - timedelta(days=1),
                    kind="text",
                    content="我们昨天说去杭州",
                    raw={},
                ),
                Message(
                    project_id="project-1",
                    import_id="import-1",
                    participant_id="person-1",
                    source_id="future",
                    timestamp=origin + timedelta(days=1),
                    kind="text",
                    content="未来秘密",
                    raw={},
                ),
            ]
        )
        session.add(
            EventNode(
                id="event-1",
                project_id="project-1",
                type="travel",
                title="杭州旅行",
                summary="确认一起去杭州",
                start_message_id="before",
                end_message_id="before",
                started_at=origin - timedelta(days=1),
                ended_at=origin - timedelta(days=1),
                before_state=None,
                after_state=None,
                emotion_labels=[],
                topic="杭州",
                conflict_level=0,
                importance=0.8,
                reason="共同经历",
                evidence_ids=["before"],
            )
        )
        session.add(
            EventNode(
                id="event-2",
                project_id="project-1",
                type="work",
                title="完全无关的工作节点",
                summary="讨论项目排期",
                start_message_id="old-0",
                end_message_id="old-0",
                started_at=origin - timedelta(days=8),
                ended_at=origin - timedelta(days=8),
                before_state=None,
                after_state=None,
                emotion_labels=[],
                topic="项目",
                conflict_level=0,
                importance=1.0,
                reason="工作",
                evidence_ids=["old-0"],
            )
        )
        session.flush()
        branch = Branch(
            id="branch-1",
            project_id="project-1",
            origin_event_id="event-1",
            model_version_id="model-1",
            title="分支",
            origin_time=origin,
            state_snapshot={},
        )
        session.add(branch)
        session.flush()

        packet = build_memory_packet(session, branch, "杭州怎么说", [])

    assert packet.memories[0].resource_id == "branch:branch-1"
    assert packet.memories[0].resource_type == "event"
    assert "当前分支设定" in packet.memories[0].content
    assert "杭州旅行" in packet.memories[0].content
    assert "确认一起去杭州" in packet.memories[0].content
    assert "未来秘密" not in packet.background


def test_memory_packet_groups_valid_branch_bubbles_by_turn(tmp_path: Path) -> None:
    from moonlightbox.branches.memory import build_memory_packet

    database = Database(f"sqlite:///{tmp_path / 'turn-memory.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        branch = Branch(
            id="branch-1",
            project_id="project-1",
            origin_event_id="event-1",
            model_version_id="model-1",
            title="分支",
            origin_time=datetime(2026, 1, 10),
            state_snapshot={},
        )
        history = [
            BranchMessage(
                branch_id="branch-1",
                sequence=0,
                role="assistant",
                content="第一条",
                turn_id="turn-1",
                bubble_index=0,
                generation_metadata={"model_version_id": "model-1"},
            ),
            BranchMessage(
                branch_id="branch-1",
                sequence=1,
                role="assistant",
                content="第二条",
                turn_id="turn-1",
                bubble_index=1,
                delay_ms=500,
                generation_metadata={"model_version_id": "model-1"},
            ),
        ]

        packet = build_memory_packet(session, branch, "继续", history)

    assert len(packet.prompt_messages) == 1
    assert packet.prompt_messages[0]["content"] == "第一条\n第二条"


def test_memory_packet_excludes_old_model_and_degraded_exchanges(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.memory import build_memory_packet

    database = Database(f"sqlite:///{tmp_path / 'filtered-memory.db'}")
    Project.metadata.create_all(database.engine)
    branch = Branch(
        id="branch-1",
        project_id="project-1",
        origin_event_id="event-1",
        model_version_id="new-model",
        title="分支",
        origin_time=datetime(2026, 1, 10),
        state_snapshot={},
    )
    history = [
        BranchMessage(
            branch_id="branch-1",
            sequence=0,
            role="user",
            content="旧问题",
            turn_id="user-old",
        ),
        BranchMessage(
            branch_id="branch-1",
            sequence=1,
            role="assistant",
            content="旧模型编造的日记",
            turn_id="assistant-old",
            generation_metadata={"model_version_id": "old-model"},
        ),
        BranchMessage(
            branch_id="branch-1",
            sequence=2,
            role="user",
            content="新问题",
            turn_id="user-new",
        ),
        BranchMessage(
            branch_id="branch-1",
            sequence=3,
            role="assistant",
            content="你说哪个呀？",
            turn_id="assistant-new",
            generation_metadata={
                "model_version_id": "new-model",
                "degraded": True,
            },
        ),
        BranchMessage(
            branch_id="branch-1",
            sequence=4,
            role="user",
            content="被错误回复的问题",
            turn_id="user-quarantined",
        ),
        BranchMessage(
            branch_id="branch-1",
            sequence=5,
            role="assistant",
            content="入",
            turn_id="assistant-quarantined",
            generation_metadata={
                "model_version_id": "new-model",
                "quarantined": True,
            },
        ),
    ]

    with Session(database.engine) as session:
        packet = build_memory_packet(session, branch, "继续说", history)

    assert packet.prompt_messages == []


def test_memory_packet_keeps_grounded_history_from_accepted_upgrade_lineage(
    tmp_path: Path,
) -> None:
    from moonlightbox.branches.memory import build_memory_packet

    database = Database(f"sqlite:///{tmp_path / 'lineage-memory.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id="project-1", name="连续性"))
        session.add_all(
            [
                ModelVersion(
                    id="old-model",
                    project_id="project-1",
                    base_model="qwen",
                    adapter_path="old",
                    dataset_hash="old",
                    status="ready",
                    metrics={},
                    training_config={},
                ),
                ModelVersion(
                    id="new-model",
                    project_id="project-1",
                    base_model="qwen",
                    adapter_path="new",
                    dataset_hash="new",
                    status="ready",
                    metrics={},
                    training_config={
                        "upgraded_from_model_version_id": "old-model"
                    },
                ),
            ]
        )
        session.flush()
        branch = Branch(
            id="branch-1",
            project_id="project-1",
            origin_event_id="event-1",
            model_version_id="new-model",
            title="分支",
            origin_time=datetime(2026, 1, 10),
            state_snapshot={},
        )
        history = [
            BranchMessage(
                branch_id="branch-1",
                sequence=0,
                role="user",
                content="我们刚才说什么",
                turn_id="user-old",
            ),
            BranchMessage(
                branch_id="branch-1",
                sequence=1,
                role="assistant",
                content="说周末去上海",
                turn_id="assistant-old",
                generation_metadata={"model_version_id": "old-model"},
            ),
        ]

        packet = build_memory_packet(session, branch, "继续", history)

    assert packet.prompt_messages == [
        {"role": "user", "content": "我们刚才说什么"},
        {"role": "assistant", "content": "说周末去上海"},
    ]
