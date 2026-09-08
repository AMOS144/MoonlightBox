from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from moonlightbox.agent.jobs import (
    COGNITIVE_CYCLE_JOB_KIND,
    create_cognitive_cycle_handler,
)
from moonlightbox.agent.models import (
    CognitiveCycle,
    MentalStateVersion,
    PerceptionEvent,
)
from moonlightbox.agent.types import CognitionDraft, CognitionRequest
from moonlightbox.api import create_app
from moonlightbox.branches.actor import (
    BRANCH_CONVERSATION_JOB_KIND,
    create_conversation_actor_handler,
    deliver_due_pending_bubbles,
    enqueue_due_actor_reviews,
)
from moonlightbox.branches.actor_models import ConversationActorState
from moonlightbox.branches.baseline_jobs import (
    BRANCH_BASELINE_JOB_KIND,
    create_branch_baseline_handler,
)
from moonlightbox.branches.identity import (
    IdentityKernelProposal,
    IdentityKernelService,
)
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobRegistry
from moonlightbox.training.models import ModelVersion
from moonlightbox.worker import Worker
from sqlalchemy.orm import Session


class FakeGenerator:
    def generate(
        self,
        _model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        assert "不主动编造" in system_prompt
        assert "2026-02-01" in system_prompt
        assert "已确认经历" in system_prompt
        assert "重新联系" in system_prompt
        assert "分支状态" not in system_prompt
        assert "用户对数字人的描述只是用户观点" in system_prompt
        assert "紧凑气泡协议" in system_prompt
        assert "统一气泡 JSON" not in system_prompt
        return GeneratedReplyTurn(
            bubbles=(
                GeneratedBubble(
                    content=f"模拟回复：{messages[-1]['content']}",
                    delay_ms=0,
                ),
                GeneratedBubble(content="第二个气泡", delay_ms=800),
            ),
            raw_output="{}",
        )


class FailingGenerator:
    def generate(
        self,
        _model_version_id: str,
        _system_prompt: str,
        _messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        raise RuntimeError("模拟生成失败")


class FailingCognitionGenerator:
    def generate(self, _request: CognitionRequest) -> CognitionDraft:
        raise RuntimeError("模拟认知失败")


def test_runtime_sticker_policy_cache_reuses_frozen_branch_policy() -> None:
    from moonlightbox.branches.actor import StickerPolicyRuntimeCache
    from moonlightbox.training.sticker_policy import StickerPolicy

    cache = StickerPolicyRuntimeCache(max_entries=2)
    build_count = 0

    def build() -> StickerPolicy:
        nonlocal build_count
        build_count += 1
        return StickerPolicy(enabled=False)

    first = cache.get_or_build(("branch-1", "model-1", "manifest-1"), build)
    second = cache.get_or_build(("branch-1", "model-1", "manifest-1"), build)

    assert first is second
    assert build_count == 1


def test_branch_create_rejects_empty_model_id_before_database_write(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'validation.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )
    with TestClient(create_app(settings, generator=FakeGenerator())) as client:
        response = client.post(
            "/api/projects/project-1/branches",
            json={
                "origin_event_id": "event-1",
                "model_version_id": "",
                "title": "无效分支",
                "origin_time": datetime.now(UTC).isoformat(),
                "state_snapshot": {},
            },
        )

    assert response.status_code == 422


def test_branch_create_rejects_client_supplied_persona_state(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'state-injection.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )
    with TestClient(create_app(settings, generator=FakeGenerator())) as client:
        response = client.post(
            "/api/projects/project-1/branches",
            json={
                "origin_event_id": "event-1",
                "model_version_id": "model-1",
                "title": "不能注入人格",
                "origin_time": datetime.now(UTC).isoformat(),
                "state_snapshot": {"relationship_state": {"trust": 100}},
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_branch_conversation_is_created_generated_and_replayed(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        database_url=f"sqlite:///{tmp_path / 'branches.db'}",
        chroma_dir=tmp_path / "chroma",
        model_dir=tmp_path / "models",
        auto_create_schema=True,
    )
    with TestClient(create_app(settings, generator=FakeGenerator())) as client:
        project = client.post("/api/projects", json={"name": "时间分支"}).json()
        database = Database(settings.database_url)
        with Session(database.engine) as session:
            source = ImportSource(
                project_id=project["id"],
                preview_id="preview-branch-api",
                source_path="/tmp/chat.txt",
                message_count=2,
                confirmed_at=datetime.now(UTC),
            )
            target = Participant(
                project_id=project["id"], name="她", role="target"
            )
            session.add_all([source, target])
            session.flush()
            now = datetime.now(UTC)
            session.add_all(
                [
                    Message(
                        project_id=project["id"],
                        import_id=source.id,
                        participant_id=target.id,
                        source_id="m1",
                        timestamp=now,
                        kind="text",
                        content="重新联系",
                        raw={},
                    ),
                    Message(
                        project_id=project["id"],
                        import_id=source.id,
                        participant_id=target.id,
                        source_id="m2",
                        timestamp=now + timedelta(seconds=1),
                        kind="text",
                        content="好",
                        raw={},
                    ),
                ]
            )
            session.commit()
        event = client.post(
            f"/api/projects/{project['id']}/events",
            json={
                "type": "reconciliation",
                "start_message_id": "m1",
                "end_message_id": "m2",
                "before_state": "疏远",
                "after_state": "朋友",
                "emotion_labels": ["期待"],
                "topic": "重新联系",
                "conflict_level": 1,
                "importance": 0.8,
                "reason": "恢复交流",
                "evidence_ids": ["m1", "m2"],
            },
        ).json()
        model = client.post(
            f"/api/projects/{project['id']}/models",
            json={
                "base_model": "qwen",
                "adapter_path": "/models/test",
                "dataset_hash": "hash",
                "metrics": {},
            },
        ).json()
        branch_payload = {
            "origin_event_id": event["id"],
            "model_version_id": model["id"],
            "title": "如果那天我道歉了",
            "origin_time": datetime(2026, 2, 1).isoformat(),
        }
        rejected = client.post(
            f"/api/projects/{project['id']}/branches",
            json=branch_payload,
        )
        assert rejected.status_code == 409
        assert rejected.json()["detail"]["code"] == "model_not_accepted"
        with Session(database.engine) as session:
            version = session.get(ModelVersion, model["id"])
            assert version is not None
            version.active = True
            IdentityKernelService(session).create_and_lock(
                project_id=project["id"],
                model_version_id=model["id"],
                proposal=IdentityKernelProposal(
                    persona="她",
                    values=["真诚"],
                    stable_preferences=[],
                    relationship_boundaries=["不接受被定义感受"],
                    language_patterns=["自然口语"],
                    typical_reactions=["保留自己的判断"],
                ),
                evidence_message_ids=["m1", "m2"],
                acceptance_report_id="accepted-report",
            )
        created = client.post(
            f"/api/projects/{project['id']}/branches",
            json=branch_payload,
        )
        assert created.status_code == 201
        branch_id = created.json()["id"]
        blocked = client.post(
            f"/api/projects/{project['id']}/branches/{branch_id}/messages",
            json={"content": "还不能发送"},
        )
        preparing = client.get(
            f"/api/projects/{project['id']}/branches/{branch_id}/preparation"
        )
        assert blocked.status_code == 409
        assert blocked.json()["detail"]["code"] == "branch_baseline_not_ready"
        assert preparing.json()["status"] == "preparing"
        registry = JobRegistry()
        registry.register(
            BRANCH_BASELINE_JOB_KIND, create_branch_baseline_handler()
        )
        assert Worker(database, registry).run_once()
        ready = client.get(
            f"/api/projects/{project['id']}/branches/{branch_id}/preparation"
        )
        assert ready.json()["status"] == "ready"

        state_updated = client.put(
            f"/api/projects/{project['id']}/branches/{branch_id}/situational-state",
            json={
                "values": {
                    "availability": "并不方便电话",
                    "emotion": "已经并不生气了",
                },
                "valid_until": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "idempotency_key": "user-state-1",
            },
        )
        state_duplicate = client.put(
            f"/api/projects/{project['id']}/branches/{branch_id}/situational-state",
            json={
                "values": {"availability": "并不方便电话"},
                "valid_until": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "idempotency_key": "user-state-1",
            },
        )
        assert state_updated.status_code == 200
        assert state_duplicate.status_code == 200
        assert state_duplicate.json()["event_id"] == state_updated.json()["event_id"]
        assert state_updated.json()["state"]["values"] == {
            "availability": "并不方便电话",
            "emotion": "已经并不生气了",
        }
        with Session(database.engine) as session:
            state_events = session.query(PerceptionEvent).filter(
                PerceptionEvent.branch_id == branch_id,
                PerceptionEvent.event_type == "situational_observation",
            ).all()
            assert len(state_events) == 1
            assert state_events[0].source == "user_configured"

        generated = client.post(
            f"/api/projects/{project['id']}/branches/{branch_id}/messages",
            json={"content": "对不起", "client_message_id": "client-message-1"},
        )
        typing = client.put(
            f"/api/projects/{project['id']}/branches/{branch_id}/typing",
            json={"typing": True},
        )
        duplicate = client.post(
            f"/api/projects/{project['id']}/branches/{branch_id}/messages",
            json={"content": "对不起", "client_message_id": "client-message-1"},
        )
        with Session(database.engine) as session:
            user_events = session.query(PerceptionEvent).filter(
                PerceptionEvent.branch_id == branch_id,
                PerceptionEvent.event_type == "user_message",
            ).all()
            cycles = session.query(CognitiveCycle).filter(
                CognitiveCycle.branch_id == branch_id
            ).all()
            mental_states = session.query(MentalStateVersion).filter(
                MentalStateVersion.branch_id == branch_id
            ).all()
            message_jobs = session.query(Job).filter(
                Job.payload["message_id"].as_string() == generated.json()["id"]
            ).all()
            assert len(user_events) == 1
            assert user_events[0].evidence == {
                "branch_message_id": generated.json()["id"],
                "content": "对不起",
            }
            assert len(cycles) == 1
            assert cycles[0].trigger_event_id == user_events[0].id
            assert len(mental_states) == 1
            assert mental_states[0].state["relationship_state"] == {
                "summary": "朋友"
            }
            assert {job.kind for job in message_jobs} == {
                BRANCH_CONVERSATION_JOB_KIND,
                COGNITIVE_CYCLE_JOB_KIND,
            }
        cognition_registry = JobRegistry()
        cognition_registry.register(
            COGNITIVE_CYCLE_JOB_KIND,
            create_cognitive_cycle_handler(FailingCognitionGenerator()),
        )
        assert Worker(
            database,
            cognition_registry,
            allowed_kinds={COGNITIVE_CYCLE_JOB_KIND},
        ).run_once()
        with Session(database.engine) as session:
            assert session.query(Job).filter(
                Job.kind == COGNITIVE_CYCLE_JOB_KIND
            ).one().status == "failed"
            assert session.query(Job).filter(
                Job.kind == BRANCH_CONVERSATION_JOB_KIND,
                Job.payload["message_id"].as_string() == generated.json()["id"],
            ).one().status == "queued"
        immediate = client.get(
            f"/api/projects/{project['id']}/branches/{branch_id}/messages"
        )
        actor_registry = JobRegistry()
        actor_registry.register(
            BRANCH_CONVERSATION_JOB_KIND,
            create_conversation_actor_handler(FakeGenerator()),
        )
        actor_worker = Worker(
            database,
            actor_registry,
            allowed_kinds={BRANCH_CONVERSATION_JOB_KIND},
        )
        assert actor_worker.run_once()
        deferred = client.get(
            f"/api/projects/{project['id']}/branches/{branch_id}/messages"
        )
        with Session(database.engine) as session:
            state = session.query(ConversationActorState).filter(
                ConversationActorState.branch_id == branch_id
            ).one()
            state.user_typing_until = None
            state.next_review_at = datetime.now(UTC) - timedelta(seconds=1)
            session.commit()
            assert enqueue_due_actor_reviews(session) == 1
            session.refresh(state)
            assert state.next_review_at is None
            assert enqueue_due_actor_reviews(session) == 0
        assert actor_worker.run_once()
        with Session(database.engine) as session:
            actor_jobs = session.query(Job).filter(
                Job.kind == BRANCH_CONVERSATION_JOB_KIND
            ).all()
            assert all(job.status == "succeeded" for job in actor_jobs)
            assert deliver_due_pending_bubbles(
                session,
                branch_id=branch_id,
                now=datetime.now(UTC) + timedelta(seconds=10),
            ) == 1
        replayed = client.get(
            f"/api/projects/{project['id']}/branches/{branch_id}/messages"
        )
        failed_message = client.post(
            f"/api/projects/{project['id']}/branches/{branch_id}/messages",
            json={"content": "触发失败", "client_message_id": "client-message-2"},
        )
        failing_registry = JobRegistry()
        failing_registry.register(
            BRANCH_CONVERSATION_JOB_KIND,
            create_conversation_actor_handler(FailingGenerator()),
        )
        assert Worker(
            database,
            failing_registry,
            allowed_kinds={BRANCH_CONVERSATION_JOB_KIND},
        ).run_once()
        with Session(database.engine) as session:
            failed_state = session.query(ConversationActorState).filter(
                ConversationActorState.branch_id == branch_id
            ).one()
            assert failed_state.status == "idle"
            assert failed_state.assistant_typing_until is None
            assert failed_state.lease_owner is None
        database.close()

    assert generated.status_code == 202
    assert typing.status_code == 200
    assert duplicate.status_code == 202
    assert duplicate.json()["id"] == generated.json()["id"]
    assert failed_message.status_code == 202
    assert generated.json()["role"] == "user"
    assert generated.json()["client_message_id"] == "client-message-1"
    assert [message["role"] for message in immediate.json()] == ["user"]
    assert [message["role"] for message in deferred.json()] == ["user"]
    assert [message["role"] for message in replayed.json()] == [
        "user",
        "assistant",
        "assistant",
    ]
    assert replayed.json()[1]["content"] == "模拟回复：对不起"
    assert replayed.json()[2]["delay_ms"] == 800
    assert replayed.json()[1]["turn_id"] == replayed.json()[2]["turn_id"]
