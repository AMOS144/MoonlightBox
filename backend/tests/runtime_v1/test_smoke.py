"""Runtime v1 最小冒烟测试：只验证关键边界，不依赖真实模型或显卡。"""

from datetime import UTC, datetime, timedelta

import moonlightbox.api  # noqa: F401  # 注册全部 ORM 表
from langchain_core.messages import AIMessage
from moonlightbox.config import Settings
from moonlightbox.db import Base, Database
from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from moonlightbox.runtime_v1.branch_models import Branch
from moonlightbox.runtime_v1.clock import create_clock, pause_clock, resume_clock
from moonlightbox.runtime_v1.db_models import (
    RuntimeMemoryIndexRow,
    RuntimeSnapshotRow,
    RuntimeWakeupRow,
)
from moonlightbox.runtime_v1.jobs import enqueue_due_runtime_cycles
from moonlightbox.runtime_v1.planning import generate_day_plan
from moonlightbox.runtime_v1.service import RuntimeService
from moonlightbox.runtime_v1.style import StyleService
from moonlightbox.training.models import ModelVersion
from moonlightbox.world.client import LightRAGMetadata, LightRAGReference, LightRAGRetrieval
from moonlightbox.world.models import (
    ConversationBundle,
    ConversationBundleMessage,
    WorldGraphVersion,
)
from sqlalchemy import select
from sqlalchemy.orm import Session


def test_clock_and_plan_smoke() -> None:
    anchor = datetime(2026, 9, 1, 8, tzinfo=UTC)
    clock = create_clock("branch", anchor, wall_anchor=anchor)
    assert clock.now(anchor + timedelta(hours=2)) == datetime(2026, 9, 1, 10, tzinfo=UTC)
    assert pause_clock(clock, wall_now=anchor + timedelta(hours=1)).status == "paused"
    assert resume_clock(clock, wall_now=anchor + timedelta(hours=1)).status == "running"
    assert len(generate_day_plan("branch", anchor.date()).blocks) >= 4


class _FakeModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def bind_tools(self, _tools: list[object], **_kwargs: object) -> "_FakeModel":
        return self

    def invoke(self, _messages: list[object]) -> AIMessage:
        self.calls += 1
        return AIMessage(content=self.content)

    def count_text_tokens(self, text: str) -> int:
        """测试替身显式提供 tokenizer 契约，生产代码不得退回字符数估算。"""

        return max(1, len(text.split()))


class _FakeLightRAG:
    """不访问 Sidecar；记录 Runtime 实际提交的动态检索问题。"""

    def __init__(self, context: str) -> None:
        self.context = context
        self.queries: list[str] = []

    def query(self, _workspace: str, query: str, **_kwargs: object) -> LightRAGRetrieval:
        self.queries.append(query)
        return LightRAGRetrieval(
            context=self.context,
            references=[LightRAGReference(file_path="bundle-1.txt")],
            metadata=LightRAGMetadata(
                lightrag_version="test",
                embedding_model="test",
                embedding_dimension=1,
                extraction_model="test",
                chunking_strategy="test",
                chunk_token_size=1,
                chunk_overlap_token_size=0,
                entity_prompt_version="test",
            ),
        )


def test_runtime_cycle_smoke_without_gpu(tmp_path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'runtime.db'}")
    Base.metadata.create_all(database.engine)
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.flush()
        session.add(
            ModelVersion(
                id="m",
                project_id="p",
                base_model="test",
                adapter_path="none",
                dataset_hash="h",
                metrics={},
            )
        )
        session.flush()
        session.add(
            EventNode(
                id="e",
                project_id="p",
                type="origin",
                start_message_id="1",
                end_message_id="1",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        session.flush()
        session.add(
            Branch(
                id="b",
                project_id="p",
                origin_event_id="e",
                model_version_id="m",
                title="分支",
                origin_time=now,
            )
        )
        # Runtime v1 只允许从已完成的人物世界启动；测试直接放入冻结快照，
        # 避免在冒烟测试中启动 LightRAG 或占用本地模型显存。
        session.add(
            RuntimeSnapshotRow(
                id="snapshot",
                branch_id="b",
                cutoff_at=now,
                timezone="UTC",
                snapshot_mode="latest_profile",
                source_message_ids=[],
                profile={"identity": {"names": []}},
                routine_profile={},
                compiler_version="test",
            )
        )
        session.commit()
        service = RuntimeService(
            session,
            director_model=_FakeModel(
                '{"action":"speak","speech_mode":"reply","communication_intent":"问候","content_points":["你好"],"state_patch":{},"private_reason":"用户消息"}'
            ),
            actor_model=_FakeModel('{"text":"你好呀","bubbles":["你好呀"],"style_applied":[]}'),
        )
        service.bootstrap("p", "b")
        # bootstrap 会建立独立 MemoryIndexVersion，不再把旧 BranchStateVersion 当作事实库。
        assert (
            session.scalar(
                select(RuntimeMemoryIndexRow).where(RuntimeMemoryIndexRow.branch_id == "b")
            )
            is not None
        )
        service.submit_user_message(
            project_id="p", branch_id="b", content="在吗", idempotency_key="msg-1"
        )
        result = service.process_next(project_id="p", branch_id="b")
        assert result is not None
        assert result["decision"].action == "speak"
        assert result["message"].content == "你好呀"

        # 同时到期的多个 Wakeup 必须合并为一次 Cycle，并在提交后续接下一条计划边界。
        virtual_now = service.get_clock("p", "b").now()
        first_plan_wakeup = session.scalar(
            select(RuntimeWakeupRow).where(
                RuntimeWakeupRow.branch_id == "b",
                RuntimeWakeupRow.status == "scheduled",
            )
        )
        assert first_plan_wakeup is not None
        first_plan_wakeup.wake_at = virtual_now
        # 模拟已经跨过此前安排的边界，使后续调度必须创建新的标准计划 key。
        first_plan_wakeup.idempotency_key = "smoke-expired-plan"
        due_wakeups = [
            RuntimeWakeupRow(
                branch_id="b",
                wake_at=virtual_now,
                reason="延迟回复",
                trigger_type="delayed_reply",
                idempotency_key="smoke-delayed-1",
            ),
        ]
        session.add_all(due_wakeups)
        session.commit()
        merged = service.process_next(project_id="p", branch_id="b")
        assert merged is not None
        assert merged["packet"].trigger["type"] == "delayed_reply"
        assert len(merged["packet"].trigger["additional_triggers"]) == 1
        assert all(
            session.get(RuntimeWakeupRow, wakeup.id).status == "completed"
            for wakeup in [first_plan_wakeup, *due_wakeups]
        )
        assert (
            session.scalar(
                select(RuntimeWakeupRow).where(
                    RuntimeWakeupRow.branch_id == "b",
                    RuntimeWakeupRow.status == "scheduled",
                    RuntimeWakeupRow.wake_at > virtual_now,
                )
            )
            is not None
        )

        # Director 的不完整 speak 决定会在 Actor 之前被 Executor 拒绝，不能浪费
        # LoRA 调用，也不能让未校验意图进入公开表达层。
        rejected_actor = _FakeModel('{"text":"不应生成","bubbles":["不应生成"]}')
        rejecting_service = RuntimeService(
            session,
            director_model=_FakeModel('{"action":"speak","state_patch":{}}'),
            actor_model=rejected_actor,
        )
        rejecting_service.submit_user_message(
            project_id="p", branch_id="b", content="测试校验", idempotency_key="msg-invalid"
        )
        rejected = rejecting_service.process_next(project_id="p", branch_id="b")
        assert rejected is not None
        assert rejected["decision"].action == "wait"
        assert rejected_actor.calls == 0

        # Wakeup 是虚拟时间：暂停后即使墙上时间继续流逝，扫描器也不应创建任务。
        scheduled = session.scalar(
            select(RuntimeWakeupRow).where(
                RuntimeWakeupRow.branch_id == "b",
                RuntimeWakeupRow.status == "scheduled",
            )
        )
        assert scheduled is not None
        scheduled.wake_at = virtual_now + timedelta(hours=1)
        session.commit()
        service.change_clock("p", "b", action="pause")
        assert enqueue_due_runtime_cycles(database, now=datetime.now(UTC) + timedelta(hours=2)) == 0
    database.close()


def test_style_tool_queries_lightrag_without_forcing_history_into_reply_pairs(tmp_path) -> None:
    """跨夜的相邻消息可以作为原文证据，但绝不被工具标成一问一答。"""

    database = Database(f"sqlite:///{tmp_path / 'style.db'}")
    Base.metadata.create_all(database.engine)
    now = datetime(2026, 9, 2, 8, tzinfo=UTC)
    with Session(database.engine) as session:
        session.add(Project(id="p", name="项目"))
        session.flush()
        session.add(
            ImportSource(
                id="import",
                project_id="p",
                preview_id="preview",
                source_path="/tmp/chat.txt",
                message_count=2,
                confirmed_at=now,
            )
        )
        session.flush()
        session.add_all(
            [
                Participant(id="self", project_id="p", name="我", role="self"),
                Participant(id="target", project_id="p", name="她", role="target"),
            ]
        )
        session.flush()
        session.add_all(
            [
                Message(
                    id="sleep",
                    project_id="p",
                    import_id="import",
                    participant_id="self",
                    source_id="1",
                    timestamp=now - timedelta(hours=10),
                    kind="text",
                    content="我睡觉了，晚安",
                    raw={},
                ),
                Message(
                    id="nightmare",
                    project_id="p",
                    import_id="import",
                    participant_id="target",
                    source_id="2",
                    timestamp=now,
                    kind="text",
                    content="bb 我做噩梦了",
                    raw={},
                ),
            ]
        )
        session.add(
            ModelVersion(
                id="m",
                project_id="p",
                base_model="test",
                adapter_path="none",
                dataset_hash="h",
                metrics={},
            )
        )
        # EventNode 外键指向原始消息，先明确写入聊天证据，避免 ORM 无关联对象
        # 在同一次 flush 中选择不稳定的插入顺序。
        session.flush()
        session.add(
            EventNode(
                id="e",
                project_id="p",
                type="origin",
                start_message_id="sleep",
                end_message_id="nightmare",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        session.add(
            Branch(
                id="b",
                project_id="p",
                origin_event_id="e",
                model_version_id="m",
                title="分支",
                origin_time=now,
            )
        )
        session.add(
            WorldGraphVersion(
                id="graph",
                project_id="p",
                trigger_import_id="import",
                workspace_key="world_test",
                status="ready",
                source_fingerprint="source",
                config_fingerprint="config",
                source_import_ids=["import"],
                compiler_version="test",
            )
        )
        session.flush()
        bundle = ConversationBundle(
            project_id="p",
            graph_version_id="graph",
            document_id="bundle-1",
            source_name="bundle-1.txt",
            ordinal=0,
            started_at=now - timedelta(hours=10),
            ended_at=now,
            content="2026-09-01 22:00 我：我睡觉了，晚安\n2026-09-02 08:00 她：bb 我做噩梦了",
            content_hash="bundle",
            primary_message_count=2,
        )
        session.add(bundle)
        session.flush()
        session.add_all(
            [
                ConversationBundleMessage(bundle_id=bundle.id, message_id="sleep", ordinal=0),
                ConversationBundleMessage(bundle_id=bundle.id, message_id="nightmare", ordinal=1),
                RuntimeSnapshotRow(
                    id="snapshot",
                    branch_id="b",
                    graph_version_id="graph",
                    cutoff_at=now,
                    timezone="UTC",
                    snapshot_mode="latest_profile",
                    source_message_ids=["sleep", "nightmare"],
                    profile={},
                    routine_profile={},
                    compiler_version="test",
                ),
            ]
        )
        session.commit()

        sidecar = _FakeLightRAG(bundle.content)
        result = StyleService(
            session,
            settings=Settings(lightrag_enabled=True),
            lightrag_client=sidecar,  # type: ignore[arg-type]
        ).tool(branch_id="b", model_version_id="m").invoke(
            {
                "situation": "用户早上问候，目标人物准备自然回应。",
                "intent": "早安问候",
                "speech_mode": "reply",
                "limit": 2,
            }
        )

        assert "用户早上问候" in sidecar.queries[0]
        example = result["data"]["examples"][0]
        assert example["selection"] == "lightrag_semantic_context"
        assert example["source_ids"] == ["sleep", "nightmare"]
        assert "prompt" not in example
        assert "text" not in example
    database.close()
