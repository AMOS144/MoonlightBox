"""真实 Controller/Worker/SQLite 的离线冒烟；不向云端发消息或改体验数据。"""

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobRegistry
from moonlightbox.node_investigation.jobs import create_investigation_handler
from moonlightbox.node_investigation.router import create_node_investigation_router
from moonlightbox.node_investigation.schemas import (
    CandidateArgs,
    ConfirmArgs,
    InputArgs,
    PreviewArgs,
)
from moonlightbox.node_investigation.store import JOB_KIND, InvestigationStore
from moonlightbox.node_investigation.tools import InvestigationTools
from moonlightbox.projects.models import Project
from moonlightbox.worker import Worker
from moonlightbox.world.models import (
    ConversationBundle,
    ConversationBundleMessage,
    WorldGraphVersion,
)
from sqlalchemy.orm import Session


@pytest.fixture
def setup(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'nodes.db'}")
    database.create_schema()
    with Session(database.engine) as session:
        session.add(Project(id="project", name="test"))
        session.commit()
        session.add(
            ImportSource(
                id="import",
                project_id="project",
                preview_id="preview",
                source_path="test",
                message_count=4,
                confirmed_at=datetime(2026, 5, 1),
            )
        )
        session.add_all(
            [
                Participant(id=role, project_id="project", name=role, role=role)
                for role in ["self", "target"]
            ]
        )
        session.commit()
        for i, (day, text) in enumerate(
            [(1, "明天出发？"), (1, "好啊"), (6, "那几天很开心"), (7, "回来有点不习惯")]
        ):
            session.add(
                Message(
                    id=f"m{i}",
                    project_id="project",
                    import_id="import",
                    participant_id="target" if i % 2 else "self",
                    source_id=str(i),
                    timestamp=datetime(2026, 5, day, 10),
                    kind="text",
                    content=text,
                    raw={},
                )
            )
        session.commit()
    store = InvestigationStore(database.engine, "project")
    store.start("Asia/Shanghai")
    yield database, store
    database.close()


def candidate(**changes):
    return CandidateArgs(
        title="一次可能的共同出行",
        summary="准备到后来回忆，中间几天缺少消息",
        occurrence="5月初，待用户补充",
        why_branch="可以从决定出发前开始",
        source_refs=["m0", "m2"],
        interpretation="可能见面旅行，也可能是其他安排",
        boundaries=[{"label": "出发前", "kind": "message", "message_ref": "m0", "side": "before"}],
        status="ready",
        change_reason="结合前后记录提出候选",
        **changes,
    ).model_dump()


class ScriptModel:
    def __init__(self, calls):
        self.calls = list(calls)
        self.histories = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        self.histories.append(list(messages))
        item = self.calls.pop(0)
        if callable(item):
            item = item(messages)
        name, args = item
        return AIMessage(
            content="",
            tool_calls=[{"id": f"call-{len(self.histories)}", "name": name, "args": args}],
        )


def run_worker(database, model):
    registry = JobRegistry()
    registry.register(
        JOB_KIND,
        create_investigation_handler(Settings.model_construct(lightrag_enabled=False), model=model),
    )
    return Worker(database, registry, allowed_kinds={JOB_KIND}).run_once()


def finish(question=None):
    return (
        "submit_investigation_turn",
        {
            "result": {
                "outcome": "waiting_for_user" if question else "completed",
                "summary": "已找到一个候选，尚未覆盖全部记录",
                "question": question,
            }
        },
    )


def test_worker_does_not_block_on_legacy_question_submission(setup):
    database, store = setup
    first_job = store.get().state["job_id"]
    model = ScriptModel(
        [
            ("read_next_messages", {"cursor": 0, "limit": 2}),
            ("upsert_node_candidate", candidate()),
            finish("中间这几天是见面出行，还是其他安排？"),
        ]
    )
    assert run_worker(database, model)
    view = store.view()
    assert view["status"] == "queued", view
    assert view["pending_question"] is None
    assert view["cursor"] == 2
    assert len(view["candidates"]) == 1
    with Session(database.engine) as session:
        assert session.get(Job, first_job).status == "succeeded"
    second = ScriptModel([("read_next_messages", {"cursor": 2}), finish()])
    assert run_worker(database, second)
    assert store.view()["status"] == "completed"
    assert store.view()["cursor"] == 4


def test_same_second_boundary_and_stale_preview(setup):
    _, store = setup
    tools = InvestigationTools(store, store.get().state["job_id"], None)
    c = tools.upsert(**candidate())
    before = store.preview(c["id"], PreviewArgs(candidate_revision=1, option_index=0))
    assert before["included_count"] == 0
    assert before["cutoff_at"].endswith("+08:00")
    updated = candidate(candidate_ref=c["id"])
    updated["boundaries"][0].update(message_ref="m1", side="after")
    tools.upsert(**updated)
    with pytest.raises(Exception, match="候选已更新"):
        store.confirm(
            c["id"], ConfirmArgs(preview_hash=before["preview_hash"], continue_investigating=False)
        )
    after = store.preview(c["id"], PreviewArgs(candidate_revision=2, option_index=0))
    assert after["included_count"] == 2
    approved = store.confirm(
        c["id"], ConfirmArgs(preview_hash=after["preview_hash"], continue_investigating=False)
    )
    assert approved["confirmed"]["last_included_ref"] == "m1"
    with pytest.raises(Exception, match="暂停|取消"):
        tools.upsert(**updated)


def test_gap_candidate_is_rejected_and_search_does_not_cover(setup):
    _, store = setup
    tools = InvestigationTools(store, store.get().state["job_id"], None)
    args = candidate()
    args["boundaries"] = [{"kind": "gap", "label": "旅行途中", "time_hint": "5月2日至5日"}]
    with pytest.raises(Exception, match="message"):
        tools.upsert(**args)
    tools.context(message_refs=["m3"])
    with pytest.raises(Exception, match="没有可查询"):
        tools.search("naive", question="一起旅行")
    assert store.view()["cursor"] == 0


def test_tool_error_returns_to_model_and_new_input_does_not_cancel(setup):
    database, store = setup

    def inject(messages):
        assert any(isinstance(m, ToolMessage) and "error" in str(m.content) for m in messages)
        store.add_input(InputArgs(request_id="during", text="并非旅行，是断联"))
        return ("read_next_messages", {"cursor": 0, "limit": 1})

    model = ScriptModel([("read_next_messages", {"cursor": 99}), inject, finish()])
    run_worker(database, model)
    assert len(model.histories) == 3
    assert "并非旅行，是断联" in str(model.histories[-1])
    assert any(isinstance(m, ToolMessage) for m in model.histories[-1])
    assert store.view()["status"] == "completed"


def test_rag_recovers_message_ids_not_bundle_mapping_ids(setup):
    database, store = setup
    with Session(database.engine) as session:
        graph = WorldGraphVersion(
            id="graph",
            project_id="project",
            trigger_import_id="import",
            workspace_key="workspace",
            status="ready",
            source_fingerprint="x",
            config_fingerprint="x",
            source_import_ids=["import"],
            compiler_version="test",
        )
        session.add(graph)
        session.flush()
        session.add(
            ConversationBundle(
                id="bundle",
                project_id="project",
                graph_version_id="graph",
                document_id="doc",
                source_name="source.txt",
                ordinal=0,
                started_at=datetime(2026, 5, 1),
                ended_at=datetime(2026, 5, 7),
                content="test",
                content_hash="x",
                primary_message_count=4,
            )
        )
        session.flush()
        for i in range(4):
            session.add(
                ConversationBundleMessage(
                    id=f"mapping-{i}",
                    bundle_id="bundle",
                    message_id=f"m{i}",
                    ordinal=i,
                    is_carry_in=False,
                )
            )
        session.commit()
    store.mutate(lambda s, _: s.update(graph_id="graph", workspace="workspace"))

    class Client:
        def query(self, workspace, question, **kwargs):
            return SimpleNamespace(
                context="旅行线索",
                references=[
                    SimpleNamespace(
                        file_path="source.txt", content="2026-05-06 10:00 self：那几天很开心"
                    )
                ],
            )

    tools = InvestigationTools(store, store.get().state["job_id"], Client())
    result = tools.search("naive", question="旅行")
    assert result["fragments"][0]["message_refs"] == ["m2"]
    assert store.view()["cursor"] == 0


def test_api_ownership_reload_and_atomic_start(setup):
    database, store = setup
    app = FastAPI()
    app.include_router(create_node_investigation_router(database))
    client = TestClient(app)
    base = "/api/projects/project/node-investigations"
    assert client.post(base, json={}).json()["id"] == store.id
    assert client.get(base).json()["id"] == store.id
    assert client.get(f"/api/projects/other/node-investigations/{store.id}").status_code == 409
    assert client.post(f"{base}/{store.id}/pause").status_code == 200
    assert client.post(f"{base}/{store.id}/resume").json()["status"] == "queued"


def test_failed_worker_restores_checkpoint_and_existing_candidates(setup):
    database, store = setup
    model = ScriptModel(
        [("read_next_messages", {"cursor": 0, "limit": 1}), ("upsert_node_candidate", candidate())]
    )
    run_worker(database, model)  # 脚本耗尽模拟供应商异常。
    assert store.view()["status"] == "failed"
    app = FastAPI()
    app.include_router(create_node_investigation_router(database))
    client = TestClient(app)
    assert (
        client.post(f"/api/projects/project/node-investigations/{store.id}/resume").status_code
        == 200
    )
    restored = ScriptModel([("read_next_messages", {"cursor": 1}), finish()])
    run_worker(database, restored)
    assert store.view()["status"] == "completed", store.view()
    assert store.view()["cursor"] == 4
    assert len(store.view()["candidates"]) == 1
    assert any(isinstance(m, ToolMessage) for m in restored.histories[0])


def test_source_mutation_invalidates_confirmation(setup):
    database, store = setup
    tools = InvestigationTools(store, store.get().state["job_id"], None)
    c = tools.upsert(**candidate())
    preview = store.preview(c["id"], PreviewArgs(candidate_revision=1, option_index=0))
    with Session(database.engine) as session:
        session.get(Message, "m0").content = "修改了的原文"
        session.commit()
    with pytest.raises(Exception, match="改变"):
        store.confirm(
            c["id"], ConfirmArgs(preview_hash=preview["preview_hash"], continue_investigating=True)
        )


def test_sensitive_content_masked_and_investigation_completes(tmp_path):
    """云端风控拒绝时只遮触发的那条消息,调查继续并完成。"""
    import json as jsonlib

    from moonlightbox.agent_runtime.sensitive_mask import MaskRegistry, SensitiveContentHandler
    from moonlightbox.runtime_v1.cloud_models import (
        RuntimeCloudChatModel,
        RuntimeCloudCompletion,
        RuntimeCloudInferenceError,
    )

    database = Database(f"sqlite:///{tmp_path / 'nodes.db'}")
    database.create_schema()
    with Session(database.engine) as session:
        session.add(Project(id="project", name="test"))
        session.commit()
        session.add(
            ImportSource(
                id="import",
                project_id="project",
                preview_id="preview",
                source_path="test",
                message_count=4,
                confirmed_at=datetime(2026, 5, 1),
            )
        )
        session.add_all(
            [
                Participant(id=role, project_id="project", name=role, role=role)
                for role in ["self", "target"]
            ]
        )
        session.commit()
        for i, (day, text) in enumerate(
            [(1, "明天出发?"), (1, "SENSITIVE-MARKER"), (6, "那几天很开心"), (7, "回来有点不习惯")]
        ):
            session.add(
                Message(
                    id=f"m{i}",
                    project_id="project",
                    import_id="import",
                    participant_id="target" if i % 2 else "self",
                    source_id=str(i),
                    timestamp=datetime(2026, 5, day, 10),
                    kind="text",
                    content=text,
                    raw={},
                )
            )
        session.commit()
    store = InvestigationStore(database.engine, "project")
    store.start("Asia/Shanghai")

    class FakeCloud:
        _timeout_seconds = 60.0
        _max_retries = 0
        model_name = "fake"

        def complete(self, *, system_prompt, messages, temperature, tools):
            payload = jsonlib.dumps(messages, ensure_ascii=False)
            if "SENSITIVE-MARKER" in payload:
                raise RuntimeCloudInferenceError(
                    "provider_input_rejected", "rejected", diagnostic={}
                )
            name, args = (
                ("submit_investigation_turn", {"result": {"outcome": "completed", "summary": "读完", "question": None}})
                if "message_ref" in payload
                else ("read_next_messages", {"cursor": 0, "limit": 10})
            )
            return RuntimeCloudCompletion(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": jsonlib.dumps(args)},
                        }
                    ],
                },
                usage={},
            )

    registry = MaskRegistry(database.engine, "project")
    handler = SensitiveContentHandler(registry, source="test")
    model = RuntimeCloudChatModel(FakeCloud(), temperature=0.3, sensitive_handler=handler)
    assert run_worker(database, model)
    view = store.view()
    assert view["status"] == "completed", view
    assert view["cursor"] == 4
    # 只有包含敏感词的那条消息进了注册表,其余原文照常发送
    assert registry.load() == frozenset({"m1"})
