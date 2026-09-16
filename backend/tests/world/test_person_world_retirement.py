"""旧版退役：阻止旧任务执行，保留数据，升级生成独立 v3 候选。"""

from datetime import UTC, datetime

import pytest
from moonlightbox.config import Settings
from moonlightbox.db import Base
from moonlightbox.imports.models import ImportSource, Participant
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandlerError
from moonlightbox.jobs.service import JobService
from moonlightbox.projects.models import Project
from moonlightbox.world.models import (
    PersonWorldAgentRun,
    PersonWorldProfile,
    PersonWorldProfileDraft,
    PersonWorldSectionTask,
    WorldGraphVersion,
)
from moonlightbox.world.person_world.jobs import enqueue_profile_recompile_job
from moonlightbox.world.person_world.section_retry import (
    SectionRetryStateError,
    create_section_retry_handler,
    enqueue_section_retry_job,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


@pytest.fixture
def candidate():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(
            [
                Project(id="p", name="测试"),
                ImportSource(
                    id="i",
                    project_id="p",
                    preview_id="preview",
                    source_path="test.csv",
                    message_count=0,
                    confirmed_at=datetime.now(UTC),
                ),
                Participant(id="t", project_id="p", name="目标", role="target"),
            ]
        )
        graph = WorldGraphVersion(
            id="g",
            project_id="p",
            trigger_import_id="i",
            workspace_key="original",
            status="awaiting_profile_review",
            source_fingerprint="s" * 64,
            config_fingerprint="c" * 64,
            source_import_ids=["i"],
            compiler_version="v2",
        )
        run = PersonWorldAgentRun(
            id="r",
            project_id="p",
            graph_version_id="g",
            status="awaiting_review",
            prompt_version="old",
        )
        task = PersonWorldSectionTask(
            id="task", agent_run_id="r", section="identity", status="cloud_error", result_summary={}
        )
        profile = PersonWorldProfile(
            id="profile",
            project_id="p",
            subject_person_id="t",
            graph_version_id="g",
            agent_run_id="r",
            profile_schema_version="v2",
            profile_v2={"historical": True},
            identity={},
            work_and_education=[],
            places=[],
            social_relationships=[],
            preferences=[],
            recurring_activities=[],
            routine_summary={},
            life_phases=[],
            relationship_with_user={},
            important_events=[],
            unresolved_candidates=[],
            source_message_ids=[],
            retrieval_manifest=[],
            compiler_version="v2",
        )
        draft = PersonWorldProfileDraft(
            id="draft",
            project_id="p",
            graph_version_id="g",
            agent_run_id="r",
            profile_schema_version="v2",
            payload={},
            profile_v2={"historical": True},
        )
        session.add_all([graph, run, task, profile, draft])
        session.commit()
        yield session, graph, run, task, profile, draft
    engine.dispose()


def _v3_retry_args(candidate):
    from types import SimpleNamespace

    from moonlightbox.world.person_world.contracts.profile_v3 import PersonWorldProfileV3

    session, graph, run, task, profile, draft = candidate
    payload = PersonWorldProfileV3(subject_participant_id="t").model_dump(mode="json")
    payload["identity"]["summary"] = "已有材料中的人物理解"
    profile.profile_schema_version = draft.profile_schema_version = "v3"
    profile.profile_v3 = draft.profile_v3 = payload
    profile.generation_summary = {"candidate_revision": 1, "failed_sections": [task.section]}
    draft.generation_summary = dict(profile.generation_summary)
    session.commit()
    return dict(
        run=run,
        task=task,
        graph=graph,
        profile=profile,
        draft=draft,
        target=SimpleNamespace(id="t", name="目标"),
        user=SimpleNamespace(id="u", name="用户"),
        sidecar=SimpleNamespace(),
        resume_key="job:retry-test",
        settings=Settings(
            person_world_section_model_timeout_seconds=31,
            person_world_section_tool_timeout_seconds=7,
        ),
    )


@pytest.mark.parametrize("fail", [False, True])
def test_v3_retry_executes_native_agent_and_failed_retry_preserves_candidate(
    candidate, monkeypatch, fail
):
    from langchain_core.messages import AIMessage
    from moonlightbox.agent_runtime.controller import AgentLoopController
    from moonlightbox.world.person_world.contracts.profile_v3 import SECTION_RESULT_MODELS
    from moonlightbox.world.person_world.section_retry_v3 import retry_v3

    args = _v3_retry_args(candidate)
    session, _, _, task, profile, draft = candidate
    original = AgentLoopController.run
    observed = []

    def inspect_policy(self, **kwargs):
        policy = kwargs["spec"]
        assert policy.resilience.request_timeout_seconds == 31
        assert {
            t.contract.timeout_seconds for t in policy.tools if t.tool.name == "search_world"
        } == {7}
        return original(self, **kwargs)

    monkeypatch.setattr(AgentLoopController, "run", inspect_policy)

    class Model:
        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            observed.append(True)
            if fail:
                raise RuntimeError("模拟供应商失败")
            result = SECTION_RESULT_MODELS["identity"](
                **{**profile.profile_v3["identity"], "summary": "重试后的新理解"},
                overview="新的整体理解",
            ).model_dump(mode="json")

            # 模拟供应商遵循当前模型可见接口；这些标识由后端绑定，不再由模型提交。
            def model_input(value):
                if isinstance(value, dict):
                    return {
                        key: model_input(child)
                        for key, child in value.items()
                            if key not in {"model", "schema_version"}
                            and (key != "dimension_id" or "model" in value)
                    }
                if isinstance(value, list):
                    return [model_input(child) for child in value]
                return value

            result = model_input(result)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "submit",
                        "name": "submit_section",
                        "args": {"result": result},
                    }
                ],
            )

    class Compiler:
        def create_agent_chat_model(self):
            return Model()

    if fail:
        with pytest.raises(RuntimeError, match="栏目重试未完成"):
            retry_v3(JobService(session), compiler=Compiler(), **args)
        assert profile.generation_summary["candidate_revision"] == 1
        assert profile.profile_v3["identity"]["summary"] == "已有材料中的人物理解"
        assert task.status == "failed"
    else:
        from moonlightbox.world.person_world.coordinator_v3 import PersonWorldCoordinatorV3

        original_batch = PersonWorldCoordinatorV3._batch

        async def interrupt_after_batch(self, sections, phase):
            await original_batch(self, sections, phase)
            raise RuntimeError("模拟已接受栏目、尚未提交候选时退出")

        monkeypatch.setattr(PersonWorldCoordinatorV3, "_batch", interrupt_after_batch)
        with pytest.raises(RuntimeError, match="尚未提交候选"):
            retry_v3(JobService(session), compiler=Compiler(), **args)
        assert profile.generation_summary["candidate_revision"] == 1
        monkeypatch.setattr(PersonWorldCoordinatorV3, "_batch", original_batch)
        retry_v3(JobService(session), compiler=Compiler(), **args)
        assert profile.generation_summary["candidate_revision"] == 2
        assert profile.profile_v3["identity"]["summary"] == "重试后的新理解"
        assert draft.profile_v3 == profile.profile_v3
        assert task.status == "completed"
    assert observed == [True]
    if fail:
        fail = False
        retry_v3(JobService(session), compiler=Compiler(), **args)
        assert observed == [True, True]
        assert profile.generation_summary["candidate_revision"] == 2
    calls = len(observed)
    retry_v3(JobService(session), compiler=Compiler(), **args)
    assert len(observed) == calls
    assert profile.generation_summary["candidate_revision"] == 2


def test_life_context_retry_awaits_enrichment(candidate, monkeypatch):
    import asyncio

    from moonlightbox.world.person_world.coordinator_v3 import PersonWorldCoordinatorV3
    from moonlightbox.world.person_world.section_retry_v3 import retry_v3

    candidate[3].section = "life_context"
    args = _v3_retry_args(candidate)
    phases = []

    async def batch(self, sections, phase):
        await asyncio.sleep(0)
        phases.append(phase)
        self.errors.pop("life_context")

    async def enrich(self, state):
        await asyncio.sleep(0)
        phases.append("enrichment")

    monkeypatch.setattr(PersonWorldCoordinatorV3, "_batch", batch)
    monkeypatch.setattr(PersonWorldCoordinatorV3, "_enrich", enrich)
    retry_v3(JobService(candidate[0]), compiler=None, **args)
    assert phases == ["retry", "enrichment"]
    assert candidate[4].generation_summary["candidate_revision"] == 2


def test_old_retry_rejected_without_enqueue_or_mutation(candidate):
    session, _, run, task, profile, _ = candidate
    with pytest.raises(SectionRetryStateError, match="重新生成 v3"):
        enqueue_section_retry_job(
            JobService(session), settings=Settings(), run=run, task=task, idempotency_key="old"
        )
    assert task.status == "cloud_error"
    assert profile.profile_v2 == {"historical": True}
    assert session.scalar(select(Job)) is None


def test_already_queued_old_job_rejected_before_model_call(candidate):
    session, graph, run, task, _, _ = candidate
    task.status = "pending"
    task.result_summary = {"retry_attempt": 1}
    session.commit()
    job = Job(
        worker_token="lease",
        payload={
            "agent_run_id": run.id,
            "section_task_id": task.id,
            "section": task.section,
            "retry_attempt": 1,
            "graph_version_id": graph.id,
            "graph_source_fingerprint": graph.source_fingerprint,
        },
    )
    with pytest.raises(JobHandlerError) as error:
        create_section_retry_handler(Settings())(JobService(session), job)
    assert error.value.code == "legacy_profile_retired"
    assert task.status == "pending"


def test_legacy_regeneration_preserves_original_and_queues_separate_candidate(candidate):
    session, graph, _, _, profile, draft = candidate
    job, new = enqueue_profile_recompile_job(JobService(session), settings=Settings(), base=graph)
    assert new.id != graph.id and new.workspace_key != graph.workspace_key
    assert new.parent_version_id == graph.id
    assert new.status == "profile_compilation_queued"
    assert graph.status == "superseded"
    assert profile.profile_v2 == draft.profile_v2 == {"historical": True}
    assert profile.profile_schema_version == "v2"
    assert job.payload["candidate_graph_version_id"] == new.id


def test_v3_retry_remains_idempotent(candidate):
    session, _, run, task, profile, draft = candidate
    profile.profile_schema_version = draft.profile_schema_version = "v3"
    session.commit()
    service = JobService(session)
    first = enqueue_section_retry_job(
        service, settings=Settings(), run=run, task=task, idempotency_key="new"
    )
    second = enqueue_section_retry_job(
        service, settings=Settings(), run=run, task=task, idempotency_key="new"
    )
    assert first.id == second.id
    assert task.result_summary["retry_attempt"] == 1


@pytest.mark.parametrize("base_changed", [False, True])
def test_upgrade_publication_respects_original_active_base(candidate, monkeypatch, base_changed):
    from moonlightbox.world.models import WorldPublication
    from moonlightbox.world.person_world import publication

    session, old_graph, _, _, profile, _ = candidate
    old_graph.parent_version_id = "original-active"
    old_graph.status = "superseded"
    new_graph = WorldGraphVersion(
        id="new",
        project_id="p",
        trigger_import_id="i",
        workspace_key="new",
        parent_version_id=old_graph.id,
        status="awaiting_profile_review",
        source_fingerprint="s" * 64,
        config_fingerprint="n" * 64,
        source_import_ids=["i"],
        compiler_version="v3",
    )
    values = {
        column.name: getattr(profile, column.name)
        for column in PersonWorldProfile.__table__.columns
        if column.name not in {"id", "graph_version_id", "agent_run_id", "created_at"}
    }
    new_profile = PersonWorldProfile(
        id="new-profile",
        graph_version_id="new",
        **values,
    )
    new_profile.profile_schema_version = "v3"
    session.add_all([new_graph, new_profile])
    session.flush()
    current = WorldPublication(
        project_id="p",
        graph_version_id="changed" if base_changed else "original-active",
        profile_id="profile",
    )
    monkeypatch.setattr(publication, "active_publication", lambda *_: current)
    # 此处只检查退役父版本与发布基线关系；正式序列化由已有 v3 发布集成测试覆盖。
    monkeypatch.setattr(publication, "_publish_pair", lambda *args, **kwargs: current)
    if base_changed:
        with pytest.raises(publication.WorldPublicationStaleBaseError):
            publication.approve_initial_profile(
                session, project_id="p", graph_version_id="new", profile_id="new-profile"
            )
    else:
        assert (
            publication.approve_initial_profile(
                session, project_id="p", graph_version_id="new", profile_id="new-profile"
            )
            is current
        )
