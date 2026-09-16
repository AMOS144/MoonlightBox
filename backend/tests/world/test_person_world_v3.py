"""v3 主链路冒烟：类型、原文可见性、快照、并行装配、纠正版本与 Runtime。"""

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage
from moonlightbox.db import Base
from moonlightbox.imports.models import ImportSource, Participant
from moonlightbox.projects.models import Project
from moonlightbox.world.models import PersonWorldProfileDraft, WorldGraphVersion
from moonlightbox.world.person_world.context_module_snapshots import (
    ModuleReadTracker,
    ModuleSnapshot,
    accept_modules,
    affected_sections,
    assign_entry_ids,
    validate_references,
)
from moonlightbox.world.person_world.contracts.context_modules import MODULE_MODELS
from moonlightbox.world.person_world.contracts.profile_v3 import (
    SECTION_MODELS,
    SECTION_RESULT_MODELS,
    PersonWorldProfileV3,
)
from moonlightbox.world.person_world.contracts.world_dimensions import SECTION_DIMENSIONS
from moonlightbox.world.person_world.coordinator_v3 import PersonWorldCoordinatorV3
from moonlightbox.world.person_world.investigation_artifacts import InvestigationArtifactStore
from moonlightbox.world.person_world.review.profile_v3 import validate_selection
from moonlightbox.world.person_world.tools.context_modules import build_context_module_tools
from moonlightbox.world.person_world.tools.evidence_page import (
    project_native_messages,
    project_native_search,
)
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


def module(kind="employment", title="主职工作"):
    return MODULE_MODELS[kind](
        title=title, summary="工作安排严格，但本人希望灵活一些。", status="current"
    ).model_dump(mode="json")


def section_result(section, **kwargs):
    kwargs = {**SECTION_MODELS[section]().model_dump(mode="json"), **kwargs}
    if section == "life_context":
        kwargs.setdefault("context_modules", [])
        kwargs.setdefault(
            "module_assessment",
            {
                "status": "identified" if kwargs["context_modules"] else "not_identified",
                "explanation": "烟测情境选择",
            },
        )
    return SECTION_RESULT_MODELS[section](**kwargs)


def test_ten_typed_modules_and_inferred_psychology_without_quotes():
    profile = PersonWorldProfileV3(subject_participant_id="target")
    entry = profile.identity.big_five_bfi2.dimensions[0]
    entry.status, entry.value, entry.description = (
        "described",
        "mixed",
        "轻松聊天时比较愿意表达，在工作安排上倾向保留自己的空间。",
    )
    profile.identity.mbti.candidates = ["ENxP"]
    assert not entry.reference_message_ids
    assert set(profile.model_dump()) == {
        "schema_version",
        "subject_participant_id",
        "overview",
        *SECTION_DIMENSIONS,
    }
    for kind in MODULE_MODELS:
        result = MODULE_MODELS[kind].model_validate(module(kind))
        assert result.kind == kind
    with pytest.raises(ValueError):
        MODULE_MODELS["education"].model_validate(module())


def test_immutable_snapshots_revision_and_dependency_scope():
    modules = accept_modules([module(), module(title="兼职"), module("education", "备考")])
    snapshot = ModuleSnapshot.create("p", modules)
    tracker = ModuleReadTracker("practices", snapshot)
    tools = build_context_module_tools(tracker, project_id="p")
    key = modules[0]["id"]
    from langchain_core.tools import ToolException
    from moonlightbox.world.person_world.tools.context_module_spec import get_context_module_spec

    assert (
        "details.working_arrangement.schedule"
        in get_context_module_spec("employment")["read_paths"]
    )
    with pytest.raises(ToolException, match="details"):
        tools["read_context_module"].invoke(
            {"module_id": key, "field_paths": ["working_arrangement.schedule"]}
        )
    tools["read_context_module"].invoke(
        {"module_id": key, "field_paths": ["details.working_arrangement.schedule"]}
    )
    renamed = deepcopy(modules)
    renamed[0]["title"] = "主业"
    renamed = accept_modules(renamed, modules)
    assert renamed[0]["id"] == key and renamed[0]["revision"] == 2
    assert not affected_sections(modules, renamed, tracker.dependencies)
    renamed[0]["details"]["working_arrangement"]["schedule"]["value"] = "弹性安排"
    assert affected_sections(modules, renamed, tracker.dependencies) == {"practices"}
    assert snapshot.modules[0]["title"] == "主职工作"
    with pytest.raises(ValueError):
        build_context_module_tools(tracker, project_id="other")
    with pytest.raises(ToolException):
        tools["read_context_module"].invoke({"module_id": "foreign"})
    with pytest.raises(ValueError):
        validate_references(
            {"context_module_refs": [{"module_id": key, "revision": 2, "field_paths": []}]},
            snapshot,
        )


def test_native_agent_reads_context_and_original_messages_not_only_ids():
    artifacts = InvestigationArtifactStore()
    evidence = artifacts.add_evidence_set(
        [{"message_id": "m", "content": "这个笨入早上十点才上班！", "participant_role": "target"}],
        retrieval_id=None,
    )
    visible = project_native_messages(artifacts, {"evidence_set_id": evidence})
    assert visible["messages"][0]["content"] == "这个笨入早上十点才上班！"
    assert (
        project_native_search({"context": "图谱的整体归纳", "references": []})["context"]
        == "图谱的整体归纳"
    )


def test_enrichment_can_understand_modules_without_repeating_source_search():
    from moonlightbox.world.person_world.tools.submit_section import _has_materials

    artifacts = InvestigationArtifactStore()
    snapshot = ModuleSnapshot.create("p", accept_modules([module()]))
    assert _has_materials({}, snapshot, artifacts, [])
    empty = ModuleSnapshot.create("p", availability="pending")
    assert _has_materials(
        {"section_summaries": {"agency": "她希望工作安排灵活"}}, empty, artifacts, []
    )
    assert not _has_materials({}, empty, artifacts, [{"error": "timeout"}])


def test_revision_uses_stable_entry_and_module_identity():
    profile = PersonWorldProfileV3(subject_participant_id="target").model_dump(mode="json")
    assign_entry_ids(profile)
    profile["life_context"]["context_modules"] = accept_modules([module()])
    entry = profile["identity"]["self_evaluations"]
    selected = {"section": "identity", "entry_id": entry["id"], "field_path": "self_evaluations"}
    assert validate_selection(profile, selected)["id"] == entry["id"]
    with pytest.raises(ValueError):
        validate_selection(profile, {**selected, "entry_id": "forged"})


def test_model_schema_exposes_dimension_names_and_value_domains():
    schema = SECTION_RESULT_MODELS["identity"].model_json_schema()
    dimensions = schema["$defs"]["BigFiveDimension"]["properties"]
    assert "moderate" in dimensions["value"]["enum"]
    assert "negative_emotionality" in dimensions["dimension_id"]["enum"]
    assert schema["$defs"]["MBTIEIDimension"]["properties"]["value"]["enum"] == [
        "E",
        "I",
        "mixed",
        "unknown",
    ]
    from moonlightbox.events.cloud_client import (
        _contains_unrepresentable_structure,
        _normalize_tagged_unions,
    )

    assert not _contains_unrepresentable_structure(_normalize_tagged_unions(schema))
    modules = SECTION_RESULT_MODELS["life_context"].model_json_schema()
    assert not _contains_unrepresentable_structure(_normalize_tagged_unions(modules))
    assert _contains_unrepresentable_structure(
        _normalize_tagged_unions({"oneOf": [{"type": "string"}, {"type": "string"}]})
    )


def test_submission_cannot_mix_module_id_and_another_kinds_fields():
    from types import SimpleNamespace

    from moonlightbox.agent_runtime.submission import submission_context
    from moonlightbox.world.person_world.tools.submit_section import build_submit_section

    modules = accept_modules([module(), module("education", "学习")])
    snapshot = ModuleSnapshot.create("p", modules)
    artifacts = InvestigationArtifactStore()
    tool = build_submit_section("identity", {}, SimpleNamespace(snapshot=snapshot), artifacts).tool
    from profile_submission_helpers import section_input
    output = section_input(section_result("identity").model_dump(mode="json"))
    refs = [
        {
            "module_id": modules[1]["id"],
            "field_paths": ["details.working_arrangement.schedule"],
        }
    ]
    output["self_evaluations"]["context_module_refs"] = refs
    with submission_context(SimpleNamespace(tool_results=[])):
        rejected = tool.invoke({"result": output})
        assert rejected["status"] == "rejected"
        assert "不存在的模块字段" in rejected["message"]
        refs[0]["module_id"] = modules[0]["id"]
        assert tool.invoke({"result": output})["status"] == "accepted"


def test_v3_union_schema_reaches_actual_provider_transport():
    import httpx
    from moonlightbox.events.cloud_client import NodeAnalysisCloudClient

    for section in ("identity", "life_context"):
        model = SECTION_RESULT_MODELS[section]
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": section_result(section).model_dump_json()},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            compiler = NodeAnalysisCloudClient(
                enabled=True,
                endpoint="https://example.com/v1/chat/completions",
                model="smoke",
                api_key="test-key",
                client=http,
                max_retries=0,
            )
            result = compiler.create_structured_completion(
                system_content="填写栏目", user_content="测试", response_model=model
            )
            assert isinstance(result, model) and len(calls) == 1


@pytest.mark.parametrize("identity_reads_module", [True, False])
def test_work_schedule_revision_merges_only_selected_and_dependent_fields(
    monkeypatch, identity_reads_module
):
    from types import SimpleNamespace

    from moonlightbox.world.person_world.review import profile_v3 as review
    from moonlightbox.world.person_world.section_agent_v3 import SectionExecutionV3

    original = PersonWorldProfileV3(subject_participant_id="t").model_dump(mode="json")
    original["life_context"]["context_modules"] = accept_modules([module()])
    assign_entry_ids(original)
    key = original["life_context"]["context_modules"][0]["id"]
    dependencies = [
        {
            "consuming_section": section,
            "module_id": key,
            "revision": 1,
            "field_paths": ["details.working_arrangement.schedule"],
            "directory": False,
        }
        for section in (("practices", "identity") if identity_reads_module else ("practices",))
    ]
    profile = SimpleNamespace(
        profile_v3=original,
        project_id="p",
        subject_person_id="t",
        generation_summary={"dependencies": dependencies},
    )
    revision = SimpleNamespace(
        id="r",
        input_revision=1,
        status="understanding_ready",
        scope={
            "selected_statements": [
                {
                    "section": "life_context",
                    "module_id": key,
                    "field_path": "details.working_arrangement.schedule",
                }
            ]
        },
        understanding_payload={"corrected_interpretation": "工作时间并不固定，可以弹性安排"},
    )
    calls = []

    def generate(**kwargs):
        section = kwargs["context"]["section"]
        calls.append(section)
        result = section_result(section, **kwargs["context"]["previous_section"])
        if section == "life_context":
            result.context_modules[0].details.working_arrangement.schedule.value = "弹性安排"
            result.context_modules[0].details.working_arrangement.schedule.status = "described"
            # Agent 返回整栏不代表它有权顺带更改用户未选择的字段。
            result.context_modules[0].details.organization.name.value = "不应被采纳的公司名"
        elif section == "practices":
            result.summary = "工作日安排可以灵活调整，不能据此推断固定出门时刻。"
        return SectionExecutionV3(
            section, result, "completed", None, f"trace-{section}", (), (), []
        )

    monkeypatch.setattr(review, "run_section_v3", generate)
    monkeypatch.setattr(
        review,
        "_build_section_tools",
        lambda *args, **kwargs: {
            name: None
            for section in SECTION_DIMENSIONS
            for name in review.load_section_prompt_v3(section).tool_names
        },
    )
    session = SimpleNamespace(scalars=lambda *args: [], refresh=lambda *args: None)
    patch = review.build_merged_preview(
        SimpleNamespace(session=session, compiler=None, lightrag=None),
        revision,
        profile,
        SimpleNamespace(id="g"),
    )
    assert calls == ["life_context", "practices", "identity"]
    changed = {item["section"]: item["value"] for item in patch}
    updated = changed["life_context"]["context_modules"][0]
    assert updated["id"] == key and updated["revision"] == 2
    assert updated["details"]["working_arrangement"]["schedule"]["basis"] == "user_corrected"
    assert updated["basis"] == "inferred"  # 只确认 schedule，不把整个工作模块标成人工确认
    assert updated["details"]["organization"]["name"]["value"] is None
    assert "practices" in changed and "identity" not in changed
    assert profile.profile_v3 == original


class NativeModel:
    def bind_tools(self, tools, **kwargs):
        self.tool_names = {tool.name for tool in tools}
        assert {"read_context_module", "read_evidence_page", "search_world"} <= self.tool_names
        return self

    def invoke(self, messages):
        if not any(item.type == "tool" for item in messages):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_world",
                        "args": {
                            "question": "目标人物在不同生活情境中怎样表达与行动？",
                        },
                        "id": "search",
                    }
                ],
            )
        import json

        section = json.loads(next(item.content for item in messages if item.type == "human"))[
            "section"
        ]
        result = section_result(section, summary="测试模型的栏目理解")
        if section == "life_context":
            result.context_modules = [MODULE_MODELS["employment"].model_validate(module())]
            result.module_assessment.status = "identified"
        if section == "identity":
            result.overview = "测试模型写出的整体画像，不由后端模板生成。"

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_section",
                    "id": "submit",
                    "args": {"result": __import__("profile_submission_helpers").section_input(result.model_dump(mode="json"))},
                }
            ],
        )


def test_confirmed_v3_preview_is_queued_without_model_call(monkeypatch):
    """HTTP 确认只冻结输入并入队，不能阻塞等待跨栏目推理或提前写图。"""
    from types import SimpleNamespace

    from moonlightbox.world.models import PersonWorldProfile
    from moonlightbox.world.person_world.review import service as review_module
    from moonlightbox.world.person_world.review.profile_preview_jobs import enqueue_profile_preview

    revision = SimpleNamespace(
        id="r",
        project_id="p",
        base_graph_version_id="g",
        base_profile_id="profile",
        status="understanding_ready",
        understanding_revision=2,
        session_revision=3,
        understanding_payload_hash="hash",
        understanding_payload={},
        input_revision=4,
        scope={},
    )
    profile = SimpleNamespace(profile_schema_version="v3")
    session = SimpleNamespace(
        get=lambda model, key: profile if model is PersonWorldProfile else object()
    )
    service = review_module.PersonWorldReviewService(session, compiler=object(), lightrag=object())
    monkeypatch.setattr(service, "_ensure_base_is_current", lambda revision: None)
    monkeypatch.setattr(review_module, "_source_message_context", lambda *args: [])
    assert (
        service.confirm_understanding(
            revision,
            understanding_revision=2,
            understanding_payload_hash="hash",
            expected_session_revision=3,
            defer_v3=True,
        )
        is None
    )
    assert revision.status == "profile_compiling" and revision.session_revision == 4
    assert revision.input_revision == 4
    recorded = {}

    def enqueue(kind, payload, **kwargs):
        recorded.update(kind=kind, payload=payload, **kwargs)
        return SimpleNamespace(id="job")

    enqueue_profile_preview(SimpleNamespace(enqueue_unique=enqueue), revision=revision)
    assert recorded["commit"] is False  # 状态转移与入队在路由同一事务提交
    assert recorded["payload"]["understanding_hash"] == "hash"
    assert recorded["payload"]["input_revision"] == 4


def test_persona_reads_frozen_profile_without_mutating_origin():
    from moonlightbox.runtime_v1.persona_context import PersonaContextAssembler

    profile = PersonWorldProfileV3(subject_participant_id="target").model_dump(mode="json")
    profile["profile_schema_version"] = "v3"
    historical = module()
    historical["status"] = "past"
    profile["life_context"]["context_modules"] = [module(), historical]
    original = deepcopy(profile)
    view = {
        "virtual_now": "2026-09-11T12:00:00+08:00",
        "branch": {},
        "origin": {"person_world_profile": profile},
        "current": {},
    }
    result = PersonaContextAssembler().assemble(view)
    # 表达层只接收身份、关系和 Director 选出的相关材料，不复制完整生活画像。
    assert "life_context" not in result["person_world_profile"]
    assert result["person_world_profile"]["identity"] == profile["identity"]
    result["person_world_profile"]["identity"]["test_mutation"] = True
    assert profile == original


class LightRAG:
    def query(self, *args, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(context="她面对严格工作安排会表达希望保持灵活。", references=[])


class Compiler:
    def create_agent_chat_model(self):
        return NativeModel()

    def create_structured_completion(self, **kwargs):
        raise AssertionError("栏目 Agent 不得再调用旧的最终结构化编译")


@pytest.mark.parametrize("interrupt_after_drafts", [False, True])
def test_v3_full_workflow_persists_candidate_and_never_publishes(
    tmp_path, monkeypatch, interrupt_after_drafts
):
    engine = create_engine(f"sqlite:///{tmp_path}/v3.db")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        project = Project(id="p", name="测试")
        source = ImportSource(
            id="i",
            project_id="p",
            preview_id="preview",
            source_path="test.csv",
            message_count=0,
            confirmed_at=datetime.now(UTC),
        )
        graph = WorldGraphVersion(
            id="g",
            project_id="p",
            trigger_import_id="i",
            workspace_key="test",
            status="compiling_profile",
            source_fingerprint="s" * 64,
            config_fingerprint="c" * 64,
            source_import_ids=["i"],
            compiler_version="v3",
        )
        session.add_all(
            [
                project,
                source,
                graph,
                Participant(id="t", project_id="p", name="目标", role="target"),
                Participant(id="u", project_id="p", name="用户", role="self"),
            ]
        )
        session.commit()
        options = dict(
            session=session,
            graph=graph,
            lightrag=LightRAG(),
            compiler=Compiler(),
            subject_name="目标",
            user_name="用户",
        )
        if interrupt_after_drafts:
            original = PersonWorldCoordinatorV3._drafts

            async def interrupt(self, state):
                await original(self, state)
                raise SystemExit("模拟栏目草稿完成后进程退出")

            monkeypatch.setattr(PersonWorldCoordinatorV3, "_drafts", interrupt)
            with pytest.raises(SystemExit):
                PersonWorldCoordinatorV3(**options).run(resume_key="job:smoke")
            monkeypatch.setattr(PersonWorldCoordinatorV3, "_drafts", original)
        result = PersonWorldCoordinatorV3(**options).run(resume_key="job:smoke")
        assert result.profile_v3.overview
        assert len(result.profile_v3.life_context.context_modules) == 1
        assert not result.generation_summary["failed_sections"]
        draft = session.scalar(select(PersonWorldProfileDraft))
        assert draft.profile_schema_version == "v3" and draft.status == "awaiting_review"
        assert graph.status == "compiling_profile"
        attempts = result.generation_summary["attempts"]
        assert len([item for item in attempts if item["phase"] == "draft"]) == 7
        assert len([item for item in attempts if item["phase"] == "enrichment"]) == 3

        def unexpected(*args, **kwargs):
            raise AssertionError("已完成的栏目不应重新启动")

        monkeypatch.setattr(PersonWorldCoordinatorV3, "_worker", unexpected)
        replay = PersonWorldCoordinatorV3(**options).run(resume_key="job:smoke")
        assert replay.run_id == result.run_id
        assert replay.profile_v3 == result.profile_v3
