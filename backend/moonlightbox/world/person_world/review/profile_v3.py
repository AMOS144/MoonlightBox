"""v3 纠正候选：先更新生活模块，再合并相关 Agent 的重理解，一次批准完整结果。"""

from copy import deepcopy

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.policy import SectionAgentRuntimePolicy
from moonlightbox.imports.models import Participant
from moonlightbox.world.models import PersonWorldAgentRun, PersonWorldProfileDraft

from ..context_module_snapshots import (
    ModuleReadTracker,
    ModuleSnapshot,
    accept_modules,
    affected_sections,
    assign_entry_ids,
    bind_owned_module_references,
    validate_references,
)
from ..contracts.profile_v3 import SECTION_MODELS, PersonWorldProfileV3
from ..contracts.world_dimensions import SECTION_DIMENSIONS, SECTION_LABELS
from ..coordinator_v3 import PersonWorldV3AgentResult, empty_legacy_projection
from ..investigation.report import PersonWorldInvestigationReport
from ..investigation_artifacts import InvestigationArtifactStore
from ..prompt_loader import load_section_prompt_v3
from ..section_agent_v3 import run_section_v3
from ..tools.context_modules import build_context_module_tools
from ..tools.section_tools import build_section_tools as _build_section_tools


def value_at(value, path):
    """字段路径用稳定 module/entry ID 或 dimension_id，不以列表下标或文案定位。"""
    current = value
    for part in filter(None, path.split(".")):
        if isinstance(current, list):
            matches = [
                item
                for item in current
                if isinstance(item, dict) and part in {item.get("id"), item.get("dimension_id")}
            ]
            if len(matches) != 1:
                raise ValueError("所选条目不存在或身份不唯一")
            current = matches[0]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError("所选画像字段不存在")
    return current


def validate_selection(profile, selection):
    section = selection.section if hasattr(selection, "section") else selection["section"]
    item = selection.model_dump() if hasattr(selection, "model_dump") else selection
    if section not in SECTION_DIMENSIONS:
        raise ValueError("v3 选择必须属于七个栏目")
    base = profile[section]
    if item.get("module_id"):
        if section != "life_context":
            raise ValueError("模块只能在现实生活栏目修改")
        base = value_at(base["context_modules"], item["module_id"])
    target = value_at(base, item.get("field_path") or "")
    if item.get("entry_id") and (
        not isinstance(target, dict) or target.get("id") != item["entry_id"]
    ):
        raise ValueError("所选条目 ID 与当前字段不匹配")
    return deepcopy(target)


def revision_base_is_current(session, revision, publication):
    """已发布基线或其尚未发布候选均可纠正，但内容 hash/发布指针必须仍一致。"""
    from moonlightbox.world.models import PersonWorldProfile, WorldGraphVersion

    from .context import canonical_context_hash

    profile = session.get(PersonWorldProfile, revision.base_profile_id)
    graph = session.get(WorldGraphVersion, revision.base_graph_version_id)
    if (
        profile is None
        or graph is None
        or profile.graph_version_id != graph.id
        or graph.project_id != revision.project_id
    ):
        return False
    if canonical_context_hash(profile.profile_v3) != revision.scope.get("base_profile_hash"):
        return False
    if profile.node_boundary_hash:
        # 待审核节点版本绑定创建时的发布指针，其他节点发布不使它失效。
        return (publication is not None and publication.profile_id == profile.id) or (
            publication.id if publication else None
        ) == profile.generation_summary.get("base_node_publication_id")
    if publication is None:
        return graph.status in {"ready", "awaiting_profile_review"}
    return (publication.graph_version_id == graph.id and publication.profile_id == profile.id) or (
        graph.status == "awaiting_profile_review"
        and graph.parent_version_id == publication.graph_version_id
    )


def replace_path(root, path, value):
    if not path:
        return deepcopy(value)
    parts = path.split(".")
    parent = value_at(root, ".".join(parts[:-1]))
    last = parts[-1]
    if isinstance(parent, list):
        old = value_at(parent, last)
        parent[parent.index(old)] = deepcopy(value)
    else:
        if last not in parent:
            raise ValueError("不能新增不属于 schema 的字段")
        parent[last] = deepcopy(value)
    return root


def mark_corrected_changes(value, previous):
    """只标记用户直接确认范围内实际变动的字段，关联推断仍保留 inferred。"""
    if isinstance(value, dict):
        old = previous if isinstance(previous, dict) else {}
        if "basis" in value and value != old:
            value["basis"] = "user_corrected"
        for key, child in value.items():
            mark_corrected_changes(child, old.get(key))
    elif isinstance(value, list):
        old_items = previous if isinstance(previous, list) else []
        for child in value:
            old = next(
                (
                    item
                    for item in old_items
                    if isinstance(item, dict)
                    and isinstance(child, dict)
                    and (item.get("id") or item.get("dimension_id"))
                    == (child.get("id") or child.get("dimension_id"))
                ),
                None,
            )
            mark_corrected_changes(child, old)


def build_merged_preview(service, revision, profile, graph):
    """仅生成可审核草稿。模型失败或旧 input_revision 绝不产生可批准的半成品。"""
    from ..node_scope import inherited_node_scope

    node_scope = inherited_node_scope(
        service.session, graph, revision.scope, profile.generation_summary
    )
    original = deepcopy(profile.profile_v3)
    candidate = deepcopy(original)
    input_revision = revision.input_revision

    def current_input():
        guard = getattr(service, "profile_preview_current_input", None)
        if guard is not None:
            return guard()
        # 单独只读 Session，避免 ORM identity map 把用户的新输入缓存成旧版本。
        if not hasattr(service.session, "get_bind"):
            return revision.input_revision  # 注入的无数据库烟测 Session
        from moonlightbox.world.models import PersonWorldRevisionSession

        with Session(bind=service.session.get_bind()) as check:
            row = check.execute(
                select(
                    PersonWorldRevisionSession.input_revision, PersonWorldRevisionSession.status
                ).where(PersonWorldRevisionSession.id == revision.id)
            ).first()
            return (
                row[0] if row and row[1] in {"understanding_ready", "profile_compiling"} else None
            )

    selected = revision.scope.get("selected_statements", [])
    for item in selected:
        validate_selection(original, item)
    sections = list(dict.fromkeys(item["section"] for item in selected))
    if not sections:
        sections = list(
            dict.fromkeys(revision.understanding_payload.get("affected_profile_sections", []))
        )
        if not sections or set(sections) - set(SECTION_DIMENSIONS):
            raise ValueError("尚未明确修改哪些栏目，请先澄清范围并重新确认理解")
    snapshot = ModuleSnapshot.create(
        profile.project_id, original["life_context"]["context_modules"]
    )
    dependencies = list(profile.generation_summary.get("dependencies", []))
    attempts = []
    participants = list(
        service.session.scalars(
            select(Participant).where(Participant.project_id == profile.project_id)
        )
    )
    target = next((item for item in participants if item.id == profile.subject_person_id), None)
    user = next((item for item in participants if item.role == "self"), None)

    def generate(section, *, dependent=False):
        nonlocal dependencies
        if current_input() != input_revision:
            raise ValueError("用户已更新输入或取消生成，旧候选停止")
        progress = getattr(service, "profile_preview_progress", None)
        if callable(progress):
            progress(section)
        artifacts = InvestigationArtifactStore()
        tracker = ModuleReadTracker(section, snapshot)
        tools = _build_section_tools(
            service.session,
            graph=graph,
            lightrag=service.lightrag,
            top_k=30,
            chunk_top_k=12,
            max_total_tokens=16_000,
            correction_change_set_ids=set(),
            artifacts=artifacts,
            temporal_scope=node_scope.retrieval_scope() if node_scope else None,
            message_periods=node_scope.message_periods() if node_scope else None,
            node_scope=node_scope.envelope() if node_scope else None,
            profile_snapshot=candidate,
        )
        tools.update(build_context_module_tools(tracker, project_id=profile.project_id))
        definition = load_section_prompt_v3(section)
        execution = run_section_v3(
            definition=definition,
            tools={key: tools[key] for key in definition.tool_names},
            compiler=service.compiler,
            context={
                "node_scope": node_scope.model_context() if node_scope else None,
                "section": section,
                "phase": "revision_dependency" if dependent else "revision",
                "target_person": target.name if target else "目标人物",
                "user": user.name if user else "用户",
                "previous_section": candidate[section],
                "section_summaries": {key: candidate[key]["summary"] for key in SECTION_DIMENSIONS},
                "confirmed_correction": revision.understanding_payload,
                "selected_fields": [item for item in selected if item["section"] == section],
                "instruction": (
                    "依据用户确认的理解修订。相关栏目应重新判断，可以保留结论；"
                    "不能因背景变化自动清空人格。完整返回栏目。"
                ),
            },
            tracker=tracker,
            artifacts=artifacts,
            policy=getattr(service, "section_runtime_policy", SectionAgentRuntimePolicy()),
            owner_id=f"{revision.id}:{input_revision}:{section}",
            project_id=profile.project_id,
            graph_id=graph.id,
            target_id=profile.subject_person_id,
            input_revision=input_revision,
            input_revision_resolver=current_input,
        )
        attempts.append(
            {
                "section": section,
                "phoenix_execution_id": execution.execution_id,
                "status": execution.status,
            }
        )
        if execution.result is None or execution.status in {
            "failed",
            "blocked",
            "cancelled",
            "stale",
        }:
            raise ValueError(f"{SECTION_LABELS[section]}重新理解未完成，候选不能批准")
        result = execution.result.model_dump(mode="json")
        result = {
            key: value
            for key, value in result.items()
            if key not in {"overview", "section", "unresolved_questions", "module_assessment"}
        }
        assign_entry_ids(result, candidate[section])
        scope = [item for item in selected if item["section"] == section]
        if scope and not dependent:
            # 只采纳用户选择范围。Agent 返回整栏只是 typed output，不授予跨范围编辑权。
            merged = deepcopy(candidate[section])
            for item in scope:
                module_id, path = item.get("module_id"), item.get("field_path") or ""
                if module_id:
                    old_module = value_at(merged["context_modules"], module_id)
                    replacement = next(
                        (m for m in result["context_modules"] if m.get("id") == module_id), None
                    )
                    if not path:
                        merged["context_modules"].remove(old_module)
                        if replacement:
                            mark_corrected_changes(replacement, old_module)
                            merged["context_modules"].append(replacement)
                        else:
                            # 类型纠正的新实例只有 id=null；已确认删除也可一个不新增。
                            additions = [m for m in result["context_modules"] if not m.get("id")]
                            for new_module in additions:
                                mark_corrected_changes(new_module, {})
                            merged["context_modules"].extend(additions)
                    else:
                        if replacement is None:
                            raise ValueError("字段级纠正不能删除整个模块")
                        selected_value = deepcopy(value_at(replacement, path))
                        mark_corrected_changes(selected_value, value_at(old_module, path))
                        replace_path(old_module, path, selected_value)
                else:
                    selected_value = deepcopy(value_at(result, path))
                    mark_corrected_changes(selected_value, value_at(merged, path))
                    merged = replace_path(merged, path, selected_value)
            result = merged
        if not dependent and not scope:
            mark_corrected_changes(result, candidate[section])
        elif dependent and scope:
            # 同栏既有直接纠正又有背景补全时，仅直接选择的字段标记 user_corrected。
            for item in scope:
                path = item.get("field_path") or ""
                mark_corrected_changes(value_at(result, path), value_at(candidate[section], path))
        if section == "life_context":
            result["context_modules"] = accept_modules(result["context_modules"], snapshot.modules)
            bind_owned_module_references(result)
        else:
            validate_references(result, snapshot, tracker=tracker)
        candidate[section] = SECTION_MODELS[section].model_validate(result).model_dump(mode="json")
        dependencies = [
            item for item in dependencies if item["consuming_section"] != section
        ] + tracker.dependencies
        return execution

    # 先处理模块直接修改，其余直接修改合并到一次消费者重理解，避免同栏重复运行。
    if "life_context" in sections:
        generate("life_context")
        updated = candidate["life_context"]["context_modules"]
        affected = affected_sections(snapshot.modules, updated, dependencies)
        snapshot = ModuleSnapshot.create(profile.project_id, updated)
    else:
        affected = set()
    targets = (set(sections) | affected) - {"life_context"}
    for section in [key for key in SECTION_DIMENSIONS if key in targets and key != "identity"]:
        generate(section, dependent=section in affected)
    # identity 实际读取过所有栏目摘要。摘要变化是输入版本变化，不应只追踪模块工具读取。
    # 仅改模块标题、排序或引用版本时摘要不变，因此不会无意义重跑人格。
    if any(
        candidate[key]["summary"] != original[key]["summary"]
        for key in SECTION_DIMENSIONS
        if key != "identity"
    ):
        targets.add("identity")
        affected.add("identity")
    if "identity" in targets:
        execution = generate("identity", dependent="identity" in affected)
        if execution.result.overview:
            candidate["overview"] = execution.result.overview
    assign_entry_ids(candidate, original)
    from ..context_module_snapshots import rebase_unchanged_references

    rebase_unchanged_references(
        candidate, original["life_context"]["context_modules"], snapshot.modules
    )
    # 模块删除后的过期引用不能偷偷发布；负责 Agent 须在重理解中消除。
    validate_references(candidate, snapshot)
    candidate = PersonWorldProfileV3.model_validate(candidate).model_dump(mode="json")
    service.session.refresh(revision)
    if revision.input_revision != input_revision or revision.status not in {
        "understanding_ready",
        "profile_compiling",
    }:
        raise ValueError("用户理解或范围已经改变，旧补全结果已废弃")
    patch = [
        {
            "operation": "REPLACE_V3_SECTION",
            "section": key,
            "value": candidate[key],
            "current_text": readable(original[key]),
            "replacement_text": readable(candidate[key]),
            "reason": "用户确认的修订及相关背景重新理解",
            "source_message_ids": [],
        }
        for key in (*SECTION_DIMENSIONS, "overview")
        if candidate[key] != original[key]
    ]
    preview_scope = {
        **revision.scope,
        "v3_preview": {
            "input_revision": input_revision,
            "dependencies": dependencies,
            "attempts": attempts,
            "module_snapshot": snapshot.to_dict(),
        },
    }
    if current_input() != input_revision:
        raise ValueError("修订输入或任务租约已经变化，旧候选已废弃")
    if hasattr(service.session, "execute"):
        from moonlightbox.world.models import PersonWorldRevisionSession

        # 比较更新持有写事务直到 ChangeSet 一起提交，关闭校验与写入之间的竞态窗口。
        updated = service.session.execute(
            update(PersonWorldRevisionSession)
            .where(
                PersonWorldRevisionSession.id == revision.id,
                PersonWorldRevisionSession.input_revision == input_revision,
                PersonWorldRevisionSession.status.in_(["understanding_ready", "profile_compiling"]),
                PersonWorldRevisionSession.session_revision == revision.session_revision,
            )
            .values(scope=preview_scope)
            .execution_options(synchronize_session=False)
        )
        if updated.rowcount != 1:
            raise ValueError("修订已被并发更新，旧候选已废弃")
    revision.scope = preview_scope
    return patch


def readable(value):
    """供审核 Diff 展示真实文字；不生成模板人设。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(filter(None, (readable(item) for item in value)))
    if isinstance(value, dict):
        if "description" in value:
            return str(value.get("description") or value.get("value") or "尚不了解")
        return "\n".join(
            filter(
                None,
                (
                    readable(item)
                    for key, item in value.items()
                    if key
                    not in {
                        "id",
                        "revision",
                        "kind",
                        "schema_version",
                        "basis",
                        "status",
                        "reference_message_ids",
                        "context_module_refs",
                        "related_module_ids",
                    }
                ),
            )
        )
    return ""


def create_v3_change_set(service, revision, profile, graph):
    from moonlightbox.world.models import WorldCorrection, WorldGraphChangeSet

    from ..graph_executor import canonical_profile_patch_hash

    patch = build_merged_preview(service, revision, profile, graph)
    number = (
        service.session.scalar(
            select(WorldGraphChangeSet.revision)
            .where(WorldGraphChangeSet.revision_session_id == revision.id)
            .order_by(WorldGraphChangeSet.revision.desc())
        )
        or 0
    ) + 1
    change_set = WorldGraphChangeSet(
        project_id=revision.project_id,
        revision_session_id=revision.id,
        base_graph_version_id=graph.id,
        revision=number,
        status="awaiting_profile_approval",
        profile_patch=patch,
        graph_operations=[],
        affected_entities=[],
        affected_relations=[],
        regression_queries=[],
        canonical_payload_hash=canonical_profile_patch_hash(
            profile_patch=patch, base_graph_version_id=graph.id, revision=number
        ),
    )
    service.session.add(change_set)
    service.session.flush()
    understanding = revision.understanding_payload
    domains = [item["section"] for item in patch if item["section"] in SECTION_DIMENSIONS]
    service.session.add(
        WorldCorrection(
            project_id=revision.project_id,
            revision_session_id=revision.id,
            correction_type="profile",
            target_key="person_world_v3",
            original_interpretation={"text": understanding["wrong_interpretation"]},
            corrected_interpretation={
                "text": understanding["corrected_interpretation"],
                "scope": {
                    "primary_domains": domains,
                    "selected_statements": revision.scope.get("selected_statements", []),
                },
            },
            user_explanation=understanding["summary_for_user"],
            source_message_ids=[],
            status="proposed",
            approved_change_set_id=change_set.id,
        )
    )
    revision.status = "profile_review"
    revision.profile_change_set_id = revision.graph_change_set_id = change_set.id
    revision.session_revision += 1
    service.session.flush()
    return change_set


def approved_v3_result(session, *, revision, change_set, candidate_graph):
    """执行阶段逐字使用已批准的合并画像；不得再次调用模型悄悄改变已批准人格。"""
    from moonlightbox.world.models import PersonWorldProfile

    base = session.get(PersonWorldProfile, revision.base_profile_id)
    if base is None or base.profile_schema_version != "v3":
        return None
    from moonlightbox.world.models import WorldGraphVersion

    from ..node_scope import inherited_node_scope

    node_scope = inherited_node_scope(
        session,
        session.get(WorldGraphVersion, base.graph_version_id),
        revision.scope,
        base.generation_summary,
    )
    if node_scope:
        node_scope = node_scope.for_candidate_graph(session, candidate_graph)
    payload = deepcopy(base.profile_v3)
    for operation in change_set.profile_patch:
        if operation.get("operation") != "REPLACE_V3_SECTION" or operation.get("section") not in {
            *SECTION_DIMENSIONS,
            "overview",
        }:
            raise ValueError("v3 批准内容包含不兼容操作")
        payload[operation["section"]] = deepcopy(operation["value"])
    profile = PersonWorldProfileV3.model_validate(payload)
    preview = revision.scope.get("v3_preview", {})
    if preview.get("input_revision") != revision.input_revision:
        raise ValueError("已批准候选的输入版本过期")
    summary = {
        **base.generation_summary,
        **preview,
        "failed_sections": [
            section
            for section in base.generation_summary.get("failed_sections", [])
            if section not in {item["section"] for item in preview.get("attempts", [])}
        ],
        "profile_schema_version": "v3",
        "approved_change_set_id": change_set.id,
        "node_scope": node_scope.envelope() if node_scope else None,
    }
    run = PersonWorldAgentRun(
        project_id=base.project_id,
        graph_version_id=candidate_graph.id,
        mode="revision",
        prompt_version="person-world-lived-world-v3",
        status="awaiting_review",
        state={"approved_preview": change_set.id, "node_scope": summary.get("node_scope")},
        trace=[],
    )
    session.add(run)
    session.flush()
    legacy = empty_legacy_projection()
    report = PersonWorldInvestigationReport.model_validate(base.investigation_report)
    renewed = {item["section"] for item in preview.get("attempts", [])}
    from ..contracts.profile_v3 import described_count

    for status in report.section_statuses:
        if status.section in renewed:
            status.state = "completed"
            status.error_code = None
            status.unresolved_questions = []
            status.accepted_fact_count = described_count(payload[status.section])
            status.trace_refs = [
                item["phoenix_execution_id"]
                for item in preview["attempts"]
                if item["section"] == status.section
            ]
    report.cloud_and_schema_errors = [
        item for item in report.cloud_and_schema_errors if item.get("section") not in renewed
    ]
    session.add(
        PersonWorldProfileDraft(
            project_id=base.project_id,
            graph_version_id=candidate_graph.id,
            agent_run_id=run.id,
            payload=legacy.model_dump(mode="json"),
            profile_v3=payload,
            profile_schema_version="v3",
            investigation_report=report.model_dump(mode="json"),
            generation_summary=summary,
            claim_ids=[],
        )
    )
    return PersonWorldV3AgentResult(
        run.id, profile, report, legacy, tuple(base.source_message_ids), (), summary
    )
