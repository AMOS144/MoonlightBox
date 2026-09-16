"""候选 v3 的栏目重试，复用相同快照/补全协议，不回落到 v2 原子事实。"""

import asyncio
from copy import deepcopy

from sqlalchemy import func, select, update

from moonlightbox.agent_runtime.policy import section_policy
from moonlightbox.world.models import (
    PersonWorldProfile,
    PersonWorldProfileDraft,
    PersonWorldSectionTask,
)

from .context_module_snapshots import ModuleSnapshot
from .contracts.profile_v3 import PersonWorldProfileV3, described_count
from .contracts.world_dimensions import SECTION_DIMENSIONS
from .coordinator_v3 import PersonWorldCoordinatorV3
from .investigation.report import PersonWorldInvestigationReport, SectionInvestigationStatus


def retry_v3(service, **kwargs):
    """同步 Job 边界只启动一次事件循环，调查与补全都必须等待完成。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aretry_v3(service, **kwargs))
    raise RuntimeError("已有事件循环，请使用 await aretry_v3()")


async def aretry_v3(
    service,
    *,
    run,
    task,
    graph,
    profile,
    draft,
    target,
    user,
    compiler,
    sidecar,
    settings,
    resume_key=None,
):
    from .node_scope import inherited_node_scope

    node_scope = inherited_node_scope(service.session, graph, run.state, draft.generation_summary)
    if run.mode == "node_compile" and node_scope is None:
        raise ValueError("节点重试缺少冻结范围")
    generation = profile.generation_summary.get("candidate_revision", 1)
    namespace = resume_key or f"retry:{task.section}:{generation}"
    scopes = deepcopy(run.state.get("retry_scopes", {}))
    saved_scope = scopes.get(namespace)
    if saved_scope:
        base_generation = saved_scope["candidate_revision"]
        if saved_scope.get("committed") and generation == base_generation + 1:
            return  # 候选已提交但 Job 尚未确认完成时重启，不重复递增版本。
        if generation != base_generation:
            raise ValueError("候选版本已更新，不能恢复旧版本的栏目重试")
    else:
        scopes[namespace] = {"candidate_revision": generation, "committed": False}
        run.state = {**run.state, "retry_scopes": scopes}
        service.session.commit()
    coordinator = PersonWorldCoordinatorV3(
        session=service.session,
        graph=graph,
        lightrag=sidecar,
        compiler=compiler,
        subject_name=target.name,
        user_name=user.name,
        target_participant_id=target.id,
        user_participant_id=user.id,
        section_concurrency=settings.person_world_section_concurrency,
        section_runtime_policy=section_policy(settings),
        node_scope=node_scope,
        timezone=node_scope.boundary["timezone"] if node_scope else "Asia/Shanghai",
    )
    coordinator.baseline = deepcopy(profile.profile_v3)
    coordinator.module_snapshot = ModuleSnapshot.create(
        graph.project_id, profile.profile_v3["life_context"]["context_modules"]
    )
    coordinator.run_row = run
    # 每个显式重试任务有自己的作用域；同一 Job 重启则复用原来的栏目存档。
    coordinator.checkpoint_namespace = namespace
    coordinator._v3_run_id = run.id
    coordinator.tasks = {
        item.section: item
        for item in service.session.scalars(
            select(PersonWorldSectionTask).where(PersonWorldSectionTask.agent_run_id == run.id)
        )
    }
    coordinator.results = {key: deepcopy(profile.profile_v3[key]) for key in SECTION_DIMENSIONS}
    coordinator.overview = profile.profile_v3["overview"]
    coordinator.executions = {}
    coordinator.errors = {
        key: "previous_failure" for key in profile.generation_summary.get("failed_sections", [])
    }
    coordinator.dependencies = list(profile.generation_summary.get("dependencies", []))
    coordinator.attempts = list(profile.generation_summary.get("attempts", []))
    if any(key.startswith(namespace + ":") for key in run.state.get("batches", {})):
        coordinator._restore_results(run.state)
        # 显式重试 Job 恢复时，仅重新打开失败批次，已接受栏目仍由 accepted 清单跳过。
        if task.section in coordinator.errors:
            batches = deepcopy(run.state["batches"])
            for key, batch in batches.items():
                if key.startswith(namespace + ":"):
                    batch["completed"] = False
            run.state = {**run.state, "batches": batches}
    await coordinator._batch([task.section], "retry")
    if task.section in coordinator.errors:
        # _batch 会保存栏目失败以便恢复，但失败不等于生成了新版候选。
        raise RuntimeError(f"栏目重试未完成：{coordinator.errors[task.section]}")
    if task.section == "life_context" and task.section not in coordinator.errors:
        await coordinator._enrich({})
    service.session.refresh(profile)
    service.session.refresh(graph)
    if (
        graph.status
        not in (
            {"ready", "superseded", "awaiting_profile_review"}
            if node_scope
            else {"awaiting_profile_review"}
        )
        or draft.status != "awaiting_review"
        or profile.generation_summary.get("candidate_revision", 1) != generation
    ):
        raise ValueError("候选版本已更新，旧栏目重试结果不会覆盖新版")
    payload = PersonWorldProfileV3(
        subject_participant_id=profile.profile_v3["subject_participant_id"],
        overview=coordinator.overview,
        **coordinator.results,
    ).model_dump(mode="json")
    report = PersonWorldInvestigationReport(
        section_statuses=[
            SectionInvestigationStatus(
                section=key,
                state="cloud_error" if key in coordinator.errors else "completed",
                accepted_fact_count=described_count(payload[key]),
                error_code=coordinator.errors.get(key),
            )
            for key in SECTION_DIMENSIONS
        ]
    ).model_dump(mode="json")
    summary = {
        **profile.generation_summary,
        "candidate_revision": generation + 1,
        "failed_sections": list(coordinator.errors),
        "dependencies": coordinator.dependencies,
        "attempts": coordinator.attempts,
        "module_snapshot": coordinator.module_snapshot.to_dict(),
    }
    # 最终写入使用数据库 CAS；仅 Python refresh 后比较仍存在并发覆盖窗口。
    model = PersonWorldProfileDraft if profile is draft else PersonWorldProfile
    updated = service.session.execute(
        update(model)
        .where(
            model.id == profile.id,
            *([model.status == "awaiting_review"] if profile is draft else []),
            func.coalesce(model.generation_summary["candidate_revision"].as_integer(), 1)
            == generation,
        )
        .values(profile_v3=payload, generation_summary=summary, investigation_report=report),
        execution_options={"synchronize_session": False},
    )
    if updated.rowcount != 1:
        service.session.rollback()
        raise ValueError("候选版本已更新，旧栏目重试结果不会覆盖新版")
    profile.profile_v3 = draft.profile_v3 = payload
    profile.generation_summary = draft.generation_summary = summary
    profile.investigation_report = draft.investigation_report = report
    if profile is not draft:
        profile.source_message_ids = list(
            dict.fromkeys(
                [
                    *profile.source_message_ids,
                    *[
                        item.message_id
                        for execution in coordinator.executions.values()
                        for item in execution.evidence
                    ],
                ]
            )
        )
    scopes = deepcopy(run.state.get("retry_scopes", {}))
    scopes[namespace] = {"candidate_revision": generation, "committed": True}
    run.state = {**run.state, "retry_scopes": scopes}
    service.session.commit()
