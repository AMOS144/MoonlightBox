"""v3 工作流：七栏草稿 → 模块固定快照 → 集中补全 → 候选审核。

Coordinator 不写人格、不审批图谱。Send 异步节点隔离同步栏目工作单元，
各自持有 Session；候选结果和快照在协调协程落库，不读取同批次半成品。
"""

import asyncio
import logging
import operator
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.agent_runtime.async_execution import run_sync_owned
from moonlightbox.agent_runtime.cancellation import cancellation_requested
from moonlightbox.agent_runtime.policy import SectionAgentRuntimePolicy
from moonlightbox.agent_runtime.resilience import ExecutionInterrupted
from moonlightbox.world.compiler import AgentCompilerClient, RetrievalManifestItem
from moonlightbox.world.models import (
    PersonWorldAgentRun,
    PersonWorldProfile,
    PersonWorldProfileDraft,
    PersonWorldSectionTask,
    WorldGraphVersion,
    WorldPublication,
)
from moonlightbox.world.schemas import (
    IdentityProfile,
    RelationshipProfile,
    RoutineProfile,
    WorldProfileDraft,
)

from .context_module_snapshots import (
    ModuleReadTracker,
    ModuleSnapshot,
    accept_modules,
    assign_entry_ids,
    bind_owned_module_references,
    validate_references,
)
from .contracts.profile_v3 import SECTION_MODELS, PersonWorldProfileV3, described_count
from .contracts.world_dimensions import SECTION_DIMENSIONS
from .investigation.report import PersonWorldInvestigationReport, SectionInvestigationStatus
from .investigation_artifacts import InvestigationArtifactStore
from .prompt_loader import load_section_prompt_v3
from .section_agent_v3 import run_section_v3
from .tools.context_modules import build_context_module_tools
from .tools.section_tools import build_section_tools as _build_section_tools
from .tools.section_tools import participant_id
from .tools.submit_section import _reference_ids


def _check_cancelled():
    """模型退出后也检查取消，禁止取消中的调查继续合并候选。"""
    if cancellation_requested():
        raise ExecutionInterrupted("cancelled")


@dataclass
class PersonWorldV3AgentResult:
    run_id: str
    profile_v3: object
    investigation_report: PersonWorldInvestigationReport
    draft: WorldProfileDraft
    source_message_ids: tuple[str, ...]
    retrieval_manifest: tuple[RetrievalManifestItem, ...]
    generation_summary: dict


class WorkflowState(TypedDict, total=False):
    stage: str


class SectionBatchState(TypedDict, total=False):
    """每个 Send 只提交自己的完成标记，由 reducer 汇合，禁止覆盖其他分支。"""

    completed_sections: Annotated[list[str], operator.add]


def empty_legacy_projection():
    """旧 API 必填壳，不将 v3 理解伪装成 v1 原话/原子事实。"""
    return WorldProfileDraft(
        identity=IdentityProfile(),
        work_and_education=[],
        places=[],
        social_relationships=[],
        preferences=[],
        recurring_activities=[],
        routine_summary=RoutineProfile(),
        life_phases=[],
        relationship_with_user=RelationshipProfile(),
        important_events=[],
        unresolved_candidates=[],
    )


class PersonWorldCoordinatorV3:
    def __init__(
        self,
        *,
        session: Session,
        graph: WorldGraphVersion,
        lightrag: Any,
        compiler: AgentCompilerClient,
        subject_name: str,
        user_name: str,
        target_participant_id: str | None = None,
        user_participant_id: str | None = None,
        timezone: str = "Asia/Shanghai",
        top_k: int = 30,
        chunk_top_k: int = 12,
        max_total_tokens: int = 16_000,
        section_concurrency: int = 3,
        section_runtime_policy: SectionAgentRuntimePolicy | None = None,
        progress: Callable[[str, int, int], None] | None = None,
        correction_change_set_ids: set[str] | None = None,
        node_scope=None,
    ) -> None:
        # v3 不继承旧 Coordinator，避免旧事实门槛或调度逻辑随父类改动重新渗入。
        self.session, self.graph, self._graph_id = session, graph, graph.id
        self.node_scope = node_scope
        if node_scope is not None and node_scope.graph_id != graph.id:
            raise ValueError("节点编译作用域不属于当前图谱")
        self.lightrag, self.compiler = lightrag, compiler
        self.subject_name, self.user_name, self.timezone = subject_name, user_name, timezone
        if not 1 <= section_concurrency <= len(SECTION_DIMENSIONS):
            raise ValueError("section_concurrency 必须在 1 到 7 之间")
        self.section_concurrency = section_concurrency
        self.section_runtime_policy = section_runtime_policy or SectionAgentRuntimePolicy()
        self.progress = progress
        self.correction_change_set_ids = set(correction_change_set_ids or ())
        self.target_participant_id = target_participant_id or participant_id(
            session,
            project_id=graph.project_id,
            role="target",
            name=subject_name,
        )
        self.user_participant_id = user_participant_id or participant_id(
            session,
            project_id=graph.project_id,
            role="self",
            name=user_name,
        )
        self._tool_options = {
            "top_k": top_k,
            "chunk_top_k": chunk_top_k,
            "max_total_tokens": max_total_tokens,
        }

    def _initialize(self, *, mode, resume_key):
        publication = self.session.scalar(
            select(WorldPublication)
            .where(
                WorldPublication.project_id == self.graph.project_id,
                WorldPublication.status == "active",
                WorldPublication.node_boundary_hash.is_(None),
            )
            .order_by(WorldPublication.published_at.desc())
        )
        published = (
            self.session.get(PersonWorldProfile, publication.profile_id) if publication else None
        )
        # 历史节点不能沿用项目末端画像；同节点的后续重试由当前 run 恢复。
        if self.node_scope is not None:
            published = None
        self.baseline = (
            dict(published.profile_v3 or {})
            if published and published.profile_schema_version == "v3"
            else {}
        )
        self.module_snapshot = ModuleSnapshot.create(
            self.graph.project_id,
            self.baseline.get("life_context", {}).get("context_modules", []),
            availability="ready" if self.baseline else "pending",
            source="published",
        )
        self.run_row = PersonWorldAgentRun(
            project_id=self.graph.project_id,
            graph_version_id=self.graph.id,
            mode=mode,
            prompt_version="person-world-lived-world-v3",
            status="researching",
            state={
                "contract_version": "person-world-profile-v3",
                "base_profile_id": published.id if published else None,
                "snapshots": [self.module_snapshot.to_dict()],
                "resume_key": resume_key,
                "baseline": self.baseline,
                "node_scope": self.node_scope.envelope() if self.node_scope else None,
            },
            trace=[],
        )
        self.session.add(self.run_row)
        self.session.flush()
        self._v3_run_id = self.run_row.id
        self.tasks = {}
        for name in SECTION_DIMENSIONS:
            task = PersonWorldSectionTask(
                agent_run_id=self.run_row.id, section=name, status="pending"
            )
            self.session.add(task)
            self.tasks[name] = task
        self.results, self.executions, self.errors = {}, {}, {}
        self.dependencies, self.attempts = [], []
        self.session.commit()

    def run(self, *, mode="initial_compile", resume_key=None):
        """同步 Worker 入口；异步调用方必须 await arun，不能嵌套事件循环。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(mode=mode, resume_key=resume_key))
        raise RuntimeError("已有事件循环，请使用 await coordinator.arun()")

    async def arun(self, *, mode="initial_compile", resume_key=None):
        """Job 身份固定调查实例；普通重启不创建新的七栏调查。"""
        existing = None
        if resume_key is not None:
            existing = next(
                (
                    row
                    for row in self.session.scalars(
                        select(PersonWorldAgentRun)
                        .where(
                            PersonWorldAgentRun.graph_version_id == self.graph.id,
                            PersonWorldAgentRun.mode == mode,
                        )
                        .order_by(PersonWorldAgentRun.created_at.desc())
                    )
                    if row.state.get("resume_key") == resume_key
                    and row.state.get("node_scope")
                    == (self.node_scope.envelope() if self.node_scope else None)
                ),
                None,
            )
        if existing is None:
            self._initialize(mode=mode, resume_key=resume_key)
        else:
            self._restore_run(existing)

        workflow = StateGraph(WorkflowState)
        workflow.add_node("section_drafts", self._drafts)
        workflow.add_node("module_enrichment", self._enrich)

        # 短事务留在协调协程，不能让 LangGraph 将 ORM Session 交给线程池节点。
        async def assemble_review(state):
            return self._assemble(state)

        workflow.add_node("assemble_review", assemble_review)
        workflow.add_edge(START, "section_drafts")
        workflow.add_edge("section_drafts", "module_enrichment")
        workflow.add_edge("module_enrichment", "assemble_review")
        workflow.add_edge("assemble_review", END)
        try:
            await workflow.compile().ainvoke({"stage": "start"})
        except asyncio.CancelledError:
            self.session.rollback()
            self.run_row.status = "cancelled"
            self.session.commit()
            raise
        except Exception as error:
            self.session.rollback()
            self.run_row.status = (
                "cancelled" if getattr(error, "code", None) == "cancelled" else "failed"
            )
            self.session.commit()
            raise
        return self.output

    def _restore_run(self, run):
        if run.state.get("node_scope") != (self.node_scope.envelope() if self.node_scope else None):
            raise ValueError("不能跨节点恢复栏目调查")
        self.run_row, self._v3_run_id = run, run.id
        state = run.state
        self.baseline = deepcopy(state["baseline"])
        self.module_snapshot = ModuleSnapshot.from_dict(state["snapshots"][0])
        self.tasks = {
            task.section: task
            for task in self.session.scalars(
                select(PersonWorldSectionTask).where(PersonWorldSectionTask.agent_run_id == run.id)
            )
        }
        self._restore_results(state)

    def _restore_results(self, state):
        """普通编译和单栏重试共用已接受结果恢复，不把批次完成误当成无需恢复材料。"""
        from types import SimpleNamespace

        self.results = deepcopy(state.get("section_drafts", {}))
        self.dependencies = deepcopy(state.get("dependencies", []))
        self.attempts = deepcopy(state.get("attempts", []))
        self.errors = dict(state.get("errors", {}))
        self.overview = state.get("overview", "")
        # 原文在 LangGraph 工作存档里；业务状态仅保留来源索引和已有审核清单。
        self.executions = {
            name: SimpleNamespace(
                evidence=tuple(SimpleNamespace(message_id=key) for key in keys),
                retrievals=tuple(self.tasks[name].attempted_queries or []),
            )
            for name, keys in state.get("section_source_ids", {}).items()
        }

    async def _drafts(self, state):
        await self._batch(list(SECTION_DIMENSIONS), "draft")
        return {"stage": "drafted"}

    async def _enrich(self, state):
        life = self.results.get("life_context")
        if life is None or "life_context" in self.errors:
            self.module_snapshot = ModuleSnapshot.create(
                self.graph.project_id, availability="failed"
            )
            self.run_row.state = {**self.run_row.state, "module_availability": "failed"}
            self.session.commit()
            return {"stage": "module_failed"}
        namespace = getattr(self, "checkpoint_namespace", "main")
        enrichment = deepcopy(self.run_row.state.get("enrichment", {}))
        saved_enrichment = enrichment.get(namespace, {})
        saved_snapshot = saved_enrichment.get("snapshot")
        self.module_snapshot = (
            ModuleSnapshot.from_dict(saved_snapshot)
            if saved_snapshot
            else ModuleSnapshot.create(self.graph.project_id, life["context_modules"])
        )
        consumers = {"practices", "agency", "identity"} | {
            item["consuming_section"] for item in self.dependencies
        }
        consumers = set(saved_enrichment.get("consumers", sorted(consumers)))
        enrichment[namespace] = {
            "snapshot": self.module_snapshot.to_dict(),
            "consumers": sorted(consumers),
        }
        snapshots = list(self.run_row.state.get("snapshots", []))
        if not any(item["snapshot_id"] == self.module_snapshot.id for item in snapshots):
            snapshots.append(self.module_snapshot.to_dict())
        self.run_row.state = {
            **self.run_row.state,
            "enrichment": enrichment,
            "snapshots": snapshots,
        }
        self.session.commit()
        # 身份 Agent 最后一次补全同时写 overview；不新增一个核验/概括模型。
        await self._batch(
            [key for key in SECTION_DIMENSIONS if key in consumers - {"life_context", "identity"}],
            "enrichment",
        )
        await self._batch(["identity"], "enrichment")
        return {"stage": "enriched"}

    def _worker(self, section, phase, snapshot, context):
        with Session(bind=self._engine, expire_on_commit=False) as session:
            graph = session.get(WorldGraphVersion, self._graph_id)
            artifacts = InvestigationArtifactStore()
            tracker = ModuleReadTracker(section, snapshot)
            tools = _build_section_tools(
                session,
                graph=graph,
                lightrag=self.lightrag,
                correction_change_set_ids=self.correction_change_set_ids,
                artifacts=artifacts,
                temporal_scope=self.node_scope.retrieval_scope() if self.node_scope else None,
                message_periods=self.node_scope.message_periods() if self.node_scope else None,
                node_scope=self.node_scope.envelope() if self.node_scope else None,
                profile_snapshot=context.get("profile_snapshot"),
                **self._tool_options,
            )
            tools.update(build_context_module_tools(tracker, project_id=graph.project_id))
            definition = load_section_prompt_v3(section)
            from moonlightbox.agent_runtime.persistence import checkpoint_path

            return run_section_v3(
                checkpoint_path=checkpoint_path(session),
                definition=definition,
                tools={key: tools[key] for key in definition.tool_names},
                compiler=self.compiler,
                context=context,
                tracker=tracker,
                artifacts=artifacts,
                policy=self.section_runtime_policy,
                owner_id=(
                    f"{self._v3_run_id}:{getattr(self, 'checkpoint_namespace', 'main')}:"
                    f"{phase}:{section}"
                ),
                project_id=graph.project_id,
                graph_id=graph.id,
                target_id=self.target_participant_id,
            )

    async def _batch(self, sections, phase):
        _check_cancelled()
        if not sections:
            return
        batch_key = f"{getattr(self, 'checkpoint_namespace', 'main')}:{phase}:{','.join(sections)}"
        batches = deepcopy(self.run_row.state.get("batches", {}))
        saved_batch = batches.get(batch_key)
        if saved_batch and saved_batch.get("completed"):
            return
        snapshot = self.module_snapshot
        # 批次开始前冻结完整栏目；后续 _accept 写入不会改变任何兄弟任务的输入。
        profile_snapshot = deepcopy({**self.baseline, **self.results})
        summaries = {key: item.get("summary", "") for key, item in self.results.items()}
        initial_messages = self.node_scope.initial_messages(self.session) if self.node_scope else []
        contexts = {
            section: {
                "section": section,
                "phase": phase,
                "target_person": self.subject_name,
                "user": self.user_name,
                "timezone": self.timezone,
                "module_snapshot": snapshot.envelope(),
                "previous_section": self.results.get(section) or self.baseline.get(section),
                "section_summaries": summaries,
                "profile_snapshot": deepcopy(profile_snapshot),
                **(
                    {
                        "node_scope": self.node_scope.model_context(),
                        "initial_messages": deepcopy(initial_messages),
                    }
                    if self.node_scope
                    else {}
                ),
                "section_work": deepcopy(
                    self.run_row.state.get("section_work", {}).get(section, {})
                ),
                "instruction": "先形成完整栏目理解。"
                if phase == "draft"
                else (
                    "模块已就绪：读取相关背景，重新考虑原稿，可保留正文。"
                    "不要把其他栏目的推断当作额外独立证据。identity 同时更新全档 overview。"
                ),
            }
            for section in sections
        }
        if saved_batch:
            snapshot = ModuleSnapshot.from_dict(saved_batch["snapshot"])
            contexts = saved_batch["contexts"]
            self.module_snapshot = snapshot
        else:
            batches[batch_key] = {
                "snapshot": snapshot.to_dict(),
                "contexts": contexts,
                "accepted": [],
            }
            self.run_row.state = {**self.run_row.state, "batches": batches}
        self._active_batch_key = batch_key
        accepted = set(batches[batch_key]["accepted"])
        sections = [section for section in sections if section not in accepted]
        for section in sections:
            self.tasks[section].status = "researching"
        self.session.commit()
        # 内存 SQLite 不能跨线程共享连接；文件库只读线程无需人为串行云请求。
        engine = self.session.get_bind()
        self._engine = engine
        in_memory = engine.dialect.name == "sqlite" and str(engine.url.database) in {
            "None",
            "",
            ":memory:",
        }

        def dispatch(state):
            return [
                Send(
                    "section_agent",
                    {
                        "section": section,
                        "context": deepcopy(contexts[section]),
                        "snapshot": snapshot.to_dict(),
                    },
                )
                for section in sections
            ] or END

        async def section_agent(task):
            section = task["section"]
            frozen = ModuleSnapshot.from_dict(task["snapshot"])
            _check_cancelled()
            try:
                if in_memory:
                    # sqlite:// 的库绑定当前线程，仅供离线测试串行执行。
                    result = self._worker(section, phase, frozen, task["context"])
                else:
                    result = await run_sync_owned(
                        self._worker, section, phase, frozen, task["context"]
                    )
                # 没有 await 的短事务只在协调协程执行；每栏结束即可保存恢复点。
                _check_cancelled()
                self._accept(section, phase, result)
            except Exception as error:
                self._failed(section, phase, error)
            return {"completed_sections": [section]}

        batch = StateGraph(SectionBatchState)
        batch.add_node("section_agent", section_agent)
        batch.add_conditional_edges(START, dispatch)
        batch.add_edge("section_agent", END)
        await batch.compile().ainvoke(
            {"completed_sections": []},
            config={"max_concurrency": 1 if in_memory else self.section_concurrency},
        )
        if self.progress:
            self.progress(f"person_world_v3_{phase}", len(self.results), 7)
        batches = deepcopy(self.run_row.state["batches"])
        batches[batch_key]["completed"] = True
        self.run_row.state = {**self.run_row.state, "batches": batches}
        self.session.commit()

    def _accept(self, section, phase, execution):
        _check_cancelled()
        previous = self.executions.get(section)
        if previous is not None:
            # 补全轮仍可引用草稿轮已读材料，不能用最后一轮覆盖来源清单。
            execution.evidence = (*previous.evidence, *execution.evidence)
            execution.retrievals = (*previous.retrievals, *execution.retrievals)
        self.executions[section] = execution
        work = getattr(execution, "section_work", {})
        if work:
            self.run_row.state = {
                **self.run_row.state,
                "section_work": {**self.run_row.state.get("section_work", {}), section: work},
            }
        self.dependencies.extend(execution.dependencies)
        self.attempts.append(
            {
                "section": section,
                "phase": phase,
                "phoenix_execution_id": execution.execution_id,
                "status": execution.status,
                "reason": execution.reason,
            }
        )
        if execution.result is None or execution.status in {
            "failed",
            "blocked",
            "cancelled",
            "stale",
        }:
            self._failed(
                section,
                phase,
                RuntimeError(execution.reason or execution.status),
                code=execution.reason or execution.status,
            )
            return
        raw = execution.result.model_dump(mode="json")
        payload = {
            key: value
            for key, value in raw.items()
            if key not in {"section", "unresolved_questions", "overview", "module_assessment"}
        }
        assign_entry_ids(payload, self.results.get(section) or self.baseline.get(section))
        if section == "life_context":
            payload["context_modules"] = accept_modules(
                payload["context_modules"], self.module_snapshot.modules
            )
            bind_owned_module_references(payload)
        else:
            tracker = ModuleReadTracker(section, self.module_snapshot)
            validate_references(payload, self.module_snapshot, tracker=tracker)
            self.dependencies.extend(tracker.dependencies)
        SECTION_MODELS[section].model_validate(payload)
        self.results[section] = payload
        if section == "identity" and raw.get("overview"):
            self.overview = raw["overview"]
        self.errors.pop(section, None)
        task = self.tasks[section]
        task.status = "completed"
        task.error_code = None
        task.research_round = 1 if phase == "draft" else 2
        task.unresolved_questions = raw.get("unresolved_questions", [])
        task.result_summary = {
            **dict(task.result_summary or {}),
            **(
                {"module_assessment": raw["module_assessment"]}
                if "module_assessment" in raw
                else {}
            ),
            "profile_schema_version": "v3",
            "fact_count": described_count(payload),
            "phoenix_execution_id": execution.execution_id,
        }
        task.attempted_queries = list(execution.retrievals)
        task.completed_at = datetime.now(UTC)
        batches = deepcopy(self.run_row.state.get("batches", {}))
        batches[self._active_batch_key]["accepted"].append(section)
        self.run_row.state = {**self.run_row.state, "batches": batches}
        self._checkpoint()

    def _failed(self, section, phase, error, *, code=None):
        _check_cancelled()
        logging.getLogger(__name__).error(
            "PersonWorld 栏目失败 section=%s phase=%s", section, phase, exc_info=error
        )
        code = str(code or getattr(error, "code", None) or type(error).__name__)
        self.errors[section] = code
        task = self.tasks[section]
        task.status = "failed"
        task.error_code = code
        task.unresolved_questions = [
            "模型供应商拒绝本次输入材料，请查看 Phoenix 的供应商错误；这不是没有人物资料。"
            if code == "provider_input_rejected"
            else "本栏目运行未完成；这不是没有资料或没有人物特征。"
        ]
        self._checkpoint()

    def _checkpoint(self):
        self.run_row.state = {
            **self.run_row.state,
            "section_drafts": deepcopy(self.results),
            "dependencies": deepcopy(self.dependencies),
            "attempts": deepcopy(self.attempts),
            "errors": dict(self.errors),
            "overview": getattr(self, "overview", ""),
            "section_source_ids": {
                name: list(dict.fromkeys(item.message_id for item in execution.evidence))
                for name, execution in self.executions.items()
            },
        }
        self.session.commit()

    def _assemble(self, state):
        _check_cancelled()
        payload = PersonWorldProfileV3(
            subject_participant_id=self.target_participant_id,
            overview=getattr(self, "overview", ""),
            **self.results,
        )
        report = PersonWorldInvestigationReport(
            section_statuses=[
                SectionInvestigationStatus(
                    section=name,
                    state=(
                        "budget_exhausted"
                        if self.errors[name]
                        in {
                            "tool_result_context_limit",
                            "tool_call_safety_limit",
                            "emergency_model_step_limit",
                            "wall_deadline",
                        }
                        else "schema_error"
                        if self.errors[name]
                        in {
                            "ValidationError",
                            "schema_error",
                            "TypeError",
                            "ValueError",
                            "invalid_profile_reference",
                            "invalid_module_assessment",
                        }
                        else "needs_more_research"
                        if self.errors[name] == "materials_unavailable"
                        else "cloud_error"
                    )
                    if name in self.errors
                    else "completed"
                    if name in self.results
                    else "not_started",
                    accepted_fact_count=described_count(self.results.get(name)),
                    error_code=self.errors.get(name),
                    unresolved_questions=list(self.tasks[name].unresolved_questions or []),
                    trace_refs=[
                        item["phoenix_execution_id"]
                        for item in self.attempts
                        if item["section"] == name
                    ],
                )
                for name in SECTION_DIMENSIONS
            ]
        )
        summary = {
            "profile_schema_version": "v3",
            "prompt_version": "lived-world-v3",
            "dependencies": self.dependencies,
            "attempts": self.attempts,
            "failed_sections": list(self.errors),
            "module_snapshot": self.module_snapshot.to_dict(),
            "node_scope": self.node_scope.envelope() if self.node_scope else None,
            "section_work": self.run_row.state.get("section_work", {}),
        }
        legacy = empty_legacy_projection()
        draft = self.session.scalar(
            select(PersonWorldProfileDraft).where(
                PersonWorldProfileDraft.agent_run_id == self.run_row.id
            )
        )
        if draft is None:
            draft = PersonWorldProfileDraft(
                project_id=self.graph.project_id,
                graph_version_id=self.graph.id,
                agent_run_id=self.run_row.id,
                claim_ids=[],
            )
            self.session.add(draft)
        elif draft.status == "approved":
            raise ValueError("不能通过恢复编译覆盖已经批准的草稿")
        draft.status = "awaiting_review"
        draft.payload = legacy.model_dump(mode="json")
        draft.profile_v3 = payload.model_dump(mode="json")
        draft.profile_schema_version = "v3"
        draft.generation_summary = summary
        draft.investigation_report = report.model_dump(mode="json")
        self.session.flush()
        self.run_row.status = "awaiting_review"
        self.run_row.state = {**self.run_row.state, "profile_draft_id": draft.id}
        self.session.commit()
        # 恢复运行时 executions 只包含本次重试，不能因此丢掉保留栏目的既有引用。
        ids = tuple(
            sorted(
                _reference_ids(payload.model_dump(mode="json"))
                | {
                    item.message_id
                    for execution in self.executions.values()
                    for item in execution.evidence
                }
            )
        )
        manifest = tuple(
            RetrievalManifestItem(
                question=str(item.get("question", "")),
                mode=str(item.get("mode", "mix")),
                document_ids=(),
            )
            for execution in self.executions.values()
            for item in execution.retrievals
        )
        self.output = PersonWorldV3AgentResult(
            self.run_row.id, payload, report, legacy, ids, manifest, summary
        )
        return {"stage": "awaiting_review"}
