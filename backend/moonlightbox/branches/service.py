from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.agent.jobs import COGNITIVE_CYCLE_JOB_KIND
from moonlightbox.agent.models import MentalStateVersion
from moonlightbox.agent.service import ShadowCognitionService
from moonlightbox.branches.actor import (
    BRANCH_CONVERSATION_JOB_KIND,
    interrupt_pending_expression,
)
from moonlightbox.branches.actor_models import (
    ConversationActorState,
    assistant_typing_active,
)
from moonlightbox.branches.baseline_boundary import BaselineBoundaryResolver
from moonlightbox.branches.baseline_jobs import BRANCH_BASELINE_JOB_KIND
from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.continuity_models import BranchMemoryItem, IdentityKernel
from moonlightbox.branches.generation import BranchGenerator
from moonlightbox.branches.memory_jobs import ProjectMemoryRepository
from moonlightbox.branches.models import (
    Branch,
    BranchMessage,
    uses_authoritative_cognition,
)
from moonlightbox.branches.reviewer import ReplyReviewer
from moonlightbox.branches.schemas import BranchCreate
from moonlightbox.jobs.service import JobService
from moonlightbox.training.models import ModelVersion


class BranchNotFoundError(LookupError):
    pass


class BranchReadOnlyError(RuntimeError):
    pass


class ActiveModelUnavailableError(RuntimeError):
    pass


class BranchBaselineNotReadyError(RuntimeError):
    pass


class BranchModelNotAcceptedError(RuntimeError):
    pass


class CandidateModelNotReadyError(RuntimeError):
    pass


class BranchService:
    def __init__(
        self,
        session: Session,
        generator: BranchGenerator,
        reviewer: ReplyReviewer | None = None,
        memory_repository: ProjectMemoryRepository | None = None,
        continuity_repository: BranchContinuityRepository | None = None,
    ) -> None:
        self._session = session
        self._generator = generator
        self._reviewer = reviewer
        self._memory_repository = memory_repository
        self._continuity_repository = continuity_repository

    def create(self, project_id: str, payload: BranchCreate) -> Branch:
        model = self._session.get(ModelVersion, payload.model_version_id)
        kernel = self._session.scalar(
            select(IdentityKernel).where(
                IdentityKernel.model_version_id == payload.model_version_id
            )
        )
        if (
            model is None
            or model.project_id != project_id
            or not model.active
            or kernel is None
            or kernel.schema_version != "subject-persona-v2"
            or kernel.acceptance_report_id is None
        ):
            raise BranchModelNotAcceptedError(
                "只能使用已通过验收并锁定人格内核的活动模型创建分支"
            )
        boundary = BaselineBoundaryResolver(self._session).resolve(
            project_id, payload.origin_event_id
        )
        branch = Branch(
            project_id=project_id,
            **payload.model_dump(),
            state_snapshot={},
            generation_policy_version="subject-v2-actor-v1",
            origin_import_id=boundary.import_id,
            origin_boundary_message_id=boundary.message_id,
            baseline_status="preparing",
        )
        self._session.add(branch)
        self._session.flush()
        job = JobService(self._session).enqueue_unique(
            BRANCH_BASELINE_JOB_KIND,
            {"project_id": project_id, "branch_id": branch.id},
            dedupe_key=f"{BRANCH_BASELINE_JOB_KIND}:{branch.id}",
        )
        branch.baseline_job_id = job.id
        self._session.commit()
        self._session.refresh(branch)
        return branch

    def list_branches(self, project_id: str) -> list[Branch]:
        return list(
            self._session.scalars(
                select(Branch)
                .where(Branch.project_id == project_id)
                .order_by(Branch.created_at.desc())
            )
        )

    def messages(self, project_id: str, branch_id: str) -> list[BranchMessage]:
        branch = self._get(project_id, branch_id)
        self._ensure_baseline_ready(branch)
        return list(
            self._session.scalars(
                select(BranchMessage)
                .where(BranchMessage.branch_id == branch_id)
                .order_by(BranchMessage.sequence)
            )
        )

    def upgrade(self, project_id: str, branch_id: str) -> Branch:
        branch = self._get(project_id, branch_id)
        if branch.replacement_branch_id is not None:
            replacement = self._session.get(Branch, branch.replacement_branch_id)
            if replacement is not None:
                return replacement
        active_model = self._session.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.project_id == project_id,
                ModelVersion.active.is_(True),
            )
            .order_by(ModelVersion.created_at.desc())
        )
        if active_model is None:
            raise ActiveModelUnavailableError("当前项目没有已通过验收的活动模型")
        active_kernel = self._session.scalar(
            select(IdentityKernel).where(
                IdentityKernel.model_version_id == active_model.id
            )
        )
        if (
            active_kernel is None
            or active_kernel.schema_version != "subject-persona-v2"
            or active_kernel.acceptance_report_id is None
        ):
            raise ActiveModelUnavailableError(
                "活动模型缺少已锁定的人格内核，不能升级分支"
            )
        replacement = Branch(
            project_id=project_id,
            origin_event_id=branch.origin_event_id,
            model_version_id=active_model.id,
            title=branch.title,
            origin_time=branch.origin_time,
            state_snapshot=_durable_upgrade_state(branch.state_snapshot),
            lifecycle_status="active",
            generation_policy_version="subject-v2-actor-v1",
            baseline_status="preparing",
            subject_agent_mode=branch.subject_agent_mode,
        )
        self._session.add(replacement)
        self._session.flush()
        _copy_branch_memories(self._session, branch, replacement)
        boundary = BaselineBoundaryResolver(self._session).resolve(
            project_id, replacement.origin_event_id
        )
        replacement.origin_import_id = boundary.import_id
        replacement.origin_boundary_message_id = boundary.message_id
        job = JobService(self._session).enqueue_unique(
            BRANCH_BASELINE_JOB_KIND,
            {"project_id": project_id, "branch_id": replacement.id},
            dedupe_key=f"{BRANCH_BASELINE_JOB_KIND}:{replacement.id}",
        )
        replacement.baseline_job_id = job.id
        branch.lifecycle_status = "upgraded"
        branch.replacement_branch_id = replacement.id
        self._session.commit()
        self._session.refresh(replacement)
        return replacement

    def add_user_message(
        self,
        project_id: str,
        branch_id: str,
        content: str,
        client_message_id: str | None = None,
    ) -> BranchMessage:
        branch = self._get(project_id, branch_id)
        self._ensure_baseline_ready(branch)
        if branch.lifecycle_status != "active":
            raise BranchReadOnlyError("旧时间分支已只读，请进入升级后的新分支")
        if client_message_id is not None:
            existing = self._session.scalar(
                select(BranchMessage).where(
                    BranchMessage.branch_id == branch.id,
                    BranchMessage.client_message_id == client_message_id,
                )
            )
            if existing is not None:
                return existing
        latest_sequence = self._session.scalar(
            select(func.max(BranchMessage.sequence)).where(
                BranchMessage.branch_id == branch.id
            )
        )
        user_turn_id = str(uuid4())
        user = BranchMessage(
            branch_id=branch.id,
            sequence=int(latest_sequence) + 1 if latest_sequence is not None else 0,
            role="user",
            content=content,
            type="text",
            turn_id=user_turn_id,
            bubble_index=0,
            delay_ms=0,
            generation_status="completed",
            generation_metadata={},
            client_message_id=client_message_id,
        )
        self._session.add(user)
        self._session.flush()
        interrupt_pending_expression(self._session, branch.id)
        event_time = datetime.now(UTC)
        try:
            with self._session.begin_nested():
                cognition = ShadowCognitionService(self._session)
                event = cognition.append_perception_event(
                    project_id=project_id,
                    branch_id=branch_id,
                    event_type="user_message",
                    occurred_at=event_time,
                    source="branch_conversation",
                    idempotency_key=f"user-message:{user.id}",
                    evidence={
                        "branch_message_id": user.id,
                        "content": content,
                    },
                    commit=False,
                )
                model_version = self._session.get(ModelVersion, branch.model_version_id)
                reply_protocol = (
                    str(
                        model_version.training_config.get(
                            "reply_protocol_version", ""
                        )
                    )
                    if model_version is not None
                    and isinstance(model_version.training_config, dict)
                    else ""
                )
                # Persona-text 的公开回复会先聚合连续消息。逐条创建认知周期会让
                # 最后一条短消息覆盖整个表达，因此只保留逐条感知事件，认知周期
                # 由 Conversation Actor 在输入停顿后按贡献簇统一创建。
                if reply_protocol.startswith("persona-text"):
                    cycle = None
                else:
                    initial_state = (
                        self._session.scalar(
                            select(MentalStateVersion).where(
                                MentalStateVersion.branch_id == branch_id,
                                MentalStateVersion.is_current.is_(True),
                            )
                        )
                        if uses_authoritative_cognition(branch.subject_agent_mode)
                        else None
                    )
                    if initial_state is None:
                        initial_state = cognition.ensure_initial_mental_state(
                            project_id=project_id,
                            branch_id=branch_id,
                            state=self._initial_mental_state(branch.state_snapshot),
                            evidence={"branch_state_snapshot": True},
                            model_version_id=branch.model_version_id,
                            commit=False,
                        )
                    cycle = cognition.create_pending_cycle(
                        project_id=project_id,
                        branch_id=branch_id,
                        trigger_event_id=event.id,
                        input_cutoff_at=event_time,
                        starting_state_version_id=initial_state.id,
                        model_version_id=branch.model_version_id,
                        evidence={"branch_message_id": user.id},
                        commit=False,
                    )
                if cycle is not None and branch.subject_agent_mode == "shadow":
                    JobService(self._session).enqueue_unique(
                        COGNITIVE_CYCLE_JOB_KIND,
                        {
                            "project_id": project_id,
                            "branch_id": branch_id,
                            "message_id": user.id,
                            "cycle_id": cycle.id,
                        },
                        dedupe_key=f"{COGNITIVE_CYCLE_JOB_KIND}:{cycle.id}",
                        commit=False,
                    )
        except Exception:
            pass
        JobService(self._session).enqueue_unique(
            BRANCH_CONVERSATION_JOB_KIND,
            {
                "project_id": project_id,
                "branch_id": branch_id,
                "message_id": user.id,
            },
            dedupe_key=f"{BRANCH_CONVERSATION_JOB_KIND}:{user.id}",
            commit=False,
        )
        self._session.commit()
        self._session.refresh(user)
        return user

    def conversation_state(
        self,
        project_id: str,
        branch_id: str,
    ) -> dict[str, object]:
        branch = self._get(project_id, branch_id)
        self._ensure_baseline_ready(branch)
        actor = self._session.scalar(
            select(ConversationActorState).where(
                ConversationActorState.branch_id == branch_id
            )
        )
        if actor is None:
            actor = ConversationActorState(branch_id=branch_id)
            self._session.add(actor)
            self._session.commit()
            self._session.refresh(actor)
        return {
            "status": actor.status,
            "typing": assistant_typing_active(actor),
            "observed_message_sequence": actor.observed_message_sequence,
            "version": actor.version,
        }

    def set_user_typing(
        self,
        project_id: str,
        branch_id: str,
        typing: bool,
    ) -> dict[str, object]:
        self.conversation_state(project_id, branch_id)
        actor = self._session.scalar(
            select(ConversationActorState).where(
                ConversationActorState.branch_id == branch_id
            )
        )
        if actor is None:
            raise RuntimeError("会话 Actor 状态初始化失败")
        actor.user_typing_until = (
            datetime.now(UTC) + timedelta(seconds=3) if typing else None
        )
        actor.version += 1
        self._session.commit()
        return self.conversation_state(project_id, branch_id)

    def _get(self, project_id: str, branch_id: str) -> Branch:
        branch = self._session.get(Branch, branch_id)
        if branch is None or branch.project_id != project_id:
            raise BranchNotFoundError(branch_id)
        return branch

    @staticmethod
    def _ensure_baseline_ready(branch: Branch) -> None:
        if branch.baseline_status != "ready":
            raise BranchBaselineNotReadyError("分支基础历史尚未准备完成")

    @staticmethod
    def _initial_mental_state(snapshot: dict[str, object]) -> dict[str, object]:
        mapping_keys = (
            "persona_state",
            "relationship_state",
            "user_model",
            "emotional_tendency",
            "current_goals",
            "current_concerns",
        )
        list_keys = ("active_belief_ids", "contested_belief_ids")
        state: dict[str, object] = {}
        for key in mapping_keys:
            value = snapshot.get(key)
            state[key] = dict(value) if isinstance(value, dict) else {}
        for key in list_keys:
            value = snapshot.get(key)
            state[key] = [
                item for item in value if isinstance(item, str)
            ] if isinstance(value, list) else []
        return state


def _durable_upgrade_state(snapshot: dict[str, object]) -> dict[str, object]:
    transient = {
        "situational_state",
        "state_version_id",
        "version",
        "updated_at",
    }
    return {
        str(key): value
        for key, value in snapshot.items()
        if key not in transient
    }


def _copy_branch_memories(
    session: Session,
    source: Branch,
    replacement: Branch,
) -> None:
    source_items = list(
        session.scalars(
            select(BranchMemoryItem)
            .where(
                BranchMemoryItem.branch_id == source.id,
                BranchMemoryItem.review_status == "approved",
                BranchMemoryItem.valid_to.is_(None),
            )
            .order_by(BranchMemoryItem.created_at)
        )
    )
    id_map: dict[str, str] = {}
    for item in source_items:
        cloned = BranchMemoryItem(
            branch_id=replacement.id,
            kind=item.kind,
            content=item.content,
            subject=item.subject,
            predicate=item.predicate,
            object=item.object,
            confidence=item.confidence,
            importance=item.importance,
            valid_from=item.valid_from,
            source_episode_ids=list(item.source_episode_ids),
            source_item_ids=[item.id, *item.source_item_ids],
            lineage_hash=hashlib.sha256(
                f"{replacement.id}:{item.lineage_hash}".encode()
            ).hexdigest(),
            review_status="approved",
            verification_status=item.verification_status,
            claim_key=item.claim_key,
            stance=item.stance,
            root_episode_hashes=list(item.root_episode_hashes),
        )
        session.add(cloned)
        session.flush()
        id_map[item.id] = cloned.id
    state = dict(replacement.state_snapshot)
    for field in ("active_belief_ids", "contested_belief_ids"):
        values = state.get(field)
        if isinstance(values, list):
            state[field] = [
                id_map[item_id]
                for item_id in values
                if isinstance(item_id, str) and item_id in id_map
            ]
    replacement.state_snapshot = state


def stage_candidate_branch(
    session: Session,
    *,
    project_id: str,
    source_branch_id: str,
    model_version_id: str,
) -> Branch:
    """Prepare an inactive accepted candidate for replay without touching live state."""

    source = session.scalar(
        select(Branch).where(
            Branch.id == source_branch_id,
            Branch.project_id == project_id,
        )
    )
    if source is None:
        raise BranchNotFoundError(source_branch_id)
    if source.lifecycle_status != "active" or source.baseline_status != "ready":
        raise CandidateModelNotReadyError("源分支尚未处于可回放状态")
    model = session.scalar(
        select(ModelVersion).where(
            ModelVersion.id == model_version_id,
            ModelVersion.project_id == project_id,
        )
    )
    kernel = session.scalar(
        select(IdentityKernel).where(IdentityKernel.model_version_id == model_version_id)
    )
    config = model.training_config if model is not None else {}
    if (
        model is None
        or model.active
        or model.status != "awaiting_human_review"
        or not bool(config.get("human_blind_required"))
        or config.get("upgraded_from_model_version_id") != source.model_version_id
        or kernel is None
        or kernel.schema_version != "subject-persona-v2"
        or kernel.acceptance_report_id is None
    ):
        raise CandidateModelNotReadyError(
            "只能暂存已通过自动验收、等待真人盲测且来源匹配的候选模型"
        )
    staged_from_key = "_candidate_staged_from_branch_id"
    existing_rows = session.scalars(
        select(Branch).where(
            Branch.project_id == project_id,
            Branch.model_version_id == model_version_id,
            Branch.generation_policy_version == "subject-v2-candidate-v1",
        )
    )
    for existing in existing_rows:
        if existing.state_snapshot.get(staged_from_key) == source.id:
            if existing.subject_agent_mode != "preview":
                existing.subject_agent_mode = "preview"
                session.commit()
                session.refresh(existing)
            return existing

    state_snapshot = _durable_upgrade_state(source.state_snapshot)
    state_snapshot[staged_from_key] = source.id
    candidate = Branch(
        project_id=project_id,
        origin_event_id=source.origin_event_id,
        model_version_id=model.id,
        title=source.title,
        origin_time=source.origin_time,
        state_snapshot=state_snapshot,
        lifecycle_status="active",
        generation_policy_version="subject-v2-candidate-v1",
        baseline_status="preparing",
        subject_agent_mode="preview",
    )
    session.add(candidate)
    session.flush()
    _copy_branch_memories(session, source, candidate)
    boundary = BaselineBoundaryResolver(session).resolve(project_id, candidate.origin_event_id)
    candidate.origin_import_id = boundary.import_id
    candidate.origin_boundary_message_id = boundary.message_id
    job = JobService(session).enqueue_unique(
        BRANCH_BASELINE_JOB_KIND,
        {"project_id": project_id, "branch_id": candidate.id},
        dedupe_key=f"{BRANCH_BASELINE_JOB_KIND}:{candidate.id}",
        commit=False,
    )
    candidate.baseline_job_id = job.id
    session.commit()
    session.refresh(candidate)
    return candidate
