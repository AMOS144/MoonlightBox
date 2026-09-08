"""影子主体认知的领域服务。"""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from moonlightbox.agent.models import (
    AgentGoal,
    AgentWakeup,
    CognitiveCycle,
    MentalStateVersion,
    PerceptionEvent,
    PrivateCognitionNote,
)
from moonlightbox.agent.persona_examples import select_authentic_dialogue_examples
from moonlightbox.agent.types import CognitionDraft, CognitionRequest, FusedAgentTurn
from moonlightbox.branches.continuity_models import (
    BranchMemoryItem,
    BranchStateVersion,
    IdentityKernel,
)
from moonlightbox.branches.models import Branch, BranchMessage, uses_authoritative_cognition
from moonlightbox.branches.situational_state import set_situational_state
from moonlightbox.jobs.service import JobService
from moonlightbox.training.expression_policy import ExpressionPolicy, should_respond
from moonlightbox.training.models import ModelVersion

DEFAULT_INITIAL_MENTAL_STATE: dict[str, object] = {}


def _aware_utc(value: datetime) -> datetime:
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value).astimezone(UTC)


def _recent_user_query(
    context: list[dict[str, object]],
    trigger: PerceptionEvent,
) -> str:
    """Build a short retrieval query from the latest other-person bubbles."""

    messages: list[str] = []
    for item in reversed(context):
        if item.get("event_type") != "user_message":
            continue
        evidence = item.get("evidence")
        content = evidence.get("content") if isinstance(evidence, dict) else None
        if isinstance(content, str) and content.strip():
            messages.append(content.strip())
        if len(messages) >= 3:
            break
    messages.reverse()
    trigger_content = trigger.evidence.get("content")
    if (
        isinstance(trigger_content, str)
        and trigger_content.strip()
        and trigger_content.strip() not in messages
    ):
        messages.append(trigger_content.strip())
    return "\n".join(messages[-3:])


def _select_relevant_memories(
    memories: list[BranchMemoryItem],
    query: str,
    *,
    limit: int,
) -> list[BranchMemoryItem]:
    """按当前贡献选择相关长期记忆，低相关时不做兜底注入。"""

    query_chars = {item for item in query.lower() if item.isalnum()}
    if not query_chars or limit <= 0:
        return []
    ranked: list[tuple[float, BranchMemoryItem]] = []
    for memory in memories:
        content = memory.content.lower()
        content_chars = {item for item in content if item.isalnum()}
        overlap = len(query_chars & content_chars) / max(
            1,
            len(query_chars | content_chars),
        )
        containment = 1.0 if query.strip() and query.strip() in memory.content else 0.0
        score = overlap + containment + min(1.0, max(0.0, memory.importance)) * 0.05
        if score >= 0.08:
            ranked.append((score, memory))
    ranked.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
    return [memory for _, memory in ranked[:limit]]


class BranchScopeError(ValueError):
    """项目与分支不匹配。"""


class CognitiveCycleNotFoundError(LookupError):
    """认知周期不存在。"""


class InvalidCognitiveCycleStateError(ValueError):
    """认知周期当前状态不允许执行请求的操作。"""


class ShadowCognitionService:
    """仅持久化影子结果、不改变主体外显行为的认知服务。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def append_perception_event(
        self,
        *,
        project_id: str,
        branch_id: str,
        event_type: str,
        occurred_at: datetime,
        source: str,
        idempotency_key: str,
        confidence: float = 1.0,
        visible_through: datetime | None = None,
        evidence: dict[str, object] | None = None,
        model_protocol_version: str = "subject-cognition-v1",
        commit: bool = True,
    ) -> PerceptionEvent:
        self._ensure_branch_scope(project_id, branch_id)
        existing = self._session.scalar(
            select(PerceptionEvent).where(
                PerceptionEvent.branch_id == branch_id,
                PerceptionEvent.idempotency_key == idempotency_key,
            )
        )
        if existing is not None:
            return existing
        event = PerceptionEvent(
            project_id=project_id,
            branch_id=branch_id,
            event_type=event_type,
            occurred_at=occurred_at,
            source=source,
            confidence=confidence,
            idempotency_key=idempotency_key,
            visible_through=visible_through or occurred_at,
            evidence=evidence or {},
            model_protocol_version=model_protocol_version,
        )
        self._session.add(event)
        if not commit:
            self._session.flush()
            return event
        return self._commit_with_idempotent_fallback(
            event,
            select(PerceptionEvent).where(
                PerceptionEvent.branch_id == branch_id,
                PerceptionEvent.idempotency_key == idempotency_key,
            ),
        )

    def append_situational_observation(
        self,
        *,
        project_id: str,
        branch_id: str,
        values: dict[str, str],
        occurred_at: datetime,
        valid_until: datetime,
        source: str,
        idempotency_key: str,
        confidence: float = 1.0,
        commit: bool = True,
    ) -> PerceptionEvent:
        """Atomically project a trusted world observation into expiring state."""

        if source not in {
            "external_observation",
            "user_configured",
            "agent_action",
            "historical_replay",
        }:
            raise ValueError("情境观察来源不可信")
        event = self.append_perception_event(
            project_id=project_id,
            branch_id=branch_id,
            event_type="situational_observation",
            occurred_at=occurred_at,
            source=source,
            idempotency_key=idempotency_key,
            confidence=confidence,
            visible_through=valid_until,
            evidence={
                "situational_values": values,
                "valid_until": valid_until.isoformat(),
            },
            commit=False,
        )
        branch = self._session.get(Branch, branch_id)
        if branch is None:
            raise LookupError("分支不存在")
        event_values = event.evidence.get("situational_values", {})
        if not isinstance(event_values, dict):
            raise ValueError("情境观察缺少结构化状态")
        set_situational_state(
            branch,
            values={str(key): str(value) for key, value in event_values.items()},
            source=event.source,  # type: ignore[arg-type]
            evidence_ids=(event.id,),
            observed_at=_aware_utc(event.occurred_at),
            valid_until=_aware_utc(event.visible_through),
            confidence=event.confidence,
        )
        if commit:
            self._session.commit()
            self._session.refresh(event)
        else:
            self._session.flush()
        return event

    def ensure_initial_mental_state(
        self,
        *,
        project_id: str,
        branch_id: str,
        state: dict[str, object],
        evidence: dict[str, object] | None = None,
        model_version_id: str | None = None,
        model_protocol_version: str = "subject-cognition-v1",
        commit: bool = True,
    ) -> MentalStateVersion:
        self._ensure_branch_scope(project_id, branch_id)
        existing = self._session.scalar(
            select(MentalStateVersion).where(
                MentalStateVersion.branch_id == branch_id,
                MentalStateVersion.version == 1,
            )
        )
        if existing is not None:
            return existing
        initial = MentalStateVersion(
            project_id=project_id,
            branch_id=branch_id,
            version=1,
            state=state,
            evidence=evidence or {},
            is_current=True,
            model_version_id=model_version_id,
            model_protocol_version=model_protocol_version,
        )
        self._session.add(initial)
        if not commit:
            self._session.flush()
            return initial
        return self._commit_with_idempotent_fallback(
            initial,
            select(MentalStateVersion).where(
                MentalStateVersion.branch_id == branch_id,
                MentalStateVersion.version == 1,
            ),
        )

    def create_pending_cycle(
        self,
        *,
        project_id: str,
        branch_id: str,
        trigger_event_id: str,
        input_cutoff_at: datetime,
        starting_state_version_id: str,
        model_version_id: str,
        memory_ids: list[str] | None = None,
        goal_ids: list[str] | None = None,
        evidence: dict[str, object] | None = None,
        model_protocol_version: str = "subject-cognition-v1",
        commit: bool = True,
    ) -> CognitiveCycle:
        self._ensure_branch_scope(project_id, branch_id)
        self._require_scoped_record(
            PerceptionEvent,
            trigger_event_id,
            project_id=project_id,
            branch_id=branch_id,
        )
        self._require_scoped_record(
            MentalStateVersion,
            starting_state_version_id,
            project_id=project_id,
            branch_id=branch_id,
        )
        existing_query = select(CognitiveCycle).where(
            CognitiveCycle.branch_id == branch_id,
            CognitiveCycle.trigger_event_id == trigger_event_id,
        )
        existing = self._session.scalar(existing_query)
        if existing is not None:
            return existing
        cycle = CognitiveCycle(
            project_id=project_id,
            branch_id=branch_id,
            trigger_event_id=trigger_event_id,
            input_cutoff_at=input_cutoff_at,
            starting_state_version_id=starting_state_version_id,
            memory_ids=memory_ids or [],
            goal_ids=goal_ids or [],
            model_version_id=model_version_id,
            model_protocol_version=model_protocol_version,
            evidence=evidence or {},
            status="pending",
        )
        self._session.add(cycle)
        if not commit:
            self._session.flush()
            return cycle
        return self._commit_with_idempotent_fallback(cycle, existing_query)

    def mark_cycle_running(
        self,
        cycle_id: str,
        *,
        started_at: datetime | None = None,
    ) -> CognitiveCycle:
        cycle = self.get_cycle(cycle_id)
        if cycle.status != "pending":
            raise InvalidCognitiveCycleStateError("认知周期不是 pending 状态")
        cycle.status = "running"
        cycle.started_at = started_at or datetime.now(UTC)
        self._session.commit()
        return cycle

    def mark_cycle_failed(
        self,
        cycle_id: str,
        *,
        completed_at: datetime | None = None,
    ) -> CognitiveCycle:
        cycle = self.get_cycle(cycle_id)
        if cycle.status in {"succeeded", "invalidated"}:
            return cycle
        cycle.status = "failed"
        cycle.completed_at = completed_at or datetime.now(UTC)
        self._session.commit()
        return cycle

    def invalidate_if_stale(
        self,
        cycle_id: str,
        *,
        completed_at: datetime | None = None,
    ) -> bool:
        cycle = self.get_cycle(cycle_id)
        state_is_current = self._session.scalar(
            select(MentalStateVersion.is_current).where(
                MentalStateVersion.id == cycle.starting_state_version_id,
                MentalStateVersion.branch_id == cycle.branch_id,
            )
        )
        has_new_user_message = (
            self._session.scalar(
                select(PerceptionEvent.id)
                .where(
                    PerceptionEvent.branch_id == cycle.branch_id,
                    PerceptionEvent.event_type == "user_message",
                    PerceptionEvent.occurred_at > cycle.input_cutoff_at,
                )
                .limit(1)
            )
            is not None
        )
        if state_is_current is True and not has_new_user_message:
            return False
        cycle.status = "invalidated"
        cycle.completed_at = completed_at or datetime.now(UTC)
        self._session.commit()
        return True

    def build_request(
        self,
        cycle_id: str,
        *,
        deadline: datetime | None = None,
    ) -> CognitionRequest:
        cycle = self.get_cycle(cycle_id)
        trigger = self._require_scoped_record(
            PerceptionEvent,
            cycle.trigger_event_id,
            project_id=cycle.project_id,
            branch_id=cycle.branch_id,
        )
        state = self._require_scoped_record(
            MentalStateVersion,
            cycle.starting_state_version_id,
            project_id=cycle.project_id,
            branch_id=cycle.branch_id,
        )
        goals = self._session.scalars(
            select(AgentGoal).where(
                AgentGoal.branch_id == cycle.branch_id,
                AgentGoal.status == "active",
            )
        )
        context = self._recent_context_payloads(cycle, trigger, limit=12)
        branch = self._session.get(Branch, cycle.branch_id)
        query_text = _recent_user_query(context, trigger)
        expression_required: bool | None = None
        if (
            branch is not None
            and uses_authoritative_cognition(branch.subject_agent_mode)
            and trigger.event_type == "user_message"
        ):
            model = self._session.get(ModelVersion, cycle.model_version_id)
            content = trigger.evidence.get("content")
            if model is not None and isinstance(content, str):
                expression_required = should_respond(
                    ExpressionPolicy.from_metadata(
                        model.training_config.get("expression_policy")
                    ),
                    content,
                )
        decision_context = self._decision_context(branch, query_text=query_text)
        approved_memories = decision_context.get("approved_memories")
        if isinstance(approved_memories, list):
            cycle.memory_ids = [
                str(item["id"])
                for item in approved_memories
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
        return CognitionRequest(
            project_id=cycle.project_id,
            branch_id=cycle.branch_id,
            model_version_id=cycle.model_version_id,
            trigger_event=self._event_payload(trigger),
            deadline=deadline or datetime.now(UTC) + timedelta(seconds=60),
            current_mental_state=dict(state.state),
            goals=tuple(
                {
                    "id": goal.id,
                    "goal_type": goal.goal_type,
                    "content": goal.content,
                    "priority": goal.priority,
                }
                for goal in goals
            ),
            relevant_context=tuple(context),
            decision_context=decision_context,
            authoritative_expression=(
                branch is not None
                and uses_authoritative_cognition(branch.subject_agent_mode)
            ),
            expression_required=expression_required,
        )

    def persist_shadow_result(
        self,
        cycle_id: str,
        draft: CognitionDraft,
        *,
        completed_at: datetime | None = None,
        commit: bool = True,
    ) -> PrivateCognitionNote:
        cycle = self.get_cycle(cycle_id)
        if cycle.status not in {"pending", "running"}:
            raise InvalidCognitiveCycleStateError("认知周期不能保存影子结果")
        note = PrivateCognitionNote(
            project_id=cycle.project_id,
            branch_id=cycle.branch_id,
            trigger_event_id=cycle.trigger_event_id,
            content=draft.private_content,
            subjective_feelings=dict(draft.subjective_feelings),
            attention_target=dict(draft.attention_target),
            desired_actions=[dict(action) for action in draft.desired_actions],
            memory_ids=list(cycle.memory_ids),
            confidence=draft.confidence,
            evidence={"shadow_mode": True},
            model_version_id=cycle.model_version_id,
            model_protocol_version=cycle.model_protocol_version,
        )
        self._session.add(note)
        self._session.flush()
        cycle.private_note_id = note.id
        cycle.structured_changes = dict(draft.structured_changes)
        cycle.final_decision = {
            "express": draft.expression_decision.express,
            "content": draft.expression_decision.content,
            "reason": draft.expression_decision.reason,
        }
        cycle.status = "succeeded"
        cycle.completed_at = completed_at or datetime.now(UTC)
        if commit:
            self._session.commit()
        return note

    def persist_active_result(
        self,
        cycle_id: str,
        turn: FusedAgentTurn,
        *,
        completed_at: datetime | None = None,
        commit: bool = True,
    ) -> PrivateCognitionNote:
        """保存实时融合认知，并把状态演化留给后台提取任务。"""

        cycle = self.get_cycle(cycle_id)
        if cycle.status not in {"pending", "running"}:
            raise InvalidCognitiveCycleStateError("认知周期不能保存融合结果")
        note = PrivateCognitionNote(
            project_id=cycle.project_id,
            branch_id=cycle.branch_id,
            trigger_event_id=cycle.trigger_event_id,
            content=turn.cognition.private_content,
            subjective_feelings=dict(turn.cognition.subjective_feelings),
            attention_target=dict(turn.cognition.attention_target),
            desired_actions=[
                dict(action) for action in turn.cognition.desired_actions
            ],
            memory_ids=list(cycle.memory_ids),
            confidence=turn.cognition.confidence,
            evidence={"active_mode": True, "cycle_id": cycle.id},
            model_version_id=cycle.model_version_id,
            model_protocol_version=cycle.model_protocol_version,
        )
        self._session.add(note)
        self._session.flush()
        cycle.private_note_id = note.id
        cycle.structured_changes = {"extraction_status": "pending"}
        cycle.final_decision = {
            "express": turn.expression_decision.express,
            "content": turn.expression_decision.content,
            "reason": turn.expression_decision.reason,
        }
        cycle.status = "succeeded"
        cycle.completed_at = completed_at or datetime.now(UTC)

        from moonlightbox.agent.extraction import COGNITIVE_EXTRACTION_JOB_KIND

        JobService(self._session).enqueue_unique(
            COGNITIVE_EXTRACTION_JOB_KIND,
            {
                "project_id": cycle.project_id,
                "branch_id": cycle.branch_id,
                "cycle_id": cycle.id,
                "note_id": note.id,
            },
            dedupe_key=f"{COGNITIVE_EXTRACTION_JOB_KIND}:{cycle.id}",
            commit=False,
        )
        if commit:
            self._session.commit()
        return note

    def schedule_wakeup(
        self,
        *,
        project_id: str,
        branch_id: str,
        wake_at: datetime,
        reason: str,
        idempotency_key: str,
        goal_id: str | None = None,
        event_id: str | None = None,
        evidence: dict[str, object] | None = None,
        model_version_id: str | None = None,
        model_protocol_version: str = "subject-cognition-v1",
        commit: bool = True,
    ) -> AgentWakeup:
        self._ensure_branch_scope(project_id, branch_id)
        existing_query = select(AgentWakeup).where(
            AgentWakeup.branch_id == branch_id,
            AgentWakeup.idempotency_key == idempotency_key,
        )
        wakeup = self._session.scalar(existing_query)
        if wakeup is None:
            wakeup = AgentWakeup(
                project_id=project_id,
                branch_id=branch_id,
                wake_at=wake_at,
                reason=reason,
                goal_id=goal_id,
                event_id=event_id,
                idempotency_key=idempotency_key,
                evidence=evidence or {},
                model_version_id=model_version_id,
                model_protocol_version=model_protocol_version,
            )
            self._session.add(wakeup)
            if commit:
                return self._commit_with_idempotent_fallback(wakeup, existing_query)
            self._session.flush()
            return wakeup
        wakeup.wake_at = wake_at
        wakeup.reason = reason
        wakeup.goal_id = goal_id
        wakeup.event_id = event_id
        wakeup.evidence = evidence or {}
        wakeup.model_version_id = model_version_id
        wakeup.model_protocol_version = model_protocol_version
        wakeup.status = "scheduled"
        if commit:
            self._session.commit()
        else:
            self._session.flush()
        return wakeup

    def get_cycle(self, cycle_id: str) -> CognitiveCycle:
        cycle = self._session.get(CognitiveCycle, cycle_id)
        if cycle is None:
            raise CognitiveCycleNotFoundError(cycle_id)
        return cycle

    def _ensure_branch_scope(self, project_id: str, branch_id: str) -> Branch:
        branch = self._session.scalar(
            select(Branch).where(
                Branch.id == branch_id,
                Branch.project_id == project_id,
            )
        )
        if branch is None:
            raise BranchScopeError("project/branch 不匹配")
        return branch

    def _require_scoped_record(
        self,
        model: type[Any],
        record_id: str | None,
        *,
        project_id: str,
        branch_id: str,
    ) -> Any:
        record = self._session.scalar(
            select(model).where(
                model.id == record_id,
                model.project_id == project_id,
                model.branch_id == branch_id,
            )
        )
        if record is None:
            raise BranchScopeError("引用记录不属于指定 project/branch")
        return record

    def _commit_with_idempotent_fallback(
        self,
        created: Any,
        fallback_query: Any,
    ) -> Any:
        try:
            self._session.commit()
            return created
        except IntegrityError:
            self._session.rollback()
            existing = self._session.scalar(fallback_query)
            if existing is None:
                raise
            return existing

    @staticmethod
    def _event_payload(event: PerceptionEvent) -> dict[str, object]:
        return {
            "id": event.id,
            "event_type": event.event_type,
            "actor": (
                "self"
                if event.event_type == "agent_expression"
                else "other"
                if event.event_type == "user_message"
                else "world"
            ),
            "occurred_at": event.occurred_at.isoformat(),
            "source": event.source,
            "confidence": event.confidence,
            "evidence": dict(event.evidence),
        }

    def _recent_context_payloads(
        self,
        cycle: CognitiveCycle,
        trigger: PerceptionEvent,
        *,
        limit: int,
    ) -> list[dict[str, object]]:
        """Build one bounded, chronological view of world events and public chat.

        Perception events remain the audit log for external observations.  The
        public branch transcript is authoritative for conversation continuity,
        because it contains both sides of the chat and also repairs branches
        created before assistant expressions were projected into perception.
        """

        recent_events = list(
            self._session.scalars(
                select(PerceptionEvent)
                .where(
                    PerceptionEvent.branch_id == cycle.branch_id,
                    PerceptionEvent.occurred_at <= cycle.input_cutoff_at,
                )
                .order_by(
                    PerceptionEvent.occurred_at.desc(),
                    PerceptionEvent.id.desc(),
                )
                .limit(max(limit * 2, limit))
            )
        )
        events_by_message_id = {
            message_id: event
            for event in recent_events
            if isinstance(
                message_id := event.evidence.get("branch_message_id"),
                str,
            )
        }

        trigger_message_id = trigger.evidence.get("branch_message_id")
        trigger_message = (
            self._session.get(BranchMessage, trigger_message_id)
            if isinstance(trigger_message_id, str)
            else None
        )
        message_query = select(BranchMessage).where(
            BranchMessage.branch_id == cycle.branch_id,
            BranchMessage.generation_status == "completed",
        )
        if (
            trigger_message is not None
            and trigger_message.branch_id == cycle.branch_id
        ):
            message_query = message_query.where(
                BranchMessage.sequence <= trigger_message.sequence
            )
        else:
            message_query = message_query.where(
                BranchMessage.created_at <= cycle.input_cutoff_at
            )
        recent_messages = list(
            reversed(
                list(
                    self._session.scalars(
                        message_query.order_by(BranchMessage.sequence.desc()).limit(limit)
                    )
                )
            )
        )
        transcript_message_ids = {message.id for message in recent_messages}

        ordered: list[tuple[datetime, int, dict[str, object]]] = []
        for event in recent_events:
            message_id = event.evidence.get("branch_message_id")
            if isinstance(message_id, str) and message_id in transcript_message_ids:
                continue
            ordered.append(
                (
                    _aware_utc(event.occurred_at),
                    0,
                    self._event_payload(event),
                )
            )
        for message in recent_messages:
            event = events_by_message_id.get(message.id)
            payload = (
                self._event_payload(event)
                if event is not None
                else self._branch_message_payload(message)
            )
            occurred_at = event.occurred_at if event is not None else message.created_at
            ordered.append(
                (
                    _aware_utc(occurred_at),
                    message.sequence + 1,
                    payload,
                )
            )
        ordered.sort(key=lambda item: (item[0], item[1], str(item[2].get("id", ""))))
        return [payload for _, _, payload in ordered[-limit:]]

    def _decision_context(
        self,
        branch: Branch | None,
        *,
        query_text: str = "",
    ) -> dict[str, object]:
        """Give the authoritative cognition grounded persona and memory state."""

        if branch is None:
            return {}
        identity = self._session.scalar(
            select(IdentityKernel).where(
                IdentityKernel.model_version_id == branch.model_version_id
            )
        )
        state = self._session.scalar(
            select(BranchStateVersion).where(
                BranchStateVersion.branch_id == branch.id,
                BranchStateVersion.is_current.is_(True),
            )
        )
        candidate_memories = list(
            self._session.scalars(
                select(BranchMemoryItem)
                .where(
                    BranchMemoryItem.branch_id == branch.id,
                    BranchMemoryItem.review_status == "approved",
                    BranchMemoryItem.valid_to.is_(None),
                    BranchMemoryItem.invalidated_at.is_(None),
                )
                .order_by(
                    BranchMemoryItem.importance.desc(),
                    BranchMemoryItem.created_at.desc(),
                )
                .limit(100)
            )
        )
        memories = _select_relevant_memories(candidate_memories, query_text, limit=6)
        branch_state = (
            {
                "persona_state": dict(state.persona_state),
                "relationship_state": dict(state.relationship_state),
                "user_model": dict(state.user_model),
                "emotional_tendency": dict(state.emotional_tendency),
                "current_goals": dict(state.current_goals),
                "current_concerns": dict(state.current_concerns),
            }
            if state is not None
            else {}
        )
        situational_state = branch.state_snapshot.get("situational_state", {})
        authentic_examples = select_authentic_dialogue_examples(
            self._session,
            branch,
            query_text,
        )
        return {
            "identity_kernel": dict(identity.content) if identity is not None else {},
            "branch_state": branch_state,
            "situational_state": (
                dict(situational_state) if isinstance(situational_state, dict) else {}
            ),
            "approved_memories": [
                {
                    "id": memory.id,
                    "kind": memory.kind,
                    "content": memory.content,
                    "confidence": memory.confidence,
                    "importance": memory.importance,
                    "verification_status": memory.verification_status,
                    "stance": memory.stance,
                }
                for memory in memories
            ],
            "authentic_dialogue_examples": [
                example.as_payload() for example in authentic_examples
            ],
        }

    @staticmethod
    def _branch_message_payload(message: BranchMessage) -> dict[str, object]:
        return {
            "id": f"branch-message:{message.id}",
            "event_type": (
                "agent_expression" if message.role == "assistant" else "user_message"
            ),
            "actor": "self" if message.role == "assistant" else "other",
            "occurred_at": message.created_at.isoformat(),
            "source": "branch_conversation",
            "confidence": 1.0,
            "evidence": {
                "branch_message_id": message.id,
                "content": message.content,
                "role": message.role,
                "message_type": message.type,
                "turn_id": message.turn_id,
                "bubble_index": message.bubble_index,
                "media_asset_id": message.media_asset_id,
            },
        }


def process_due_wakeups(
    session: Session,
    *,
    limit: int = 100,
    now: datetime | None = None,
) -> int:
    """原子消费到期唤醒，并创建唯一的认知任务。"""

    if limit <= 0:
        return 0
    processed_at = now or datetime.now(UTC)
    candidate_ids = list(
        session.scalars(
            select(AgentWakeup.id)
            .where(
                AgentWakeup.status == "scheduled",
                AgentWakeup.wake_at <= processed_at,
            )
            .order_by(AgentWakeup.wake_at.asc(), AgentWakeup.id.asc())
            .limit(limit)
        )
    )
    processed = 0
    for wakeup_id in candidate_ids:
        try:
            claimed = session.execute(
                update(AgentWakeup)
                .where(
                    AgentWakeup.id == wakeup_id,
                    AgentWakeup.status == "scheduled",
                    AgentWakeup.wake_at <= processed_at,
                )
                .values(status="claimed", updated_at=processed_at)
                .execution_options(synchronize_session=False)
            )
        except OperationalError as error:
            session.rollback()
            if "locked" in str(error).lower() or "busy" in str(error).lower():
                continue
            raise
        if claimed.rowcount != 1:
            session.rollback()
            continue

        try:
            wakeup = session.scalar(
                select(AgentWakeup)
                .where(AgentWakeup.id == wakeup_id)
                .execution_options(populate_existing=True)
            )
            if wakeup is None:
                raise RuntimeError("已领取的主体唤醒不存在")
            branch = session.scalar(
                select(Branch).where(
                    Branch.id == wakeup.branch_id,
                    Branch.project_id == wakeup.project_id,
                )
            )
            if branch is None:
                raise BranchScopeError("project/branch 不匹配")

            cognition = ShadowCognitionService(session)
            event = cognition.append_perception_event(
                project_id=wakeup.project_id,
                branch_id=wakeup.branch_id,
                event_type="elapsed_time",
                occurred_at=wakeup.wake_at,
                source="agent_wakeup",
                idempotency_key=f"wakeup:{wakeup.id}",
                visible_through=processed_at,
                evidence={"wakeup_id": wakeup.id, "reason": wakeup.reason},
                model_protocol_version=wakeup.model_protocol_version,
                commit=False,
            )
            state = session.scalar(
                select(MentalStateVersion).where(
                    MentalStateVersion.branch_id == wakeup.branch_id,
                    MentalStateVersion.is_current.is_(True),
                )
            )
            if state is None:
                state = cognition.ensure_initial_mental_state(
                    project_id=wakeup.project_id,
                    branch_id=wakeup.branch_id,
                    state=dict(DEFAULT_INITIAL_MENTAL_STATE),
                    evidence={"source": "agent_wakeup", "wakeup_id": wakeup.id},
                    model_version_id=wakeup.model_version_id or branch.model_version_id,
                    model_protocol_version=wakeup.model_protocol_version,
                    commit=False,
                )
            cycle = cognition.create_pending_cycle(
                project_id=wakeup.project_id,
                branch_id=wakeup.branch_id,
                trigger_event_id=event.id,
                input_cutoff_at=processed_at,
                starting_state_version_id=state.id,
                model_version_id=wakeup.model_version_id or branch.model_version_id,
                evidence={"wakeup_id": wakeup.id},
                model_protocol_version=wakeup.model_protocol_version,
                commit=False,
            )
            from moonlightbox.agent.jobs import COGNITIVE_CYCLE_JOB_KIND

            JobService(session).enqueue_unique(
                COGNITIVE_CYCLE_JOB_KIND,
                {"cycle_id": cycle.id},
                dedupe_key=f"subject-cognitive-cycle:{cycle.id}",
                commit=False,
            )
            completed = session.execute(
                update(AgentWakeup)
                .where(
                    AgentWakeup.id == wakeup.id,
                    AgentWakeup.status == "claimed",
                )
                .values(status="completed", updated_at=processed_at)
                .execution_options(synchronize_session=False)
            )
            if completed.rowcount != 1:
                raise RuntimeError("主体唤醒完成状态更新失败")
            session.commit()
            processed += 1
        except Exception:
            session.rollback()
            raise
    return processed
