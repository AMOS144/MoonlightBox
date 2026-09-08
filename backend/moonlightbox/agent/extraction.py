"""私密认知记录的后台结构化提取。"""

import logging
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from moonlightbox.agent.models import (
    AgentGoal,
    AgentIntention,
    AgentWakeup,
    CognitiveCycle,
    MentalStateVersion,
    PrivateCognitionNote,
)
from moonlightbox.branches.memory_proposer import MemoryJsonGenerator
from moonlightbox.branches.models import Branch, uses_authoritative_cognition
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler, JobHandlerError
from moonlightbox.jobs.service import JobService

COGNITIVE_EXTRACTION_JOB_KIND = "subject_cognition_extract"
logger = logging.getLogger(__name__)

COGNITIVE_EXTRACTION_SYSTEM_PROMPT = """
你是私密认知记录的严格结构化提取器。
只能从提供的 private_cognition 原文中提取以下字段：
subjective_feelings、attention_target、desired_actions、mental_state_changes、
goal_changes、intention_changes、next_wakeup。
不得增加、推断或改写原文不存在的事实。原文没有明确依据时，字典返回空对象，
列表返回空数组，next_wakeup 返回 null。
只返回完整 JSON，不要解释，不要增加其他字段。
""".strip()


class CognitiveExtractionResult(BaseModel):
    """严格校验后的私密认知结构。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    subjective_feelings: dict[str, object]
    attention_target: dict[str, object]
    desired_actions: list[dict[str, object]]
    mental_state_changes: dict[str, object]
    goal_changes: list[dict[str, object]]
    intention_changes: list[dict[str, object]]
    next_wakeup: dict[str, object] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_concise_fields(cls, value: object) -> object:
        """只转换原有文字的容器形态，不补充任何新语义。"""

        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for key in (
            "subjective_feelings",
            "attention_target",
            "mental_state_changes",
        ):
            current = normalized.get(key)
            if isinstance(current, str):
                normalized[key] = {"summary": current}
            elif isinstance(current, list):
                normalized[key] = {"items": current}
        for key in ("desired_actions", "goal_changes", "intention_changes"):
            current = normalized.get(key)
            if isinstance(current, str):
                normalized[key] = [{"content": current}]
            elif isinstance(current, list):
                normalized[key] = [
                    {"content": item} if isinstance(item, str) else item
                    for item in current
                ]
        return normalized


class CognitiveExtractor(Protocol):
    """结构化提取器协议。"""

    def extract(self, note: PrivateCognitionNote) -> CognitiveExtractionResult:
        """从单条私密认知记录提取结构。"""


class CognitiveStructureExtractor:
    """使用干净基座 JSON 生成器提取认知结构。"""

    def __init__(self, generator: MemoryJsonGenerator) -> None:
        self._generator = generator

    def extract(self, note: PrivateCognitionNote) -> CognitiveExtractionResult:
        """生成并严格校验结构，任何不合法输出均关闭失败。"""

        raw = self._generator.generate_json(
            model_version_id=note.model_version_id,
            system_prompt=COGNITIVE_EXTRACTION_SYSTEM_PROMPT,
            payload={"private_cognition": note.content},
        )
        try:
            return CognitiveExtractionResult.model_validate(raw)
        except ValidationError as error:
            logger.warning(
                "认知提取结构无效：字段类型=%s 校验错误=%s",
                {key: type(value).__name__ for key, value in raw.items()},
                error.errors(include_input=False),
            )
            raise


def create_cognitive_extraction_handler(
    extractor: CognitiveExtractor,
) -> JobHandler:
    """创建仅更新影子审计字段的结构化提取处理器。"""

    def handle(job_service: JobService, job: Job) -> None:
        session = job_service.session
        cycle_id = _required_scope_value(job, "cycle_id")
        project_id = _required_scope_value(job, "project_id")
        branch_id = _required_scope_value(job, "branch_id")
        note_id = _required_scope_value(job, "note_id")

        cycle = session.get(CognitiveCycle, cycle_id)
        note = session.get(PrivateCognitionNote, note_id)
        if not _scope_is_valid(
            cycle=cycle,
            note=note,
            project_id=project_id,
            branch_id=branch_id,
            note_id=note_id,
        ):
            raise JobHandlerError(
                "invalid_subject_cognition_extraction_job",
                "私密认知提取任务范围无效",
            )
        assert cycle is not None
        assert note is not None

        if cycle.structured_changes.get("extraction_status") == "completed":
            return

        try:
            authoritative = cycle.structured_changes.get("authoritative_extraction")
            result = (
                CognitiveExtractionResult.model_validate(authoritative)
                if isinstance(authoritative, dict)
                else extractor.extract(note)
            )
            note.subjective_feelings = result.subjective_feelings
            note.attention_target = result.attention_target
            note.desired_actions = result.desired_actions
            cycle.structured_changes = {
                "extraction_status": "completed",
                "mental_state_changes": result.mental_state_changes,
                "goal_changes": result.goal_changes,
                "intention_changes": result.intention_changes,
                "next_wakeup": result.next_wakeup,
            }
            branch = session.get(Branch, branch_id)
            if branch is not None and uses_authoritative_cognition(
                branch.subject_agent_mode
            ):
                _apply_active_changes(session, cycle, result)
            session.commit()
        except Exception as error:
            session.rollback()
            error_code = (
                "invalid_extraction_output"
                if isinstance(error, ValidationError)
                else "extraction_generation_failed"
            )
            failed_cycle = session.get(CognitiveCycle, cycle_id)
            if failed_cycle is not None:
                failed_cycle.structured_changes = {
                    "extraction_status": "failed",
                    "error_code": error_code,
                }
                session.commit()
            raise JobHandlerError(
                "subject_cognition_extraction_failed",
                "私密认知结构化提取失败",
            ) from error

    return handle


def _apply_active_changes(
    session: Session,
    cycle: CognitiveCycle,
    result: CognitiveExtractionResult,
) -> None:
    """以起始状态为 CAS 基准提交主动分支的心理演化。"""

    starting = session.get(MentalStateVersion, cycle.starting_state_version_id)
    if starting is None:
        cycle.structured_changes = {
            **cycle.structured_changes,
            "state_apply_status": "stale",
        }
        return
    claimed = session.execute(
        update(MentalStateVersion)
        .where(
            MentalStateVersion.id == starting.id,
            MentalStateVersion.branch_id == cycle.branch_id,
            MentalStateVersion.is_current.is_(True),
        )
        .values(is_current=False)
        .execution_options(synchronize_session=False)
    )
    if claimed.rowcount != 1:
        cycle.structured_changes = {
            **cycle.structured_changes,
            "state_apply_status": "stale",
        }
        return

    merged_state = {
        **dict(starting.state),
        **dict(result.mental_state_changes),
    }
    session.add(
        MentalStateVersion(
            project_id=cycle.project_id,
            branch_id=cycle.branch_id,
            version=starting.version + 1,
            state=merged_state,
            previous_version_id=starting.id,
            source_cycle_id=cycle.id,
            evidence={
                "cycle_id": cycle.id,
                "event_id": cycle.trigger_event_id,
                "model_version_id": cycle.model_version_id,
            },
            is_current=True,
            model_version_id=cycle.model_version_id,
            model_protocol_version=cycle.model_protocol_version,
        )
    )
    session.flush()
    _apply_goal_changes(session, cycle, result.goal_changes)
    _apply_intention_changes(session, cycle, result.intention_changes)
    if result.next_wakeup is not None:
        _apply_wakeup(session, cycle, result.next_wakeup)
    cycle.structured_changes = {
        **cycle.structured_changes,
        "state_apply_status": "applied",
    }


def _apply_goal_changes(
    session: Session,
    cycle: CognitiveCycle,
    changes: list[dict[str, object]],
) -> None:
    for change in changes:
        goal_id = change.get("id")
        goal = (
            session.scalar(
                select(AgentGoal).where(
                    AgentGoal.id == goal_id,
                    AgentGoal.branch_id == cycle.branch_id,
                )
            )
            if isinstance(goal_id, str)
            else None
        )
        content = _required_text(change, "content", "goal")
        goal_type = _optional_text(change, "goal_type", "type") or "temporary"
        status = _optional_text(change, "status") or "active"
        priority = _optional_number(change.get("priority"), default=0.0)
        if goal is None and goal_id is None:
            goal = session.scalar(
                select(AgentGoal).where(
                    AgentGoal.branch_id == cycle.branch_id,
                    AgentGoal.goal_type == goal_type,
                    AgentGoal.status.in_(("active", "suspended", "conflicted")),
                )
            )
        source = {
            "cycle_id": cycle.id,
            "event_id": cycle.trigger_event_id,
            "model_version_id": cycle.model_version_id,
        }
        if goal is None:
            if isinstance(goal_id, str):
                raise ValueError("目标更新引用了其它分支或不存在的目标")
            if status in {"completed", "cancelled"}:
                continue
            goal = AgentGoal(
                project_id=cycle.project_id,
                branch_id=cycle.branch_id,
                goal_type=goal_type,
                content=content,
                source=source,
                priority=priority,
                status=status,
                evidence=source,
                model_version_id=cycle.model_version_id,
                model_protocol_version=cycle.model_protocol_version,
            )
            session.add(goal)
        else:
            goal.goal_type = goal_type
            goal.content = content
            goal.source = source
            goal.priority = priority
            goal.status = status
            goal.evidence = source
            goal.model_version_id = cycle.model_version_id
            goal.model_protocol_version = cycle.model_protocol_version


def _apply_intention_changes(
    session: Session,
    cycle: CognitiveCycle,
    changes: list[dict[str, object]],
) -> None:
    for change in changes:
        intention_id = change.get("id")
        intention = (
            session.scalar(
                select(AgentIntention).where(
                    AgentIntention.id == intention_id,
                    AgentIntention.branch_id == cycle.branch_id,
                )
            )
            if isinstance(intention_id, str)
            else None
        )
        goal_id = change.get("goal_id")
        if goal_id is not None:
            if not isinstance(goal_id, str) or session.scalar(
                select(AgentGoal.id).where(
                    AgentGoal.id == goal_id,
                    AgentGoal.branch_id == cycle.branch_id,
                )
            ) is None:
                raise ValueError("意图引用了其它分支或不存在的目标")
        content = _required_text(change, "content", "intention")
        intention_type = (
            _optional_text(change, "intention_type", "type") or "consider"
        )
        status = _optional_text(change, "status") or "active"
        if intention is None and intention_id is None:
            intention = session.scalar(
                select(AgentIntention).where(
                    AgentIntention.branch_id == cycle.branch_id,
                    AgentIntention.intention_type == intention_type,
                    AgentIntention.status.in_(("active", "suspended")),
                )
            )
        expression_plan = change.get("expression_plan")
        if not isinstance(expression_plan, dict):
            expression_plan = {}
        evidence = {
            "cycle_id": cycle.id,
            "event_id": cycle.trigger_event_id,
            "model_version_id": cycle.model_version_id,
        }
        if intention is None:
            if isinstance(intention_id, str):
                raise ValueError("意图更新引用了其它分支或不存在的意图")
            if status in {"fulfilled", "cancelled"}:
                continue
            intention = AgentIntention(
                project_id=cycle.project_id,
                branch_id=cycle.branch_id,
                intention_type=intention_type,
                content=content,
                goal_id=goal_id,
                trigger_event_id=cycle.trigger_event_id,
                expression_plan=dict(expression_plan),
                evidence=evidence,
                status=status,
                model_version_id=cycle.model_version_id,
                model_protocol_version=cycle.model_protocol_version,
            )
            session.add(intention)
        else:
            intention.intention_type = intention_type
            intention.content = content
            intention.goal_id = goal_id
            intention.trigger_event_id = cycle.trigger_event_id
            intention.expression_plan = dict(expression_plan)
            intention.evidence = evidence
            intention.status = status
            intention.model_version_id = cycle.model_version_id
            intention.model_protocol_version = cycle.model_protocol_version


def _apply_wakeup(
    session: Session,
    cycle: CognitiveCycle,
    change: dict[str, object],
) -> None:
    wake_at_value = _required_text(change, "wake_at")
    wake_at = datetime.fromisoformat(wake_at_value)
    if wake_at.tzinfo is None:
        wake_at = wake_at.replace(tzinfo=UTC)
    reason = _required_text(change, "reason")
    idempotency_key = _required_text(change, "idempotency_key")
    goal_id = change.get("goal_id")
    if goal_id is not None:
        if not isinstance(goal_id, str) or session.scalar(
            select(AgentGoal.id).where(
                AgentGoal.id == goal_id,
                AgentGoal.branch_id == cycle.branch_id,
            )
        ) is None:
            raise ValueError("唤醒引用了其它分支或不存在的目标")
    wakeup = session.scalar(
        select(AgentWakeup).where(
            AgentWakeup.branch_id == cycle.branch_id,
            AgentWakeup.idempotency_key == idempotency_key,
        )
    )
    evidence = {
        "cycle_id": cycle.id,
        "event_id": cycle.trigger_event_id,
        "model_version_id": cycle.model_version_id,
    }
    if wakeup is None:
        wakeup = AgentWakeup(
            project_id=cycle.project_id,
            branch_id=cycle.branch_id,
            wake_at=wake_at,
            reason=reason,
            goal_id=goal_id,
            event_id=cycle.trigger_event_id,
            idempotency_key=idempotency_key,
            evidence=evidence,
            model_version_id=cycle.model_version_id,
            model_protocol_version=cycle.model_protocol_version,
        )
        session.add(wakeup)
    else:
        wakeup.wake_at = wake_at
        wakeup.reason = reason
        wakeup.goal_id = goal_id
        wakeup.event_id = cycle.trigger_event_id
        wakeup.evidence = evidence
        wakeup.status = "scheduled"
        wakeup.model_version_id = cycle.model_version_id
        wakeup.model_protocol_version = cycle.model_protocol_version


def _required_text(
    value: dict[str, object],
    *keys: str,
) -> str:
    result = _optional_text(value, *keys)
    if result is None:
        raise ValueError(f"认知变化缺少字段：{'/'.join(keys)}")
    return result


def _optional_text(
    value: dict[str, object],
    *keys: str,
) -> str | None:
    for key in keys:
        result = value.get(key)
        if isinstance(result, str) and result.strip():
            return result.strip()
    return None


def _optional_number(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        raise ValueError("布尔值不能作为数值")
    if isinstance(value, int | float):
        return float(value)
    if value is None:
        return default
    raise ValueError("认知变化数值字段无效")


def _required_scope_value(job: Job, key: str) -> str:
    value = job.payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JobHandlerError(
            "invalid_subject_cognition_extraction_job",
            "私密认知提取任务参数无效",
        )
    return value


def _scope_is_valid(
    *,
    cycle: CognitiveCycle | None,
    note: PrivateCognitionNote | None,
    project_id: str,
    branch_id: str,
    note_id: str,
) -> bool:
    if cycle is None or note is None:
        return False
    return (
        cycle.status == "succeeded"
        and cycle.project_id == project_id
        and cycle.branch_id == branch_id
        and cycle.private_note_id == note_id
        and note.project_id == project_id
        and note.branch_id == branch_id
        and note.trigger_event_id == cycle.trigger_event_id
    )
