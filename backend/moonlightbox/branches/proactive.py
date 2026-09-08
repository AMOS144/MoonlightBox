from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class ProactiveDecision:
    should_speak: bool
    readiness: float
    reason: str
    next_review_at: datetime


@dataclass(frozen=True)
class EndogenousImpetus:
    trigger: str
    score: float
    reason: str


def derive_endogenous_impetus(
    *,
    state: dict[str, object],
    behavioral_rhythm: dict[str, object] | None,
    active_goals: tuple[tuple[str, float], ...] = (),
    recent_reflection_id: str | None = None,
    situational_evidence_id: str | None = None,
    now: datetime | None = None,
) -> EndogenousImpetus | None:
    """Create a new, deduplicated motive from the subject's ongoing life."""

    evaluated_at = now or datetime.now(UTC)
    approach = _mapping(state.get("approach_motivation"))
    open_sequences = state.get("open_sequences")
    if isinstance(open_sequences, list) and open_sequences:
        latest = open_sequences[-1]
        source = latest.get("source_plan_id") if isinstance(latest, dict) else None
        return _prefer_new_impetus(
            EndogenousImpetus(
                trigger=f"open-sequence:{source or 'current'}",
                score=0.68,
                reason="想起上一轮还没有聊完的话题",
            ),
            approach,
        )
    day_key = evaluated_at.date().isoformat()
    if situational_evidence_id:
        return _prefer_new_impetus(
            EndogenousImpetus(
                trigger=f"situational:{situational_evidence_id}",
                score=0.72,
                reason="当前生活状态发生了值得分享的变化",
            ),
            approach,
        )
    if active_goals:
        goal_id, priority = max(active_goals, key=lambda item: item[1])
        return _prefer_new_impetus(
            EndogenousImpetus(
                trigger=f"goal:{goal_id}:{day_key}",
                score=min(0.8, 0.58 + max(0.0, priority) * 0.2),
                reason="持续关注的事情在这一刻重新进入注意中心",
            ),
            approach,
        )
    rhythm = behavioral_rhythm or {}
    sample_count = int(_number(rhythm.get("sample_count"), 0.0))
    proactive_rate = max(0.0, min(1.0, _number(rhythm.get("proactive_turn_rate"), 0.0)))
    active_hours = _active_hours(rhythm.get("active_hours"))
    timezone_offset = int(_number(rhythm.get("timezone_offset_minutes"), 0.0))
    local_hour = (evaluated_at + timedelta(minutes=timezone_offset)).hour
    if (
        sample_count >= 20
        and proactive_rate >= 0.15
        and (not active_hours or local_hour in active_hours)
    ):
        return _prefer_new_impetus(
            EndogenousImpetus(
                trigger=f"habitual-sharing:{day_key}:{local_hour}",
                score=min(0.78, 0.55 + proactive_rate * 0.25),
                reason="符合本人历史节奏的自然分享冲动",
            ),
            approach,
        )
    if recent_reflection_id:
        return _prefer_new_impetus(
            EndogenousImpetus(
                trigger=f"reflection:{recent_reflection_id}:{day_key}",
                score=0.58,
                reason="最近形成的想法让本人想主动联系对方",
            ),
            approach,
        )
    return None


def _unless_consumed(
    impetus: EndogenousImpetus,
    consumed_trigger: object,
) -> EndogenousImpetus | None:
    return None if impetus.trigger == consumed_trigger else impetus


def _prefer_new_impetus(
    impetus: EndogenousImpetus,
    approach: dict[str, object],
) -> EndogenousImpetus | None:
    """Upgrade a stale weak motive while preserving stronger pending intent."""

    candidate = _unless_consumed(impetus, approach.get("consumed_trigger"))
    if candidate is None:
        return None
    current_trigger = approach.get("trigger")
    current_unconsumed = (
        isinstance(current_trigger, str)
        and current_trigger != approach.get("consumed_trigger")
    )
    if not current_unconsumed:
        return candidate
    if current_trigger == candidate.trigger:
        return None
    current_score = _number(approach.get("score"), 0.0)
    return candidate if candidate.score > current_score else None


class ProactiveDecisionEngine:
    """用接近动机与打扰抑制的竞争决定是否主动联系。"""

    def decide(
        self,
        *,
        state: dict[str, object],
        last_message_at: datetime | None,
        unanswered_assistant_count: int,
        behavioral_rhythm: dict[str, object] | None = None,
        now: datetime | None = None,
    ) -> ProactiveDecision:
        evaluated_at = now or datetime.now(UTC)
        approach = _mapping(state.get("approach_motivation"))
        inhibition = _mapping(state.get("inhibition"))
        trigger = approach.get("trigger")
        consumed_trigger = approach.get("consumed_trigger")
        has_new_impetus = isinstance(trigger, str) and trigger != consumed_trigger
        open_sequences = state.get("open_sequences")
        has_open_sequence = isinstance(open_sequences, list) and bool(open_sequences)
        elapsed_hours = (
            max(0.0, (evaluated_at - _aware(last_message_at)).total_seconds() / 3600)
            if last_message_at is not None
            else 24.0
        )
        approach_score = _number(approach.get("score"), 0.0)
        inhibition_score = _number(inhibition.get("score"), 0.0)
        readiness = (
            approach_score
            + (0.15 if has_open_sequence else 0.0)
            + min(0.2, elapsed_hours * 0.05)
            - inhibition_score
            - min(0.6, unanswered_assistant_count * 0.15)
        )
        rhythm = behavioral_rhythm or {}
        rhythm_confidence = max(0.0, min(1.0, _number(rhythm.get("confidence"), 0.0)))
        sample_count = int(_number(rhythm.get("sample_count"), 0.0))
        proactive_rate = max(
            0.0,
            min(1.0, _number(rhythm.get("proactive_turn_rate"), 0.5)),
        )
        active_hours = _active_hours(rhythm.get("active_hours"))
        timezone_offset = int(_number(rhythm.get("timezone_offset_minutes"), 0.0))
        local_hour = (evaluated_at + timedelta(minutes=timezone_offset)).hour
        outside_usual_hours = bool(
            sample_count >= 10 and active_hours and local_hour not in active_hours
        )
        if sample_count >= 10:
            readiness += (proactive_rate - 0.5) * 0.4 * rhythm_confidence
            if outside_usual_hours:
                readiness -= 0.25 * rhythm_confidence
        should_speak = has_new_impetus and readiness >= 0.65
        if should_speak:
            reason = "存在新的关系动机或未完成话题"
            next_review = evaluated_at + timedelta(hours=2)
        elif outside_usual_hours and has_new_impetus:
            reason = "当前不在本人历史上常主动交流的时段"
            next_review = _next_active_review(
                evaluated_at,
                active_hours=active_hours,
                timezone_offset_minutes=timezone_offset,
            )
        elif not has_new_impetus:
            reason = "没有新的心理状态变化，不重复主动发送"
            next_review = evaluated_at + timedelta(hours=6)
        else:
            reason = "打扰抑制暂时高于表达冲动"
            next_review = evaluated_at + timedelta(hours=1)
        return ProactiveDecision(
            should_speak=should_speak,
            readiness=round(readiness, 4),
            reason=reason,
            next_review_at=next_review,
        )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _number(value: object, default: float) -> float:
    return float(value) if isinstance(value, int | float) else default


def _active_hours(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        sorted(
            {
                int(item)
                for item in value
                if isinstance(item, int | float) and 0 <= int(item) <= 23
            }
        )
    )


def _next_active_review(
    evaluated_at: datetime,
    *,
    active_hours: tuple[int, ...],
    timezone_offset_minutes: int,
) -> datetime:
    if not active_hours:
        return evaluated_at + timedelta(hours=2)
    local = evaluated_at + timedelta(minutes=timezone_offset_minutes)
    for hours_ahead in range(1, 25):
        candidate = local + timedelta(hours=hours_ahead)
        if candidate.hour in active_hours:
            rounded = candidate.replace(minute=0, second=0, microsecond=0)
            return rounded - timedelta(minutes=timezone_offset_minutes)
    return evaluated_at + timedelta(hours=6)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
