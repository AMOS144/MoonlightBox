import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.events.reviewer import (
    EventCandidate,
    EventCandidateReview,
    stable_candidate_key,
)
from moonlightbox.events.windowing import AnalysisWindow, PersistenceContext

ALLOWED_EVENT_TYPES = frozenset(
    {
        "relationship_started",
        "intimacy_increased",
        "commitment",
        "boundary_change",
        "conflict",
        "distancing",
        "reconciliation",
        "separation",
        "reconnection",
    }
)
DECISIVE_EVENT_TYPES = frozenset(
    {
        "relationship_started",
        "commitment",
        "boundary_change",
        "separation",
        "reconciliation",
    }
)
DEFAULT_EVIDENCE_ALIGNMENT_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """硬校验所需的项目归属与时序上下文。"""

    expected_project_id: str
    messages: Mapping[str, NormalizedMessage]
    message_project_ids: Mapping[str, str]
    analysis_window: AnalysisWindow
    persistence_context: PersistenceContext

    def __post_init__(self) -> None:
        source_ids = [message.source_id for message in self.messages.values()]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("ValidationContext 包含重复 source_id")
        window_ids = [message.source_id for message in self.analysis_window.messages]
        if len(window_ids) != len(set(window_ids)):
            raise ValueError("AnalysisWindow 包含重复 source_id")

    @property
    def window_message_ids(self) -> tuple[str, ...]:
        """只从实际分析窗口派生有序消息 ID。"""

        return tuple(message.source_id for message in self.analysis_window.messages)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """单个候选的确定性校验结果。"""

    candidate_key: str
    is_valid: bool
    accepted: bool
    rejection_reasons: tuple[str, ...]
    review_reason: str

    @property
    def status(self) -> Literal["accepted", "rejected"]:
        return "accepted" if self.accepted else "rejected"


def validate_candidate(
    candidate: EventCandidate,
    review: EventCandidateReview,
    context: ValidationContext,
    *,
    evidence_alignment_threshold: float = DEFAULT_EVIDENCE_ALIGNMENT_THRESHOLD,
) -> ValidationResult:
    """校验一个候选；任何畸形字段都转为拒绝结果而不是向外抛错。"""

    try:
        return _validate_candidate(
            candidate,
            review,
            context,
            evidence_alignment_threshold,
        )
    except (AttributeError, TypeError, ValueError):
        return _malformed_result(candidate, review)


def validate_candidates(
    candidates_and_reviews: Sequence[tuple[EventCandidate, EventCandidateReview]],
    context: ValidationContext,
    *,
    evidence_alignment_threshold: float = DEFAULT_EVIDENCE_ALIGNMENT_THRESHOLD,
) -> list[ValidationResult]:
    """逐个校验，确保单个异常候选不会中止整个窗口。"""

    return [
        validate_candidate(
            candidate,
            review,
            context,
            evidence_alignment_threshold=evidence_alignment_threshold,
        )
        for candidate, review in candidates_and_reviews
    ]


def _validate_candidate(
    candidate: EventCandidate,
    review: EventCandidateReview,
    context: ValidationContext,
    evidence_alignment_threshold: float,
) -> ValidationResult:
    reasons: list[str] = []
    candidate_key = candidate.candidate_key
    evidence_ids = _string_ids(candidate.evidence_ids)
    review_evidence_ids = _string_ids(review.evidence_ids)
    event_ids = (candidate.start_message_id, candidate.end_message_id)

    if candidate.type not in ALLOWED_EVENT_TYPES:
        reasons.append("invalid_event_type")
    if candidate_key != stable_candidate_key(candidate):
        reasons.append("unstable_candidate_key")

    referenced_ids = set(event_ids) | set(evidence_ids)
    if any(message_id not in context.messages for message_id in referenced_ids):
        reasons.append("unknown_message_id")
    if any(message_id not in context.messages for message_id in review_evidence_ids):
        reasons.append("unknown_review_evidence")
    window_id_set = set(context.window_message_ids)
    if any(message_id not in window_id_set for message_id in referenced_ids):
        reasons.append("candidate_message_outside_window")

    all_referenced_ids = referenced_ids | set(review_evidence_ids)
    if any(message_id not in context.message_project_ids for message_id in all_referenced_ids):
        reasons.append("missing_project_ownership")
    if any(
        owner != context.expected_project_id
        for message_id in all_referenced_ids
        if (owner := context.message_project_ids.get(message_id)) is not None
    ):
        reasons.append("project_ownership_mismatch")

    start_message = context.messages.get(candidate.start_message_id)
    end_message = context.messages.get(candidate.end_message_id)
    if (
        start_message is not None
        and end_message is not None
        and start_message.timestamp > end_message.timestamp
    ):
        reasons.append("reversed_time_range")

    if not evidence_ids:
        reasons.append("empty_evidence")
    if len(set(evidence_ids)) < 2:
        reasons.append("insufficient_evidence")
    if len(evidence_ids) != len(set(evidence_ids)):
        reasons.append("duplicate_event_evidence")
    if not set(event_ids).issubset(evidence_ids):
        reasons.append("event_bounds_not_in_evidence")
    _validate_window_range(candidate, evidence_ids, context, reasons)

    if _canonical_state(candidate.before_state) == _canonical_state(candidate.after_state):
        reasons.append("unchanged_state")

    if not _is_int_in_range(candidate.conflict_level, 0, 5):
        reasons.append("score_out_of_range")
    if not _is_number_in_range(candidate.state_change_strength, 0, 1):
        reasons.append("score_out_of_range")
    if not _is_number_in_range(candidate.model_confidence, 0, 1):
        reasons.append("score_out_of_range")
    if not _is_number_in_range(review.persistence, 0, 1):
        reasons.append("score_out_of_range")
    if not _is_number_in_range(review.evidence_alignment, 0, 1):
        reasons.append("score_out_of_range")
    if not review.reason.strip():
        reasons.append("empty_review_reason")
    if not review.type_supported:
        reasons.append("unsupported_event_type")
    if (
        not _is_number_in_range(review.evidence_alignment, 0, 1)
        or review.evidence_alignment < evidence_alignment_threshold
    ):
        reasons.append("insufficient_evidence_alignment")

    if review.accepted:
        valid_follow_up_ids = _valid_follow_up_ids(candidate, context)
        valid_follow_up_set = set(valid_follow_up_ids)
        has_real_persistence = (
            _is_number_in_range(review.persistence, 0, 1)
            and review.persistence > 0
            and bool(review_evidence_ids)
            and set(review_evidence_ids).issubset(valid_follow_up_set)
        )
        decisive_supported = _supports_decisive_event(
            candidate,
            review,
            evidence_ids,
            context,
            evidence_alignment_threshold,
        )
        if review.decisive_event and not decisive_supported:
            reasons.append("unsupported_decisive_event")
        if not has_real_persistence and not decisive_supported:
            reasons.append("unsupported_acceptance")
        elif has_real_persistence and not _strictly_preserves_order(
            review_evidence_ids,
            valid_follow_up_ids,
        ):
            reasons.append("unordered_review_evidence")
    else:
        reasons.append("review_rejected")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return ValidationResult(
        candidate_key=candidate_key,
        is_valid=not unique_reasons,
        accepted=review.accepted and not unique_reasons,
        rejection_reasons=unique_reasons,
        review_reason=review.reason,
    )


def _supports_decisive_event(
    candidate: EventCandidate,
    review: EventCandidateReview,
    evidence_ids: tuple[str, ...],
    context: ValidationContext,
    evidence_alignment_threshold: float,
) -> bool:
    """仅允许明确决策及双方行动/回应绕过后续持续性要求。"""

    evidence_messages = [
        context.messages[message_id]
        for message_id in evidence_ids
        if message_id in context.messages
    ]
    return (
        review.decisive_event
        and candidate.type in DECISIVE_EVENT_TYPES
        and review.type_supported
        and review.evidence_alignment >= evidence_alignment_threshold
        and len(evidence_ids) >= 2
        and len({message.sender for message in evidence_messages}) >= 2
    )


def _string_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("证据 ID 必须是列表")
    if any(not isinstance(item, str) or not item for item in value):
        raise TypeError("证据 ID 必须是非空字符串")
    return tuple(value)


def _canonical_state(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("状态必须是字符串")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    without_formatting = "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )
    return re.sub(r"^(?:已经|已)", "", without_formatting, count=1)


def _validate_window_range(
    candidate: EventCandidate,
    evidence_ids: tuple[str, ...],
    context: ValidationContext,
    reasons: list[str],
) -> None:
    positions = {message_id: index for index, message_id in enumerate(context.window_message_ids)}
    start_position = positions.get(candidate.start_message_id)
    end_position = positions.get(candidate.end_message_id)
    if start_position is None or end_position is None:
        return
    if start_position > end_position:
        reasons.append("reversed_window_range")
    evidence_positions = [
        positions[message_id] for message_id in evidence_ids if message_id in positions
    ]
    if len(evidence_positions) == len(evidence_ids) and any(
        left >= right
        for left, right in zip(
            evidence_positions,
            evidence_positions[1:],
            strict=False,
        )
    ):
        reasons.append("unordered_event_evidence")
    if any(position < start_position or position > end_position for position in evidence_positions):
        reasons.append("event_evidence_outside_range")


def _is_int_in_range(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _is_number_in_range(value: Any, minimum: float, maximum: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and minimum <= value <= maximum
    )


def _valid_follow_up_ids(
    candidate: EventCandidate,
    context: ValidationContext,
) -> tuple[str, ...]:
    """按窗口后续、持续性上下文的顺序构造合法证据 ID。"""

    end_message = context.messages.get(candidate.end_message_id)
    if end_message is None:
        return ()
    same_window_messages: tuple[NormalizedMessage, ...] = ()
    for index, message in enumerate(context.analysis_window.messages):
        if message.source_id == candidate.end_message_id:
            same_window_messages = context.analysis_window.messages[index + 1 :]
            break
    ordered_messages = same_window_messages + context.persistence_context.messages
    last_order_key = (end_message.timestamp, end_message.source_id)
    valid_ids: list[str] = []
    for message in ordered_messages:
        order_key = (message.timestamp, message.source_id)
        if order_key <= last_order_key:
            continue
        valid_ids.append(message.source_id)
        last_order_key = order_key
    return tuple(valid_ids)


def _strictly_preserves_order(
    evidence_ids: tuple[str, ...],
    valid_follow_up_ids: tuple[str, ...],
) -> bool:
    positions = {message_id: index for index, message_id in enumerate(valid_follow_up_ids)}
    evidence_positions = [positions[message_id] for message_id in evidence_ids]
    return all(
        left < right
        for left, right in zip(
            evidence_positions,
            evidence_positions[1:],
            strict=False,
        )
    )


def _malformed_result(
    candidate: EventCandidate,
    review: EventCandidateReview,
) -> ValidationResult:
    candidate_key = getattr(candidate, "candidate_key", "")
    review_reason = getattr(review, "reason", "")
    return ValidationResult(
        candidate_key=candidate_key if isinstance(candidate_key, str) else "",
        is_valid=False,
        accepted=False,
        rejection_reasons=("malformed_candidate",),
        review_reason=review_reason if isinstance(review_reason, str) else "",
    )
