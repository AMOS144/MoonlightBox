from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from moonlightbox.events.normalization import NormalizedMessage, normalize_message
from moonlightbox.events.v3_reviewer import (
    V3CandidateIntegrityError,
    V3CandidateReview,
    V3EventCandidate,
    stable_v3_candidate_key,
)
from moonlightbox.events.validation import ValidationContext
from moonlightbox.imports.types import ImportedMessage, MessageKind

V3_REJECTION_REASON_ORDER = (
    "message_not_found",
    "message_project_mismatch",
    "invalid_time_order",
    "event_bounds_not_in_evidence",
    "event_evidence_outside_range",
    "noise_only_evidence",
    "unchanged_relationship_state",
    "facts_unsupported",
    "occurrence_unsupported",
    "unconfirmed_plan",
)

_NON_DISPLAYABLE_CONTENT = frozenset(
    {
        "[图片]",
        "[语音]",
        "[视频]",
        "[文件]",
        "[表情]",
        "[动画表情]",
    }
)


@dataclass(frozen=True, slots=True)
class V3ValidationResult:
    """V3 候选的确定性硬校验结果。"""

    candidate_key: str
    is_valid: bool
    rejection_reasons: tuple[str, ...]
    review_reason: str = ""


@dataclass(frozen=True, slots=True)
class ValidationIndex:
    """批次共享的只读 V3 校验索引。"""

    expected_project_id: str
    message_ids: frozenset[str]
    window_ids: frozenset[str]
    review_allowed_ids: frozenset[str]
    order_keys: Mapping[str, tuple[datetime, str]]
    project_ids: Mapping[str, str]
    noise_flags: Mapping[str, bool]


class V3ValidationDataError(ValueError):
    """单个候选或复核包含预期内的数据结构错误。"""


def build_validation_index(context: ValidationContext) -> ValidationIndex:
    """单次遍历上下文，构建批量复用的不可变索引。"""

    messages = dict(context.messages)
    window_messages = tuple(context.analysis_window.messages)
    persistence_messages = list(context.persistence_context.remaining_session_messages)
    for session in context.persistence_context.following_sessions:
        persistence_messages.extend(session.messages)

    window_ids = frozenset(message.source_id for message in window_messages)
    persistence_ids = frozenset(message.source_id for message in persistence_messages)
    order_keys = {message_id: _message_order(message) for message_id, message in messages.items()}
    noise_flags = {
        message_id: _is_noise_only_message(message) for message_id, message in messages.items()
    }
    return ValidationIndex(
        expected_project_id=context.expected_project_id,
        message_ids=frozenset(messages),
        window_ids=window_ids,
        review_allowed_ids=window_ids | persistence_ids,
        order_keys=MappingProxyType(order_keys),
        project_ids=MappingProxyType(dict(context.message_project_ids)),
        noise_flags=MappingProxyType(noise_flags),
    )


def validate_v3_candidate(
    candidate: V3EventCandidate,
    review: V3CandidateReview,
    context: ValidationContext,
) -> V3ValidationResult:
    """在公开入口重验完整性，并把畸形单项隔离为安全结果。"""

    index = build_validation_index(context)
    return _validate_with_index(candidate, review, index)


def _validate_with_index(
    candidate: V3EventCandidate,
    review: V3CandidateReview,
    index: ValidationIndex,
) -> V3ValidationResult:
    candidate_key = _safe_candidate_key(candidate)
    try:
        validated_candidate = _revalidate_candidate(candidate)
    except (ValidationError, V3CandidateIntegrityError):
        return _integrity_failure(candidate_key, "candidate_integrity_invalid")

    try:
        validated_review = _revalidate_review(review)
    except (ValidationError, V3ValidationDataError):
        return _integrity_failure(candidate_key, "review_integrity_invalid")

    try:
        return _validate_candidate(
            validated_candidate,
            validated_review,
            index,
        )
    except V3ValidationDataError:
        return _integrity_failure(candidate_key, "validation_failed")


def validate_v3_candidates(
    candidates_and_reviews: Sequence[tuple[V3EventCandidate, V3CandidateReview]],
    context: ValidationContext,
) -> list[V3ValidationResult]:
    """逐项校验，确保一个畸形候选不会中断整个批次。"""

    index = build_validation_index(context)
    return [
        _validate_with_index(candidate, review, index)
        for candidate, review in candidates_and_reviews
    ]


def validate_candidate(
    candidate: V3EventCandidate,
    review: V3CandidateReview,
    context: ValidationContext,
) -> V3ValidationResult:
    """提供与 V2 校验模块一致的单项入口名称。"""

    return validate_v3_candidate(candidate, review, context)


def validate_candidates(
    candidates_and_reviews: Sequence[tuple[V3EventCandidate, V3CandidateReview]],
    context: ValidationContext,
) -> list[V3ValidationResult]:
    """提供与 V2 校验模块一致的批量入口名称。"""

    return validate_v3_candidates(candidates_and_reviews, context)


def _revalidate_candidate(candidate: V3EventCandidate) -> V3EventCandidate:
    try:
        payload = dict(vars(candidate))
    except TypeError:
        raise V3CandidateIntegrityError from None
    provided_key = payload.get("candidate_key")
    validated = V3EventCandidate.model_validate(payload)
    if (
        not isinstance(provided_key, str)
        or provided_key != validated.candidate_key
        or provided_key != stable_v3_candidate_key(validated)
    ):
        raise V3CandidateIntegrityError
    return validated


def _revalidate_review(review: V3CandidateReview) -> V3CandidateReview:
    try:
        payload = dict(vars(review))
    except TypeError:
        raise V3ValidationDataError("复核模型结构无效") from None
    return V3CandidateReview.model_validate(payload)


def _validate_candidate(
    candidate: V3EventCandidate,
    review: V3CandidateReview,
    index: ValidationIndex,
) -> V3ValidationResult:
    candidate_evidence_ids = _stable_unique(candidate.evidence_ids)
    review_evidence_ids = _stable_unique(review.evidence_ids)
    bound_ids = (candidate.start_message_id, candidate.end_message_id)
    all_referenced_ids = _stable_unique((*bound_ids, *candidate_evidence_ids, *review_evidence_ids))
    reasons: set[str] = set()

    if any(message_id not in index.message_ids for message_id in all_referenced_ids):
        reasons.add("message_not_found")

    if any(
        index.project_ids.get(message_id) != index.expected_project_id
        for message_id in all_referenced_ids
        if message_id in index.message_ids
    ):
        reasons.add("message_project_mismatch")

    if any(
        message_id not in index.window_ids for message_id in (*bound_ids, *candidate_evidence_ids)
    ):
        reasons.add("event_evidence_outside_range")
    if any(message_id not in index.review_allowed_ids for message_id in review_evidence_ids):
        reasons.add("event_evidence_outside_range")

    if not _evidence_preserves_time_order(
        candidate_evidence_ids,
        index.order_keys,
    ) or not _evidence_preserves_time_order(
        review_evidence_ids,
        index.order_keys,
    ):
        reasons.add("invalid_time_order")

    start_order = index.order_keys.get(candidate.start_message_id)
    end_order = index.order_keys.get(candidate.end_message_id)
    if start_order is not None and end_order is not None:
        if start_order > end_order:
            reasons.add("invalid_time_order")
        _validate_candidate_evidence_range(
            candidate_evidence_ids,
            candidate.start_message_id,
            candidate.end_message_id,
            index.order_keys,
            reasons,
        )

    if not set(bound_ids).issubset(candidate_evidence_ids):
        reasons.add("event_bounds_not_in_evidence")

    known_evidence_noise_flags = [
        index.noise_flags[message_id]
        for message_id in candidate_evidence_ids
        if message_id in index.noise_flags
    ]
    if known_evidence_noise_flags and all(known_evidence_noise_flags):
        reasons.add("noise_only_evidence")

    # A relationship node must describe a transition, not merely a warm or
    # difficult moment inside the relationship's existing baseline.  This is
    # deliberately a hard invariant: a model assigning high soft scores must
    # not turn repeated "I love you", routine goodnights, or an isolated mood
    # into a new point on the timeline while reporting identical states.
    if (
        candidate.lane == "relationship"
        and candidate.before_state is not None
        and candidate.after_state is not None
        and _canonical_state(candidate.before_state)
        == _canonical_state(candidate.after_state)
    ):
        reasons.add("unchanged_relationship_state")

    if not review.facts_supported:
        reasons.add("facts_unsupported")
    if candidate.event_status == "occurred" and not review.occurrence_supported:
        reasons.add("occurrence_unsupported")
    if candidate.event_status == "confirmed" and not review.bilateral_confirmation:
        reasons.add("unconfirmed_plan")

    ordered_reasons = tuple(reason for reason in V3_REJECTION_REASON_ORDER if reason in reasons)
    return V3ValidationResult(
        candidate_key=candidate.candidate_key,
        is_valid=not ordered_reasons,
        rejection_reasons=ordered_reasons,
        review_reason=review.reason,
    )


def _validate_candidate_evidence_range(
    evidence_ids: tuple[str, ...],
    start_message_id: str,
    end_message_id: str,
    order_keys: Mapping[str, tuple[datetime, str]],
    reasons: set[str],
) -> None:
    start_order = order_keys.get(start_message_id)
    end_order = order_keys.get(end_message_id)
    if start_order is None or end_order is None:
        return
    if start_order > end_order:
        reasons.add("invalid_time_order")
        return
    if any(
        order_keys[message_id] < start_order or order_keys[message_id] > end_order
        for message_id in evidence_ids
        if message_id in order_keys
    ):
        reasons.add("event_evidence_outside_range")


def _evidence_preserves_time_order(
    evidence_ids: tuple[str, ...],
    order_keys: Mapping[str, tuple[datetime, str]],
) -> bool:
    evidence_order_keys = [
        order_keys[message_id] for message_id in evidence_ids if message_id in order_keys
    ]
    return all(
        left <= right
        for left, right in zip(
            evidence_order_keys,
            evidence_order_keys[1:],
            strict=False,
        )
    )


def _message_order(message: NormalizedMessage) -> tuple[datetime, str]:
    return (message.timestamp, message.source_id)


def _stable_unique(values: Sequence[str] | Any) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise V3ValidationDataError("证据 ID 必须是序列")
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            raise V3ValidationDataError("证据 ID 必须是非空字符串")
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return tuple(unique_values)


def _canonical_state(value: str) -> str:
    return "".join(value.casefold().split())


def _is_noise_only_message(message: Any) -> bool:
    if not all(
        hasattr(message, field_name)
        for field_name in ("source_id", "timestamp", "sender", "kind", "content")
    ):
        return True
    if message.kind is MessageKind.SYSTEM:
        return True
    imported = ImportedMessage(
        source_id=message.source_id,
        timestamp=message.timestamp,
        sender=message.sender,
        kind=message.kind,
        content=message.content,
        raw={},
    )
    normalized = normalize_message(imported)
    return normalized is None or normalized.content in _NON_DISPLAYABLE_CONTENT


def _safe_candidate_key(candidate: object) -> str:
    candidate_key = getattr(candidate, "candidate_key", "")
    return candidate_key if isinstance(candidate_key, str) else ""


def _integrity_failure(
    candidate_key: str,
    review_reason: str,
) -> V3ValidationResult:
    return V3ValidationResult(
        candidate_key=candidate_key,
        is_valid=False,
        rejection_reasons=(),
        review_reason=review_reason,
    )
