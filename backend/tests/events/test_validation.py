from datetime import datetime, timedelta
from typing import Any

import pytest
from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.events.windowing import AnalysisWindow, PersistenceContext
from moonlightbox.imports.types import MessageKind

BASE_TIME = datetime(2026, 7, 18, 12, 0)


def make_message(source_id: str, minutes: int) -> NormalizedMessage:
    return NormalizedMessage(
        source_id=source_id,
        timestamp=BASE_TIME + timedelta(minutes=minutes),
        sender="甲" if minutes % 2 == 0 else "乙",
        kind=MessageKind.TEXT,
        content=f"消息 {source_id}",
    )


def make_candidate(**updates: Any) -> Any:
    from moonlightbox.events.reviewer import EventCandidate, stable_candidate_key

    payload: dict[str, Any] = {
        "candidate_key": "relationship_started:m1:m2",
        "type": "relationship_started",
        "start_message_id": "m1",
        "end_message_id": "m2",
        "before_state": "互相试探",
        "after_state": "正式交往",
        "emotion_labels": ["期待"],
        "topic": "确认关系",
        "conflict_level": 0,
        "state_change_strength": 0.9,
        "model_confidence": 0.85,
        "reason": "双方明确确认关系",
        "evidence_ids": ["m1", "m2"],
    }
    payload.update(updates)
    candidate = EventCandidate.model_construct(**payload)
    if candidate.evidence_ids is None:
        return candidate
    return candidate.model_copy(update={"candidate_key": stable_candidate_key(candidate)})


def make_review(**updates: Any) -> Any:
    from moonlightbox.events.reviewer import EventCandidateReview

    payload: dict[str, Any] = {
        "accepted": True,
        "type_supported": True,
        "evidence_alignment": 0.9,
        "decisive_event": False,
        "persistence": 0.8,
        "evidence_ids": ["m3"],
        "reason": "后续仍按新关系状态互动",
    }
    payload.update(updates)
    return EventCandidateReview.model_construct(**payload)


@pytest.fixture
def allowed_messages() -> dict[str, NormalizedMessage]:
    return {
        "m0": make_message("m0", -1),
        "m1": make_message("m1", 0),
        "m2": make_message("m2", 1),
        "m3": make_message("m3", 2),
        "m4": make_message("m4", 3),
    }


@pytest.fixture
def validation_context(allowed_messages: dict[str, NormalizedMessage]) -> Any:
    from moonlightbox.events.validation import ValidationContext

    window_messages = tuple(allowed_messages[key] for key in ("m0", "m1", "m2", "m3"))
    window = AnalysisWindow(
        window_id="window-1",
        session_id="session-1",
        window_ordinal=0,
        start_message_ordinal=0,
        end_message_ordinal=3,
        messages=window_messages,
        start_message_id="m0",
        end_message_id="m3",
        character_count=sum(len(message.content) for message in window_messages),
    )
    return ValidationContext(
        expected_project_id="project-1",
        messages=allowed_messages,
        message_project_ids={message_id: "project-1" for message_id in allowed_messages},
        analysis_window=window,
        persistence_context=PersistenceContext(
            remaining_session_messages=(allowed_messages["m4"],),
            following_sessions=(),
        ),
    )


@pytest.mark.parametrize(
    ("candidate_updates", "review_updates", "expected_issue"),
    [
        ({"evidence_ids": ["m1", "m99"]}, {}, "unknown_message_id"),
        (
            {"end_message_id": "outside", "evidence_ids": ["m1", "outside"]},
            {},
            "unknown_message_id",
        ),
        (
            {
                "start_message_id": "m2",
                "end_message_id": "m1",
                "evidence_ids": ["m2", "m1"],
            },
            {},
            "reversed_time_range",
        ),
        ({"evidence_ids": []}, {}, "empty_evidence"),
        (
            {
                "start_message_id": "m1",
                "end_message_id": "m1",
                "evidence_ids": ["m1"],
            },
            {},
            "insufficient_evidence",
        ),
        ({"before_state": " 正式交往 ", "after_state": "正式交往"}, {}, "unchanged_state"),
        ({}, {"evidence_ids": ["m99"]}, "unknown_review_evidence"),
        ({}, {"persistence": 1.2}, "score_out_of_range"),
        ({"conflict_level": 6}, {}, "score_out_of_range"),
        ({"type": "ordinary_gap"}, {}, "invalid_event_type"),
    ],
)
def test_hard_validation_rejects_invalid_candidate_without_raising(
    validation_context: Any,
    candidate_updates: dict[str, Any],
    review_updates: dict[str, Any],
    expected_issue: str,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(**candidate_updates),
        make_review(**review_updates),
        validation_context,
    )

    assert result.accepted is False
    assert expected_issue in result.rejection_reasons


def test_accepted_review_requires_real_follow_up_persistence_evidence(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    validation_context = validation_context.__class__(
        expected_project_id=validation_context.expected_project_id,
        messages=validation_context.messages,
        message_project_ids=validation_context.message_project_ids,
        analysis_window=validation_context.analysis_window,
        persistence_context=PersistenceContext(
            remaining_session_messages=(),
            following_sessions=(),
        ),
    )
    result = validate_candidate(
        make_candidate(),
        make_review(evidence_ids=[], persistence=0.8),
        validation_context,
    )

    assert result.accepted is False
    assert "unsupported_acceptance" in result.rejection_reasons


@pytest.mark.parametrize(
    ("review_updates", "expected_issue"),
    [
        ({"type_supported": False}, "unsupported_event_type"),
        ({"evidence_alignment": 0.69}, "insufficient_evidence_alignment"),
    ],
)
def test_semantic_review_gate_rejects_unsupported_or_misaligned_evidence(
    validation_context: Any,
    review_updates: dict[str, Any],
    expected_issue: str,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(type="conflict"),
        make_review(**review_updates),
        validation_context,
    )

    assert result.accepted is False
    assert expected_issue in result.rejection_reasons


def test_decisive_separation_with_action_and_response_can_end_the_chat(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import ValidationContext, validate_candidate

    context = ValidationContext(
        expected_project_id=validation_context.expected_project_id,
        messages=validation_context.messages,
        message_project_ids=validation_context.message_project_ids,
        analysis_window=validation_context.analysis_window,
        persistence_context=PersistenceContext(
            remaining_session_messages=(),
            following_sessions=(),
        ),
    )
    result = validate_candidate(
        make_candidate(
            type="separation",
            before_state="恋爱关系",
            after_state="明确分手",
        ),
        make_review(
            decisive_event=True,
            persistence=0,
            evidence_ids=[],
            reason="一方明确分手，另一方回应确认",
        ),
        context,
    )

    assert result.accepted is True


def test_conflict_cannot_claim_decisive_event_to_bypass_persistence(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import ValidationContext, validate_candidate

    context = ValidationContext(
        expected_project_id=validation_context.expected_project_id,
        messages=validation_context.messages,
        message_project_ids=validation_context.message_project_ids,
        analysis_window=validation_context.analysis_window,
        persistence_context=PersistenceContext(
            remaining_session_messages=(),
            following_sessions=(),
        ),
    )
    result = validate_candidate(
        make_candidate(type="conflict"),
        make_review(
            decisive_event=True,
            persistence=0,
            evidence_ids=[],
            reason="模型声称这是决定性冲突",
        ),
        context,
    )

    assert result.accepted is False
    assert "unsupported_decisive_event" in result.rejection_reasons


def test_start_and_end_must_both_be_evidence(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(evidence_ids=["m1", "m3"]),
        make_review(),
        validation_context,
    )

    assert result.accepted is False
    assert "event_bounds_not_in_evidence" in result.rejection_reasons


@pytest.mark.parametrize(
    ("evidence_ids", "expected_issue"),
    [
        (["m1", "m2", "m4"], "candidate_message_outside_window"),
        (["m1", "m1", "m2"], "duplicate_event_evidence"),
        (["m2", "m1"], "unordered_event_evidence"),
        (["m0", "m1", "m2"], "event_evidence_outside_range"),
    ],
)
def test_event_evidence_must_be_unique_ordered_and_inside_window_range(
    validation_context: Any,
    evidence_ids: list[str],
    expected_issue: str,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(evidence_ids=evidence_ids),
        make_review(),
        validation_context,
    )

    assert result.accepted is False
    assert expected_issue in result.rejection_reasons


def test_window_ids_are_derived_only_from_analysis_window(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    assert validation_context.window_message_ids == ("m0", "m1", "m2", "m3")

    result = validate_candidate(
        make_candidate(evidence_ids=["m1", "m2", "m4"]),
        make_review(),
        validation_context,
    )

    assert result.accepted is False
    assert "candidate_message_outside_window" in result.rejection_reasons


def test_candidate_bounds_must_follow_window_order(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(
            start_message_id="m2",
            end_message_id="m1",
            evidence_ids=["m2", "m1"],
        ),
        make_review(),
        validation_context,
    )

    assert result.accepted is False
    assert "reversed_window_range" in result.rejection_reasons


def test_same_window_message_after_candidate_end_is_valid_follow_up(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(),
        make_review(evidence_ids=["m3"]),
        validation_context,
    )

    assert result.accepted is True


def test_message_before_candidate_end_is_not_valid_follow_up(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(),
        make_review(evidence_ids=["m0"]),
        validation_context,
    )

    assert result.accepted is False
    assert "unsupported_acceptance" in result.rejection_reasons


def test_persistence_context_cannot_reclassify_earlier_message_as_follow_up(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import ValidationContext, validate_candidate

    context = ValidationContext(
        expected_project_id=validation_context.expected_project_id,
        messages=validation_context.messages,
        message_project_ids=validation_context.message_project_ids,
        analysis_window=validation_context.analysis_window,
        persistence_context=PersistenceContext(
            remaining_session_messages=(validation_context.messages["m0"],),
            following_sessions=(),
        ),
    )

    result = validate_candidate(
        make_candidate(),
        make_review(evidence_ids=["m0"]),
        context,
    )

    assert result.accepted is False
    assert "unsupported_acceptance" in result.rejection_reasons


def test_review_evidence_must_preserve_follow_up_time_order(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(),
        make_review(evidence_ids=["m4", "m3"]),
        validation_context,
    )

    assert result.accepted is False
    assert "unordered_review_evidence" in result.rejection_reasons


def test_conservative_state_canonicalization_handles_prefix_and_punctuation(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(
            before_state=" 正式，交往 ",
            after_state="已经正式交往。",
        ),
        make_review(),
        validation_context,
    )

    assert result.accepted is False
    assert "unchanged_state" in result.rejection_reasons


def test_review_reason_is_required_and_rejected_result_is_explicit(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(),
        make_review(accepted=False, persistence=0, evidence_ids=[], reason="  "),
        validation_context,
    )

    assert result.accepted is False
    assert result.status == "rejected"
    assert "empty_review_reason" in result.rejection_reasons
    assert "review_rejected" in result.rejection_reasons


@pytest.mark.parametrize(
    ("message_id", "owner", "expected_issue"),
    [
        ("m2", "project-2", "project_ownership_mismatch"),
        ("m3", None, "missing_project_ownership"),
    ],
)
def test_all_referenced_messages_require_expected_project_ownership(
    validation_context: Any,
    message_id: str,
    owner: str | None,
    expected_issue: str,
) -> None:
    from moonlightbox.events.validation import ValidationContext, validate_candidate

    owners = dict(validation_context.message_project_ids)
    if owner is None:
        owners.pop(message_id)
    else:
        owners[message_id] = owner
    context = ValidationContext(
        expected_project_id=validation_context.expected_project_id,
        messages=validation_context.messages,
        message_project_ids=owners,
        analysis_window=validation_context.analysis_window,
        persistence_context=validation_context.persistence_context,
    )

    result = validate_candidate(make_candidate(), make_review(), context)

    assert result.accepted is False
    assert expected_issue in result.rejection_reasons


@pytest.mark.parametrize(
    ("event_type", "before_state", "after_state"),
    [
        ("relationship_started", "互相试探", "正式交往"),
        ("distancing", "每天联系", "持续减少联系"),
        ("reconciliation", "冲突冷淡", "恢复沟通"),
        ("boundary_change", "随时联系", "约定工作时不打扰"),
    ],
)
def test_hard_validation_accepts_supported_persistent_changes(
    validation_context: Any,
    event_type: str,
    before_state: str,
    after_state: str,
) -> None:
    from moonlightbox.events.validation import validate_candidate

    result = validate_candidate(
        make_candidate(
            type=event_type,
            candidate_key=f"{event_type}:m1:m2",
            before_state=before_state,
            after_state=after_state,
        ),
        make_review(),
        validation_context,
    )

    assert result.is_valid is True
    assert result.accepted is True
    assert result.rejection_reasons == ()


def test_batch_validation_keeps_processing_after_one_bad_candidate(
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import validate_candidates

    results = validate_candidates(
        [
            (make_candidate(evidence_ids=None), make_review()),
            (make_candidate(), make_review()),
        ],
        validation_context,
    )

    assert len(results) == 2
    assert results[0].accepted is False
    assert "malformed_candidate" in results[0].rejection_reasons
    assert results[1].accepted is True


def test_validation_context_rejects_duplicate_source_ids(
    allowed_messages: dict[str, NormalizedMessage],
    validation_context: Any,
) -> None:
    from moonlightbox.events.validation import ValidationContext

    duplicate_messages = dict(allowed_messages)
    duplicate_messages["alias"] = allowed_messages["m1"]

    with pytest.raises(ValueError, match="重复"):
        ValidationContext(
            expected_project_id="project-1",
            messages=duplicate_messages,
            message_project_ids={message_id: "project-1" for message_id in duplicate_messages},
            analysis_window=validation_context.analysis_window,
            persistence_context=validation_context.persistence_context,
        )
