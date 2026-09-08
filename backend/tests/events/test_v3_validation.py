from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import pytest
from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.events.v3_reviewer import V3CandidateReview, V3EventCandidate
from moonlightbox.events.validation import ValidationContext
from moonlightbox.events.windowing import AnalysisWindow, PersistenceContext
from moonlightbox.imports.types import MessageKind

BASE_TIME = datetime(2026, 7, 19, 12, 0)


class CountingMessages(tuple[NormalizedMessage, ...]):
    """记录消息序列被完整遍历的次数。"""

    iterations: int

    def __new__(
        cls,
        messages: tuple[NormalizedMessage, ...],
    ) -> "CountingMessages":
        instance = super().__new__(cls, messages)
        instance.iterations = 0
        return instance

    def __iter__(self) -> Iterator[NormalizedMessage]:
        self.iterations += 1
        return super().__iter__()


class ExplodingMessages(tuple[NormalizedMessage, ...]):
    """模拟上下文基础设施在遍历时发生系统错误。"""

    def __iter__(self) -> Iterator[NormalizedMessage]:
        raise RuntimeError("context traversal failed")


def make_message(
    source_id: str,
    minutes: int,
    *,
    content: str | None = None,
    kind: MessageKind = MessageKind.TEXT,
) -> NormalizedMessage:
    return NormalizedMessage(
        source_id=source_id,
        timestamp=BASE_TIME + timedelta(minutes=minutes),
        sender="甲" if minutes % 2 == 0 else "乙",
        kind=kind,
        content=content if content is not None else f"真实消息 {source_id}",
    )


@pytest.fixture
def validation_context() -> ValidationContext:
    messages = {
        "m0": make_message("m0", 0),
        "m1": make_message("m1", 1),
        "m2": make_message("m2", 2),
        "m3": make_message("m3", 3),
        "m4": make_message("m4", 4),
        "m5": make_message("m5", 5),
        "xml": make_message("xml", 2, content="<msg><title>内部协议</title></msg>"),
        "system": make_message(
            "system",
            3,
            content="以下为新消息",
            kind=MessageKind.SYSTEM,
        ),
    }
    window_messages = tuple(
        messages[key] for key in ("m0", "m1", "xml", "m2", "system", "m3", "m4")
    )
    return ValidationContext(
        expected_project_id="project-1",
        messages=messages,
        message_project_ids={key: "project-1" for key in messages},
        analysis_window=AnalysisWindow(
            window_id="window-1",
            session_id="session-1",
            window_ordinal=0,
            start_message_ordinal=0,
            end_message_ordinal=len(window_messages) - 1,
            messages=window_messages,
            start_message_id="m0",
            end_message_id="m4",
            character_count=sum(len(message.content) for message in window_messages),
        ),
        persistence_context=PersistenceContext(
            remaining_session_messages=(messages["m5"],),
            following_sessions=(),
        ),
    )


def make_candidate(**updates: Any) -> V3EventCandidate:
    payload: dict[str, Any] = {
        "candidate_key": "由本地重算",
        "lane": "shared_experience",
        "type": "outing",
        "title": "共同看展",
        "event_status": "occurred",
        "start_message_id": "m1",
        "end_message_id": "m3",
        "summary": "双方共同看展并在事后回顾",
        "before_state": None,
        "after_state": None,
        "emotion_labels": ["开心"],
        "topic": "看展",
        "conflict_level": 0,
        "event_significance": 0.8,
        "relationship_impact": 0.6,
        "model_confidence": 0.9,
        "reason": "消息直接支持共同经历",
        "evidence_ids": ["m1", "m2", "m3"],
    }
    payload.update(updates)
    return V3EventCandidate.model_validate(payload)


def make_review(**updates: Any) -> V3CandidateReview:
    payload: dict[str, Any] = {
        "facts_supported": True,
        "occurrence_supported": True,
        "bilateral_confirmation": True,
        "evidence_alignment": 0.9,
        "persistence": 0.8,
        "type_support": 0.9,
        "relationship_impact": 0.7,
        "event_significance": 0.8,
        "model_confidence": 0.9,
        "evidence_ids": ["m2", "m3", "m5"],
        "reason": "事实与发生状态均有消息支持",
    }
    payload.update(updates)
    return V3CandidateReview.model_validate(payload)


def replace_window_messages(
    context: ValidationContext,
    messages: tuple[NormalizedMessage, ...],
) -> AnalysisWindow:
    window = context.analysis_window
    return AnalysisWindow(
        window_id=window.window_id,
        session_id=window.session_id,
        window_ordinal=window.window_ordinal,
        start_message_ordinal=window.start_message_ordinal,
        end_message_ordinal=window.start_message_ordinal + len(messages) - 1,
        messages=messages,
        start_message_id=messages[0].source_id,
        end_message_id=messages[-1].source_id,
        character_count=sum(len(message.content) for message in messages),
    )


def validate(
    candidate: V3EventCandidate,
    review: V3CandidateReview,
    context: ValidationContext,
) -> Any:
    from moonlightbox.events.v3_validation import validate_v3_candidate

    return validate_v3_candidate(candidate, review, context)


def test_valid_result_is_frozen_and_has_stable_shape(
    validation_context: ValidationContext,
) -> None:
    from dataclasses import FrozenInstanceError

    result = validate(make_candidate(), make_review(), validation_context)

    assert result.is_valid is True
    assert result.rejection_reasons == ()
    with pytest.raises(FrozenInstanceError):
        result.is_valid = False


def test_relationship_candidate_must_change_the_existing_state(
    validation_context: ValidationContext,
) -> None:
    candidate = make_candidate(
        lane="relationship",
        type="intimacy_increased",
        before_state="亲密，互相关心",
        after_state=" 亲密，互相关心 ",
    )

    result = validate(candidate, make_review(), validation_context)

    assert result.is_valid is False
    assert result.rejection_reasons == ("unchanged_relationship_state",)


@pytest.mark.parametrize(
    ("candidate_updates", "review_updates", "expected_reason"),
    [
        ({"evidence_ids": ["m1", "missing", "m3"]}, {}, "message_not_found"),
        (
            {"start_message_id": "m3", "end_message_id": "m1"},
            {},
            "invalid_time_order",
        ),
        ({"evidence_ids": ["m1", "m2"]}, {}, "event_bounds_not_in_evidence"),
        ({"evidence_ids": ["m0", "m1", "m2", "m3"]}, {}, "event_evidence_outside_range"),
        (
            {
                "evidence_ids": ["xml"],
                "start_message_id": "xml",
                "end_message_id": "xml",
            },
            {},
            "noise_only_evidence",
        ),
        ({}, {"facts_supported": False}, "facts_unsupported"),
        ({}, {"occurrence_supported": False}, "occurrence_unsupported"),
        (
            {"event_status": "confirmed"},
            {"bilateral_confirmation": False},
            "unconfirmed_plan",
        ),
    ],
)
def test_each_semantic_hard_rejection_code_is_emitted(
    validation_context: ValidationContext,
    candidate_updates: dict[str, Any],
    review_updates: dict[str, Any],
    expected_reason: str,
) -> None:
    result = validate(
        make_candidate(**candidate_updates),
        make_review(**review_updates),
        validation_context,
    )

    assert result.is_valid is False
    assert expected_reason in result.rejection_reasons


def test_cross_project_candidate_or_review_evidence_is_rejected(
    validation_context: ValidationContext,
) -> None:
    owners = dict(validation_context.message_project_ids)
    owners["m2"] = "project-2"
    context = ValidationContext(
        expected_project_id=validation_context.expected_project_id,
        messages=validation_context.messages,
        message_project_ids=owners,
        analysis_window=validation_context.analysis_window,
        persistence_context=validation_context.persistence_context,
    )

    result = validate(make_candidate(), make_review(), context)

    assert "message_project_mismatch" in result.rejection_reasons


def test_review_evidence_must_stay_in_window_and_persistence_context(
    validation_context: ValidationContext,
) -> None:
    outside = make_message("outside", 20)
    context = ValidationContext(
        expected_project_id=validation_context.expected_project_id,
        messages={**validation_context.messages, "outside": outside},
        message_project_ids={
            **validation_context.message_project_ids,
            "outside": "project-1",
        },
        analysis_window=validation_context.analysis_window,
        persistence_context=validation_context.persistence_context,
    )

    result = validate(
        make_candidate(),
        make_review(evidence_ids=["m2", "outside"]),
        context,
    )

    assert "event_evidence_outside_range" in result.rejection_reasons


def test_duplicate_evidence_ids_are_stably_deduplicated(
    validation_context: ValidationContext,
) -> None:
    result = validate(
        make_candidate(evidence_ids=["m1", "m1", "m2", "m3", "m3"]),
        make_review(evidence_ids=["m3", "m3", "m5"]),
        validation_context,
    )

    assert result.is_valid is True


def test_review_evidence_alone_rejects_reverse_time_order(
    validation_context: ValidationContext,
) -> None:
    result = validate(
        make_candidate(),
        make_review(evidence_ids=["m5", "m3"]),
        validation_context,
    )

    assert result.rejection_reasons == ("invalid_time_order",)


def test_ordered_candidate_does_not_hide_unordered_review_evidence(
    validation_context: ValidationContext,
) -> None:
    result = validate(
        make_candidate(evidence_ids=["m1", "m2", "m3"]),
        make_review(evidence_ids=["m2", "m5", "m3"]),
        validation_context,
    )

    assert result.rejection_reasons.count("invalid_time_order") == 1


def test_same_timestamp_evidence_uses_source_id_as_order_tiebreaker(
    validation_context: ValidationContext,
) -> None:
    same_a = make_message("same-a", 2)
    same_b = make_message("same-b", 2)
    messages = {
        **validation_context.messages,
        "same-a": same_a,
        "same-b": same_b,
    }
    window_messages = (
        validation_context.messages["m0"],
        validation_context.messages["m1"],
        same_b,
        same_a,
        validation_context.messages["m3"],
        validation_context.messages["m4"],
    )
    context = ValidationContext(
        expected_project_id="project-1",
        messages=messages,
        message_project_ids={key: "project-1" for key in messages},
        analysis_window=replace_window_messages(validation_context, window_messages),
        persistence_context=validation_context.persistence_context,
    )

    result = validate(
        make_candidate(evidence_ids=["m1", "same-b", "same-a", "m3"]),
        make_review(evidence_ids=["m3", "m5"]),
        context,
    )

    assert result.rejection_reasons == ("invalid_time_order",)


@pytest.mark.parametrize(
    "review_updates",
    [
        {
            "persistence": 0,
            "type_support": 0,
            "evidence_alignment": 0,
            "relationship_impact": 0,
            "event_significance": 0,
            "model_confidence": 0,
        },
    ],
)
def test_low_soft_scores_do_not_hard_reject(
    validation_context: ValidationContext,
    review_updates: dict[str, Any],
) -> None:
    result = validate(
        make_candidate(),
        make_review(**review_updates),
        validation_context,
    )

    assert result.is_valid is True
    assert result.rejection_reasons == ()


@pytest.mark.parametrize(
    ("candidate_updates", "review_updates"),
    [
        ({"event_status": "occurred"}, {"occurrence_supported": True}),
        (
            {"event_status": "confirmed"},
            {"occurrence_supported": False, "bilateral_confirmation": True},
        ),
    ],
)
def test_supported_occurred_and_confirmed_events_pass(
    validation_context: ValidationContext,
    candidate_updates: dict[str, Any],
    review_updates: dict[str, Any],
) -> None:
    assert validate(
        make_candidate(**candidate_updates),
        make_review(**review_updates),
        validation_context,
    ).is_valid


@pytest.mark.parametrize(
    "noise_content",
    [
        "wxpay://transfer?id=1",
        "room_type=3&red_dot=1",
        "<msg><title>协议</title></msg>",
        "以下为新消息",
        "[图片]",
    ],
)
def test_noise_cannot_be_the_only_event_evidence(
    validation_context: ValidationContext,
    noise_content: str,
) -> None:
    noise = make_message("noise", 2, content=noise_content)
    context = ValidationContext(
        expected_project_id=validation_context.expected_project_id,
        messages={**validation_context.messages, "noise": noise},
        message_project_ids={**validation_context.message_project_ids, "noise": "project-1"},
        analysis_window=replace_window_messages(
            validation_context,
            (
                *validation_context.analysis_window.messages[:3],
                noise,
                *validation_context.analysis_window.messages[3:],
            ),
        ),
        persistence_context=validation_context.persistence_context,
    )

    result = validate(
        make_candidate(
            start_message_id="noise",
            end_message_id="noise",
            evidence_ids=["noise"],
        ),
        make_review(),
        context,
    )

    assert "noise_only_evidence" in result.rejection_reasons


@pytest.mark.parametrize("real_content", ["今天 < 明天 > 后天", "喜欢你 <3"])
def test_normal_angle_bracket_text_and_mixed_real_evidence_pass(
    validation_context: ValidationContext,
    real_content: str,
) -> None:
    real = make_message("zz-real", 2, content=real_content)
    messages = {**validation_context.messages, "zz-real": real}
    window_messages = (
        validation_context.messages["m0"],
        validation_context.messages["m1"],
        validation_context.messages["xml"],
        real,
        validation_context.messages["m3"],
        validation_context.messages["m4"],
    )
    context = ValidationContext(
        expected_project_id="project-1",
        messages=messages,
        message_project_ids={key: "project-1" for key in messages},
        analysis_window=AnalysisWindow(
            window_id="window-real",
            session_id="session-1",
            window_ordinal=0,
            start_message_ordinal=0,
            end_message_ordinal=len(window_messages) - 1,
            messages=window_messages,
            start_message_id="m0",
            end_message_id="m4",
            character_count=sum(len(message.content) for message in window_messages),
        ),
        persistence_context=validation_context.persistence_context,
    )

    result = validate(
        make_candidate(
            start_message_id="m1",
            end_message_id="zz-real",
            evidence_ids=["m1", "xml", "zz-real"],
        ),
        make_review(evidence_ids=["zz-real"]),
        context,
    )

    assert result.is_valid is True


def test_rejection_reasons_have_fixed_order_without_duplicates(
    validation_context: ValidationContext,
) -> None:
    result = validate(
        make_candidate(
            event_status="confirmed",
            evidence_ids=["missing", "missing"],
            start_message_id="m3",
            end_message_id="m1",
        ),
        make_review(
            facts_supported=False,
            bilateral_confirmation=False,
            evidence_ids=["missing"],
        ),
        validation_context,
    )

    assert result.rejection_reasons == (
        "message_not_found",
        "invalid_time_order",
        "event_bounds_not_in_evidence",
        "event_evidence_outside_range",
        "facts_unsupported",
        "unconfirmed_plan",
    )


def test_forged_candidate_key_and_models_are_safely_rejected(
    validation_context: ValidationContext,
) -> None:
    secret = "SENSITIVE-VALIDATION-ERROR"
    forged_key = make_candidate().model_copy(update={"candidate_key": "stale-key"})
    forged_model = make_candidate().model_copy(update={"lane": secret})
    forged_review = make_review().model_copy(update={"facts_supported": secret, "reason": secret})

    results = [
        validate(forged_key, make_review(), validation_context),
        validate(forged_model, make_review(), validation_context),
        validate(make_candidate(), forged_review, validation_context),
    ]

    assert all(not result.is_valid for result in results)
    assert all(result.rejection_reasons == () for result in results)
    assert all(secret not in repr(result) for result in results)


def test_batch_validation_isolates_one_malformed_item(
    validation_context: ValidationContext,
) -> None:
    from moonlightbox.events.v3_validation import validate_v3_candidates

    malformed = make_candidate().model_copy(update={"evidence_ids": None})
    results = validate_v3_candidates(
        [
            (malformed, make_review()),
            (make_candidate(), make_review()),
        ],
        validation_context,
    )

    assert len(results) == 2
    assert results[0].is_valid is False
    assert results[1].is_valid is True


def test_candidate_cannot_use_persistence_only_messages(
    validation_context: ValidationContext,
) -> None:
    candidate = make_candidate(
        start_message_id="m5",
        end_message_id="m5",
        evidence_ids=["m5"],
    )

    result = validate(candidate, make_review(evidence_ids=["m5"]), validation_context)

    assert result.is_valid is False
    assert result.rejection_reasons == ("event_evidence_outside_range",)


def test_unknown_context_runtime_error_is_observable(
    validation_context: ValidationContext,
) -> None:
    object.__setattr__(
        validation_context.analysis_window,
        "messages",
        ExplodingMessages(validation_context.analysis_window.messages),
    )

    with pytest.raises(RuntimeError, match="context traversal failed"):
        validate(make_candidate(), make_review(), validation_context)


def test_validation_index_is_deeply_immutable(
    validation_context: ValidationContext,
) -> None:
    from dataclasses import FrozenInstanceError

    from moonlightbox.events.v3_validation import build_validation_index

    index = build_validation_index(validation_context)

    with pytest.raises(FrozenInstanceError):
        index.expected_project_id = "project-2"
    with pytest.raises(TypeError):
        index.project_ids["m1"] = "project-2"
    with pytest.raises(TypeError):
        index.order_keys["m1"] = (BASE_TIME, "forged")
    with pytest.raises(TypeError):
        index.noise_flags["m1"] = True


def test_batch_builds_context_index_once_for_large_candidate_set(
    validation_context: ValidationContext,
) -> None:
    from moonlightbox.events.v3_validation import validate_v3_candidates

    window_messages = CountingMessages(validation_context.analysis_window.messages)
    persistence_messages = CountingMessages(
        validation_context.persistence_context.remaining_session_messages
    )
    object.__setattr__(
        validation_context.analysis_window,
        "messages",
        window_messages,
    )
    object.__setattr__(
        validation_context.persistence_context,
        "remaining_session_messages",
        persistence_messages,
    )
    pairs = [(make_candidate(), make_review()) for _ in range(200)]

    results = validate_v3_candidates(pairs, validation_context)

    assert all(result.is_valid for result in results)
    assert window_messages.iterations == 1
    assert persistence_messages.iterations == 1
