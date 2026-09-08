import json
from datetime import datetime, timedelta
from typing import Any

import pytest
from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.events.windowing import AnalysisWindow
from moonlightbox.imports.types import MessageKind
from pydantic import BaseModel, ValidationError

BASE_TIME = datetime(2026, 7, 19, 12, 0)


def make_message(source_id: str, sender: str, content: str, minutes: int) -> NormalizedMessage:
    return NormalizedMessage(
        source_id=source_id,
        timestamp=BASE_TIME + timedelta(minutes=minutes),
        sender=sender,
        kind=MessageKind.TEXT,
        content=content,
    )


def make_window() -> AnalysisWindow:
    messages = (
        make_message("m1", "甲", "我们周六去看展吧", 0),
        make_message("m2", "乙", "好，十点在门口见", 1),
        make_message("m3", "甲", "今天看展很开心", 180),
    )
    return AnalysisWindow(
        window_id="window-1",
        session_id="session-1",
        window_ordinal=0,
        start_message_ordinal=0,
        end_message_ordinal=2,
        messages=messages,
        start_message_id="m1",
        end_message_id="m3",
        character_count=sum(len(message.content) for message in messages),
    )


def relationship_candidate(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_key": "模型键",
        "lane": "relationship",
        "type": "commitment",
        "title": "确认长期关系",
        "event_status": "occurred",
        "start_message_id": "m1",
        "end_message_id": "m2",
        "summary": "双方明确作出长期承诺",
        "before_state": "尚未承诺",
        "after_state": "形成长期承诺",
        "emotion_labels": ["认真", "期待"],
        "topic": "关系承诺",
        "conflict_level": 0,
        "event_significance": 0.9,
        "relationship_impact": 0.8,
        "model_confidence": 0.85,
        "reason": "双方均有明确表达",
        "evidence_ids": ["m1", "m2"],
    }
    payload.update(changes)
    return payload


def shared_candidate(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_key": "模型键",
        "lane": "shared_experience",
        "type": "outing",
        "title": "共同看展",
        "event_status": "occurred",
        "start_message_id": "m1",
        "end_message_id": "m3",
        "summary": "双方按计划共同看展并在事后回顾",
        "before_state": None,
        "after_state": None,
        "emotion_labels": ["开心"],
        "topic": "看展",
        "conflict_level": 0,
        "event_significance": 0.8,
        "relationship_impact": 0.6,
        "model_confidence": 0.9,
        "reason": "计划、确认和事后感受相互印证",
        "evidence_ids": ["m1", "m2", "m3"],
    }
    payload.update(changes)
    return payload


def review_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "facts_supported": True,
        "occurrence_supported": True,
        "bilateral_confirmation": True,
        "evidence_alignment": 0.9,
        "persistence": 0.4,
        "type_support": 0.85,
        "relationship_impact": 0.7,
        "event_significance": 0.8,
        "model_confidence": 0.9,
        "evidence_ids": ["m1", "m2", "m3"],
        "reason": "原始消息支持候选事实",
    }
    payload.update(changes)
    return payload


class FakeAnalysisClient:
    def __init__(self, responses: list[dict[str, Any]], *, overhead: int = 0) -> None:
        self.responses = responses
        self.overhead = overhead
        self.calls: list[dict[str, Any]] = []

    def estimate_structured_prompt_overhead(
        self,
        *,
        response_model: type[BaseModel],
        json_schema: dict[str, Any] | None = None,
    ) -> int:
        return self.overhead

    def create_structured_completion(
        self,
        *,
        system_content: str,
        user_content: str,
        response_model: type[BaseModel],
        json_schema: dict[str, Any] | None = None,
        operation_id: str | None = None,
        window_id: str | None = None,
        run_id: str | None = None,
        max_prompt_chars: int | None = None,
    ) -> BaseModel:
        self.calls.append(
            {
                "system_content": system_content,
                "user_content": user_content,
                "response_model": response_model,
                "json_schema": json_schema,
                "operation_id": operation_id,
                "window_id": window_id,
                "run_id": run_id,
                "max_prompt_chars": max_prompt_chars,
            }
        )
        return response_model.model_validate(self.responses.pop(0))


def test_extracts_each_lane_with_independent_operation_and_prompt() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer

    client = FakeAnalysisClient(
        [{"candidates": [relationship_candidate()]}, {"candidates": [shared_candidate()]}]
    )
    reviewer = DualChannelEventReviewer(client)

    relationship = reviewer.extract_candidates(make_window(), lane="relationship")
    shared = reviewer.extract_candidates(make_window(), lane="shared_experience")

    assert relationship.candidates[0].lane == "relationship"
    assert shared.candidates[0].lane == "shared_experience"
    assert client.calls[0]["operation_id"] == "v3_relationship_extraction"
    assert client.calls[1]["operation_id"] == "v3_shared_experience_extraction"
    assert "旅行" not in client.calls[0]["system_content"]
    assert "旅行" in client.calls[1]["system_content"]


def test_reviews_each_lane_with_independent_operation_and_prompt() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer, V3EventCandidate

    client = FakeAnalysisClient([review_payload(), review_payload()])
    reviewer = DualChannelEventReviewer(client)
    relationship = V3EventCandidate.model_validate(relationship_candidate())
    shared = V3EventCandidate.model_validate(shared_candidate())

    reviewer.review_candidate(relationship, make_window())
    reviewer.review_candidate(shared, make_window())

    assert client.calls[0]["operation_id"] == "v3_relationship_review"
    assert client.calls[1]["operation_id"] == "v3_shared_experience_review"
    assert "旅行" not in client.calls[0]["system_content"]
    assert "旅行" in client.calls[1]["system_content"]


def test_prompts_define_shared_meaning_statuses_and_forbid_invention() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer, V3EventCandidate

    client = FakeAnalysisClient([{"candidates": []}, review_payload()])
    reviewer = DualChannelEventReviewer(client)
    reviewer.extract_candidates(make_window(), lane="shared_experience")
    reviewer.review_candidate(
        V3EventCandidate.model_validate(shared_candidate()),
        make_window(),
    )

    for call in client.calls:
        prompt = call["system_content"]
        assert "不得补写" in prompt
        assert "消息 ID" in prompt
        assert "人物" in prompt
        assert "地点" in prompt
        assert "行为" in prompt
    shared_prompt = client.calls[0]["system_content"]
    for label in (
        "约会",
        "出游",
        "旅行",
        "庆祝",
        "礼物",
        "见亲友",
        "照顾",
        "共同项目",
        "重要计划",
        "人生里程碑",
    ):
        assert label in shared_prompt
    assert "meaningful" in shared_prompt
    assert "occurred" in shared_prompt
    assert "confirmed" in shared_prompt


@pytest.mark.parametrize(
    ("lane", "event_type"),
    [
        ("relationship", "relationship_started"),
        ("relationship", "intimacy_increased"),
        ("relationship", "commitment"),
        ("relationship", "boundary_change"),
        ("relationship", "conflict"),
        ("relationship", "distancing"),
        ("relationship", "reconciliation"),
        ("relationship", "separation"),
        ("relationship", "reconnection"),
        ("shared_experience", "date"),
        ("shared_experience", "outing"),
        ("shared_experience", "travel"),
        ("shared_experience", "celebration"),
        ("shared_experience", "gift"),
        ("shared_experience", "family_social"),
        ("shared_experience", "support_care"),
        ("shared_experience", "shared_project"),
        ("shared_experience", "important_plan"),
        ("shared_experience", "life_milestone"),
    ],
)
def test_candidate_accepts_only_catalog_type_for_its_lane(lane: str, event_type: str) -> None:
    from moonlightbox.events.v3_reviewer import V3EventCandidate

    payload = (
        relationship_candidate(type=event_type)
        if lane == "relationship"
        else shared_candidate(type=event_type)
    )
    assert V3EventCandidate.model_validate(payload).type == event_type


@pytest.mark.parametrize(
    "payload",
    [
        relationship_candidate(type="travel"),
        shared_candidate(type="conflict"),
        relationship_candidate(type="unknown"),
    ],
)
def test_candidate_rejects_cross_lane_or_unknown_type(payload: dict[str, object]) -> None:
    from moonlightbox.events.v3_reviewer import V3EventCandidate

    with pytest.raises(ValidationError):
        V3EventCandidate.model_validate(payload)


@pytest.mark.parametrize("event_status", ["occurred", "confirmed"])
def test_candidate_accepts_both_event_statuses(event_status: str) -> None:
    from moonlightbox.events.v3_reviewer import V3EventCandidate

    candidate = V3EventCandidate.model_validate(shared_candidate(event_status=event_status))
    assert candidate.event_status == event_status


def test_candidate_rejects_unknown_status() -> None:
    from moonlightbox.events.v3_reviewer import V3EventCandidate

    with pytest.raises(ValidationError):
        V3EventCandidate.model_validate(shared_candidate(event_status="pending"))


def test_shared_states_may_be_empty_but_relationship_states_are_required() -> None:
    from moonlightbox.events.v3_reviewer import V3EventCandidate

    shared = V3EventCandidate.model_validate(shared_candidate())
    assert shared.before_state is None
    assert shared.after_state is None

    with pytest.raises(ValidationError, match="关系候选"):
        V3EventCandidate.model_validate(relationship_candidate(before_state=None, after_state=None))


def test_candidate_key_is_stable_and_includes_lane() -> None:
    from moonlightbox.events.v3_reviewer import V3EventCandidate, stable_v3_candidate_key

    candidate = V3EventCandidate.model_validate(relationship_candidate())
    same = candidate.model_copy(update={"candidate_key": "另一个模型键"})

    assert stable_v3_candidate_key(candidate) == stable_v3_candidate_key(same)
    assert candidate.candidate_key.startswith("candidate-relationship-")
    changed_lane = candidate.model_copy(update={"lane": "shared_experience"})
    assert stable_v3_candidate_key(candidate) != stable_v3_candidate_key(changed_lane)


def test_candidate_key_hashes_complete_normalized_payload() -> None:
    from moonlightbox.events.v3_reviewer import V3EventCandidate

    candidate = V3EventCandidate.model_validate(relationship_candidate())
    changed_reason = V3EventCandidate.model_validate(
        relationship_candidate(reason="使用另一组完整证据说明")
    )
    changed_score = V3EventCandidate.model_validate(relationship_candidate(model_confidence=0.84))

    assert changed_reason.candidate_key != candidate.candidate_key
    assert changed_score.candidate_key != candidate.candidate_key


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("event_significance", 0, 0.0),
        ("event_significance", 1, 1.0),
        ("event_significance", 1.1, 0.11),
        ("relationship_impact", 10, 1.0),
        ("model_confidence", 7.5, 0.75),
        ("conflict_level", 5.0, 5),
    ],
)
def test_extraction_normalizes_finite_numeric_boundaries(
    field: str,
    value: object,
    expected: float,
) -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer

    payload = relationship_candidate()
    payload[field] = value
    result = DualChannelEventReviewer(
        FakeAnalysisClient([{"candidates": [payload]}])
    ).extract_candidates(make_window(), lane="relationship")

    assert getattr(result.candidates[0], field) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_significance", "0.8"),
        ("event_significance", True),
        ("event_significance", -0.1),
        ("event_significance", 10.1),
        ("event_significance", float("inf")),
        ("relationship_impact", "8"),
        ("model_confidence", False),
        ("conflict_level", "5"),
        ("conflict_level", True),
        ("conflict_level", -1),
        ("conflict_level", 5.1),
    ],
)
def test_extraction_rejects_invalid_numeric_values(field: str, value: object) -> None:
    from moonlightbox.events.v3_reviewer import (
        DualChannelEventReviewer,
        V3CandidateBatchStructureError,
    )

    payload = relationship_candidate()
    payload[field] = value
    with pytest.raises(V3CandidateBatchStructureError):
        DualChannelEventReviewer(
            FakeAnalysisClient([{"candidates": [payload]}])
        ).extract_candidates(make_window(), lane="relationship")


def test_huge_integer_isolated_while_valid_candidate_survives() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer

    huge_integer = 10**400
    invalid = relationship_candidate(event_significance=huge_integer)
    result = DualChannelEventReviewer(
        FakeAnalysisClient([{"candidates": [invalid, relationship_candidate(title="有效候选")]}])
    ).extract_candidates(make_window(), lane="relationship")

    assert [candidate.title for candidate in result.candidates] == ["有效候选"]
    assert result.diagnostics[0].field_locations == (("event_significance",),)
    assert str(huge_integer) not in repr(result.diagnostics)


def test_all_huge_integer_candidates_raise_safe_structure_error() -> None:
    from moonlightbox.events.v3_reviewer import (
        DualChannelEventReviewer,
        V3CandidateBatchStructureError,
    )

    huge_integer = 10**400
    with pytest.raises(V3CandidateBatchStructureError) as caught:
        DualChannelEventReviewer(
            FakeAnalysisClient(
                [
                    {
                        "candidates": [
                            relationship_candidate(event_significance=huge_integer),
                            relationship_candidate(conflict_level=huge_integer),
                        ]
                    }
                ]
            )
        ).extract_candidates(make_window(), lane="relationship")

    assert caught.value.rejected_count == 2
    assert str(huge_integer) not in str(caught.value)
    assert str(huge_integer) not in repr(caught.value.diagnostics)


def test_invalid_item_is_isolated_without_sensitive_error_content() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer

    secret = "绝不能进入异常的聊天正文"
    invalid = relationship_candidate(type="travel", reason=secret)
    reviewer = DualChannelEventReviewer(
        FakeAnalysisClient([{"candidates": [invalid, relationship_candidate(title="有效候选")]}])
    )

    result = reviewer.extract_candidates(make_window(), lane="relationship")

    assert [candidate.title for candidate in result.candidates] == ["有效候选"]
    assert result.rejected_count == 1
    assert secret not in repr(result.rejected_candidates)


def test_evidence_ids_string_is_isolated_and_valid_candidate_survives() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer

    sensitive_marker = "SENSITIVE-EVIDENCE-MARKER"
    invalid = relationship_candidate(evidence_ids=sensitive_marker)
    reviewer = DualChannelEventReviewer(
        FakeAnalysisClient([{"candidates": [invalid, relationship_candidate(title="有效候选")]}])
    )

    result = reviewer.extract_candidates(make_window(), lane="relationship")

    assert [candidate.title for candidate in result.candidates] == ["有效候选"]
    assert result.rejected_count == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.index == 0
    assert diagnostic.error_code == "candidate_validation_failed"
    assert ("evidence_ids",) in diagnostic.field_locations
    assert sensitive_marker not in repr(diagnostic)


def test_extra_field_is_isolated_and_valid_candidate_survives() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer

    sensitive_marker = "SENSITIVE-EXTRA-MARKER"
    invalid = relationship_candidate()
    invalid["invented_detail"] = sensitive_marker
    reviewer = DualChannelEventReviewer(
        FakeAnalysisClient([{"candidates": [invalid, relationship_candidate(title="有效候选")]}])
    )

    result = reviewer.extract_candidates(make_window(), lane="relationship")

    assert [candidate.title for candidate in result.candidates] == ["有效候选"]
    diagnostic = result.diagnostics[0]
    assert diagnostic.field_locations == (("<extra_field>",),)
    assert sensitive_marker not in repr(diagnostic)


def test_attacker_controlled_extra_field_name_is_redacted_everywhere() -> None:
    from moonlightbox.events.v3_reviewer import (
        DualChannelEventReviewer,
        V3CandidateBatchStructureError,
    )

    sensitive_marker = "SENSITIVE-ORIGINAL-CONTENT"
    invalid = relationship_candidate()
    invalid[sensitive_marker] = sensitive_marker

    with pytest.raises(V3CandidateBatchStructureError) as caught:
        DualChannelEventReviewer(
            FakeAnalysisClient([{"candidates": [invalid]}])
        ).extract_candidates(make_window(), lane="relationship")

    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.field_locations == (("<extra_field>",),)
    rendered_outputs = (
        repr(caught.value.diagnostics),
        str(caught.value),
        json.dumps(diagnostic.model_dump(), ensure_ascii=False),
    )
    assert all(sensitive_marker not in rendered for rendered in rendered_outputs)


def test_all_invalid_items_raise_safe_structure_error() -> None:
    from moonlightbox.events.v3_reviewer import (
        DualChannelEventReviewer,
        V3CandidateBatchStructureError,
    )

    secret = "SENSITIVE-ALL-INVALID-MARKER"
    wrong_evidence = relationship_candidate(evidence_ids=secret)
    extra_field = relationship_candidate()
    extra_field["invented_detail"] = secret

    with pytest.raises(V3CandidateBatchStructureError) as caught:
        DualChannelEventReviewer(
            FakeAnalysisClient([{"candidates": [wrong_evidence, extra_field]}])
        ).extract_candidates(make_window(), lane="relationship")

    assert caught.value.rejected_count == 2
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value.diagnostics)
    assert caught.value.diagnostics[0].field_locations == (("evidence_ids",),)
    assert caught.value.diagnostics[1].field_locations == (("<extra_field>",),)


def test_review_normalizes_scores_and_rejects_invalid_score() -> None:
    from moonlightbox.events.v3_reviewer import (
        DualChannelEventReviewer,
        V3EventCandidate,
        V3ReviewStructureError,
    )

    candidate = V3EventCandidate.model_validate(shared_candidate())
    valid = DualChannelEventReviewer(
        FakeAnalysisClient([review_payload(evidence_alignment=8)])
    ).review_candidate(candidate, make_window())
    assert valid.evidence_alignment == 0.8

    with pytest.raises(V3ReviewStructureError):
        DualChannelEventReviewer(
            FakeAnalysisClient([review_payload(type_support="0.8")])
        ).review_candidate(candidate, make_window())


def test_review_rejects_huge_integer_with_safe_structure_error() -> None:
    from moonlightbox.events.v3_reviewer import (
        DualChannelEventReviewer,
        V3EventCandidate,
        V3ReviewStructureError,
    )

    huge_integer = 10**400
    candidate = V3EventCandidate.model_validate(shared_candidate())

    with pytest.raises(V3ReviewStructureError) as caught:
        DualChannelEventReviewer(
            FakeAnalysisClient([review_payload(evidence_alignment=huge_integer)])
        ).review_candidate(candidate, make_window())

    assert str(huge_integer) not in str(caught.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"lane": "unknown"},
        {"type": "travel"},
        {"candidate_key": "stale-key"},
    ],
)
def test_review_revalidates_model_copy_and_rejects_forged_candidate(
    changes: dict[str, str],
) -> None:
    from moonlightbox.events.v3_reviewer import (
        DualChannelEventReviewer,
        V3CandidateIntegrityError,
        V3EventCandidate,
    )

    client = FakeAnalysisClient([review_payload()])
    candidate = V3EventCandidate.model_validate(relationship_candidate())
    forged = candidate.model_copy(update=changes)

    with pytest.raises(V3CandidateIntegrityError):
        DualChannelEventReviewer(client).review_candidate(forged, make_window())

    assert client.calls == []


def test_candidate_and_response_models_are_frozen() -> None:
    from moonlightbox.events.v3_reviewer import (
        RawV3CandidateReview,
        RawV3EventCandidate,
        RawV3EventCandidateBatch,
        V3CandidateReview,
        V3EventCandidate,
        V3RejectedCandidate,
    )

    models_and_assignments = [
        (
            V3EventCandidate.model_validate(relationship_candidate()),
            "lane",
            "shared_experience",
        ),
        (RawV3EventCandidate.model_validate(relationship_candidate()), "lane", "unknown"),
        (RawV3EventCandidateBatch(candidates=[]), "candidates", []),
        (V3CandidateReview.model_validate(review_payload()), "facts_supported", False),
        (RawV3CandidateReview.model_validate(review_payload()), "facts_supported", False),
        (
            V3RejectedCandidate(
                index=0,
                error_code="candidate_validation_failed",
                field_locations=(("type",),),
            ),
            "index",
            1,
        ),
    ]

    for model, field_name, value in models_and_assignments:
        with pytest.raises(ValidationError):
            setattr(model, field_name, value)


def test_review_budget_reserves_structured_output_overhead() -> None:
    from moonlightbox.events.v3_reviewer import (
        DualChannelEventReviewer,
        V3EventCandidate,
        V3PromptBudgetExceededError,
    )

    candidate = V3EventCandidate.model_validate(shared_candidate())
    probe = FakeAnalysisClient([review_payload()], overhead=100)
    with pytest.raises(V3PromptBudgetExceededError):
        DualChannelEventReviewer(
            probe,
            review_character_budget=100,
        ).review_candidate(candidate, make_window())
    assert probe.calls == []


def test_review_prompt_contains_candidate_and_complete_messages() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer, V3EventCandidate

    client = FakeAnalysisClient([review_payload()])
    candidate = V3EventCandidate.model_validate(shared_candidate())
    DualChannelEventReviewer(client).review_candidate(candidate, make_window(), run_id="run-1")

    payload = json.loads(client.calls[0]["user_content"])
    assert payload["候选"]["candidate_key"] == candidate.candidate_key
    assert [message["id"] for message in payload["消息"]] == ["m1", "m2", "m3"]
    assert payload["元数据"]["run_id"] == "run-1"
    assert client.calls[0]["max_prompt_chars"] == 12000


def test_review_prompt_compacts_irrelevant_messages_to_fit_budget() -> None:
    from moonlightbox.events.v3_reviewer import DualChannelEventReviewer, V3EventCandidate

    messages = tuple(
        make_message(
            f"long-{index}",
            "甲" if index % 2 == 0 else "乙",
            ("无关上下文" if index not in {10, 11} else "关键事实") + "甲" * 500,
            index,
        )
        for index in range(20)
    )
    window = AnalysisWindow(
        window_id="long-window",
        session_id="long-session",
        window_ordinal=0,
        start_message_ordinal=0,
        end_message_ordinal=19,
        messages=messages,
        start_message_id="long-0",
        end_message_id="long-19",
        character_count=sum(len(message.content) for message in messages),
    )
    candidate = V3EventCandidate.model_validate(
        relationship_candidate(
            start_message_id="long-10",
            end_message_id="long-11",
            evidence_ids=["long-10", "long-11"],
        )
    )
    client = FakeAnalysisClient(
        [review_payload(evidence_ids=["long-10", "long-11"])],
        overhead=200,
    )

    DualChannelEventReviewer(
        client,
        review_character_budget=5000,
    ).review_candidate(candidate, window)

    payload = json.loads(client.calls[0]["user_content"])
    included_ids = [message["id"] for message in payload["消息"]]
    assert {"long-10", "long-11"} <= set(included_ids)
    assert len(included_ids) < len(messages)
    assert included_ids == sorted(
        included_ids,
        key=lambda source_id: int(source_id.split("-")[1]),
    )


def test_raw_batch_minimal_object_shape_validates_without_all_empty_candidate() -> None:
    from moonlightbox.events.cloud_client import _minimal_json_example
    from moonlightbox.events.v3_reviewer import (
        DualChannelEventReviewer,
        RawV3EventCandidate,
        RawV3EventCandidateBatch,
    )

    client = FakeAnalysisClient([{"candidates": []}])
    DualChannelEventReviewer(client).extract_candidates(
        make_window(),
        lane="relationship",
    )
    schema = client.calls[0]["json_schema"]
    assert isinstance(schema, dict)
    example = _minimal_json_example(schema, root_schema=schema)

    validated = RawV3EventCandidateBatch.model_validate(example)
    assert validated.candidates
    item = RawV3EventCandidate.model_validate(validated.candidates[0])
    assert item.title
    assert item.evidence_ids
