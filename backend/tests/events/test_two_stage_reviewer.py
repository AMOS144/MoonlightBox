import json
from datetime import datetime, timedelta
from typing import Any

import httpx
import pytest
from moonlightbox.events.normalization import NormalizedMessage
from moonlightbox.events.windowing import AnalysisWindow, MessageSession, PersistenceContext
from moonlightbox.imports.types import MessageKind
from pydantic import BaseModel

BASE_TIME = datetime(2026, 7, 18, 12, 0)


def make_message(
    source_id: str,
    sender: str,
    content: str,
    *,
    minutes: int,
) -> NormalizedMessage:
    return NormalizedMessage(
        source_id=source_id,
        timestamp=BASE_TIME + timedelta(minutes=minutes),
        sender=sender,
        kind=MessageKind.TEXT,
        content=content,
    )


def make_window(messages: tuple[NormalizedMessage, ...]) -> AnalysisWindow:
    return AnalysisWindow(
        window_id="window-1",
        session_id="session-1",
        window_ordinal=0,
        start_message_ordinal=0,
        end_message_ordinal=len(messages) - 1,
        messages=messages,
        start_message_id=messages[0].source_id,
        end_message_id=messages[-1].source_id,
        character_count=sum(len(message.content) for message in messages),
    )


class FakeAnalysisClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

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


class OverheadAwareFakeAnalysisClient(FakeAnalysisClient):
    def __init__(
        self,
        responses: list[dict[str, Any]],
        *,
        prompt_overhead: int,
    ) -> None:
        super().__init__(responses)
        self.prompt_overhead = prompt_overhead

    def estimate_structured_prompt_overhead(
        self,
        *,
        response_model: type[BaseModel],
        json_schema: dict[str, Any] | None = None,
    ) -> int:
        return self.prompt_overhead


def candidate_payload(
    event_type: str = "relationship_started",
    *,
    start_message_id: str = "m1",
    end_message_id: str = "m2",
) -> dict[str, Any]:
    return {
        "candidate_key": "模型给出的不稳定键",
        "type": event_type,
        "start_message_id": start_message_id,
        "end_message_id": end_message_id,
        "before_state": "彼此试探",
        "after_state": "确认恋爱关系",
        "emotion_labels": ["期待", "开心"],
        "topic": "关系确认",
        "conflict_level": 0,
        "state_change_strength": 0.95,
        "model_confidence": 0.9,
        "reason": "双方明确确认关系",
        "evidence_ids": ["m1", "m2"],
    }


def test_two_stage_reviewer_calls_cloud_client_and_preserves_complete_message_metadata() -> None:
    from moonlightbox.events.reviewer import TwoStageEventReviewer

    messages = (
        make_message("m1", "甲", "我们正式在一起吧", minutes=0),
        make_message("m2", "乙", "好，我愿意", minutes=1),
        make_message("m3", "甲", "以后一起规划周末", minutes=2),
    )
    window = make_window(messages)
    persistence_message = make_message("m4", "乙", "周末一起去看展", minutes=60)
    client = FakeAnalysisClient(
        [
            {"candidates": [candidate_payload()]},
            {
                "accepted": True,
                "type_supported": True,
                "evidence_alignment": 0.95,
                "decisive_event": False,
                "persistence": 0.9,
                "evidence_ids": ["m4"],
                "reason": "后续持续以伴侣方式规划活动",
            },
        ]
    )
    reviewer = TwoStageEventReviewer(client)

    extraction = reviewer.extract_candidates(window, run_id="run-1")
    candidates = extraction.candidates
    review = reviewer.review_candidate(
        candidates[0],
        window,
        PersistenceContext(
            remaining_session_messages=(),
            following_sessions=(
                MessageSession(
                    session_id="session-2",
                    messages=(persistence_message,),
                ),
            ),
        ),
        run_id="run-1",
    )

    assert len(client.calls) == 2
    assert client.calls[0]["operation_id"] == "event_candidate_extraction"
    assert client.calls[1]["operation_id"] == "event_candidate_review"
    assert client.calls[0]["run_id"] == "run-1"
    assert client.calls[1]["run_id"] == "run-1"
    assert client.calls[0]["window_id"] == "window-1"
    assert client.calls[1]["window_id"] == "window-1"
    assert candidates[0].candidate_key.startswith("candidate-")
    assert extraction.rejected_count == 0
    assert review.accepted is True

    extraction_prompt = json.loads(client.calls[0]["user_content"])
    assert extraction_prompt["元数据"] == {
        "run_id": "run-1",
        "window_id": "window-1",
    }
    assert extraction_prompt["消息"] == [
        {
            "id": message.source_id,
            "角色": message.sender,
            "时间": message.timestamp.isoformat(),
            "内容": message.content,
        }
        for message in messages
    ]
    review_prompt = json.loads(client.calls[1]["user_content"])
    assert review_prompt["元数据"] == {
        "run_id": "run-1",
        "window_id": "window-1",
        "character_budget": 12000,
        "truncated": False,
    }
    assert review_prompt["原始证据消息"] == [
        _evidence_payload_for_assertion(messages[0]),
        _evidence_payload_for_assertion(messages[1]),
    ]
    assert review_prompt["同窗口后续消息"] == [_message_payload_for_assertion(messages[2])]
    assert review_prompt["持续性上下文"] == [_message_payload_for_assertion(persistence_message)]
    assert "普通长间隔" in client.calls[1]["system_content"]
    assert "一次性争吵" in client.calls[1]["system_content"]
    assert "短暂情绪" in client.calls[1]["system_content"]
    assert "不能虚构" in client.calls[1]["system_content"]


def test_extraction_prompt_requires_complete_window_review_without_forcing_candidates() -> None:
    from moonlightbox.events.reviewer import TwoStageEventReviewer

    messages = (
        make_message("m1", "甲", "我们在一起吧", minutes=0),
        make_message("m2", "乙", "好", minutes=1),
    )
    client = FakeAnalysisClient([{"candidates": []}])

    TwoStageEventReviewer(client).extract_candidates(make_window(messages))

    prompt = client.calls[0]["system_content"]
    assert "通读完整窗口" in prompt
    assert "至少返回一个候选" in prompt
    assert "确实不存在" in prompt
    assert "不得凑数" in prompt
    assert "普通长间隔" in prompt


def test_extraction_prompt_lists_exact_event_ids_and_numeric_scales() -> None:
    from moonlightbox.events.reviewer import TwoStageEventReviewer

    messages = (
        make_message("m1", "甲", "我们在一起吧", minutes=0),
        make_message("m2", "乙", "好", minutes=1),
    )
    client = FakeAnalysisClient([{"candidates": []}])

    TwoStageEventReviewer(client).extract_candidates(make_window(messages))

    prompt = client.calls[0]["system_content"]
    expected_types = {
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
    assert all(event_type in prompt for event_type in expected_types)
    assert "必须精确使用" in prompt
    assert "0..1" in prompt
    assert "0.8" in prompt
    assert "不是 8 或 80" in prompt
    assert "conflict_level" in prompt
    assert "0..5" in prompt


def test_extraction_normalizes_known_deepseek_aliases_and_ten_point_scales() -> None:
    from moonlightbox.events.reviewer import TwoStageEventReviewer

    payload = candidate_payload("conflict_escalation")
    payload.update(
        {
            "conflict_level": 5,
            "state_change_strength": 8,
            "model_confidence": 8,
        }
    )
    reviewer = TwoStageEventReviewer(FakeAnalysisClient([{"candidates": [payload]}]))
    messages = (
        make_message("m1", "甲", "你总是忽略我", minutes=0),
        make_message("m2", "乙", "我们需要谈谈", minutes=1),
    )

    extraction = reviewer.extract_candidates(make_window(messages))

    assert extraction.rejected_count == 0
    assert extraction.candidates[0].type == "conflict"
    assert extraction.candidates[0].state_change_strength == 0.8
    assert extraction.candidates[0].model_confidence == 0.8
    assert extraction.candidates[0].conflict_level == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("type", "关系升级"),
        ("type", "unknown_free_text"),
        ("state_change_strength", 80),
        ("state_change_strength", -1),
        ("state_change_strength", "8"),
        ("model_confidence", 80),
        ("model_confidence", -0.1),
        ("model_confidence", "0.8"),
        ("conflict_level", "5"),
        ("conflict_level", 4.5),
    ],
)
def test_extraction_rejects_ambiguous_or_invalid_normalization_inputs(
    field: str,
    value: object,
) -> None:
    from moonlightbox.events.reviewer import (
        CandidateBatchStructureError,
        TwoStageEventReviewer,
    )

    payload = candidate_payload()
    payload[field] = value
    reviewer = TwoStageEventReviewer(FakeAnalysisClient([{"candidates": [payload]}]))
    messages = (
        make_message("m1", "甲", "普通聊天", minutes=0),
        make_message("m2", "乙", "收到", minutes=1),
    )

    with pytest.raises(CandidateBatchStructureError):
        reviewer.extract_candidates(make_window(messages))


@pytest.mark.parametrize(
    "field",
    ["state_change_strength", "model_confidence", "conflict_level"],
)
def test_raw_candidate_rejects_boolean_numeric_fields(field: str) -> None:
    from moonlightbox.events.reviewer import EventCandidateBatch
    from pydantic import ValidationError

    payload = candidate_payload()
    payload[field] = True

    with pytest.raises(ValidationError):
        EventCandidateBatch.model_validate({"candidates": [payload]})


def test_extraction_keeps_unit_scale_and_accepts_integral_float_conflict_level() -> None:
    from moonlightbox.events.reviewer import TwoStageEventReviewer

    payload = candidate_payload()
    payload.update(
        {
            "conflict_level": 5.0,
            "state_change_strength": 0.75,
            "model_confidence": 1.0,
        }
    )
    reviewer = TwoStageEventReviewer(FakeAnalysisClient([{"candidates": [payload]}]))
    messages = (
        make_message("m1", "甲", "我们在一起吧", minutes=0),
        make_message("m2", "乙", "好", minutes=1),
    )

    candidate = reviewer.extract_candidates(make_window(messages)).candidates[0]

    assert candidate.conflict_level == 5
    assert candidate.state_change_strength == 0.75
    assert candidate.model_confidence == 1.0


def test_second_stage_can_reject_ordinary_gap_or_one_time_conflict() -> None:
    from moonlightbox.events.reviewer import EventCandidate, EventCandidateReview

    gap_review = EventCandidateReview(
        accepted=False,
        type_supported=False,
        evidence_alignment=0.1,
        decisive_event=False,
        persistence=0.0,
        evidence_ids=[],
        reason="只有普通长间隔，没有关系状态变化证据",
    )
    conflict_review = EventCandidateReview(
        accepted=False,
        type_supported=False,
        evidence_alignment=0.2,
        decisive_event=False,
        persistence=0.0,
        evidence_ids=[],
        reason="仅有一次性争吵，后续恢复正常",
    )

    assert gap_review.accepted is False
    assert conflict_review.accepted is False
    assert gap_review.type_supported is False
    assert conflict_review.evidence_alignment == 0.2
    assert EventCandidate.model_validate(candidate_payload("conflict")).type == "conflict"


def test_review_prompt_requires_semantic_alignment_instead_of_legal_ids_only() -> None:
    from moonlightbox.events.reviewer import EventCandidate, TwoStageEventReviewer

    messages = (
        make_message("m1", "甲", "今天下雨了", minutes=0),
        make_message("m2", "乙", "记得带伞", minutes=1),
    )
    client = FakeAnalysisClient(
        [
            {
                "accepted": False,
                "type_supported": False,
                "evidence_alignment": 0.1,
                "decisive_event": False,
                "persistence": 0,
                "evidence_ids": [],
                "reason": "天气闲聊不支持冲突类型或关系状态变化",
            }
        ]
    )

    review = TwoStageEventReviewer(client).review_candidate(
        EventCandidate.model_validate(candidate_payload("conflict")),
        make_window(messages),
        PersistenceContext(remaining_session_messages=(), following_sessions=()),
    )

    assert review.type_supported is False
    assert review.evidence_alignment == 0.1
    prompt = client.calls[0]["system_content"]
    assert "天气闲聊不能支持 conflict" in prompt
    assert "不能仅因证据 ID 合法" in prompt


@pytest.mark.parametrize(
    ("event_type", "first_content", "second_content", "type_supported"),
    [
        ("conflict", "今天下雨了", "记得带伞", False),
        ("separation", "我们分手吧", "好，我接受", True),
    ],
)
def test_real_cloud_review_receives_complete_original_evidence(
    event_type: str,
    first_content: str,
    second_content: str,
    type_supported: bool,
) -> None:
    from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
    from moonlightbox.events.reviewer import EventCandidate, TwoStageEventReviewer

    captured_evidence: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user_payload = json.loads(body["messages"][1]["content"])
        captured_evidence.extend(user_payload["原始证据消息"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "parsed": {
                                "accepted": type_supported,
                                "type_supported": type_supported,
                                "evidence_alignment": 0.95 if type_supported else 0.1,
                                "decisive_event": type_supported,
                                "persistence": 0,
                                "evidence_ids": [],
                                "reason": "证据原文支持判断",
                            }
                        }
                    }
                ]
            },
        )

    cloud_client = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://example.com/v1/chat/completions",
        model="analysis-model",
        api_key="secret",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    messages = (
        make_message("m1", "甲", first_content, minutes=0),
        make_message("m2", "乙", second_content, minutes=1),
    )
    candidate_data = candidate_payload(event_type)
    if event_type == "separation":
        candidate_data.update(
            {
                "before_state": "恋爱关系",
                "after_state": "明确分手",
                "topic": "结束关系",
            }
        )

    review = TwoStageEventReviewer(cloud_client).review_candidate(
        EventCandidate.model_validate(candidate_data),
        make_window(messages),
        PersistenceContext(remaining_session_messages=(), following_sessions=()),
    )

    assert captured_evidence == [
        _evidence_payload_for_assertion(messages[0]),
        _evidence_payload_for_assertion(messages[1]),
    ]
    assert review.type_supported is type_supported


def test_structured_review_rejects_blank_reason() -> None:
    import pytest
    from moonlightbox.events.reviewer import EventCandidateReview
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EventCandidateReview(
            accepted=False,
            persistence=0,
            evidence_ids=[],
            reason=" \n ",
        )


def test_fixed_candidate_types_reject_unknown_value() -> None:
    import pytest
    from moonlightbox.events.reviewer import EventCandidate
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EventCandidate.model_validate(candidate_payload("ordinary_gap"))


def test_cloud_batch_discards_one_invalid_candidate_and_keeps_valid_candidate() -> None:
    from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
    from moonlightbox.events.reviewer import TwoStageEventReviewer

    captured_schema: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_schema.update(body["response_format"]["json_schema"]["schema"])
        invalid = candidate_payload()
        invalid["type"] = "ordinary_gap"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "parsed": {
                                "candidates": [invalid, candidate_payload()],
                            }
                        }
                    }
                ]
            },
        )

    cloud_client = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://example.com/v1/chat/completions",
        model="analysis-model",
        api_key="secret",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    messages = (
        make_message("m1", "甲", "我们在一起吧", minutes=0),
        make_message("m2", "乙", "好", minutes=1),
    )

    extraction = TwoStageEventReviewer(cloud_client).extract_candidates(
        make_window(messages),
        run_id="run-1",
    )

    assert [candidate.type for candidate in extraction.candidates] == ["relationship_started"]
    assert extraction.rejected_count == 1
    assert extraction.rejected_candidates[0].index == 0
    assert extraction.rejected_candidates[0].error_code == "candidate_validation_failed"
    candidate_schema = captured_schema["$defs"]["RawEventCandidate"]
    assert set(candidate_schema["required"]) == set(candidate_schema["properties"])
    assert candidate_schema["additionalProperties"] is False


def test_invalid_batch_envelope_still_fails_validation() -> None:
    import pytest
    from moonlightbox.events.reviewer import EventCandidateBatch
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EventCandidateBatch.model_validate({"wrong_key": []})


def test_all_invalid_candidates_raise_explicit_structure_error() -> None:
    import pytest
    from moonlightbox.events.reviewer import (
        CandidateBatchStructureError,
        TwoStageEventReviewer,
    )

    invalid = candidate_payload()
    invalid["type"] = "ordinary_gap"
    reviewer = TwoStageEventReviewer(FakeAnalysisClient([{"candidates": [invalid]}]))
    messages = (
        make_message("m1", "甲", "普通聊天", minutes=0),
        make_message("m2", "乙", "收到", minutes=1),
    )

    with pytest.raises(CandidateBatchStructureError) as caught:
        reviewer.extract_candidates(make_window(messages))

    assert caught.value.rejected_count == 1
    assert caught.value.diagnostics[0].index == 0


def test_numeric_strings_are_rejected_without_coercion_and_valid_json_numbers_survive() -> None:
    from moonlightbox.events.reviewer import TwoStageEventReviewer

    conflict_string = candidate_payload()
    conflict_string["conflict_level"] = "2"
    strength_string = candidate_payload()
    strength_string["state_change_strength"] = "0.8"
    reviewer = TwoStageEventReviewer(
        FakeAnalysisClient(
            [
                {
                    "candidates": [
                        conflict_string,
                        strength_string,
                        candidate_payload(),
                    ]
                }
            ]
        )
    )
    messages = (
        make_message("m1", "甲", "我们在一起吧", minutes=0),
        make_message("m2", "乙", "好", minutes=1),
    )

    extraction = reviewer.extract_candidates(make_window(messages))

    assert len(extraction.candidates) == 1
    assert [diagnostic.index for diagnostic in extraction.diagnostics] == [0, 1]


def test_second_stage_rejects_transient_emotion_through_real_call() -> None:
    from moonlightbox.events.reviewer import EventCandidate, TwoStageEventReviewer

    messages = (
        make_message("m1", "甲", "今天特别想你", minutes=0),
        make_message("m2", "乙", "我也是", minutes=1),
        make_message("m3", "甲", "先去忙了", minutes=2),
    )
    client = FakeAnalysisClient(
        [
            {
                "accepted": False,
                "type_supported": False,
                "evidence_alignment": 0.4,
                "decisive_event": False,
                "persistence": 0.0,
                "evidence_ids": [],
                "reason": "只是短暂情绪，没有持续关系变化",
            }
        ]
    )
    candidate = EventCandidate.model_validate(
        candidate_payload(
            "intimacy_increased",
            start_message_id="m1",
            end_message_id="m2",
        )
    )

    review = TwoStageEventReviewer(client).review_candidate(
        candidate,
        make_window(messages),
        PersistenceContext(remaining_session_messages=(), following_sessions=()),
        run_id="run-emotion",
    )

    assert review.accepted is False
    assert client.calls[0]["operation_id"] == "event_candidate_review"
    assert client.calls[0]["run_id"] == "run-emotion"
    assert client.calls[0]["window_id"] == "window-1"


def test_candidate_key_is_stable_after_conservative_normalization() -> None:
    from moonlightbox.events.reviewer import EventCandidate, stable_candidate_key

    first = EventCandidate.model_validate(candidate_payload())
    second_payload = candidate_payload()
    second_payload.update(
        {
            "type": "RELATIONSHIP_STARTED",
            "topic": " 关系确认 ",
            "before_state": "彼此　试探",
            "after_state": "确认恋爱关系。",
        }
    )
    second = EventCandidate.model_construct(**second_payload)

    assert stable_candidate_key(first) == stable_candidate_key(second)


def test_candidate_key_distinguishes_same_boundary_with_different_semantics() -> None:
    from moonlightbox.events.reviewer import EventCandidate, stable_candidate_key

    relationship = EventCandidate.model_validate(candidate_payload())
    conflict_payload = candidate_payload("conflict")
    conflict_payload.update(
        {
            "topic": "边界争议",
            "before_state": "平静沟通",
            "after_state": "持续冲突",
        }
    )
    conflict = EventCandidate.model_validate(conflict_payload)

    assert stable_candidate_key(relationship) != stable_candidate_key(conflict)


def test_candidate_key_preserves_distinct_source_id_punctuation() -> None:
    from moonlightbox.events.reviewer import EventCandidate, stable_candidate_key

    dashed = EventCandidate.model_validate(
        candidate_payload(start_message_id="m-1", end_message_id="m2")
        | {"evidence_ids": ["m-1", "m2"]}
    )
    plain = EventCandidate.model_validate(
        candidate_payload(start_message_id="m1", end_message_id="m2")
    )

    assert stable_candidate_key(dashed) != stable_candidate_key(plain)


def test_review_prompt_truncates_complete_messages_with_positive_budget() -> None:
    from moonlightbox.events.reviewer import EventCandidate, TwoStageEventReviewer

    base_messages = (
        make_message("m1", "甲", "开始", minutes=0),
        make_message("m2", "乙", "确认", minutes=1),
        make_message(
            "最近证据-" * 8,
            "很长的发送者-" * 8,
            '中文内容需要计算 JSON 的 "引号" 与反斜杠\\',
            minutes=2,
        ),
    )
    candidate_payload_with_long_reason = candidate_payload()
    candidate_payload_with_long_reason["reason"] = "候选理由" * 30
    candidate = EventCandidate.model_validate(candidate_payload_with_long_reason)
    probe_client = FakeAnalysisClient(
        [
            {
                "accepted": False,
                "type_supported": True,
                "evidence_alignment": 0.9,
                "decisive_event": False,
                "persistence": 0,
                "evidence_ids": [],
                "reason": "后续证据不足",
            }
        ]
    )
    TwoStageEventReviewer(
        probe_client,
        persistence_character_budget=100000,
    ).review_candidate(
        candidate,
        make_window(base_messages),
        PersistenceContext(remaining_session_messages=(), following_sessions=()),
    )
    probe_prompt = json.loads(probe_client.calls[0]["user_content"])
    probe_prompt["元数据"]["truncated"] = True
    one_message_user = json.dumps(probe_prompt, ensure_ascii=False)
    budget = len(probe_client.calls[0]["system_content"]) + len(one_message_user)
    long_later_message = make_message(
        "later-" * 8,
        "乙" * 20,
        "续" * (budget - len(base_messages[2].content) - 1),
        minutes=3,
    )
    messages = base_messages + (long_later_message,)
    client = FakeAnalysisClient(
        [
            {
                "accepted": False,
                "type_supported": True,
                "evidence_alignment": 0.9,
                "decisive_event": False,
                "persistence": 0,
                "evidence_ids": [],
                "reason": "后续证据不足",
            }
        ]
    )
    reviewer = TwoStageEventReviewer(
        client,
        persistence_character_budget=budget,
    )

    reviewer.review_candidate(
        candidate,
        make_window(messages),
        PersistenceContext(remaining_session_messages=(), following_sessions=()),
    )

    prompt = json.loads(client.calls[0]["user_content"])
    assert prompt["同窗口后续消息"] == [_message_payload_for_assertion(base_messages[2])]
    assert prompt["持续性上下文"] == []
    assert prompt["元数据"]["truncated"] is True
    assert long_later_message.source_id not in client.calls[0]["user_content"]
    assert len(client.calls[0]["system_content"]) + len(client.calls[0]["user_content"]) <= budget


def test_review_fixed_payload_over_budget_fails_before_cloud_call() -> None:
    import pytest
    from moonlightbox.events.reviewer import (
        EventCandidate,
        PromptBudgetExceededError,
        TwoStageEventReviewer,
    )

    payload = candidate_payload()
    payload["reason"] = "非常长的候选理由" * 200
    candidate = EventCandidate.model_validate(payload)
    client = FakeAnalysisClient([])
    reviewer = TwoStageEventReviewer(client, persistence_character_budget=300)
    messages = (
        make_message("m1", "甲", "不可截断的原始证据" * 100, minutes=0),
        make_message("m2", "乙", "不可截断的确认回应" * 100, minutes=1),
    )

    with pytest.raises(PromptBudgetExceededError) as caught:
        reviewer.review_candidate(
            candidate,
            make_window(messages),
            PersistenceContext(
                remaining_session_messages=(),
                following_sessions=(),
            ),
        )

    assert caught.value.character_budget == 300
    assert client.calls == []


def test_review_reserves_client_prompt_overhead_and_passes_final_budget() -> None:
    from moonlightbox.events.reviewer import EventCandidate, TwoStageEventReviewer

    response = {
        "accepted": False,
        "type_supported": True,
        "evidence_alignment": 0.9,
        "decisive_event": False,
        "persistence": 0,
        "evidence_ids": [],
        "reason": "后续证据不足",
    }
    candidate = EventCandidate.model_validate(candidate_payload())
    evidence_messages = (
        make_message("m1", "甲", "开始", minutes=0),
        make_message("m2", "乙", "确认", minutes=1),
    )
    probe = FakeAnalysisClient([response])
    TwoStageEventReviewer(
        probe,
        persistence_character_budget=100000,
    ).review_candidate(
        candidate,
        make_window(evidence_messages),
        PersistenceContext(remaining_session_messages=(), following_sessions=()),
    )
    overhead = 200
    budget = len(probe.calls[0]["system_content"]) + len(probe.calls[0]["user_content"]) + overhead
    optional_message = make_message(
        "m3",
        "甲",
        "会让最终提示超过预算的后续内容" * 20,
        minutes=2,
    )
    client = OverheadAwareFakeAnalysisClient(
        [response],
        prompt_overhead=overhead,
    )

    TwoStageEventReviewer(
        client,
        persistence_character_budget=budget,
    ).review_candidate(
        candidate,
        make_window(evidence_messages + (optional_message,)),
        PersistenceContext(remaining_session_messages=(), following_sessions=()),
    )

    prompt = json.loads(client.calls[0]["user_content"])
    assert prompt["同窗口后续消息"] == []
    assert prompt["元数据"]["truncated"] is True
    assert client.calls[0]["max_prompt_chars"] == budget
    assert (
        len(client.calls[0]["system_content"]) + len(client.calls[0]["user_content"]) + overhead
        <= budget
    )


def test_reviewer_rejects_non_positive_persistence_budget() -> None:
    import pytest
    from moonlightbox.events.reviewer import TwoStageEventReviewer

    with pytest.raises(ValueError, match="字符预算"):
        TwoStageEventReviewer(FakeAnalysisClient([]), persistence_character_budget=0)


def test_review_prompt_limits_persistence_to_three_following_sessions() -> None:
    from moonlightbox.events.reviewer import EventCandidate, TwoStageEventReviewer

    messages = (
        make_message("m1", "甲", "开始", minutes=0),
        make_message("m2", "乙", "确认", minutes=1),
    )
    following = tuple(
        MessageSession(
            session_id=f"session-{index}",
            messages=(
                make_message(
                    f"follow-{index}",
                    "甲",
                    f"后续{index}",
                    minutes=10 + index,
                ),
            ),
        )
        for index in range(4)
    )
    client = FakeAnalysisClient(
        [
            {
                "accepted": False,
                "type_supported": True,
                "evidence_alignment": 0.9,
                "decisive_event": False,
                "persistence": 0,
                "evidence_ids": [],
                "reason": "后续证据不足",
            }
        ]
    )

    TwoStageEventReviewer(client).review_candidate(
        EventCandidate.model_validate(candidate_payload()),
        make_window(messages),
        PersistenceContext(
            remaining_session_messages=(),
            following_sessions=following,
        ),
    )

    prompt = json.loads(client.calls[0]["user_content"])
    assert [item["id"] for item in prompt["持续性上下文"]] == [
        "follow-0",
        "follow-1",
        "follow-2",
    ]
    assert prompt["元数据"]["truncated"] is True


def test_second_stage_parses_strict_schema_through_real_cloud_client() -> None:
    from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
    from moonlightbox.events.reviewer import EventCandidate, TwoStageEventReviewer

    captured_schema: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_schema.update(body["response_format"]["json_schema"]["schema"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "parsed": {
                                "accepted": True,
                                "type_supported": True,
                                "evidence_alignment": 0.9,
                                "decisive_event": False,
                                "persistence": 0.75,
                                "evidence_ids": ["m3"],
                                "reason": "后续持续以伴侣方式互动",
                            }
                        }
                    }
                ]
            },
        )

    cloud_client = NodeAnalysisCloudClient(
        enabled=True,
        endpoint="https://example.com/v1/chat/completions",
        model="analysis-model",
        api_key="secret",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    messages = (
        make_message("m1", "甲", "我们在一起吧", minutes=0),
        make_message("m2", "乙", "好", minutes=1),
        make_message("m3", "甲", "周末一起出门", minutes=2),
    )

    review = TwoStageEventReviewer(cloud_client).review_candidate(
        EventCandidate.model_validate(candidate_payload()),
        make_window(messages),
        PersistenceContext(remaining_session_messages=(), following_sessions=()),
    )

    assert review.accepted is True
    assert set(captured_schema["required"]) == set(captured_schema["properties"])
    assert captured_schema["additionalProperties"] is False


def _message_payload_for_assertion(message: NormalizedMessage) -> dict[str, str]:
    return {
        "id": message.source_id,
        "角色": message.sender,
        "时间": message.timestamp.isoformat(),
        "内容": message.content,
    }


def _evidence_payload_for_assertion(
    message: NormalizedMessage,
) -> dict[str, str]:
    return {
        "id": message.source_id,
        "sender": message.sender,
        "timestamp": message.timestamp.isoformat(),
        "kind": message.kind.value,
        "content": message.content,
    }
