import json
from datetime import UTC, datetime

import pytest
from moonlightbox.branches.continuity_models import BranchMemoryEpisode
from moonlightbox.branches.continuity_types import (
    MemoryCandidate,
    MemoryProposal,
    StateDeltaProposal,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class FakeClient:
    def __init__(self, result: dict[str, object]) -> None:
        self.result = result
        self.last_request: dict[str, object] | None = None

    def post(self, *args: object, **kwargs: object) -> FakeResponse:
        self.last_request = kwargs["json"]  # type: ignore[assignment]
        return FakeResponse(
            {
                "choices": [
                    {"message": {"content": json.dumps(self.result, ensure_ascii=False)}}
                ]
            }
        )


def _episode() -> BranchMemoryEpisode:
    return BranchMemoryEpisode(
        id="episode-1",
        branch_id="branch-1",
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
        user_content="今天聊得挺开心",
        assistant_bubbles=[{"type": "text", "content": "我也是"}],
        model_version_id="model-1",
        episode_hash="hash",
        importance=5,
        processing_status="pending",
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
    )


def _proposal(evidence_id: str = "assistant-turn") -> MemoryProposal:
    return MemoryProposal(
        candidates=(
            MemoryCandidate(
                kind="self_narrative",
                content="我享受这次交流",
                subject="digital_human",
                predicate="感受",
                object="开心",
                confidence=0.7,
                importance=5,
                evidence_message_ids=(evidence_id,),
                source_role="digital_human",
            ),
        ),
        state_delta=StateDeltaProposal(
            relationship_delta={"trust": 1},
            supporting_candidate_indexes=(0,),
        ),
    )


def test_reviewer_rejects_candidate_with_unknown_evidence() -> None:
    from moonlightbox.branches.memory_reviewer import (
        DeepSeekMemoryReviewer,
        MemoryReviewFailedError,
    )

    reviewer = DeepSeekMemoryReviewer(
        endpoint="https://example.com/chat",
        model="deepseek",
        api_key="secret",
        client=FakeClient(
            {
                "verdict": "approve",
                "approved_candidate_indexes": [0],
                "approved_state_delta": None,
                "rejected_reasons": [],
            }
        ),
    )

    with pytest.raises(MemoryReviewFailedError):
        reviewer.review(_episode(), _proposal("not-in-episode"))


def test_reviewer_cannot_add_candidates() -> None:
    from moonlightbox.branches.memory_reviewer import (
        DeepSeekMemoryReviewer,
        MemoryReviewFailedError,
    )

    reviewer = DeepSeekMemoryReviewer(
        endpoint="https://example.com/chat",
        model="deepseek",
        api_key="secret",
        client=FakeClient(
            {
                "verdict": "approve",
                "approved_candidate_indexes": [0, 1],
                "approved_state_delta": None,
                "rejected_reasons": [],
            }
        ),
    )

    with pytest.raises(MemoryReviewFailedError):
        reviewer.review(_episode(), _proposal())


def test_reviewer_approves_subset_with_minimal_evidence_packet() -> None:
    from moonlightbox.branches.memory_reviewer import DeepSeekMemoryReviewer

    client = FakeClient(
        {
            "verdict": "approve",
            "approved_candidate_indexes": [0],
            "approved_state_delta": {
                "relationship_delta": {"trust": 0.5},
                "emotional_delta": {},
                "user_model_updates": {},
                "supporting_candidate_indexes": [0],
            },
            "rejected_reasons": [],
        }
    )
    reviewer = DeepSeekMemoryReviewer(
        endpoint="https://example.com/chat",
        model="deepseek",
        api_key="secret",
        client=client,
    )

    result = reviewer.review(
        _episode(),
        _proposal(),
        identity_kernel={"relationship_boundaries": ["不接受被定义感受"]},
        current_state={"active_belief_ids": ["belief-1"]},
        competing_beliefs=("belief-2",),
    )

    assert result.approved_candidate_indexes == (0,)
    assert result.approved_state_delta is not None
    assert result.approved_state_delta.relationship_delta == {"trust": 0.5}
    assert client.last_request is not None
    messages = client.last_request["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    review_input = json.loads(messages[1]["content"])
    assert review_input["identity_kernel"]["relationship_boundaries"] == [
        "不接受被定义感受"
    ]
    assert review_input["current_state"]["active_belief_ids"] == ["belief-1"]
    assert review_input["competing_beliefs"] == ["belief-2"]


def test_reviewer_normalizes_missing_verdict() -> None:
    from moonlightbox.branches.memory_reviewer import DeepSeekMemoryReviewer

    reviewer = DeepSeekMemoryReviewer(
        endpoint="https://example.com/chat",
        model="deepseek",
        api_key="secret",
        client=FakeClient(
            {
                "approved_candidate_indexes": [0],
                "approved_state_delta": None,
                "rejected_reasons": [],
            }
        ),
    )

    result = reviewer.review(_episode(), _proposal())

    assert result.verdict == "approve"
    assert result.approved_candidate_indexes == (0,)
