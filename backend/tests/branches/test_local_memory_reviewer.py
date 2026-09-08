from datetime import UTC, datetime

from moonlightbox.branches.continuity_models import BranchMemoryEpisode
from moonlightbox.branches.continuity_types import MemoryCandidate, MemoryProposal
from moonlightbox.branches.memory_reviewer import ConservativeLocalMemoryReviewer


def _episode() -> BranchMemoryEpisode:
    now = datetime.now(UTC)
    return BranchMemoryEpisode(
        branch_id="branch-1",
        user_turn_id="user-turn",
        assistant_turn_id="assistant-turn",
        user_content="我周五回来",
        user_messages=[],
        assistant_bubbles=[{"type": "text", "content": "好，我等你"}],
        model_version_id="model-1",
        episode_hash="episode-hash",
        importance=8,
        processing_status="pending",
        started_at=now,
        ended_at=now,
    )


def test_local_reviewer_keeps_grounded_memory_but_never_state_delta() -> None:
    proposal = MemoryProposal(
        candidates=(
            MemoryCandidate(
                kind="self_narrative",
                content="我愿意等用户回来",
                subject="digital_human",
                predicate="交流意愿",
                object="等待",
                confidence=0.8,
                importance=7,
                evidence_message_ids=("assistant-turn",),
                source_role="digital_human",
            ),
        )
    )

    result = ConservativeLocalMemoryReviewer().review(_episode(), proposal)

    assert result.verdict == "approve"
    assert result.approved_candidate_indexes == (0,)
    assert result.approved_state_delta is None


def test_local_reviewer_rejects_out_of_episode_or_unattributed_claims() -> None:
    proposal = MemoryProposal(
        candidates=(
            MemoryCandidate(
                kind="fact",
                content="用户住在上海",
                subject="user",
                predicate="居住地",
                object="上海",
                confidence=0.8,
                importance=7,
                evidence_message_ids=("other-turn",),
                source_role="user",
            ),
            MemoryCandidate(
                kind="self_narrative",
                content="用户愿意回来",
                subject="user",
                predicate="意愿",
                object="回来",
                confidence=0.8,
                importance=7,
                evidence_message_ids=("user-turn",),
                source_role="user",
            ),
        )
    )

    result = ConservativeLocalMemoryReviewer().review(_episode(), proposal)

    assert result.verdict == "reject"
    assert result.approved_candidate_indexes == ()


def test_local_reflection_requires_multiple_approved_inputs() -> None:
    proposal = MemoryProposal(
        candidates=(
            MemoryCandidate(
                kind="reflection",
                content="我逐渐更珍惜稳定的互动",
                subject="digital_human",
                predicate="关系反思",
                object="珍惜稳定互动",
                confidence=0.8,
                importance=8,
                evidence_message_ids=("assistant-turn",),
                source_role="digital_human",
            ),
        )
    )
    reviewer = ConservativeLocalMemoryReviewer()

    rejected = reviewer.review(
        _episode(), proposal, current_state={"reflection_inputs": [{"id": "one"}]}
    )
    approved = reviewer.review(
        _episode(),
        proposal,
        current_state={"reflection_inputs": [{"id": "one"}, {"id": "two"}]},
    )

    assert rejected.verdict == "reject"
    assert approved.approved_candidate_indexes == (0,)
