from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from moonlightbox.events.reviewer import EventCandidate, EventCandidateReview

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _candidate(
    key: str,
    *,
    event_type: str = "conflict",
    start_id: str = "m1",
    end_id: str = "m2",
    evidence_ids: list[str] | None = None,
    strength: float = 0.8,
    confidence: float = 0.8,
    topic: str = "边界冲突",
) -> EventCandidate:
    return EventCandidate.model_validate(
        {
            "candidate_key": key,
            "type": event_type,
            "start_message_id": start_id,
            "end_message_id": end_id,
            "before_state": "正常交流",
            "after_state": "关系紧张",
            "emotion_labels": ["生气"],
            "topic": topic,
            "conflict_level": 4,
            "state_change_strength": strength,
            "model_confidence": confidence,
            "reason": "双方就边界持续争执",
            "evidence_ids": evidence_ids or [start_id, end_id],
        }
    )


def _review(
    *,
    persistence: float = 0.8,
    evidence_ids: list[str] | None = None,
) -> EventCandidateReview:
    return EventCandidateReview(
        accepted=True,
        type_supported=True,
        evidence_alignment=0.9,
        decisive_event=False,
        persistence=persistence,
        evidence_ids=evidence_ids or ["m3"],
        reason="后续仍在讨论同一冲突",
    )


def _rankable(
    key: str,
    *,
    offset: int = 0,
    event_type: str = "conflict",
    start_id: str = "m1",
    end_id: str = "m2",
    evidence_ids: list[str] | None = None,
    strength: float = 0.8,
    persistence: float = 0.8,
    confidence: float = 0.8,
    topic: str = "边界冲突",
):
    from moonlightbox.events.ranking import RankableCandidate

    return RankableCandidate(
        candidate=_candidate(
            key,
            event_type=event_type,
            start_id=start_id,
            end_id=end_id,
            evidence_ids=evidence_ids,
            strength=strength,
            confidence=confidence,
            topic=topic,
        ),
        review=_review(persistence=persistence),
        started_at=BASE_TIME + timedelta(minutes=offset),
        ended_at=BASE_TIME + timedelta(minutes=offset + 5),
        valid_follow_up_ids=("m3", "m4"),
    )


def test_score_uses_required_weights_and_exposes_components() -> None:
    from moonlightbox.events.ranking import score_candidate

    scored = score_candidate(
        _rankable(
            "weighted",
            strength=1.0,
            persistence=0.5,
            confidence=0.4,
        )
    )

    assert scored.scores == {
        "state_change_strength": 1.0,
        "persistence": 0.5,
        "evidence_quality": 1.0,
        "model_confidence": 0.4,
        "total": pytest.approx(0.76),
    }


def test_evidence_quality_is_deterministic_and_rewards_legal_follow_up() -> None:
    from moonlightbox.events.ranking import score_candidate

    complete = score_candidate(_rankable("complete"))
    invalid_follow_up = replace(
        _rankable("invalid"),
        review=_review(evidence_ids=["unknown"]),
    )

    assert complete.scores["evidence_quality"] == 1.0
    assert score_candidate(invalid_follow_up).scores["evidence_quality"] == 0.6


def test_threshold_boundary_is_inclusive_and_does_not_pad_results() -> None:
    from moonlightbox.events.ranking import rank_candidates

    # 0.8*0.35 + 0.6*0.30 + 1*0.20 + 0.4*0.15 = 0.72
    boundary = _rankable(
        "boundary",
        strength=0.8,
        persistence=0.6,
        confidence=0.4,
    )
    below = _rankable(
        "below",
        strength=0.8,
        persistence=0.6,
        confidence=0.399,
    )

    ranked = rank_candidates([below, boundary])

    assert [item.candidate.candidate_key for item in ranked] == ["boundary"]


def test_ranking_caps_at_25_and_is_stable_for_ties() -> None:
    from moonlightbox.events.ranking import rank_candidates

    candidates = [_rankable(f"candidate-{index:02}", offset=index % 2) for index in range(30)]

    ranked = rank_candidates(reversed(candidates), threshold=0.0)

    assert len(ranked) == 25
    assert [item.candidate.candidate_key for item in ranked[:3]] == [
        "candidate-00",
        "candidate-02",
        "candidate-04",
    ]


def test_duplicate_event_candidates_keep_highest_score_representative() -> None:
    from moonlightbox.events.ranking import merge_ranked_candidates, rank_candidates

    first = _rankable("first", evidence_ids=["m1", "m2", "m3"], confidence=0.9)
    second = _rankable(
        "second",
        offset=2,
        evidence_ids=["m2", "m3", "m4"],
        confidence=0.8,
    )

    merged = merge_ranked_candidates(
        rank_candidates([first, second], threshold=0.0),
        evidence_order={"m1": 1, "m2": 2, "m3": 3, "m4": 4},
    )

    assert len(merged) == 1
    assert merged[0].candidate.candidate_key == "first"
    assert merged[0].candidate.evidence_ids == ["m1", "m2", "m3"]


def test_normalized_semantic_key_merges_overlapping_ranges() -> None:
    from moonlightbox.events.ranking import merge_ranked_candidates, rank_candidates

    first = _rankable(
        "first",
        start_id="m1",
        end_id="m2",
        evidence_ids=["m1", "m2"],
        topic="边界 冲突！",
    )
    second = _rankable(
        "second",
        offset=2,
        start_id="m10",
        end_id="m11",
        evidence_ids=["m10", "m11"],
        topic="边界冲突",
    )

    merged = merge_ranked_candidates(rank_candidates([first, second], threshold=0.0))

    assert len(merged) == 1


def test_different_type_or_distant_events_do_not_merge() -> None:
    from moonlightbox.events.ranking import merge_ranked_candidates, rank_candidates

    same_evidence_different_type = _rankable(
        "reconciliation",
        event_type="reconciliation",
        evidence_ids=["m1", "m2", "m3"],
    )
    distant = _rankable(
        "distant",
        offset=10_000,
        start_id="m10",
        end_id="m11",
        evidence_ids=["m1", "m2", "m3"],
    )
    original = _rankable("original", evidence_ids=["m1", "m2", "m3"])

    merged = merge_ranked_candidates(
        rank_candidates(
            [same_evidence_different_type, distant, original],
            threshold=0.0,
        )
    )

    assert len(merged) == 3


def test_merge_is_order_independent() -> None:
    from moonlightbox.events.ranking import merge_ranked_candidates, rank_candidates

    values = [
        _rankable("a", evidence_ids=["m1", "m2", "m3"], confidence=0.9),
        _rankable("b", offset=1, evidence_ids=["m2", "m3", "m4"]),
        _rankable("c", offset=2, evidence_ids=["m3", "m4", "m5"]),
    ]

    forward = merge_ranked_candidates(
        rank_candidates(values, threshold=0.0),
        evidence_order={f"m{index}": index for index in range(1, 6)},
    )
    backward = merge_ranked_candidates(
        rank_candidates(reversed(values), threshold=0.0),
        evidence_order={f"m{index}": index for index in range(1, 6)},
    )

    assert [item.candidate.model_dump() for item in forward] == [
        item.candidate.model_dump() for item in backward
    ]


def test_complete_link_does_not_merge_similarity_chain() -> None:
    from moonlightbox.events.ranking import merge_ranked_candidates, rank_candidates

    values = [
        _rankable(
            "a",
            evidence_ids=["m1", "m2"],
            confidence=0.9,
            topic="事件 A",
        ),
        _rankable(
            "b",
            evidence_ids=["m1", "m2", "m3"],
            confidence=0.8,
            topic="事件 B",
        ),
        _rankable(
            "c",
            evidence_ids=["m2", "m3"],
            confidence=0.7,
            topic="事件 C",
        ),
    ]

    merged = merge_ranked_candidates(rank_candidates(values, threshold=0.0))

    assert [item.candidate.candidate_key for item in merged] == ["a", "c"]


def test_merge_keeps_representative_evidence_scores_and_boundaries() -> None:
    from moonlightbox.events.ranking import merge_ranked_candidates, rank_candidates

    representative = _rankable(
        "representative",
        evidence_ids=["m1", "m2"],
        confidence=0.9,
    )
    duplicate = _rankable(
        "duplicate",
        offset=2,
        evidence_ids=["m1", "m2", "outside-boundary"],
        confidence=0.8,
    )
    ranked = rank_candidates([representative, duplicate], threshold=0.0)

    merged = merge_ranked_candidates(ranked)

    assert len(merged) == 1
    assert merged[0].candidate.evidence_ids == ["m1", "m2"]
    assert merged[0].scores == ranked[0].scores
    assert merged[0].started_at == ranked[0].started_at
    assert merged[0].ended_at == ranked[0].ended_at


def test_semantic_bucketing_avoids_unbounded_pairwise_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.ranking as ranking

    comparisons = 0
    original = ranking._same_event

    def counted(*args: object, **kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ranking, "_same_event", counted)
    values = [
        _rankable(
            f"candidate-{index:04}",
            offset=index % 3,
            evidence_ids=[f"m{index}", f"m{index + 1}"],
        )
        for index in range(1000)
    ] + [
        _rankable(
            "reconciliation",
            event_type="reconciliation",
            evidence_ids=["r1", "r2"],
        )
    ]

    merged = ranking.merge_ranked_candidates(
        ranking.rank_candidates(values, threshold=0.0, maximum_nodes=None)
    )

    assert {item.candidate.type for item in merged} == {
        "conflict",
        "reconciliation",
    }
    assert comparisons < 100


def test_twenty_sixth_cluster_is_not_truncated_before_real_duplicate() -> None:
    from moonlightbox.events.ranking import merge_ranked_candidates, rank_candidates

    interferers = [
        _rankable(
            f"interferer-{index:02}",
            offset=index * 60,
            evidence_ids=[
                "target-1",
                f"i{index}-1",
                f"i{index}-2",
                f"i{index}-3",
            ],
            confidence=0.99,
            topic=f"干扰事件-{index}",
        )
        for index in range(25)
    ]
    representative = _rankable(
        "representative",
        offset=2_000,
        evidence_ids=["target-1", "target-2"],
        confidence=0.8,
        topic="目标代表",
    )
    duplicate = _rankable(
        "duplicate",
        offset=2_001,
        evidence_ids=["target-1", "target-2"],
        confidence=0.7,
        topic="目标重复",
    )

    merged = merge_ranked_candidates(
        rank_candidates(
            [*interferers, representative, duplicate],
            threshold=0.0,
            maximum_nodes=None,
        )
    )

    assert len(merged) == 26
    assert (
        sum(item.candidate.candidate_key in {"representative", "duplicate"} for item in merged) == 1
    )


@pytest.mark.parametrize("long_has_higher_score", [True, False])
def test_event_spanning_more_than_24_months_merges_with_inner_short_event(
    long_has_higher_score: bool,
) -> None:
    from moonlightbox.events.ranking import merge_ranked_candidates, rank_candidates

    long_event = replace(
        _rankable(
            "long",
            confidence=0.9 if long_has_higher_score else 0.7,
            evidence_ids=["long-1", "long-2"],
        ),
        started_at=datetime(2020, 1, 1, tzinfo=UTC),
        ended_at=datetime(2023, 1, 1, tzinfo=UTC),
    )
    short_event = replace(
        _rankable(
            "short",
            confidence=0.7 if long_has_higher_score else 0.9,
            evidence_ids=["short-1", "short-2"],
        ),
        started_at=datetime(2021, 6, 1, tzinfo=UTC),
        ended_at=datetime(2021, 6, 2, tzinfo=UTC),
    )

    merged = merge_ranked_candidates(
        rank_candidates(
            [long_event, short_event],
            threshold=0.0,
            maximum_nodes=None,
        )
    )

    assert len(merged) == 1
    expected = "long" if long_has_higher_score else "short"
    assert merged[0].candidate.candidate_key == expected


def test_thousands_of_disjoint_candidates_have_bounded_similarity_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.ranking as ranking

    comparisons = 0
    original = ranking._same_event

    def counted(*args: object, **kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ranking, "_same_event", counted)
    values = [
        _rankable(
            f"disjoint-{index:04}",
            offset=index,
            evidence_ids=[f"e{index}-1", f"e{index}-2"],
            topic=f"事件-{index}",
        )
        for index in range(2000)
    ]

    merged = ranking.merge_ranked_candidates(
        ranking.rank_candidates(values, threshold=0.0, maximum_nodes=None)
    )

    assert len(merged) == 2000
    assert comparisons < 100


def test_thousands_of_exact_evidence_duplicates_avoid_pairwise_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.ranking as ranking

    comparisons = 0
    original = ranking._same_event

    def counted(*args: object, **kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ranking, "_same_event", counted)
    values = [
        _rankable(
            f"duplicate-{index:04}",
            evidence_ids=["shared-1", "shared-2"],
            topic=f"不同表述-{index}",
        )
        for index in range(1000)
    ]

    merged = ranking.merge_ranked_candidates(
        ranking.rank_candidates(values, threshold=0.0, maximum_nodes=None)
    )

    assert len(merged) == 1
    assert comparisons < 100
