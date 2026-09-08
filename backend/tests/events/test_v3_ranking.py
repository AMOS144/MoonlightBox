from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from itertools import combinations as real_combinations
from random import Random

import pytest
from moonlightbox.events.v3_reviewer import V3CandidateReview, V3EventCandidate

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _candidate(
    key: str,
    *,
    event_type: str = "conflict",
    lane: str | None = None,
    evidence_ids: list[str] | None = None,
    strength: float = 0.8,
    impact: float | None = None,
    confidence: float = 0.8,
    topic: str = "边界变化",
    title: str | None = None,
) -> V3EventCandidate:
    resolved_lane = lane or ("shared_experience" if event_type == "travel" else "relationship")
    return V3EventCandidate.model_validate(
        {
            "candidate_key": key,
            "lane": resolved_lane,
            "type": event_type,
            "title": title or key,
            "event_status": "occurred",
            "start_message_id": (evidence_ids or ["m1", "m2"])[0],
            "end_message_id": (evidence_ids or ["m1", "m2"])[-1],
            "summary": f"{key} 的事件摘要",
            "before_state": "原状态",
            "after_state": "新状态",
            "emotion_labels": ["认真"],
            "topic": topic,
            "conflict_level": 0,
            "event_significance": strength,
            "relationship_impact": impact if impact is not None else strength,
            "model_confidence": confidence,
            "reason": "发生了可验证的关系变化",
            "evidence_ids": evidence_ids or ["m1", "m2"],
        }
    )


def _review(
    *,
    persistence: float = 0.8,
    alignment: float = 0.9,
    type_support: float = 0.9,
    significance: float = 0.8,
    impact: float = 0.8,
    confidence: float = 0.8,
    facts_supported: bool = True,
    occurrence_supported: bool = True,
    bilateral_confirmation: bool = True,
    evidence_ids: list[str] | None = None,
) -> V3CandidateReview:
    return V3CandidateReview.model_validate(
        {
            "facts_supported": facts_supported,
            "occurrence_supported": occurrence_supported,
            "bilateral_confirmation": bilateral_confirmation,
            "evidence_alignment": alignment,
            "persistence": persistence,
            "type_support": type_support,
            "event_significance": significance,
            "relationship_impact": impact,
            "model_confidence": confidence,
            "evidence_ids": evidence_ids if evidence_ids is not None else ["m3"],
            "reason": "审查支持该事件",
        }
    )


def _rankable(
    key: str,
    *,
    event_type: str = "conflict",
    lane: str | None = None,
    offset_hours: float = 0,
    duration_hours: float = 1,
    evidence_ids: list[str] | None = None,
    strength: float = 0.8,
    confidence: float = 0.8,
    persistence: float = 0.8,
    alignment: float = 0.9,
    review_significance: float | None = None,
    review_impact: float | None = None,
    review_confidence: float | None = None,
    review_evidence_ids: list[str] | None = None,
    valid_follow_up_ids: tuple[str, ...] = ("m3",),
    bilateral_confirmation: bool = True,
    topic: str = "边界变化",
    title: str | None = None,
):
    from moonlightbox.events.v3_ranking import V3RankableCandidate

    return V3RankableCandidate(
        candidate=_candidate(
            key,
            event_type=event_type,
            lane=lane,
            evidence_ids=evidence_ids,
            strength=strength,
            confidence=confidence,
            topic=topic,
            title=title,
        ),
        review=_review(
            persistence=persistence,
            alignment=alignment,
            significance=review_significance if review_significance is not None else strength,
            impact=review_impact if review_impact is not None else strength,
            confidence=review_confidence if review_confidence is not None else confidence,
            evidence_ids=review_evidence_ids,
            bilateral_confirmation=bilateral_confirmation,
        ),
        started_at=BASE_TIME + timedelta(hours=offset_hours),
        ended_at=BASE_TIME + timedelta(hours=offset_hours + duration_hours),
        valid_follow_up_ids=valid_follow_up_ids,
        source_candidate_id=f"source-{key}",
    )


def test_v3_scores_are_frozen_and_use_the_exact_formula() -> None:
    from moonlightbox.events.v3_ranking import V3Scores, score_candidate

    scored = score_candidate(
        _rankable(
            "weighted",
            strength=0.8,
            persistence=0.4,
            alignment=0.5,
            confidence=0.6,
            review_significance=0.7,
            review_impact=0.6,
            review_confidence=0.5,
        )
    )

    expected = V3Scores(
        event_significance=0.7,
        relationship_impact=0.6,
        evidence_quality=0.5,
        persistence=0.4,
        type_support=0.9,
        model_confidence=0.5,
        total=0.6,
    )
    assert scored.scores == expected
    assert scored.scores.total == pytest.approx(0.6)
    with pytest.raises(FrozenInstanceError):
        scored.scores.total = 0.0  # type: ignore[misc]


def test_evidence_quality_is_bounded_by_completeness_not_quantity() -> None:
    from moonlightbox.events.v3_ranking import score_candidate

    normal = score_candidate(_rankable("normal", alignment=1.0))
    many_ids = [f"m{index}" for index in range(20)]
    many = replace(
        _rankable("many", evidence_ids=many_ids, alignment=1.0),
        valid_follow_up_ids=("m3",),
    )

    assert normal.scores.evidence_quality == 1.0
    assert score_candidate(many).scores.evidence_quality == 1.0


def test_evidence_quality_only_rewards_legal_follow_up_coverage() -> None:
    from moonlightbox.events.v3_ranking import score_candidate

    complete = score_candidate(
        _rankable(
            "complete-followup",
            alignment=1.0,
            review_evidence_ids=["m3", "m4"],
            valid_follow_up_ids=("m3", "m4"),
        )
    )
    partial = score_candidate(
        _rankable(
            "partial-followup",
            alignment=1.0,
            review_evidence_ids=["m3", "unrelated"],
            valid_follow_up_ids=("m3", "m4"),
        )
    )
    unrelated = score_candidate(
        _rankable(
            "unrelated-followup",
            alignment=1.0,
            review_evidence_ids=["unrelated"],
            valid_follow_up_ids=("m3",),
        )
    )
    empty_context = score_candidate(
        _rankable(
            "empty-followup",
            alignment=1.0,
            review_evidence_ids=["m3"],
            valid_follow_up_ids=(),
        )
    )

    assert complete.scores.evidence_quality == 1.0
    assert partial.scores.evidence_quality == 0.875
    assert unrelated.scores.evidence_quality == 0.75
    assert empty_context.scores.evidence_quality == 0.75


def test_low_persistence_high_evidence_travel_can_pass_default_threshold() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    travel = _rankable(
        "travel",
        event_type="travel",
        strength=0.8,
        persistence=0.05,
        alignment=1.0,
        confidence=0.8,
    )

    assert [item.candidate.title for item in rank_candidates([travel])] == ["travel"]


def test_threshold_filter_happens_before_diversity() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    below = _rankable(
        "below",
        event_type="travel",
        strength=0.59,
        persistence=0,
        alignment=0,
        confidence=0,
    )
    accepted = rank_candidates([below], threshold=0.60)

    assert accepted == []


def test_mmr_changes_selection_order_but_not_total() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    values = [
        _rankable("a", event_type="conflict", confidence=0.90),
        _rankable(
            "b",
            event_type="conflict",
            offset_hours=48,
            confidence=0.89,
        ),
        _rankable(
            "c",
            event_type="travel",
            offset_hours=96,
            confidence=0.60,
        ),
    ]

    ranked = rank_candidates(values, threshold=0)

    assert [item.candidate.title for item in ranked] == ["a", "c", "b"]
    assert ranked[2].scores.total > ranked[1].scores.total


def test_cross_lane_travel_and_intimacy_merge_and_union_provenance() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    travel = _rankable(
        "travel",
        event_type="travel",
        evidence_ids=["m1", "m2", "m3"],
        confidence=0.9,
        topic="一起去杭州旅行",
    )
    intimacy = _rankable(
        "intimacy",
        event_type="intimacy_increased",
        offset_hours=0.5,
        evidence_ids=["m2", "m3", "m4"],
        confidence=0.7,
        topic="一起去杭州旅行",
    )

    ranked = rank_candidates([intimacy, travel], threshold=0)

    assert len(ranked) == 1
    assert ranked[0].candidate.title == "travel"
    assert ranked[0].candidate.evidence_ids == ["m1", "m2", "m3", "m4"]
    assert ranked[0].source_lanes == ("relationship", "shared_experience")
    assert ranked[0].source_candidate_ids == ("source-intimacy", "source-travel")


def test_merge_preserves_source_scores_reviews_and_recomputes_conservative_scores() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    travel = _rankable(
        "travel-sources",
        event_type="travel",
        evidence_ids=["m1", "m2", "m3"],
        strength=0.9,
        confidence=0.9,
        alignment=1.0,
        persistence=0.8,
        review_evidence_ids=["m3"],
        valid_follow_up_ids=("m3",),
    )
    intimacy = _rankable(
        "intimacy-sources",
        event_type="intimacy_increased",
        evidence_ids=["m2", "m3"],
        strength=0.7,
        confidence=0.7,
        alignment=0.5,
        persistence=0.4,
        review_evidence_ids=["unrelated"],
        valid_follow_up_ids=("m3",),
        bilateral_confirmation=False,
    )

    merged = rank_candidates(
        [intimacy, travel],
        threshold=0,
        maximum_nodes=None,
    )[0]
    source_scores = dict(merged.source_scores)
    source_reviews = dict(merged.source_reviews)

    assert set(source_scores) == {
        "source-intimacy-sources",
        "source-travel-sources",
    }
    assert source_scores["source-travel-sources"].evidence_quality == 1.0
    assert source_scores["source-intimacy-sources"].evidence_quality == 0.375
    assert source_reviews["source-intimacy-sources"].evidence_ids == ("unrelated",)
    assert merged.review.evidence_ids == ["m3", "unrelated"]
    assert merged.review.bilateral_confirmation is False
    assert merged.scores.event_significance == 0.7
    assert merged.scores.relationship_impact == 0.7
    assert merged.scores.evidence_quality == 0.75
    assert merged.scores.persistence == 0.4
    assert merged.scores.type_support == 0.9
    assert merged.scores.model_confidence == 0.7
    assert merged.scores.total == pytest.approx(0.7025)


def test_complete_link_prevents_transitive_chain_merge() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    values = [
        _rankable("a", evidence_ids=["m1", "m2"], confidence=0.9, topic="事件 A"),
        _rankable(
            "b",
            evidence_ids=["m1", "m2", "m3"],
            confidence=0.8,
            topic="事件 B",
        ),
        _rankable("c", evidence_ids=["m2", "m3"], confidence=0.7, topic="事件 C"),
    ]

    ranked = rank_candidates(values, threshold=0)

    assert len(ranked) == 2
    assert sorted(source_id for item in ranked for source_id in item.source_candidate_ids) == [
        "source-a",
        "source-b",
        "source-c",
    ]
    assert max(len(item.source_candidate_ids) for item in ranked) == 2


def test_same_lane_different_types_need_high_time_and_evidence_overlap() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    weak_overlap = [
        _rankable(
            "conflict",
            event_type="conflict",
            evidence_ids=["m1", "m2", "m3"],
        ),
        _rankable(
            "separation",
            event_type="separation",
            offset_hours=0.5,
            evidence_ids=["m2", "m3", "m4"],
        ),
    ]

    assert len(rank_candidates(weak_overlap, threshold=0)) == 2


def test_same_lane_different_types_with_exact_evidence_do_not_merge_without_time_overlap() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    values = [
        _rankable(
            "conflict",
            event_type="conflict",
            evidence_ids=["shared-1", "shared-2"],
        ),
        _rankable(
            "separation",
            event_type="separation",
            offset_hours=20,
            evidence_ids=["shared-1", "shared-2"],
        ),
    ]

    assert len(rank_candidates(values, threshold=0)) == 2


def test_same_lane_different_types_with_exact_evidence_merge_on_high_time_overlap() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    values = [
        _rankable(
            "conflict",
            event_type="conflict",
            duration_hours=10,
            evidence_ids=["shared-1", "shared-2"],
        ),
        _rankable(
            "separation",
            event_type="separation",
            offset_hours=1,
            duration_hours=10,
            evidence_ids=["shared-1", "shared-2"],
        ),
    ]

    assert len(rank_candidates(values, threshold=0)) == 1


def test_same_lane_different_types_use_stricter_configured_jaccard_threshold() -> None:
    import moonlightbox.events.v3_ranking as ranking

    shared = [f"shared-{index}" for index in range(8)]
    left = ranking.score_candidate(
        _rankable(
            "conflict",
            event_type="conflict",
            duration_hours=10,
            evidence_ids=[*shared, "left-only"],
        )
    )
    jaccard_point_eight = ranking.score_candidate(
        _rankable(
            "separation-08",
            event_type="separation",
            duration_hours=10,
            evidence_ids=[*shared, "right-only"],
        )
    )
    jaccard_point_nine = ranking.score_candidate(
        _rankable(
            "separation-09",
            event_type="separation",
            duration_hours=10,
            evidence_ids=[*shared, "left-only", "right-only"],
        )
    )

    assert ranking._same_event(left, jaccard_point_eight, 0.9) is False
    assert (
        len(
            ranking.merge_ranked_candidates(
                [left, jaccard_point_eight],
                evidence_jaccard_threshold=0.9,
            )
        )
        == 2
    )
    assert ranking._same_event(left, jaccard_point_nine, 0.9) is True
    assert (
        len(
            ranking.merge_ranked_candidates(
                [left, jaccard_point_nine],
                evidence_jaccard_threshold=0.9,
            )
        )
        == 1
    )


def test_cross_lane_pair_uses_configured_jaccard_threshold() -> None:
    import moonlightbox.events.v3_ranking as ranking

    shared = [f"shared-{index}" for index in range(8)]
    travel = ranking.score_candidate(
        _rankable(
            "travel",
            event_type="travel",
            evidence_ids=[*shared, "travel-only"],
        )
    )
    intimacy = ranking.score_candidate(
        _rankable(
            "intimacy",
            event_type="intimacy_increased",
            evidence_ids=[*shared, "intimacy-only"],
        )
    )

    assert ranking._same_event(travel, intimacy, 0.9) is False
    assert ranking._same_event(travel, intimacy, 0.8) is True


def test_exact_signature_compression_preserves_global_first_fit_cluster_choice() -> None:
    import moonlightbox.events.v3_ranking as ranking

    values = [
        _rankable(
            "earlier-cluster",
            evidence_ids=["shared-1", "shared-2", "extra"],
            confidence=0.9,
        ),
        _rankable(
            "signature-first",
            offset_hours=40,
            evidence_ids=["shared-1", "shared-2"],
            confidence=0.8,
        ),
        _rankable(
            "signature-second",
            offset_hours=20,
            evidence_ids=["shared-1", "shared-2"],
            confidence=0.7,
        ),
    ]
    scored = sorted(
        (ranking.score_candidate(value) for value in values),
        key=ranking._cluster_order_key,
    )

    merged = ranking.merge_ranked_candidates(scored)

    assert sorted(item.source_candidate_ids for item in merged) == [
        ("source-earlier-cluster", "source-signature-second"),
        ("source-signature-first",),
    ]


def test_obviously_different_events_do_not_merge() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    values = [
        _rankable("first", event_type="conflict", evidence_ids=["m1", "m2"]),
        _rankable(
            "second",
            event_type="conflict",
            offset_hours=240,
            evidence_ids=["m1", "m2"],
        ),
    ]

    assert len(rank_candidates(values, threshold=0)) == 2


def test_shuffle_produces_stable_order_and_ties_use_total_start_key() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    values = [
        _rankable(
            f"candidate-{index:02}",
            event_type="travel" if index % 2 else "conflict",
            offset_hours=index * 48,
        )
        for index in range(12)
    ]
    expected = [
        item.candidate.title for item in rank_candidates(values, threshold=0, maximum_nodes=None)
    ]
    shuffled = list(values)
    Random(2026).shuffle(shuffled)

    assert [
        item.candidate.title for item in rank_candidates(shuffled, threshold=0, maximum_nodes=None)
    ] == expected


def test_default_maximum_is_25_and_input_is_not_modified() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    values = [
        _rankable(
            f"candidate-{index:02}",
            offset_hours=index * 48,
            evidence_ids=[f"m{index}-1", f"m{index}-2"],
        )
        for index in range(30)
    ]
    snapshot = [
        (value, value.candidate.model_dump(), value.review.model_dump()) for value in values
    ]

    ranked = rank_candidates(values, threshold=0)

    assert len(ranked) == 25
    assert [
        (value, value.candidate.model_dump(), value.review.model_dump()) for value in values
    ] == snapshot


@pytest.mark.parametrize("invalid_limit", [True, False, 1.5, "2", -1])
def test_maximum_nodes_requires_non_negative_strict_integer(
    invalid_limit: object,
) -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    with pytest.raises(ValueError):
        rank_candidates(
            [_rankable("limit")],
            threshold=0,
            maximum_nodes=invalid_limit,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_threshold", [float("nan"), float("inf"), -float("inf")])
def test_ranking_threshold_must_be_finite(invalid_threshold: float) -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    with pytest.raises(ValueError):
        rank_candidates(
            [_rankable("threshold")],
            threshold=invalid_threshold,
        )


def test_century_long_event_uses_bounded_month_buckets_and_finds_inner_overlap() -> None:
    import moonlightbox.events.v3_ranking as ranking

    long_event = replace(
        _rankable(
            "century",
            evidence_ids=["shared-1", "shared-2"],
            confidence=0.9,
        ),
        started_at=datetime(1900, 1, 1, tzinfo=UTC),
        ended_at=datetime(2000, 1, 1, tzinfo=UTC),
    )
    inner_event = replace(
        _rankable(
            "inner",
            evidence_ids=["shared-1", "shared-2"],
            confidence=0.8,
        ),
        started_at=datetime(1950, 6, 1, tzinfo=UTC),
        ended_at=datetime(1950, 6, 2, tzinfo=UTC),
    )

    bucket_keys = ranking._time_bucket_keys(
        long_event.started_at,
        long_event.ended_at,
        expand=True,
    )
    merged = ranking.rank_candidates(
        [long_event, inner_event],
        threshold=0,
        maximum_nodes=None,
    )

    assert len(bucket_keys) <= 6
    assert len(merged) == 1


def test_active_postings_expire_far_shared_evidence_near_linearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.v3_ranking as ranking

    scanned_postings = 0
    comparisons = 0
    original_active = ranking._active_prefix_postings
    original_same_event = ranking._same_event

    def counted_active(*args: object, **kwargs: object):
        nonlocal scanned_postings
        active = original_active(*args, **kwargs)  # type: ignore[arg-type]
        scanned_postings += len(active)
        return active

    def counted_same_event(*args: object, **kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original_same_event(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ranking, "_active_prefix_postings", counted_active)
    monkeypatch.setattr(ranking, "_same_event", counted_same_event)
    values = [
        _rankable(
            f"far-{index:04}",
            offset_hours=index * 72,
            duration_hours=0,
            evidence_ids=["shared-frequency-id", f"unique-{index:04}"],
        )
        for index in range(1000)
    ]

    ranked = ranking.rank_candidates(
        list(reversed(values)),
        threshold=0,
        maximum_nodes=None,
    )

    assert len(ranked) == 1000
    assert scanned_postings < 3000
    assert comparisons == 0


def test_semantic_candidates_only_scan_active_time_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.v3_ranking as ranking

    scanned_postings = 0
    original_active = ranking._active_interval_postings

    def counted_active(*args: object, **kwargs: object):
        nonlocal scanned_postings
        active = original_active(*args, **kwargs)  # type: ignore[arg-type]
        scanned_postings += len(active)
        return active

    monkeypatch.setattr(ranking, "_active_interval_postings", counted_active)
    values = [
        _rankable(
            f"semantic-far-{index:04}",
            offset_hours=index * 72,
            duration_hours=0,
            evidence_ids=[f"semantic-{index:04}-a", f"semantic-{index:04}-b"],
            title="相同标题",
            topic="相同语义主题",
        )
        for index in range(1000)
    ]

    ranked = ranking.rank_candidates(
        list(reversed(values)),
        threshold=0,
        maximum_nodes=None,
    )

    assert len(ranked) == 1000
    assert scanned_postings < 3000


def test_zero_jaccard_candidates_only_scan_active_global_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.v3_ranking as ranking

    scanned_postings = 0
    original_active = ranking._active_interval_postings

    def counted_active(*args: object, **kwargs: object):
        nonlocal scanned_postings
        active = original_active(*args, **kwargs)  # type: ignore[arg-type]
        scanned_postings += len(active)
        return active

    monkeypatch.setattr(ranking, "_active_interval_postings", counted_active)
    scored = [
        ranking.score_candidate(
            _rankable(
                f"global-far-{index:04}",
                offset_hours=index * 72,
                duration_hours=0,
                evidence_ids=[f"global-{index:04}-a", f"global-{index:04}-b"],
                title=f"不同标题-{index:04}",
                topic=f"不同主题-{index:04}",
            )
        )
        for index in range(1000)
    ]

    merged = ranking.merge_ranked_candidates(
        list(reversed(scored)),
        evidence_jaccard_threshold=0,
    )

    assert len(merged) == 1000
    assert scanned_postings < 3000


def test_semantic_and_global_active_indexes_keep_long_overlaps() -> None:
    import moonlightbox.events.v3_ranking as ranking

    long_semantic = replace(
        _rankable(
            "long-semantic",
            evidence_ids=["semantic-long-a", "semantic-long-b"],
            title="长期事件",
            topic="长期主题",
        ),
        started_at=datetime(1900, 1, 1, tzinfo=UTC),
        ended_at=datetime(2000, 1, 1, tzinfo=UTC),
    )
    short_semantic = replace(
        _rankable(
            "short-semantic",
            evidence_ids=["semantic-short-a", "semantic-short-b"],
            title="长期事件",
            topic="长期主题",
        ),
        started_at=datetime(1950, 1, 1, tzinfo=UTC),
        ended_at=datetime(1950, 1, 2, tzinfo=UTC),
    )
    global_values = [
        replace(
            _rankable(
                "long-global",
                evidence_ids=["global-long-a", "global-long-b"],
                title="全局长期事件",
            ),
            started_at=datetime(1900, 1, 1, tzinfo=UTC),
            ended_at=datetime(2000, 1, 1, tzinfo=UTC),
        ),
        replace(
            _rankable(
                "short-global",
                evidence_ids=["global-short-a", "global-short-b"],
                title="全局短期事件",
            ),
            started_at=datetime(1950, 1, 1, tzinfo=UTC),
            ended_at=datetime(1950, 1, 2, tzinfo=UTC),
        ),
    ]

    assert (
        len(
            ranking.rank_candidates(
                [short_semantic, long_semantic],
                threshold=0,
                maximum_nodes=None,
            )
        )
        == 1
    )
    assert (
        len(
            ranking.merge_ranked_candidates(
                [ranking.score_candidate(value) for value in reversed(global_values)],
                evidence_jaccard_threshold=0,
            )
        )
        == 1
    )


def test_active_postings_recall_adjacent_month_year_and_long_overlap() -> None:
    from moonlightbox.events.v3_ranking import rank_candidates

    adjacent_year = [
        replace(
            _rankable("year-left", evidence_ids=["year-1", "year-2"]),
            started_at=datetime(2025, 12, 31, 23, tzinfo=UTC),
            ended_at=datetime(2025, 12, 31, 23, 30, tzinfo=UTC),
        ),
        replace(
            _rankable("year-right", evidence_ids=["year-1", "year-2"]),
            started_at=datetime(2026, 1, 1, 0, tzinfo=UTC),
            ended_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
        ),
    ]
    adjacent_month = [
        replace(
            _rankable("month-left", evidence_ids=["month-1", "month-2"]),
            started_at=datetime(2026, 1, 31, 23, tzinfo=UTC),
            ended_at=datetime(2026, 1, 31, 23, 30, tzinfo=UTC),
        ),
        replace(
            _rankable("month-right", evidence_ids=["month-1", "month-2"]),
            started_at=datetime(2026, 2, 1, 0, tzinfo=UTC),
            ended_at=datetime(2026, 2, 1, 1, tzinfo=UTC),
        ),
    ]
    long_and_inner = [
        replace(
            _rankable("long-active", evidence_ids=["long-1", "long-2"]),
            started_at=datetime(1900, 1, 1, tzinfo=UTC),
            ended_at=datetime(2000, 1, 1, tzinfo=UTC),
        ),
        replace(
            _rankable("short-inner", evidence_ids=["long-1", "long-2"]),
            started_at=datetime(1950, 1, 1, tzinfo=UTC),
            ended_at=datetime(1950, 1, 2, tzinfo=UTC),
        ),
    ]

    assert len(rank_candidates(adjacent_year, threshold=0)) == 1
    assert len(rank_candidates(adjacent_month, threshold=0)) == 1
    assert len(rank_candidates(long_and_inner, threshold=0)) == 1


def test_time_comparison_handles_datetime_min_max_without_overflow() -> None:
    import moonlightbox.events.v3_ranking as ranking

    minimum = datetime.min.replace(tzinfo=UTC)
    maximum = datetime.max.replace(tzinfo=UTC)

    assert ranking._time_values_are_close(
        minimum,
        minimum,
        minimum,
        minimum,
    )
    assert ranking._time_values_are_close(
        maximum,
        maximum,
        maximum,
        maximum,
    )
    assert not ranking._time_values_are_close(
        minimum,
        minimum,
        maximum,
        maximum,
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1])
def test_scores_reject_non_finite_or_out_of_range_values(invalid: float) -> None:
    from moonlightbox.events.v3_ranking import V3Scores

    with pytest.raises(ValueError):
        V3Scores(
            event_significance=invalid,
            relationship_impact=0.5,
            evidence_quality=0.5,
            persistence=0.5,
            type_support=1,
            model_confidence=0.5,
            total=0.5,
        )


def test_one_thousand_candidates_use_bounded_complete_link_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.v3_ranking as ranking

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
            offset_hours=index * 48,
            evidence_ids=[f"m{index}-1", f"m{index}-2"],
            topic="相同主题但不同时间的事件",
        )
        for index in range(1000)
    ]

    ranked = ranking.rank_candidates(
        values,
        threshold=0,
        maximum_nodes=None,
    )

    assert len(ranked) == 1000
    assert comparisons < 200


def test_high_frequency_evidence_avoids_pairwise_scan_but_signatures_still_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.v3_ranking as ranking

    comparisons = 0
    original = ranking._same_event

    def counted(*args: object, **kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ranking, "_same_event", counted)
    distinct = [
        _rankable(
            f"distinct-{index:03}",
            offset_hours=(index % 20) / 100,
            evidence_ids=["common-id", f"unique-{index:03}"],
        )
        for index in range(300)
    ]
    duplicates = [
        _rankable(
            f"duplicate-{index:02}",
            offset_hours=index / 1000,
            evidence_ids=["common-id", "duplicate-id"],
            topic="同一个重复事件",
        )
        for index in range(20)
    ]

    ranked = ranking.rank_candidates(
        [*distinct, *duplicates],
        threshold=0,
        maximum_nodes=None,
    )

    assert len(ranked) == 301
    assert comparisons < 1200


def test_two_shared_frequent_ids_reach_jaccard_threshold_without_rare_posting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.v3_ranking as ranking

    comparisons = 0
    original = ranking._same_event

    def counted(*args: object, **kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ranking, "_same_event", counted)
    values = [
        _rankable(
            f"similar-{index:02}",
            offset_hours=index / 1000,
            evidence_ids=["frequent-a", "frequent-b", f"unique-{index:02}"],
        )
        for index in range(20)
    ]

    ranked = ranking.rank_candidates(values, threshold=0, maximum_nodes=None)

    assert len(ranked) == 1
    assert comparisons <= 190


def test_one_shared_id_below_jaccard_threshold_generates_no_pair_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.v3_ranking as ranking

    generated_candidates = 0
    comparisons = 0
    original_generator = ranking._overlap_candidate_cluster_ids
    original_same_event = ranking._same_event

    def counted_generator(*args: object, **kwargs: object) -> set[int]:
        nonlocal generated_candidates
        generated = original_generator(*args, **kwargs)  # type: ignore[arg-type]
        generated_candidates += len(generated)
        return generated

    def counted_same_event(*args: object, **kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original_same_event(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ranking, "_overlap_candidate_cluster_ids", counted_generator)
    monkeypatch.setattr(ranking, "_same_event", counted_same_event)
    values = [
        _rankable(
            f"below-{index:03}",
            offset_hours=(index % 20) / 100,
            evidence_ids=["common-id", f"unique-{index:03}"],
        )
        for index in range(300)
    ]

    ranked = ranking.rank_candidates(values, threshold=0, maximum_nodes=None)

    assert len(ranked) == 300
    assert generated_candidates < 600
    assert comparisons < 10


def test_three_hundred_identical_signatures_merge_without_pairwise_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.v3_ranking as ranking

    comparisons = 0
    original = ranking._same_event

    def counted(*args: object, **kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ranking, "_same_event", counted)
    values = [
        _rankable(
            f"identical-{index:03}",
            evidence_ids=["same-a", "same-b", "same-c"],
            title="完全相同事件",
        )
        for index in range(300)
    ]

    ranked = ranking.rank_candidates(values, threshold=0, maximum_nodes=None)

    assert len(ranked) == 1
    assert comparisons == 0


@pytest.mark.parametrize(
    ("evidence_count", "threshold", "expected"),
    [
        (20, 0.5, 11),
        (3, 0.5, 2),
        (2, 0.5, 2),
        (1, 0.5, 1),
        (20, 1.0, 1),
    ],
)
def test_jaccard_prefix_length_formula(
    evidence_count: int,
    threshold: float,
    expected: int,
) -> None:
    from moonlightbox.events.v3_ranking import _prefix_length

    assert _prefix_length(evidence_count, threshold) == expected


@pytest.mark.parametrize("candidate_count", [10, 1000])
def test_disjoint_twenty_evidence_sets_do_not_enumerate_combinations(
    candidate_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import moonlightbox.events.v3_ranking as ranking

    comparisons = 0
    original_same_event = ranking._same_event

    def reject_large_combinations(
        values: list[str],
        size: int,
    ):
        if size > 1:
            raise AssertionError("禁止枚举多 token 组合")
        return real_combinations(values, size)

    def counted_same_event(*args: object, **kwargs: object) -> bool:
        nonlocal comparisons
        comparisons += 1
        return original_same_event(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ranking,
        "combinations",
        reject_large_combinations,
        raising=False,
    )
    monkeypatch.setattr(ranking, "_same_event", counted_same_event)
    values = [
        _rankable(
            f"wide-{index:04}",
            offset_hours=(index % 20) / 100,
            evidence_ids=[
                f"candidate-{index:04}-evidence-{evidence_index:02}" for evidence_index in range(20)
            ],
        )
        for index in range(candidate_count)
    ]

    ranked = ranking.rank_candidates(values, threshold=0, maximum_nodes=None)

    assert len(ranked) == candidate_count
    assert comparisons == 0


@pytest.mark.parametrize(
    "seed",
    [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
)
def test_allpairs_matches_bruteforce_complete_link_oracle(seed: int) -> None:
    import moonlightbox.events.v3_ranking as ranking

    random = Random(seed)
    universe = [f"evidence-{index:02}" for index in range(12)]
    for round_index in range(30):
        values = [
            _rankable(
                f"oracle-{round_index:02}-{index:02}",
                offset_hours=index / 1000,
                evidence_ids=random.sample(
                    universe,
                    random.randint(2, 6),
                ),
                topic=f"随机事件-{round_index}-{index}",
            )
            for index in range(12)
        ]
        values.extend(
            [
                _rankable(
                    f"oracle-{round_index:02}-duplicate-{index}",
                    evidence_ids=list(values[0].candidate.evidence_ids),
                    topic=values[0].candidate.topic,
                    title=values[0].candidate.title,
                )
                for index in range(2)
            ]
        )
        scored = sorted(
            (ranking.score_candidate(value) for value in values),
            key=ranking._cluster_order_key,
        )
        oracle_clusters: list[list[object]] = []
        for candidate in scored:
            matching_cluster = next(
                (
                    cluster
                    for cluster in oracle_clusters
                    if all(ranking._same_event(candidate, member, 0.5) for member in cluster)
                ),
                None,
            )
            if matching_cluster is None:
                oracle_clusters.append([candidate])
            else:
                matching_cluster.append(candidate)
        expected = sorted(
            tuple(
                sorted(
                    source_id
                    for member in cluster
                    for source_id in member.source_candidate_ids  # type: ignore[attr-defined]
                )
            )
            for cluster in oracle_clusters
        )

        merged = ranking.merge_ranked_candidates(scored)
        actual = sorted(item.source_candidate_ids for item in merged)
        shuffled = list(scored)
        random.shuffle(shuffled)
        shuffled_actual = sorted(
            item.source_candidate_ids for item in ranking.merge_ranked_candidates(shuffled)
        )

        assert actual == expected
        assert shuffled_actual == expected
