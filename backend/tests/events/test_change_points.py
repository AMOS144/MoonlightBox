def test_combined_change_signals_rank_as_important() -> None:
    from moonlightbox.events.scoring import ChangeSignals, score_change

    signals = ChangeSignals(
        topic_delta=0.8,
        emotion_delta=0.9,
        intent_delta=0.7,
        response_gap_delta=0.4,
        persistence=0.8,
    )

    assert score_change(signals) > 0.75


def test_single_noisy_signal_does_not_become_turning_point() -> None:
    from moonlightbox.events.scoring import ChangeSignals, score_change

    signals = ChangeSignals(
        topic_delta=1.0,
        emotion_delta=0.0,
        intent_delta=0.0,
        response_gap_delta=0.0,
        persistence=0.0,
    )

    assert score_change(signals) < 0.4


def test_only_high_scoring_episode_boundaries_become_candidates() -> None:
    from datetime import datetime

    from moonlightbox.events.change_points import detect_candidates
    from moonlightbox.events.models import Episode
    from moonlightbox.events.scoring import ChangeSignals

    episodes = [
        Episode(["m1"], datetime(2026, 1, 1), datetime(2026, 1, 1), []),
        Episode(["m2"], datetime(2026, 1, 2), datetime(2026, 1, 2), ["time_gap"]),
        Episode(["m3"], datetime(2026, 1, 3), datetime(2026, 1, 3), ["semantic_change"]),
    ]
    high = ChangeSignals(0.8, 0.9, 0.7, 0.4, 0.8)
    low = ChangeSignals(0.1, 0.1, 0.1, 0.1, 0.1)

    candidates = detect_candidates(episodes, {1: high, 2: low}, threshold=0.6)

    assert len(candidates) == 1
    assert candidates[0].left_episode == 0
    assert candidates[0].right_episode == 1
    assert candidates[0].evidence_ids == ["m1", "m2"]
