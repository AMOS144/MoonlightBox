import random


def test_blind_pair_randomizes_answers_without_exposing_identity() -> None:
    from moonlightbox.evaluation.blind_test import make_blind_pair

    pair = make_blind_pair(
        prompt="今晚吃什么？",
        real_answer="随便啦",
        model_answer="都可以呀",
        model_version_id="model-1",
        rng=random.Random(7),
    )

    assert set(pair.options.values()) == {"随便啦", "都可以呀"}
    assert pair.real_option in {"a", "b"}
    assert "real" not in pair.public_view().model_dump()


def test_recommendation_requires_quality_gate_before_blind_win_rate() -> None:
    from moonlightbox.evaluation.recommendation import ModelScore, recommend

    scores = [
        ModelScore("unsafe", blind_win_rate=0.9, style_score=0.9, safety_score=0.2),
        ModelScore("balanced", blind_win_rate=0.6, style_score=0.8, safety_score=0.9),
    ]

    assert recommend(scores).model_version_id == "balanced"
