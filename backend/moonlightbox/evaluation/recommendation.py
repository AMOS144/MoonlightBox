from dataclasses import dataclass


@dataclass(frozen=True)
class ModelScore:
    model_version_id: str
    blind_win_rate: float
    style_score: float
    safety_score: float


class NoQualifiedModelError(LookupError):
    pass


def recommend(
    scores: list[ModelScore],
    minimum_style: float = 0.5,
    minimum_safety: float = 0.7,
) -> ModelScore:
    qualified = [
        score
        for score in scores
        if score.style_score >= minimum_style and score.safety_score >= minimum_safety
    ]
    if not qualified:
        raise NoQualifiedModelError("没有通过质量门槛的模型")
    return max(
        qualified,
        key=lambda score: 0.6 * score.blind_win_rate + 0.4 * score.style_score,
    )
