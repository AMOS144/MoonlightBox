from dataclasses import dataclass


@dataclass(frozen=True)
class ChangeSignals:
    topic_delta: float
    emotion_delta: float
    intent_delta: float
    response_gap_delta: float
    persistence: float


WEIGHTS = {
    "topic_delta": 0.20,
    "emotion_delta": 0.25,
    "intent_delta": 0.20,
    "response_gap_delta": 0.10,
    "persistence": 0.25,
}


def score_change(signals: ChangeSignals) -> float:
    values = {
        "topic_delta": signals.topic_delta,
        "emotion_delta": signals.emotion_delta,
        "intent_delta": signals.intent_delta,
        "response_gap_delta": signals.response_gap_delta,
        "persistence": signals.persistence,
    }
    score = sum(max(0.0, min(1.0, value)) * WEIGHTS[name] for name, value in values.items())
    return round(score, 6)
