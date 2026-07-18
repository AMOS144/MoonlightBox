from dataclasses import asdict, dataclass

from moonlightbox.events.models import Episode
from moonlightbox.events.scoring import ChangeSignals, score_change


@dataclass(frozen=True)
class CandidateBoundary:
    left_episode: int
    right_episode: int
    score: float
    signals: dict[str, float]
    evidence_ids: list[str]


def detect_candidates(
    episodes: list[Episode],
    boundary_signals: dict[int, ChangeSignals],
    threshold: float,
) -> list[CandidateBoundary]:
    candidates: list[CandidateBoundary] = []
    for right_index, signals in sorted(boundary_signals.items()):
        if right_index <= 0 or right_index >= len(episodes):
            continue
        score = score_change(signals)
        if score < threshold:
            continue
        left = episodes[right_index - 1]
        right = episodes[right_index]
        candidates.append(
            CandidateBoundary(
                left_episode=right_index - 1,
                right_episode=right_index,
                score=score,
                signals=asdict(signals),
                evidence_ids=left.message_ids[-3:] + right.message_ids[:3],
            )
        )
    return candidates
