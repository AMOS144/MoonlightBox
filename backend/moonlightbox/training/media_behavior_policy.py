import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from moonlightbox.imports.types import ImportedMessage

MediaModality = Literal["image", "audio", "video"]
_MODALITIES: tuple[MediaModality, ...] = ("image", "audio", "video")


@dataclass(frozen=True)
class MediaBehaviorPolicy:
    """Learn when the target historically used non-sticker media."""

    version: str
    total_target_messages: int
    modality_counts: dict[str, int]
    biases: dict[str, float]
    weights: dict[str, dict[str, float]]

    @property
    def enabled(self) -> bool:
        return self.total_target_messages >= 20 and any(
            self.modality_counts.get(modality, 0) >= 3
            for modality in _MODALITIES
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "version": self.version,
            "enabled": self.enabled,
            "total_target_messages": self.total_target_messages,
            "modality_counts": self.modality_counts,
            "biases": self.biases,
            "weights": self.weights,
            "raw_content_retained": False,
            "requires_semantic_annotation": True,
            "requires_explicit_reuse_approval": True,
        }

    @classmethod
    def from_metadata(cls, value: object) -> "MediaBehaviorPolicy":
        if not isinstance(value, dict):
            return cls("historical-media-behavior-v1", 0, {}, {}, {})
        counts = value.get("modality_counts")
        biases = value.get("biases")
        weights = value.get("weights")
        return cls(
            version=str(value.get("version", "historical-media-behavior-v1")),
            total_target_messages=_integer(value.get("total_target_messages")),
            modality_counts=_integer_map(counts),
            biases=_float_map(biases),
            weights=(
                {
                    str(modality): _float_map(modality_weights)
                    for modality, modality_weights in weights.items()
                    if isinstance(modality, str) and isinstance(modality_weights, dict)
                }
                if isinstance(weights, dict)
                else {}
            ),
        )


def build_media_behavior_policy(
    messages: list[ImportedMessage],
    *,
    target_sender: str,
) -> MediaBehaviorPolicy:
    ordered = sorted(messages, key=lambda item: (item.timestamp, item.source_id))
    examples: list[tuple[str, MediaModality | None]] = []
    for index, message in enumerate(ordered):
        if message.sender != target_sender:
            continue
        modality = (
            message.kind.value
            if message.kind.value in _MODALITIES
            else None
        )
        examples.append(
            (_preceding_partner_text(ordered, index, target_sender), modality)
        )
    counts = Counter(label for _context, label in examples if label is not None)
    biases: dict[str, float] = {}
    all_weights: dict[str, dict[str, float]] = {}
    for modality in _MODALITIES:
        positive = [context for context, label in examples if label == modality and context]
        negative = [context for context, label in examples if label != modality and context]
        if len(positive) < 3 or len(negative) < 20:
            continue
        positive_features: Counter[str] = Counter()
        negative_features: Counter[str] = Counter()
        for content in positive:
            positive_features.update(set(_features(content)))
        for content in negative:
            negative_features.update(set(_features(content)))
        weights: dict[str, float] = {}
        for feature in set(positive_features) | set(negative_features):
            positive_probability = (positive_features[feature] + 1) / (
                len(positive) + 2
            )
            negative_probability = (negative_features[feature] + 1) / (
                len(negative) + 2
            )
            weight = math.log(positive_probability / negative_probability)
            if abs(weight) >= 0.2:
                weights[feature] = round(max(-4.0, min(4.0, weight)), 6)
        biases[modality] = round(
            math.log((len(positive) + 1) / (len(negative) + 1)),
            6,
        )
        all_weights[modality] = weights
    return MediaBehaviorPolicy(
        version="historical-media-behavior-v1",
        total_target_messages=len(examples),
        modality_counts={
            modality: counts.get(modality, 0) for modality in _MODALITIES
        },
        biases=biases,
        weights=all_weights,
    )


def choose_media_modality(
    policy: MediaBehaviorPolicy,
    content: str,
    *,
    seed: str,
    proactive: bool = False,
) -> MediaModality | None:
    if not policy.enabled or not content.strip():
        return None
    features = set(_features(content))
    candidates: list[tuple[float, MediaModality]] = []
    for modality in _MODALITIES:
        count = policy.modality_counts.get(modality, 0)
        if count < 3 or modality not in policy.biases:
            continue
        # Historical recordings/images are never selected for an unprompted
        # outreach merely because an n-gram happened to match.
        if proactive:
            continue
        odds = policy.biases[modality] + sum(
            policy.weights.get(modality, {}).get(feature, 0.0)
            for feature in features
        )
        probability = 1 / (1 + math.exp(-max(-20.0, min(20.0, odds))))
        historical_rate = count / max(1, policy.total_target_messages)
        probability = min(probability, historical_rate * 2)
        draw = _stable_fraction(f"{seed}:{modality}")
        if draw < probability:
            candidates.append((probability - draw, modality))
    return max(candidates)[1] if candidates else None


def _preceding_partner_text(
    messages: list[ImportedMessage],
    index: int,
    target_sender: str,
) -> str:
    parts: list[str] = []
    for item in reversed(messages[max(0, index - 6) : index]):
        if item.sender == target_sender:
            if parts:
                break
            continue
        if item.kind.value == "text" and item.content.strip():
            parts.append(item.content.strip())
            if len(parts) == 3:
                break
    return "\n".join(reversed(parts))


def _features(content: str) -> tuple[str, ...]:
    compact = re.sub(r"\s+", "", content.casefold())
    return tuple(
        hashlib.sha256(compact[index : index + width].encode()).hexdigest()[:12]
        for width in (1, 2, 3)
        for index in range(max(0, len(compact) - width + 1))
    )


def _stable_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16) / float(16**12)


def _integer_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): _integer(item)
        for key, item in value.items()
        if isinstance(key, str)
    }


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): float(item)
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, int | float)
    }


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int | float) and value >= 0 else 0
