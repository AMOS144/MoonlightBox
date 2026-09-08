from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from moonlightbox.branches.replies import parse_reply_turn
from moonlightbox.training.style_features import (
    StyleBubble,
    StyleTurn,
    extract_style_features,
)

STYLE_DISTANCE_VERSION = "moonlightbox.style-distance.v1"
MEMORIZATION_VERSION = "moonlightbox.memorization.char-ngram.v1"


@dataclass(frozen=True)
class BlindABCase:
    case_id: str
    context: tuple[str, ...]
    candidate_reply: str
    human_reply: str


@dataclass(frozen=True)
class BlindABReport:
    seed: int
    case_count: int
    valid_count: int
    invalid_count: int
    candidate_wins: int
    preference_rate: float
    confidence_interval: tuple[float, float]
    passed: bool
    failure_reasons: tuple[str, ...]
    audit: tuple[dict[str, object], ...]


BlindJudge = Callable[[tuple[str, ...], str, str], str]


def run_blind_ab(
    cases: Sequence[BlindABCase],
    judge: BlindJudge,
    *,
    seed: int,
    minimum_valid: int = 1,
    minimum_preference: float = 0.5,
) -> BlindABReport:
    """按固定随机种子匿名交换选项，评审只能看到左右内容。"""

    rng = random.Random(seed)
    wins = 0
    invalid = 0
    audit: list[dict[str, object]] = []
    for case in cases:
        candidate_on_left = bool(rng.getrandbits(1))
        left = case.candidate_reply if candidate_on_left else case.human_reply
        right = case.human_reply if candidate_on_left else case.candidate_reply
        verdict = ""
        try:
            verdict = judge(case.context, left, right).strip().casefold()
        except Exception:
            invalid += 1
        if verdict not in {"left", "right", "tie"}:
            if verdict:
                invalid += 1
            audit.append(
                {
                    "case_id": case.case_id,
                    "order_hash": _hash_text(left + "\0" + right),
                    "valid": False,
                }
            )
            continue
        candidate_won = (
            verdict == "left" and candidate_on_left
        ) or (
            verdict == "right" and not candidate_on_left
        )
        wins += int(candidate_won)
        audit.append(
            {
                "case_id": case.case_id,
                "order_hash": _hash_text(left + "\0" + right),
                "valid": True,
                "tie": verdict == "tie",
                "candidate_won": candidate_won,
            }
        )
    valid = len(cases) - invalid
    rate = wins / valid if valid else 0.0
    interval = _wilson_interval(wins, valid)
    reasons: list[str] = []
    if valid < minimum_valid:
        reasons.append("有效评审不足")
    if rate < minimum_preference:
        reasons.append("候选偏好率未达阈值")
    return BlindABReport(
        seed=seed,
        case_count=len(cases),
        valid_count=valid,
        invalid_count=invalid,
        candidate_wins=wins,
        preference_rate=rate,
        confidence_interval=interval,
        passed=not reasons,
        failure_reasons=tuple(reasons),
        audit=tuple(audit),
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True)
class SpeakerSample:
    split: Literal["train", "valid", "test"]
    speaker: str
    text: str


@dataclass(frozen=True)
class SpeakerEvaluation:
    sample_count: int
    accuracy: float
    mean_positive_probability: float
    brier_score: float
    probabilities: tuple[float, ...]
    calibration: tuple[dict[str, float | int], ...]


class SpeakerStyleIdentifier:
    """使用字符 ngram 与功能词的确定性朴素贝叶斯说话者识别器。"""

    _FUNCTION_WORDS = ("的", "了", "呢", "吗", "吧", "呀", "啊", "哦", "嘛", "就", "也")

    def __init__(self, ngram_size: int = 3, smoothing: float = 1.0) -> None:
        if ngram_size < 1 or smoothing <= 0:
            raise ValueError("ngram_size 和 smoothing 必须为正数")
        self._ngram_size = ngram_size
        self._smoothing = smoothing
        self._counts: dict[str, Counter[str]] = {}
        self._totals: Counter[str] = Counter()
        self._documents: Counter[str] = Counter()
        self._vocabulary: set[str] = set()

    def fit(self, samples: Sequence[SpeakerSample]) -> None:
        if any(sample.split == "test" for sample in samples):
            raise ValueError("test 样本严禁用于说话者识别器拟合")
        if not samples:
            raise ValueError("说话者识别器训练样本不足")
        for sample in samples:
            features = self._features(sample.text)
            self._counts.setdefault(sample.speaker, Counter()).update(features)
            self._totals[sample.speaker] += sum(features.values())
            self._documents[sample.speaker] += 1
            self._vocabulary.update(features)

    def probabilities(self, text: str) -> dict[str, float]:
        if len(self._counts) < 2:
            raise ValueError("说话者识别器至少需要两个类别")
        features = self._features(text)
        document_count = sum(self._documents.values())
        vocabulary_size = max(1, len(self._vocabulary))
        scores: dict[str, float] = {}
        for speaker, counts in self._counts.items():
            score = math.log(self._documents[speaker] / document_count)
            denominator = self._totals[speaker] + self._smoothing * vocabulary_size
            for feature, count in features.items():
                probability = (counts[feature] + self._smoothing) / denominator
                score += count * math.log(probability)
            scores[speaker] = score
        maximum = max(scores.values())
        weights = {speaker: math.exp(score - maximum) for speaker, score in scores.items()}
        total = sum(weights.values())
        return {speaker: weight / total for speaker, weight in weights.items()}

    def evaluate(
        self,
        samples: Sequence[SpeakerSample],
        *,
        positive_speaker: str,
    ) -> SpeakerEvaluation:
        if not samples:
            raise ValueError("说话者识别评估样本不足")
        probabilities: list[float] = []
        positive_probabilities: list[float] = []
        correct = 0
        squared_errors = 0.0
        calibration_bins: dict[int, list[tuple[float, int]]] = {}
        for sample in samples:
            result = self.probabilities(sample.text)
            predicted = max(result, key=lambda speaker: (result[speaker], speaker))
            correct += int(predicted == sample.speaker)
            probability = result.get(positive_speaker, 0.0)
            label = int(sample.speaker == positive_speaker)
            probabilities.append(probability)
            if label:
                positive_probabilities.append(probability)
            squared_errors += (probability - label) ** 2
            calibration_bins.setdefault(min(9, int(probability * 10)), []).append(
                (probability, label)
            )
        calibration = tuple(
            {
                "lower": bucket / 10,
                "upper": (bucket + 1) / 10,
                "count": len(items),
                "mean_probability": sum(item[0] for item in items) / len(items),
                "positive_rate": sum(item[1] for item in items) / len(items),
            }
            for bucket, items in sorted(calibration_bins.items())
        )
        return SpeakerEvaluation(
            sample_count=len(samples),
            accuracy=correct / len(samples),
            mean_positive_probability=(
                sum(positive_probabilities) / len(positive_probabilities)
                if positive_probabilities
                else 0.0
            ),
            brier_score=squared_errors / len(samples),
            probabilities=tuple(probabilities),
            calibration=calibration,
        )

    def _features(self, text: str) -> Counter[str]:
        normalized = "".join(character for character in text.casefold() if not character.isspace())
        features = Counter(
            f"n:{normalized[index:index + self._ngram_size]}"
            for index in range(max(0, len(normalized) - self._ngram_size + 1))
        )
        for word in self._FUNCTION_WORDS:
            count = normalized.count(word)
            if count:
                features[f"f:{word}"] = count
        return features


@dataclass(frozen=True)
class StyleDistanceReport:
    version: str
    threshold: float
    distances: dict[str, float]
    overall_distance: float
    passed: bool
    generated_features: dict[str, object]
    test_features: dict[str, object]


def compare_style_distributions(
    generated: Sequence[StyleTurn],
    test_real: Sequence[StyleTurn],
    *,
    threshold: float = 0.35,
) -> StyleDistanceReport:
    """使用统一 style_features 计算版本化、归一到零至一的距离。"""

    generated_features = extract_style_features(generated)
    test_features = extract_style_features(test_real)
    distances = {
        "length": _number_distance(
            _number(generated_features, "structural", "bubble_length", "mean"),
            _number(test_features, "structural", "bubble_length", "mean"),
        ),
        "bubble_count": _number_distance(
            _number(generated_features, "structural", "bubbles_per_turn", "mean"),
            _number(test_features, "structural", "bubbles_per_turn", "mean"),
        ),
        "punctuation": _number_distance(
            _number(generated_features, "structural", "terminal_period_omission_rate"),
            _number(test_features, "structural", "terminal_period_omission_rate"),
        ),
        "catchphrase": _ranked_distance(
            generated_features, test_features, "lexical", "catchphrases", key="text"
        ),
        "emoji": _ranked_distance(
            generated_features, test_features, "media", "emoji_frequencies", key="emoji"
        ),
        "pragmatic": _mapping_distance(
            _mapping(generated_features, "pragmatic", "rates"),
            _mapping(test_features, "pragmatic", "rates"),
        ),
        "sticker": _ranked_distance(
            generated_features,
            test_features,
            "media",
            "sticker_asset_frequencies",
            key="asset_id",
        ),
    }
    overall = sum(distances.values()) / len(distances)
    return StyleDistanceReport(
        version=STYLE_DISTANCE_VERSION,
        threshold=threshold,
        distances=distances,
        overall_distance=overall,
        passed=bool(generated and test_real) and overall <= threshold,
        generated_features=generated_features,
        test_features=test_features,
    )


def _number(payload: Mapping[str, object], *path: str) -> float:
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping):
            return 0.0
        current = current.get(key)
    return float(current) if isinstance(current, int | float) else 0.0


def _mapping(payload: Mapping[str, object], *path: str) -> dict[str, float]:
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    if not isinstance(current, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in current.items()
        if isinstance(value, int | float)
    }


def _number_distance(left: float, right: float) -> float:
    return min(1.0, abs(left - right) / max(1.0, abs(left), abs(right)))


def _mapping_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return min(1.0, sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys) / 2)


def _ranked_distance(
    left: Mapping[str, object],
    right: Mapping[str, object],
    section: str,
    feature: str,
    *,
    key: str,
) -> float:
    return _mapping_distance(
        _ranked_rates(left, section, feature, key),
        _ranked_rates(right, section, feature, key),
    )


def _ranked_rates(
    payload: Mapping[str, object],
    section: str,
    feature: str,
    key: str,
) -> dict[str, float]:
    section_value = payload.get(section)
    values = section_value.get(feature) if isinstance(section_value, Mapping) else None
    if not isinstance(values, list):
        return {}
    counts = {
        str(item[key]): float(item["count"])
        for item in values
        if isinstance(item, Mapping)
        and key in item
        and isinstance(item.get("count"), int | float)
    }
    total = sum(counts.values())
    return {name: count / total for name, count in counts.items()} if total else {}


@dataclass(frozen=True)
class MemorizationSource:
    split: Literal["train", "valid", "test"]
    source_hash: str
    text: str


@dataclass(frozen=True)
class MemorizationResult:
    version: str
    text_hash: str
    text_length: int
    maximum_overlap: float
    nearest_source_hash: str | None
    threshold: float
    length_exempt: bool
    passed: bool


class MemorizationIndex:
    """只索引训练 target；短回复豁免，长文本按字符 ngram Jaccard 拒绝。"""

    def __init__(
        self,
        sources: Sequence[MemorizationSource],
        *,
        ngram_size: int = 4,
        threshold: float = 0.8,
        minimum_checked_length: int = 6,
    ) -> None:
        if any(source.split != "train" for source in sources):
            raise ValueError("查重索引只能包含 train target，严禁包含 test")
        self._ngram_size = ngram_size
        self._threshold = threshold
        self._minimum_length = minimum_checked_length
        self._sources = tuple(
            (source.source_hash, _ngrams(source.text, ngram_size))
            for source in sources
        )

    def check(self, text: str) -> MemorizationResult:
        grams = _ngrams(text, self._ngram_size)
        nearest_hash: str | None = None
        maximum = 0.0
        for source_hash, source_grams in self._sources:
            union = grams | source_grams
            overlap = len(grams & source_grams) / len(union) if union else 0.0
            if overlap > maximum:
                maximum = overlap
                nearest_hash = source_hash
        exempt = len(text.strip()) < self._minimum_length
        return MemorizationResult(
            version=MEMORIZATION_VERSION,
            text_hash=_hash_text(text),
            text_length=len(text),
            maximum_overlap=maximum,
            nearest_source_hash=nearest_hash,
            threshold=self._threshold,
            length_exempt=exempt,
            passed=exempt or maximum <= self._threshold,
        )

    def check_many(self, texts: Sequence[str]) -> dict[str, object]:
        results = tuple(self.check(text) for text in texts)
        overlaps = sorted(result.maximum_overlap for result in results)
        return {
            "version": MEMORIZATION_VERSION,
            "sample_count": len(results),
            "maximum_overlap": max(overlaps, default=0.0),
            "p95_overlap": _percentile(overlaps, 0.95),
            "passed": bool(results) and all(result.passed for result in results),
            "samples": [asdict(result) for result in results],
        }


def _ngrams(text: str, size: int) -> set[str]:
    normalized = "".join(character for character in text.casefold() if not character.isspace())
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {
        normalized[index : index + size]
        for index in range(len(normalized) - size + 1)
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, math.ceil(len(values) * percentile) - 1)
    return values[min(index, len(values) - 1)]


@dataclass(frozen=True)
class HeldOutCase:
    case_id: str
    split: Literal["train", "valid", "test"]
    context: tuple[dict[str, str], ...]
    human_target: str
    source_hash: str


@dataclass(frozen=True)
class HeldOutCandidate:
    case_id: str
    source_hash: str
    raw_output: str
    raw_output_hash: str
    text: str
    style_turn: StyleTurn


RawCandidateGenerator = Callable[[tuple[dict[str, str], ...]], str]


def generate_held_out_candidates(
    cases: Sequence[HeldOutCase],
    generator: RawCandidateGenerator,
) -> tuple[HeldOutCandidate, ...]:
    """仅以完全留出的 test 真实上下文生成并保留本地模型原始输出。"""

    if any(case.split != "test" for case in cases):
        raise ValueError("留出集生成只接受 test 样本")
    generated: list[HeldOutCandidate] = []
    for case in cases:
        raw = generator(case.context)
        reply = parse_reply_turn(raw)
        text = "\n".join(bubble.content or "" for bubble in reply.bubbles)
        generated.append(
            HeldOutCandidate(
                case_id=case.case_id,
                source_hash=case.source_hash,
                raw_output=raw,
                raw_output_hash=_hash_text(raw),
                text=text,
                style_turn=StyleTurn(
                    previous_text=case.context[-1]["content"] if case.context else "",
                    bubbles=tuple(
                        _reply_style_bubble(
                            bubble.type,
                            bubble.content,
                            bubble.delay_ms,
                            bubble.asset_id,
                        )
                        for bubble in reply.bubbles
                    ),
                ),
            )
        )
    return tuple(generated)


def _reply_style_bubble(
    kind: Literal["text", "emoji", "sticker"],
    content: str | None,
    delay_ms: int,
    asset_id: str | None,
) -> StyleBubble:
    return StyleBubble(
        text=content or "",
        delay_ms=delay_ms,
        kind=kind,
        asset_id=asset_id,
    )


@dataclass(frozen=True)
class AcceptanceJSONReport:
    dataset: dict[str, object]
    model_ids: dict[str, str]
    seed: int
    gates: dict[str, object]
    comparisons: dict[str, object]
    failure_reasons: tuple[str, ...]
    sample_audit: tuple[dict[str, object], ...]
    passed: bool
    schema_version: str = "moonlightbox.style-acceptance-report.v1"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, indent=2)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
