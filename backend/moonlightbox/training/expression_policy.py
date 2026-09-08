import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta

from moonlightbox.imports.types import ImportedMessage
from moonlightbox.training.turns import build_conversation_turns

_SILENCE_WINDOW = timedelta(minutes=30)


@dataclass(frozen=True)
class ExpressionPolicy:
    version: str
    enabled: bool
    positive_count: int
    negative_count: int
    bias: float
    weights: dict[str, float]
    silence_threshold: float = 0.2

    def to_metadata(self) -> dict[str, object]:
        return {
            "version": self.version,
            "enabled": self.enabled,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "bias": self.bias,
            "weights": self.weights,
            "silence_threshold": self.silence_threshold,
        }

    @classmethod
    def from_metadata(cls, value: object) -> "ExpressionPolicy":
        if not isinstance(value, dict):
            return cls("historical-expression-nb-v2", False, 0, 0, 0.0, {})
        weights = value.get("weights")
        return cls(
            version=str(value.get("version", "historical-expression-nb-v2")),
            enabled=value.get("enabled") is True,
            positive_count=_integer(value.get("positive_count")),
            negative_count=_integer(value.get("negative_count")),
            bias=_number(value.get("bias")),
            weights=(
                {
                    str(key): _number(weight)
                    for key, weight in weights.items()
                    if isinstance(key, str) and isinstance(weight, int | float)
                }
                if isinstance(weights, dict)
                else {}
            ),
            silence_threshold=max(
                0.05,
                min(0.45, _number(value.get("silence_threshold"), 0.2)),
            ),
        )


def build_expression_policy(
    messages: list[ImportedMessage],
    *,
    target_sender: str,
) -> ExpressionPolicy:
    turns, _profiles = build_conversation_turns(messages)
    labeled: list[tuple[str, bool]] = []
    for index, turn in enumerate(turns):
        if turn.speaker == target_sender:
            continue
        content = "\n".join(
            bubble.content for bubble in turn.bubbles if bubble.kind == "text"
        ).strip()
        if not content:
            continue
        following = turns[index + 1] if index + 1 < len(turns) else None
        responded = bool(
            following is not None
            and following.speaker == target_sender
            and following.started_at - turn.ended_at <= _SILENCE_WINDOW
        )
        labeled.append((content, responded))
    positive = [text for text, responded in labeled if responded]
    negative = [text for text, responded in labeled if not responded]
    if len(positive) < 20 or len(negative) < 5:
        return ExpressionPolicy(
            "historical-expression-nb-v2",
            False,
            len(positive),
            len(negative),
            0.0,
            {},
        )
    positive_features: Counter[str] = Counter()
    negative_features: Counter[str] = Counter()
    for text in positive:
        positive_features.update(set(_features(text)))
    for text in negative:
        negative_features.update(set(_features(text)))
    vocabulary = set(positive_features) | set(negative_features)
    weights: dict[str, float] = {}
    for feature in vocabulary:
        positive_probability = (positive_features[feature] + 1) / (len(positive) + 2)
        negative_probability = (negative_features[feature] + 1) / (len(negative) + 2)
        weight = math.log(positive_probability / negative_probability)
        if abs(weight) >= 0.15:
            weights[feature] = round(max(-4.0, min(4.0, weight)), 6)
    return ExpressionPolicy(
        version="historical-expression-nb-v2",
        enabled=True,
        positive_count=len(positive),
        negative_count=len(negative),
        bias=round(math.log((len(positive) + 1) / (len(negative) + 1)), 6),
        weights=weights,
    )


def should_respond(policy: ExpressionPolicy, content: str) -> bool:
    if not policy.enabled:
        return True
    compact = re.sub(r"\s+", "", content)
    if not compact:
        return True
    if _requires_response(compact):
        return True
    features = set(_features(compact))
    evidence = sum(policy.weights.get(feature, 0.0) for feature in features)
    # v1 直接累加每个字符 n-gram，长消息会仅因为更长就迅速
    # 掉到极端沉默分数。v2 把词汇证据归一到约十个独立特征，
    # 保留内容方向，但不再让气泡数和句子长度决定是否回复。
    if policy.version == "historical-expression-nb-v2" and features:
        evidence = evidence / len(features) * 10
    log_odds = policy.bias + evidence
    probability = 1 / (1 + math.exp(-max(-20.0, min(20.0, log_odds))))
    return probability >= policy.silence_threshold


def _features(content: str) -> tuple[str, ...]:
    compact = re.sub(r"\s+", "", content.casefold())
    raw = [
        compact[index : index + width]
        for width in (1, 2, 3)
        for index in range(max(0, len(compact) - width + 1))
    ]
    return tuple(hashlib.sha256(item.encode()).hexdigest()[:12] for item in raw)


def _requires_response(content: str) -> bool:
    if any(
        marker in content
        for marker in (
            "?",
            "？",
            "怎么",
            "为什么",
            "哪",
            "谁",
            "多少",
            "什么意思",
            "啥意思",
            "没看懂",
            "看不懂",
            "没明白",
            "不明白",
            "没听懂",
            "听不懂",
            "说清楚",
            "解释一下",
        )
    ):
        return True
    return bool(
        re.search(
            r"(?:吗|么|嘛|呢|在不在|在吗|救命|急|帮我|告诉我|回我|理我)[。！!…]*$",
            content,
        )
    )


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float) else default


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0
