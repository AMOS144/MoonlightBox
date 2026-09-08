import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from moonlightbox.imports.types import ImportedMessage
from moonlightbox.imports.wechat_rendering import normalize_wechat_display

ConversationAction = Literal["quote", "call", "reaction", "retract"]
_ACTIONS: tuple[ConversationAction, ...] = ("quote", "call", "reaction", "retract")


@dataclass(frozen=True)
class ConversationActionPolicy:
    version: str
    total_target_messages: int
    action_counts: dict[str, int]
    biases: dict[str, float]
    weights: dict[str, dict[str, float]]

    @property
    def enabled(self) -> bool:
        return self.total_target_messages >= 20 and any(
            self.action_counts.get(action, 0) >= 3 for action in _ACTIONS
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "version": self.version,
            "enabled": self.enabled,
            "total_target_messages": self.total_target_messages,
            "action_counts": self.action_counts,
            "biases": self.biases,
            "weights": self.weights,
            "raw_content_retained": False,
        }

    @classmethod
    def from_metadata(cls, value: object) -> "ConversationActionPolicy":
        if not isinstance(value, dict):
            return cls("historical-conversation-actions-v1", 0, {}, {}, {})
        counts = value.get("action_counts")
        biases = value.get("biases")
        weights = value.get("weights")
        return cls(
            version=str(value.get("version", "historical-conversation-actions-v1")),
            total_target_messages=_integer(value.get("total_target_messages")),
            action_counts=(
                {
                    str(key): _integer(item)
                    for key, item in counts.items()
                    if isinstance(key, str)
                }
                if isinstance(counts, dict)
                else {}
            ),
            biases=(
                {
                    str(key): _number(item)
                    for key, item in biases.items()
                    if isinstance(key, str)
                }
                if isinstance(biases, dict)
                else {}
            ),
            weights=(
                {
                    str(action): {
                        str(feature): _number(weight)
                        for feature, weight in action_weights.items()
                        if isinstance(feature, str)
                    }
                    for action, action_weights in weights.items()
                    if isinstance(action, str) and isinstance(action_weights, dict)
                }
                if isinstance(weights, dict)
                else {}
            ),
        )


def build_conversation_action_policy(
    messages: list[ImportedMessage],
    *,
    target_sender: str,
) -> ConversationActionPolicy:
    ordered = sorted(messages, key=lambda item: (item.timestamp, item.source_id))
    retraction_senders = _infer_retraction_senders(ordered, target_sender)
    examples: list[tuple[str, ConversationAction | None]] = []
    for index, message in enumerate(ordered):
        inferred_retraction_sender = retraction_senders.get(message.source_id)
        if message.sender != target_sender and inferred_retraction_sender != target_sender:
            continue
        action = (
            "retract" if inferred_retraction_sender == target_sender else _action_for(message)
        )
        context = _preceding_partner_text(ordered, index, target_sender)
        examples.append((context, action))
    action_counts = Counter(action for _context, action in examples if action is not None)
    biases: dict[str, float] = {}
    all_weights: dict[str, dict[str, float]] = {}
    for action in _ACTIONS:
        positive = [context for context, label in examples if label == action and context]
        negative = [context for context, label in examples if label != action and context]
        if len(positive) < 3 or len(negative) < 20:
            continue
        positive_features: Counter[str] = Counter()
        negative_features: Counter[str] = Counter()
        for content in positive:
            positive_features.update(set(_features(content)))
        for content in negative:
            negative_features.update(set(_features(content)))
        vocabulary = set(positive_features) | set(negative_features)
        action_weights: dict[str, float] = {}
        for feature in vocabulary:
            positive_probability = (positive_features[feature] + 1) / (len(positive) + 2)
            negative_probability = (negative_features[feature] + 1) / (len(negative) + 2)
            weight = math.log(positive_probability / negative_probability)
            if abs(weight) >= 0.2:
                action_weights[feature] = round(max(-4.0, min(4.0, weight)), 6)
        biases[action] = round(math.log((len(positive) + 1) / (len(negative) + 1)), 6)
        all_weights[action] = action_weights
    return ConversationActionPolicy(
        version="historical-conversation-actions-v1",
        total_target_messages=len(examples),
        action_counts={action: action_counts.get(action, 0) for action in _ACTIONS},
        biases=biases,
        weights=all_weights,
    )


def _infer_retraction_senders(
    messages: list[ImportedMessage],
    target_sender: str,
) -> dict[str, str]:
    """Attribute WeChat system revocations through stable quote aliases."""

    ordinary_senders = Counter(
        item.sender
        for item in messages
        if item.sender != target_sender and not _is_system_message(item)
    )
    partner_sender = ordinary_senders.most_common(1)[0][0] if ordinary_senders else ""
    alias_votes: dict[str, Counter[str]] = {}
    for message in messages:
        if _is_system_message(message):
            continue
        aliases = re.findall(r"(?:^|\n)>\s*([^:：\n]{1,80})[:：]", message.content)
        quoted_sender = (
            partner_sender if message.sender == target_sender else target_sender
        )
        if not quoted_sender:
            continue
        for alias in aliases:
            normalized = alias.strip().casefold()
            if normalized:
                alias_votes.setdefault(normalized, Counter())[quoted_sender] += 1
    alias_mapping: dict[str, str] = {}
    for alias, votes in alias_votes.items():
        ranked = votes.most_common(2)
        if not ranked:
            continue
        sender, count = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        if count >= 2 and count >= runner_up * 2:
            alias_mapping[alias] = sender
    inferred: dict[str, str] = {}
    for message in messages:
        alias = _retraction_alias(message.content)
        if alias is not None and (sender := alias_mapping.get(alias.casefold())):
            inferred[message.source_id] = sender
    return inferred


def _retraction_alias(content: str) -> str | None:
    match = re.search(
        r"<revokemsg>.*?<content>[\"“]?([^<\"”]+?)[\"”]?\s*撤回了一条消息</content>",
        content,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match is not None else None


def _is_system_message(message: ImportedMessage) -> bool:
    kind, _content = normalize_wechat_display(message.kind.value, message.content)
    return kind == "system"


def choose_conversation_action(
    policy: ConversationActionPolicy,
    content: str,
    *,
    seed: str,
    proactive: bool = False,
) -> ConversationAction | None:
    if not policy.enabled or not content.strip():
        return None
    # This policy is conditioned on replies to a partner contribution. It has
    # no evidence that an action was used to initiate a conversation, so do
    # not extrapolate calls/retractions into proactive outreach.
    if proactive:
        return None
    candidates: list[tuple[float, ConversationAction]] = []
    features = set(_features(content))
    for action in _ACTIONS:
        count = policy.action_counts.get(action, 0)
        if count < 3 or action not in policy.biases:
            continue
        odds = policy.biases[action] + sum(
            policy.weights.get(action, {}).get(feature, 0.0) for feature in features
        )
        probability = 1 / (1 + math.exp(-max(-20.0, min(20.0, odds))))
        # Never let sparse historical actions become a dominant modality merely
        # because a short context happened to match one feature.
        historical_rate = count / max(1, policy.total_target_messages)
        probability = min(probability, historical_rate * 3)
        draw = _stable_fraction(f"{seed}:{action}")
        if draw < probability:
            candidates.append((probability - draw, action))
    if not candidates:
        return None
    return max(candidates)[1]


def quote_content(text: str, quoted_content: str) -> str:
    return json.dumps(
        {
            "text": text,
            "quoted_content": re.sub(r"\s+", " ", quoted_content).strip()[:160],
        },
        ensure_ascii=False,
    )


def _action_for(message: ImportedMessage) -> ConversationAction | None:
    kind, content = normalize_wechat_display(message.kind.value, message.content)
    if kind == "quote":
        return "quote"
    if kind == "call":
        return "call"
    if kind == "reaction":
        return "reaction"
    if kind == "system" and "撤回" in content:
        return "retract"
    return None


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
        kind, content = normalize_wechat_display(item.kind.value, item.content)
        if kind in {"text", "quote"}:
            parts.append(_visible_text(content))
            if len(parts) == 3:
                break
    return "\n".join(reversed([part for part in parts if part]))


def _visible_text(content: str) -> str:
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return str(payload["text"])
    return content


def _features(content: str) -> tuple[str, ...]:
    compact = re.sub(r"\s+", "", content.casefold())
    return tuple(
        hashlib.sha256(compact[index : index + width].encode()).hexdigest()[:12]
        for width in (1, 2, 3)
        for index in range(max(0, len(compact) - width + 1))
    )


def _stable_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16) / float(16**12)


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float) else default


def _integer(value: object) -> int:
    return int(value) if isinstance(value, int | float) and value >= 0 else 0
