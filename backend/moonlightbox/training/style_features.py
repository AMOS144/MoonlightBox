import re
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Literal, cast

import regex

STYLE_FEATURE_SCHEMA_VERSION = "moonlightbox.style-features.v4"

_PHRASE_CAPACITY = 128
_NGRAM_CAPACITY = 256
_REPETITION_CAPACITY = 128
_PUNCTUATION_CAPACITY = 128
_EMOJI_CAPACITY = 128
_COMBINATION_CAPACITY = 256
_ASSET_CAPACITY = 128
_CONTEXT_CAPACITY = 64
_GRAPHEME_PATTERN = regex.compile(r"\X")
_EMOJI_GRAPHEME_PATTERN = regex.compile(
    r"(?:"
    r"\p{Regional_Indicator}{2}|"
    r"\p{Emoji_Presentation}|"
    r"\p{Extended_Pictographic}|"
    r"\p{Emoji}\uFE0F|"
    r"[0-9#*]\uFE0F?\u20E3"
    r")"
)

_FUNCTION_WORDS = (
    "的",
    "了",
    "呢",
    "吗",
    "吧",
    "呀",
    "啊",
    "哦",
    "嘛",
    "就",
    "也",
    "还",
    "都",
    "很",
)
_ADDRESS_TERMS = (
    "宝宝",
    "宝贝",
    "亲爱的",
    "笨笨",
    "笨蛋",
    "傻瓜",
    "哥哥",
    "姐姐",
    "老公",
    "老婆",
)
_CATCHPHRASES = (
    "哈哈",
    "嘿嘿",
    "嘻嘻",
    "嗯嗯",
    "好呀",
    "好啦",
    "知道啦",
    "怎么啦",
    "真的嘛",
    "哎呀",
)
_PUNCTUATION_PATTERN = re.compile(r"[，。！？!?…~～、；：,.]{2,}")
_REPETITION_PATTERN = re.compile(r"(.)\1+")
_PHRASE_SEGMENT_PATTERN = re.compile(r"[\u3400-\u9fffA-Za-z0-9]+")
_HISTOGRAM_UPPER_BOUNDS = (
    0,
    1,
    4,
    9,
    19,
    39,
    79,
    159,
    319,
    639,
    1_279,
    2_559,
    4_999,
    9_999,
    59_999,
    299_999,
)


class BoundedFrequency[Key: Hashable](Mapping[Key, int]):
    """确定性 Space-Saving 频率表，容量满后替换当前最小项。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("频率表容量必须大于零")
        self.capacity = capacity
        self._counts: dict[Key, int] = {}
        self._errors: dict[Key, int] = {}
        self.total_count = 0
        self.dropped_unique_count = 0

    def add(self, key: Key, count: int = 1) -> Key | None:
        if count < 1:
            raise ValueError("频率增量必须大于零")
        self.total_count += count
        if key in self._counts:
            self._counts[key] += count
            return None
        if len(self._counts) < self.capacity:
            self._counts[key] = count
            self._errors[key] = 0
            return None
        victim = min(
            self._counts,
            key=lambda item: (self._counts[item], repr(item)),
        )
        victim_count = self._counts.pop(victim)
        self._errors.pop(victim)
        self._counts[key] = victim_count + count
        self._errors[key] = victim_count
        self.dropped_unique_count += 1
        return victim

    def update(self, values: Iterable[Key]) -> None:
        for value in values:
            self.add(value)

    def snapshot(self) -> tuple[tuple[Key, int], ...]:
        return tuple(
            sorted(
                (
                    (key, self[key])
                    for key in self._counts
                ),
                key=lambda item: repr(item[0]),
            )
        )

    def audit(self) -> dict[str, object]:
        return {
            "total_count": self.total_count,
            "tracked_unique": len(self._counts),
            "internal_capacity": self.capacity,
            "dropped_unique_count": self.dropped_unique_count,
            "approximate": self.dropped_unique_count > 0,
            "count_semantics": "lower-bound-v1",
            "estimated_total_count": sum(self._counts.values()),
            "tracked_lower_bound_count": sum(self.values()),
            "maximum_error": self.maximum_error,
        }

    @property
    def maximum_error(self) -> int:
        return max(self._errors.values(), default=0)

    def __contains__(self, key: object) -> bool:
        return key in self._counts

    def __getitem__(self, key: Key) -> int:
        return self._counts[key] - self._errors[key]

    def __iter__(self) -> Iterator[Key]:
        return iter(self._counts)

    def __len__(self) -> int:
        return len(self._counts)


@dataclass(frozen=True)
class StyleBubble:
    text: str = ""
    delay_ms: int = 0
    kind: Literal["text", "emoji", "sticker"] = "text"
    asset_id: str | None = None


@dataclass(frozen=True)
class StyleTurn:
    bubbles: tuple[StyleBubble, ...]
    previous_text: str = ""


def extract_style_features(turns: Sequence[StyleTurn]) -> dict[str, object]:
    """抽取训练与评估共用的确定性风格特征。"""

    ordered_turns = tuple(turns)
    bubbles = tuple(bubble for turn in ordered_turns for bubble in turn.bubbles)
    text_bubbles = tuple(
        bubble.text.strip()
        for bubble in bubbles
        if bubble.kind in {"text", "emoji"} and bubble.text.strip()
    )
    lexical = _lexical_features(text_bubbles)
    structural = _structural_features(ordered_turns, bubbles, text_bubbles)
    pragmatic = _pragmatic_features(text_bubbles)
    media = _media_features(ordered_turns)
    return {
        "schema_version": STYLE_FEATURE_SCHEMA_VERSION,
        "sample_count": len(text_bubbles),
        "turn_count": len(ordered_turns),
        "bubble_count": len(bubbles),
        "lexical": lexical,
        "structural": structural,
        "pragmatic": pragmatic,
        "media": media,
    }


def _lexical_features(texts: Sequence[str]) -> dict[str, object]:
    phrase_counts = BoundedFrequency[str](_PHRASE_CAPACITY)
    ngram_counts = {
        2: BoundedFrequency[str](_NGRAM_CAPACITY),
        3: BoundedFrequency[str](_NGRAM_CAPACITY),
    }
    for text in texts:
        for segment in _PHRASE_SEGMENT_PATTERN.findall(text):
            if 2 <= len(segment) <= 16:
                phrase_counts.add(segment)
            for size in ngram_counts:
                ngram_counts[size].update(
                    segment[index : index + size]
                    for index in range(max(0, len(segment) - size + 1))
                )
    frequent_phrases = {
        text: count
        for text, count in phrase_counts.items()
        if count >= 2
    }
    repetitions = BoundedFrequency[str](_REPETITION_CAPACITY)
    repetitions.update(
        match.group(0)
        for text in texts
        for match in _REPETITION_PATTERN.finditer(text)
    )
    return {
        "phrase_semantics": "punctuation-delimited-segments-v1",
        "frequent_phrases": _ranked(frequent_phrases, "text", 32),
        "character_ngrams": {
            str(size): _ranked(counts, "text", 64)
            for size, counts in sorted(ngram_counts.items())
        },
        "function_words": _term_counts(texts, _FUNCTION_WORDS),
        "address_terms": _term_counts(texts, _ADDRESS_TERMS),
        "catchphrases": _term_counts(texts, _CATCHPHRASES),
        "repetition_patterns": _ranked(repetitions, "pattern", 32),
        "ranking_audit": {
            "frequent_phrases": {
                **phrase_counts.audit(),
                "returned_count": min(len(frequent_phrases), 32),
                "limit": 32,
                "truncated": len(frequent_phrases) > 32
                or phrase_counts.dropped_unique_count > 0,
            },
            "character_ngrams": {
                str(size): _ranking_audit(counts, 64)
                for size, counts in sorted(ngram_counts.items())
            },
            "repetition_patterns": _ranking_audit(repetitions, 32),
        },
    }


def _structural_features(
    turns: Sequence[StyleTurn],
    bubbles: Sequence[StyleBubble],
    texts: Sequence[str],
) -> dict[str, object]:
    lengths = [len(text) for text in texts]
    bubbles_per_turn = [len(turn.bubbles) for turn in turns]
    delays = [max(0, bubble.delay_ms) for bubble in bubbles]
    punctuation = BoundedFrequency[str](_PUNCTUATION_CAPACITY)
    punctuation.update(
        match.group(0)
        for text in texts
        for match in _PUNCTUATION_PATTERN.finditer(text)
    )
    return {
        "bubble_length": _distribution(lengths),
        "bubbles_per_turn": _distribution(bubbles_per_turn),
        "punctuation_combinations": _ranked(punctuation, "text", 32),
        "terminal_period_omission_rate": (
            sum(not text.endswith(("。", ".")) for text in texts) / len(texts)
            if texts
            else 0.0
        ),
        "newline_rate": (
            sum("\n" in text for text in texts) / len(texts)
            if texts
            else 0.0
        ),
        "delays_ms": _distribution(delays),
        "ranking_audit": {
            "punctuation_combinations": _ranking_audit(
                punctuation,
                32,
            )
        },
    }


def _pragmatic_features(texts: Sequence[str]) -> dict[str, object]:
    names = ("回应", "追问", "调侃", "安慰", "拒绝", "转移话题")
    counts = Counter({name: 0 for name in names})
    for text in texts:
        counts[_classify_pragmatic(text)] += 1
    denominator = len(texts)
    ordered_counts = {name: counts[name] for name in names}
    return {
        "rule_version": "local-rules-v1",
        "counts": ordered_counts,
        "rates": {
            name: counts[name] / denominator if denominator else 0.0
            for name in names
        },
    }


def _classify_pragmatic(text: str) -> str:
    if any(marker in text for marker in ("换个话题", "不说这个", "说点别的", "对了")):
        return "转移话题"
    if any(marker in text for marker in ("不要", "不想", "不行", "不能", "别逼")):
        return "拒绝"
    if any(marker in text for marker in ("抱抱", "别难过", "没事的", "会好的", "心疼")):
        return "安慰"
    if any(marker in text for marker in ("笨蛋", "傻瓜", "逗你", "哈哈", "笑死")):
        return "调侃"
    if text.endswith(("吗", "呢", "？", "?")) or any(
        marker in text for marker in ("你呢", "然后呢", "怎么啦", "为什么")
    ):
        return "追问"
    return "回应"


def _media_features(turns: Sequence[StyleTurn]) -> dict[str, object]:
    emoji_counts = BoundedFrequency[str](_EMOJI_CAPACITY)
    combinations = BoundedFrequency[tuple[str, str]](_COMBINATION_CAPACITY)
    sticker_counts = BoundedFrequency[str](_ASSET_CAPACITY)
    emoji_asset_counts = BoundedFrequency[str](_ASSET_CAPACITY)
    sticker_previous_contexts: dict[str, BoundedFrequency[str]] = {}
    sticker_same_turn_contexts: dict[str, BoundedFrequency[str]] = {}
    emoji_previous_contexts: dict[str, BoundedFrequency[str]] = {}
    emoji_same_turn_contexts: dict[str, BoundedFrequency[str]] = {}
    for turn in turns:
        last_turn_text = ""
        for bubble in turn.bubbles:
            if bubble.text:
                last_turn_text = bubble.text.strip()
                graphemes = _split_graphemes(bubble.text)
                emojis = [
                    grapheme
                    for grapheme in graphemes
                    if _is_emoji_grapheme(grapheme)
                ]
                emoji_counts.update(emojis)
                plain_text = "".join(
                    grapheme
                    for grapheme in graphemes
                    if not _is_emoji_grapheme(grapheme)
                ).strip()
                if plain_text:
                    combinations.update(
                        (emoji, plain_text)
                        for emoji in emojis
                    )
            if bubble.kind == "sticker" and bubble.asset_id:
                asset_id = bubble.asset_id
                evicted = sticker_counts.add(asset_id)
                _remove_evicted_contexts(
                    evicted,
                    sticker_previous_contexts,
                    sticker_same_turn_contexts,
                )
                _record_asset_context(
                    asset_id,
                    turn.previous_text,
                    last_turn_text,
                    sticker_previous_contexts,
                    sticker_same_turn_contexts,
                )
            if bubble.kind == "emoji" and bubble.asset_id:
                asset_id = bubble.asset_id
                evicted = emoji_asset_counts.add(asset_id)
                _remove_evicted_contexts(
                    evicted,
                    emoji_previous_contexts,
                    emoji_same_turn_contexts,
                )
                _record_asset_context(
                    asset_id,
                    turn.previous_text,
                    last_turn_text,
                    emoji_previous_contexts,
                    emoji_same_turn_contexts,
                )
    return {
        "emoji_frequencies": _ranked(emoji_counts, "emoji", 64),
        "emoji_text_combinations": [
            {"emoji": emoji, "text": text, "count": count}
            for (emoji, text), count in sorted(
                combinations.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )[:64]
        ],
        "emoji_asset_frequencies": _ranked(
            emoji_asset_counts,
            "asset_id",
            64,
        ),
        "emoji_asset_contexts": _asset_contexts(
            emoji_asset_counts,
            emoji_previous_contexts,
            emoji_same_turn_contexts,
        ),
        "sticker_asset_frequencies": _ranked(
            sticker_counts,
            "asset_id",
            64,
        ),
        "sticker_contexts": _asset_contexts(
            sticker_counts,
            sticker_previous_contexts,
            sticker_same_turn_contexts,
        ),
        "ranking_audit": {
            "emoji_frequencies": _ranking_audit(emoji_counts, 64),
            "emoji_text_combinations": _pair_ranking_audit(combinations, 64),
            "emoji_asset_frequencies": _ranking_audit(emoji_asset_counts, 64),
            "emoji_asset_contexts": _context_ranking_audit(
                emoji_asset_counts,
                emoji_previous_contexts,
                emoji_same_turn_contexts,
            ),
            "sticker_asset_frequencies": _ranking_audit(sticker_counts, 64),
            "sticker_contexts": _context_ranking_audit(
                sticker_counts,
                sticker_previous_contexts,
                sticker_same_turn_contexts,
            ),
        },
    }


def _record_asset_context(
    asset_id: str,
    previous_text: str,
    same_turn_text: str,
    previous_contexts: dict[str, BoundedFrequency[str]],
    same_turn_contexts: dict[str, BoundedFrequency[str]],
) -> None:
    if previous_text.strip():
        previous_contexts.setdefault(
            asset_id,
            BoundedFrequency(_CONTEXT_CAPACITY),
        ).add(previous_text.strip())
    if same_turn_text:
        same_turn_contexts.setdefault(
            asset_id,
            BoundedFrequency(_CONTEXT_CAPACITY),
        ).add(same_turn_text)


def _remove_evicted_contexts(
    evicted: str | None,
    previous_contexts: dict[str, BoundedFrequency[str]],
    same_turn_contexts: dict[str, BoundedFrequency[str]],
) -> None:
    if evicted is None:
        return
    previous_contexts.pop(evicted, None)
    same_turn_contexts.pop(evicted, None)


def _asset_contexts(
    asset_counts: BoundedFrequency[str],
    previous_contexts: dict[str, BoundedFrequency[str]],
    same_turn_contexts: dict[str, BoundedFrequency[str]],
) -> dict[str, object]:
    selected_assets = [
        asset_id
        for asset_id, _count in sorted(
            asset_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:64]
    ]
    return {
        asset_id: {
            "previous_text": _ranked(
                previous_contexts.get(asset_id, {}),
                "text",
                16,
            ),
            "same_turn_text": _ranked(
                same_turn_contexts.get(asset_id, {}),
                "text",
                16,
            ),
        }
        for asset_id in selected_assets
    }


def _split_graphemes(text: str) -> list[str]:
    return cast(list[str], _GRAPHEME_PATTERN.findall(text))


def _is_emoji_grapheme(grapheme: str) -> bool:
    return bool(_EMOJI_GRAPHEME_PATTERN.search(grapheme))


def _term_counts(texts: Sequence[str], terms: Sequence[str]) -> list[dict[str, object]]:
    counts = Counter(
        {
            term: sum(text.count(term) for text in texts)
            for term in terms
            if any(term in text for text in texts)
        }
    )
    return _ranked(counts, "text", len(terms))


def _ranked(
    counts: Mapping[str, int],
    key_name: str,
    limit: int,
) -> list[dict[str, object]]:
    return [
        {key_name: key, "count": count}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _ranking_audit(
    counts: Mapping[str, int],
    limit: int,
    returned_count: int | None = None,
) -> dict[str, object]:
    actual_returned = min(len(counts), limit) if returned_count is None else returned_count
    bounded_audit = (
        counts.audit()
        if isinstance(counts, BoundedFrequency)
        else {
            "total_count": sum(counts.values()),
            "tracked_unique": len(counts),
            "internal_capacity": len(counts),
            "dropped_unique_count": 0,
            "approximate": False,
            "count_semantics": "exact-v1",
        }
    )
    return {
        **bounded_audit,
        "returned_count": actual_returned,
        "limit": limit,
        "truncated": len(counts) > limit
        or (
            isinstance(counts, BoundedFrequency)
            and counts.dropped_unique_count > 0
        ),
    }


def _pair_ranking_audit(
    counts: BoundedFrequency[tuple[str, str]],
    limit: int,
) -> dict[str, object]:
    return {
        **counts.audit(),
        "returned_count": min(len(counts), limit),
        "limit": limit,
        "truncated": len(counts) > limit or counts.dropped_unique_count > 0,
    }


def _context_ranking_audit(
    asset_counts: BoundedFrequency[str],
    previous_contexts: dict[str, BoundedFrequency[str]],
    same_turn_contexts: dict[str, BoundedFrequency[str]],
) -> dict[str, object]:
    context_unique = sum(
        len(previous_contexts.get(asset_id, {}))
        + len(same_turn_contexts.get(asset_id, {}))
        for asset_id in asset_counts
    )
    context_count = sum(
        sum(previous_contexts.get(asset_id, {}).values())
        + sum(same_turn_contexts.get(asset_id, {}).values())
        for asset_id in asset_counts
    )
    dropped_contexts = sum(
        context.dropped_unique_count
        for context in (
            *previous_contexts.values(),
            *same_turn_contexts.values(),
        )
    )
    maximum_context_error = max(
        (
            context.maximum_error
            for context in (
                *previous_contexts.values(),
                *same_turn_contexts.values(),
            )
        ),
        default=0,
    )
    return {
        "tracked_assets": len(asset_counts),
        "asset_internal_capacity": asset_counts.capacity,
        "dropped_asset_unique_count": asset_counts.dropped_unique_count,
        "returned_assets": min(len(asset_counts), 64),
        "asset_limit": 64,
        "total_context_count": context_count,
        "tracked_unique_contexts": context_unique,
        "dropped_context_unique_count": dropped_contexts,
        "count_semantics": "lower-bound-v1",
        "maximum_asset_error": asset_counts.maximum_error,
        "maximum_context_error": maximum_context_error,
        "context_limit_per_condition": 16,
        "approximate": asset_counts.dropped_unique_count > 0
        or dropped_contexts > 0,
        "truncated": len(asset_counts) > 64
        or asset_counts.dropped_unique_count > 0
        or dropped_contexts > 0
        or any(
            len(previous_contexts.get(asset_id, {})) > 16
            or len(same_turn_contexts.get(asset_id, {})) > 16
            for asset_id in asset_counts
        ),
    }


def nearest_rank(values: Sequence[int], percentile: float) -> float:
    """按 nearest-rank 定义计算分位数。"""

    if not values:
        return 0.0
    if not 0 < percentile <= 1:
        raise ValueError("分位数必须在 0 到 1 之间")
    ordered = sorted(values)
    return _nearest_rank_ordered(ordered, percentile)


def _nearest_rank_ordered(
    ordered: Sequence[int],
    percentile: float,
) -> float:
    rank = max(1, ceil(len(ordered) * percentile))
    return float(ordered[min(rank, len(ordered)) - 1])


def _distribution(values: Sequence[int]) -> dict[str, object]:
    ordered = sorted(values)
    if not ordered:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "histogram": [],
            "storage": "bounded-histogram-v1",
        }
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": _nearest_rank_ordered(ordered, 0.5),
        "p90": _nearest_rank_ordered(ordered, 0.9),
        "histogram": _histogram(ordered),
        "storage": "bounded-histogram-v1",
    }


def _histogram(values: Sequence[int]) -> list[dict[str, object]]:
    counts = [0] * (len(_HISTOGRAM_UPPER_BOUNDS) + 1)
    for value in values:
        index = next(
            (
                bucket_index
                for bucket_index, upper in enumerate(_HISTOGRAM_UPPER_BOUNDS)
                if value <= upper
            ),
            len(_HISTOGRAM_UPPER_BOUNDS),
        )
        counts[index] += 1
    result: list[dict[str, object]] = []
    lower = 0
    for index, count in enumerate(counts):
        if not count:
            if index < len(_HISTOGRAM_UPPER_BOUNDS):
                lower = _HISTOGRAM_UPPER_BOUNDS[index] + 1
            continue
        upper = (
            _HISTOGRAM_UPPER_BOUNDS[index]
            if index < len(_HISTOGRAM_UPPER_BOUNDS)
            else None
        )
        result.append({"lower": lower, "upper": upper, "count": count})
        if upper is not None:
            lower = upper + 1
    return result
