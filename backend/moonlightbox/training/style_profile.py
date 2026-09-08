from collections import Counter
from collections.abc import Sequence

from moonlightbox.training.style_features import (
    StyleBubble,
    StyleTurn,
    extract_style_features,
    nearest_rank,
)

STYLE_MARKERS = ("哒", "啦~", "啦～", "嘛~", "嘛～", "哦~", "哦～")
AI_REGISTER_MARKERS = (
    "我理解你的感受",
    "我能理解你的感受",
    "听起来你",
    "这确实很",
    "如果你愿意",
    "无论如何",
    "别担心",
    "我会一直陪着你",
    "你并不孤单",
    "慢慢来",
    "照顾好自己",
    "我担心你",
    "我希望你",
    "你要知道",
    "我只是想",
    "我觉得我们",
    "这让我觉得",
    "我不想让你",
    "我会尊重",
    "你的感受",
    "对我来说",
    "我很抱歉",
    "我太抱歉了",
    "我明白你的意思",
    "我理解你的意思",
    "建议你",
    "你可以考虑",
    "你可以试试",
    "这样可能会",
    "我不会让你失望",
    "我也不好意思",
    "毕竟你",
    "我明白了",
    "很抱歉",
    "你得先",
    "我会努力",
    "我保证",
    "我会尽量",
    "我只负责",
    "怎么突然这么主动",
    "我错怪你",
    "我撑得住",
    "骑车带你回家",
    "转到你心坎里",
    "我最近特别关注",
    "我不能参与这样的对话",
    "不能参与这样的对话",
    "请您",
    "能否请您",
    "您是想",
    "您想了解",
    "我可以帮你",
    "可以帮您",
    "需要帮忙吗",
    "查找相关信息",
    "具体说明",
    "建议更加",
    "随时可以接听",
)


def build_style_profile(texts: list[str]) -> dict[str, object]:
    cleaned = [text.strip() for text in texts if text.strip()]
    unified = extract_style_features(
        [
            StyleTurn(bubbles=(StyleBubble(text=text),))
            for text in cleaned
        ]
    )
    return {**unified, **_compatibility_fields(cleaned)}


def _compatibility_fields(cleaned: list[str]) -> dict[str, object]:
    """保留旧调用方依赖的顶层字段，统计来源仍是同一批规范化文本。"""

    lengths = [len(text) for text in cleaned]
    marker_counts = {
        marker: sum(marker in text for text in cleaned)
        for marker in STYLE_MARKERS
    }
    ai_register_counts = {
        marker: sum(marker in text for text in cleaned)
        for marker in AI_REGISTER_MARKERS
    }
    endings = Counter(
        text[-2:] if len(text) >= 2 else text
        for text in cleaned
    )
    ranked_endings = sorted(
        endings.items(),
        key=lambda item: (-item[1], item[0]),
    )[:8]
    p90 = nearest_rank(lengths, 0.9)
    return {
        "sample_count": len(cleaned),
        "average_length": (
            sum(lengths) / len(lengths)
            if lengths
            else 0.0
        ),
        "p90_length": p90,
        "question_rate": (
            sum(text.endswith(("吗", "呢", "？", "?")) for text in cleaned)
            / len(cleaned)
            if cleaned
            else 0.0
        ),
        "common_endings": [
            {"text": ending, "count": count}
            for ending, count in ranked_endings
        ],
        "compatibility_ranking_audit": {
            "common_endings": {
                "total_count": sum(endings.values()),
                "total_unique": len(endings),
                "returned_count": len(ranked_endings),
                "limit": 8,
                "truncated": len(endings) > 8,
            }
        },
        "marker_counts": marker_counts,
        "ai_register_counts": ai_register_counts,
        "forbidden_unobserved_ai_register": [
            marker for marker, count in ai_register_counts.items() if count == 0
        ],
        "forbidden_unobserved_markers": [
            marker for marker, count in marker_counts.items() if count == 0
        ],
        "discouraged_rare_markers": [
            marker
            for marker, count in marker_counts.items()
            if cleaned and count / len(cleaned) < 0.01
        ],
    }


def build_style_profile_from_turns(
    turns: Sequence[StyleTurn],
) -> dict[str, object]:
    """直接从气泡轮次构建统一画像，供训练清单和评估共用。"""

    unified = extract_style_features(turns)
    texts = [
        bubble.text.strip()
        for turn in turns
        for bubble in turn.bubbles
        if bubble.kind in {"text", "emoji"} and bubble.text.strip()
    ]
    return {**unified, **_compatibility_fields(texts)}


def style_violations(
    text: str,
    profile: dict[str, object] | None,
) -> list[str]:
    if not profile:
        return []
    forbidden = profile.get("forbidden_unobserved_markers", [])
    discouraged = profile.get("discouraged_rare_markers", [])
    markers: list[object] = []
    if isinstance(forbidden, list):
        markers.extend(forbidden)
    if isinstance(discouraged, list):
        markers.extend(discouraged)
    marker_counts = profile.get("marker_counts")
    ai_register = profile.get("forbidden_unobserved_ai_register", [])
    if isinstance(ai_register, list):
        markers.extend(ai_register)
    ai_register_counts = profile.get("ai_register_counts", {})
    if isinstance(ai_register_counts, dict):
        markers.extend(
            marker
            for marker in AI_REGISTER_MARKERS
            if not isinstance(ai_register_counts.get(marker), int)
            or ai_register_counts.get(marker) == 0
        )
    sample_count = profile.get("sample_count")
    if (
        isinstance(marker_counts, dict)
        and isinstance(sample_count, int)
        and sample_count > 0
    ):
        markers.extend(
            marker
            for marker, count in marker_counts.items()
            if isinstance(marker, str)
            and isinstance(count, int)
            and count / sample_count < 0.01
        )
    matched = [
        marker
        for marker in dict.fromkeys(markers)
        if isinstance(marker, str) and marker in text
    ]
    return [
        marker
        for marker in matched
        if not any(
            marker != other and marker in other
            for other in matched
        )
    ]
