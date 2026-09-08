"""与具体模型后端无关的回复语义硬约束。"""

from __future__ import annotations

import re
from collections import Counter

_NEGATION_MARKERS = ("不", "没", "无", "别", "未", "不能", "不会", "不是")
_RELATIONSHIP_STANCE_GROUPS = (
    ("保持距离", "距离", "少联系", "别联系", "不联系", "冷静一段", "缓一缓", "需要空间"),
    ("修复关系", "修复", "和好", "重新开始", "继续这段关系"),
)
_PERSON_REFERENTS = re.compile(r"我们|咱们|你们|您们|他们|她们|它们|我|咱|你|您|他|她|它")


def rewrite_preserves_hard_semantics(draft: str, rewritten: str) -> bool:
    """拒绝风格改写中的否定翻转、数字/指代漂移和内容坍缩。"""

    if re.findall(r"\d+(?:\.\d+)?", draft) != re.findall(r"\d+(?:\.\d+)?", rewritten):
        return False
    if Counter(_PERSON_REFERENTS.findall(draft)) != Counter(_PERSON_REFERENTS.findall(rewritten)):
        return False
    for marker in _NEGATION_MARKERS:
        if draft.count(marker) != rewritten.count(marker):
            return False
    for stance_group in _RELATIONSHIP_STANCE_GROUPS:
        if any(marker in draft for marker in stance_group) and not any(
            marker in rewritten for marker in stance_group
        ):
            return False
    compact_draft = re.sub(r"\s|[，。！？!?；;、]", "", draft)
    compact_rewritten = re.sub(r"\s|[，。！？!?；;、]", "", rewritten)
    if not compact_draft or not compact_rewritten:
        return False
    ratio = len(compact_rewritten) / len(compact_draft)
    return 0.5 <= ratio <= 1.8
