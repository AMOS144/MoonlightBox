from dataclasses import dataclass


@dataclass(frozen=True)
class NodeLabel:
    type: str
    evidence_ids: frozenset[str]


@dataclass(frozen=True)
class NodeMetrics:
    precision: float
    recall: float
    f1: float
    matched: int


def evaluate_nodes(
    predicted: list[NodeLabel],
    gold: list[NodeLabel],
    minimum_overlap: float = 0.5,
) -> NodeMetrics:
    unmatched_gold = set(range(len(gold)))
    matched = 0

    for candidate in predicted:
        possible = [
            (
                _jaccard(candidate.evidence_ids, gold[index].evidence_ids),
                index,
            )
            for index in unmatched_gold
            if candidate.type == gold[index].type
        ]
        if not possible:
            continue
        overlap, index = max(possible)
        if overlap >= minimum_overlap:
            matched += 1
            unmatched_gold.remove(index)

    if not predicted and not gold:
        return NodeMetrics(1.0, 1.0, 1.0, 0)
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return NodeMetrics(
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        matched=matched,
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0
