"""当前栏目 Agent 的运行策略与资料转换；不包含旧版生成流程。"""

from __future__ import annotations

from typing import Any

from .investigation_artifacts import InvestigationArtifactStore
from .schemas import EvidenceMessage


def _evidence_from_artifacts(artifacts: InvestigationArtifactStore) -> list[EvidenceMessage]:
    """将工具已恢复的 SQL 行转换为证据类型；无法通过 schema 的行不参与事实结论。"""

    evidence: list[EvidenceMessage] = []
    for row in artifacts.evidence_rows():
        try:
            evidence.append(EvidenceMessage.model_validate(row, strict=False))
        except ValueError:
            pass
    return evidence


def _dedupe_evidence(values: list[EvidenceMessage]) -> list[EvidenceMessage]:
    result: list[EvidenceMessage] = []
    indexes: dict[str, int] = {}
    for value in values:
        current = indexes.get(value.message_id)
        if current is None:
            indexes[value.message_id] = len(result)
            result.append(value)
            continue
        previous = result[current]
        result[current] = previous.model_copy(
            update={
                "is_primary_match": previous.is_primary_match or value.is_primary_match,
                "retrieval_query_ids": list(
                    dict.fromkeys([*previous.retrieval_query_ids, *value.retrieval_query_ids])
                ),
                "reference_ranks": list(
                    dict.fromkeys([*previous.reference_ranks, *value.reference_ranks])
                ),
                "document_ranks": list(
                    dict.fromkeys([*previous.document_ranks, *value.document_ranks])
                ),
                "context_window_ids": list(
                    dict.fromkeys([*previous.context_window_ids, *value.context_window_ids])
                ),
                "retrieval_provenance": _dedupe_provenance(
                    [*previous.retrieval_provenance, *value.retrieval_provenance]
                ),
            }
        )
    return result


def _dedupe_provenance(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[tuple[str, int, int, str]] = set()
    for item in values:
        key = (
            str(item.query_id),
            int(item.reference_rank),
            int(item.document_rank),
            str(item.context_window_id),
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _retrievals_from_artifacts(artifacts: InvestigationArtifactStore) -> list[dict[str, object]]:
    """审核层读取完整检索清单，但模型上下文只看其投影。"""

    return [
        {
            "query_id": item.retrieval_id,
            "retrieval_id": item.retrieval_id,
            "question": item.question,
            "mode": item.mode,
            "references": list(item.references),
        }
        for item in artifacts.retrievals()
    ]
