"""PersonWorld 单栏目运行内的检索与原始消息工件。

这里不是数据库、缓存或第二套 trace。它的生命周期严格等同于一次 Section/Revision
Agent 执行：LightRAG 的完整引用只在这里与 Phoenix 中保留，模型上下文只收到一个小的
检索清单；定位工具再用检索 ID 读取完整引用并恢复 SQL 原始消息。
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RetrievalArtifact:
    """一次 LightRAG 查询的完整受限引用，不能直接作为模型证据。"""

    retrieval_id: str
    question: str
    mode: str
    references: tuple[dict[str, object], ...]


class InvestigationArtifactStore:
    """执行期的桥接存储，按检索 ID 和证据集 ID 保留可审计对象。"""

    def __init__(self) -> None:
        self._retrievals: OrderedDict[str, RetrievalArtifact] = OrderedDict()
        self._evidence_sets: OrderedDict[str, tuple[dict[str, object], ...]] = OrderedDict()
        self._evidence_set_retrieval_ids: dict[str, str | None] = {}
        self.section_work: dict[str, object] = {}

    def add_retrieval(
        self,
        *,
        question: str,
        mode: str,
        references: Iterable[Mapping[str, object]],
    ) -> RetrievalArtifact:
        """写入已通过来源白名单过滤的完整 LightRAG 引用。"""

        retrieval_id = f"retrieval-{len(self._retrievals) + 1}"
        artifact = RetrievalArtifact(
            retrieval_id=retrieval_id,
            question=question,
            mode=mode,
            references=tuple(dict(item) for item in references),
        )
        self._retrievals[retrieval_id] = artifact
        return artifact

    def get_retrieval(self, retrieval_id: str) -> RetrievalArtifact | None:
        return self._retrievals.get(retrieval_id)

    def snapshot(self):
        """正文随统一循环保存，恢复后旧检索 ID 和分页 ID 仍指向原材料。"""
        return deepcopy(
            {
                "version": 1,
                "section_work": self.section_work,
                "retrievals": [asdict(item) for item in self._retrievals.values()],
                "evidence_sets": dict(self._evidence_sets),
                "evidence_set_retrieval_ids": dict(self._evidence_set_retrieval_ids),
            }
        )

    def restore(self, state):
        """完整替换工作材料，避免多次恢复导致编号漂移。"""
        if state.get("version") != 1:
            raise ValueError("不支持的调查材料存档版本")
        state = deepcopy(state)
        self.section_work = state.get("section_work", {})
        self._retrievals = OrderedDict(
            (item["retrieval_id"], RetrievalArtifact(**item)) for item in state["retrievals"]
        )
        self._evidence_sets = OrderedDict(
            (key, tuple(rows)) for key, rows in state["evidence_sets"].items()
        )
        self._evidence_set_retrieval_ids = state["evidence_set_retrieval_ids"]

    def retrievals(self) -> tuple[RetrievalArtifact, ...]:
        return tuple(self._retrievals.values())

    def add_evidence_set(
        self,
        rows: Iterable[Mapping[str, object]],
        *,
        retrieval_id: str | None,
    ) -> str:
        """保存由 SQL 恢复的消息；不在此处解释消息内容。"""

        evidence_set_id = f"evidence-{len(self._evidence_sets) + 1}"
        self._evidence_sets[evidence_set_id] = tuple(dict(row) for row in rows)
        self._evidence_set_retrieval_ids[evidence_set_id] = retrieval_id
        return evidence_set_id

    def evidence_rows(self) -> tuple[dict[str, object], ...]:
        """以进入工具的先后顺序返回全部真实 SQL 消息，供最终事实校验使用。"""

        return tuple(row for rows in self._evidence_sets.values() for row in rows)

    def evidence_set_retrieval_id(self, evidence_set_id: str) -> str | None:
        return self._evidence_set_retrieval_ids.get(evidence_set_id)

    def evidence_set_rows(self, evidence_set_id: str) -> tuple[dict[str, object], ...]:
        """原生工具 Agent 分页读取当前执行的原文，不能访问其他运行的工件。"""
        if evidence_set_id not in self._evidence_sets:
            raise ValueError("原始消息集合不属于当前运行")
        return self._evidence_sets[evidence_set_id]


def project_search_world_result(result: object) -> object:
    """给模型的检索清单；完整 chunk 永远不回填到 ToolMessage。"""

    if not isinstance(result, Mapping):
        return result
    references = result.get("references")
    items = references if isinstance(references, list) else []
    documents = [
        {
            "document_name": item.get("document_name"),
            "reference_rank": item.get("reference_rank"),
            "document_rank": item.get("document_rank"),
        }
        for item in items[:12]
        if isinstance(item, Mapping) and isinstance(item.get("document_name"), str)
    ]
    return {
        "retrieval_id": result.get("retrieval_id"),
        "question": result.get("question"),
        "mode": result.get("mode"),
        "reference_count": len(items),
        "documents": documents,
    }


def project_evidence_set_result(result: object) -> object:
    """给模型的原始消息清单；消息正文由 Section Agent 从工件库按需选入提示。"""

    if not isinstance(result, Mapping):
        return result
    message_ids = result.get("message_ids")
    return {
        "retrieval_id": result.get("retrieval_id"),
        "evidence_set_id": result.get("evidence_set_id"),
        "message_count": result.get("message_count", 0),
        "primary_message_ids": result.get("primary_message_ids", []),
        "message_ids": message_ids[:24] if isinstance(message_ids, list) else [],
    }
