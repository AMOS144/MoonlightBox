"""隔离验证：复用真实 Nano 检索和 LightRAG 关联片段选择，不调用模型或真实库。

运行：sidecars/lightrag_sidecar/.venv/bin/python scripts/probe_lightrag_temporal.py
合成向量只验证候选限制，不代表真实语义质量；精确来源映射是本实验的输入假设。
"""

import asyncio
import hashlib
import inspect
import json
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from lightrag import QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import (
    _find_related_text_unit_from_entities,
    _find_related_text_unit_from_relations,
)
from nano_vectordb import NanoVectorDB


class ScopedChunks:
    """请求局部的原文视图，既有图和 chunk 不变，不做全局 monkey patch。"""

    global_config = {"kg_chunk_pick_method": "WEIGHT", "related_chunk_number": 1}

    def __init__(self, chunks):
        self.chunks = chunks

    async def get_by_ids(self, ids):
        await asyncio.sleep(0)  # 主动让出执行权，检查交错请求不会串用范围。
        return [self.chunks.get(key) for key in ids]


def scoped_graph_rows(rows, allowed):
    """在 LightRAG 的关联片段数量选择之前限制来源，不修改原图记录。"""
    result = []
    for row in rows:
        sources = [s for s in row["source_id"].split(GRAPH_FIELD_SEP) if s in allowed]
        if sources:
            result.append({**row, "source_id": GRAPH_FIELD_SEP.join(sources)})
    return result


async def main():
    started = perf_counter()
    # m2/m3 时间相同，按批准的消息序号分界，不能仅靠 timestamp。
    messages = {
        "m1": {"ordinal": 0, "sent_at": "2026-05-01T09:00:00+08:00", "text": "之前在旧岗位"},
        "m2": {"ordinal": 1, "sent_at": "2026-05-01T10:00:00+08:00", "text": "今晚按原安排"},
        "m3": {"ordinal": 2, "sent_at": "2026-05-01T10:00:00+08:00", "text": "这是边界后的新消息"},
        "m4": {"ordinal": 3, "sent_at": "2026-06-01T10:00:00+08:00", "text": "后来换了岗位"},
    }
    mapping = {"early": ["m1"], "boundary": ["m2", "m3"], "late": ["m4"]}
    chunks = {key: {"content": " / ".join(messages[m]["text"] for m in refs)}
              for key, refs in mapping.items()}
    chunks["unmapped"] = {"content": "无法准确定位，不应猜测时期"}
    vectors = {"early": [0.6, 0.8], "boundary": [0.8, 0.6], "late": [1., 0.],
               "unmapped": [1., 0.]}
    before = {key for key, refs in mapping.items()
              if all(messages[m]["ordinal"] < 2 for m in refs)}
    after = {key for key, refs in mapping.items()
             if all(messages[m]["ordinal"] >= 2 for m in refs)}
    observations = []

    def check(name, actual, expected):
        assert actual == expected, (name, actual, expected)
        observations.append({"case": name, "actual": actual, "passed": True})

    with TemporaryDirectory(prefix="moonlight-temporal-probe-") as directory:
        db = NanoVectorDB(2, storage_file=str(Path(directory) / "vectors.json"))
        db.upsert([{"__id__": key, "__vector__": np.array(value)} for key, value in vectors.items()])

        def query(allowed=None):
            # 空集合提前返回：不依赖底层对空过滤矩阵的特殊处理。
            if allowed is not None and not allowed:
                return []
            return [r["__id__"] for r in db.query(
                np.array([1., 0.]), top_k=1,
                filter_lambda=None if allowed is None else lambda row: row["__id__"] in allowed,
            )]

        global_result = query()
        check("全库top1后过滤漏掉早期材料", [key for key in global_result if key in before], [])
        check("Nano前置过滤找回早期材料", query(before), ["early"])
        check("Nano后段范围", query(after), ["late"])
        check("空范围", query(set()), [])
        check("交替请求无共享过滤状态", [query(before), query(after), query(before)],
              [["early"], ["late"], ["early"]])

        # 跨边界原文裁剪可确定；本实验不声称已验证裁剪后重新 embedding。
        clipped = [m for m in mapping["boundary"] if messages[m]["ordinal"] < 2]
        check("相同时间按消息边界裁剪", clipped, ["m2"])
        check("跨边界向量不冒充纯前段向量", "boundary" in before, False)
        check("未知来源不进入时间范围", "unmapped" in before | after, False)

        # 用真实 LightRAG WEIGHT 选择器，数量限制为1，未来来源排在最前。
        sources = GRAPH_FIELD_SEP.join(["late", "early"])
        entities = [{"entity_name": "target", "source_id": sources}]
        relations = [{"src_id": "target", "tgt_id": "work", "source_id": sources}]
        originals = deepcopy((entities, relations))

        async def related(kind, allowed):
            rows = entities if kind == "entity" else relations
            if allowed is not None:
                rows = scoped_graph_rows(rows, allowed)
            view = ScopedChunks({key: value for key, value in chunks.items()
                                 if allowed is None or key in allowed})
            if kind == "entity":
                result = await _find_related_text_unit_from_entities(rows, QueryParam(), view, None)
            else:
                result = await _find_related_text_unit_from_relations(rows, QueryParam(), view, [])
            return [r["chunk_id"] for r in result]

        for kind in ("entity", "relation"):
            baseline = await related(kind, None)
            check(f"{kind}原选择器优先未来片段", baseline, ["late"])
            check(f"{kind}选择前限制来源", await related(kind, before), ["early"])
            check(f"{kind}后段来源", await related(kind, after), ["late"])
            check(f"{kind}空来源", await related(kind, set()), [])
        check("请求处理不修改共享图来源", (entities, relations) == originals, True)
        concurrent = await asyncio.gather(related("entity", before), related("entity", after))
        check("并发请求范围隔离", concurrent, [["early"], ["late"]])

    source = Path(inspect.getfile(_find_related_text_unit_from_entities))
    print(json.dumps({
        "status": "passed", "cases": len(observations), "observations": observations,
        "lightrag_version": version("lightrag-hku"),
        "nano_version": version("nano-vectordb"),
        "operate_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "elapsed_seconds": round(perf_counter() - started, 4),
        "real_asset_writes": 0, "model_calls": 0,
        "limitations": ["使用人工精确映射和二维合成向量", "未验证全量实体候选的时间前置过滤",
                        "仅运行真实关联片段WEIGHT路径，未测试完整aquery或VECTOR路径",
                        "未验证真实chunk回填、跨界重嵌入、缓存或大数据性能"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
