from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Never
from typing import cast as type_cast
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import Engine, Float, and_, cast, create_engine, func, select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from moonlightbox.evaluation.node_metrics import NodeLabel, evaluate_nodes
from moonlightbox.events.models import (
    AnalysisRevision,
    AnalysisRun,
    EventCandidate,
    EventNode,
)
from moonlightbox.events.v3_types import EventLane, EventStatus
from moonlightbox.imports.models import Message, Participant
from moonlightbox.projects.models import Project

DEFAULT_PRECISION_THRESHOLD = 0.8
MAXIMUM_REVIEW_NODES = 25
MAXIMUM_EVIDENCE_IDS_PER_NODE = 100
EVIDENCE_QUERY_CHUNK_SIZE = 500
MAXIMUM_REVIEW_FILE_BYTES = 10 * 1024 * 1024
_HASH_ALGORITHM = "sha256"
_IMAGE_LINK = re.compile(r"!\[[^\]]*]\([^)]*\)")
_URL = re.compile(r"https?://[^\s)]+")
_TYPE_LABELS = {
    "relationship_started": "关系建立",
    "intimacy_increased": "亲密升级",
    "commitment": "承诺",
    "boundary_change": "边界变化",
    "conflict": "冲突",
    "distancing": "疏远",
    "reconciliation": "和解",
    "separation": "分离",
    "reconnection": "重新连接",
    "date": "约会",
    "outing": "出游",
    "travel": "旅行",
    "celebration": "庆祝",
    "gift": "礼物",
    "family_social": "见亲友",
    "support_care": "照顾陪伴",
    "shared_project": "共同项目",
    "important_plan": "重要计划",
    "life_milestone": "人生里程碑",
}


class NodeAcceptanceScopeError(LookupError):
    """项目、运行或历史发布数据不能用于当前验收。"""


class NodeAcceptanceReviewError(ValueError):
    """人工审核包结构或完整性不可信。"""


class NodeAcceptanceOutputError(OSError):
    """审核包输出路径不安全或写入不完整。"""


class NodeAcceptanceCliInputError(ValueError):
    """命令行参数或 SQLite 数据库路径无效。"""


class _NodeAcceptanceArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        _ = message
        raise NodeAcceptanceCliInputError("命令参数错误")


class EvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    sender: str
    timestamp: datetime
    content: str


class NodeScoreComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state_change_strength: float = Field(ge=0, le=1)
    persistence: float = Field(ge=0, le=1)
    evidence_quality: float = Field(ge=0, le=1)
    model_confidence: float = Field(ge=0, le=1)


class V3NodeScoreComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_significance: float = Field(ge=0, le=1)
    relationship_impact: float = Field(ge=0, le=1)
    evidence_quality: float = Field(ge=0, le=1)
    persistence: float = Field(ge=0, le=1)
    type_support: float = Field(ge=0, le=1)
    model_confidence: float = Field(ge=0, le=1)


class NodeAnalysisVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis_version: str
    prompt_version: str
    model: str


def _default_review_source_lanes() -> list[EventLane]:
    return ["relationship"]


class NodeReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    lane: EventLane = "relationship"
    type: str
    type_label: str
    title: str = ""
    event_status: EventStatus = "occurred"
    summary: str = ""
    source_lanes: list[EventLane] = Field(default_factory=_default_review_source_lanes)
    before_state: str | None
    after_state: str | None
    score: float = Field(ge=0, le=1)
    score_components: NodeScoreComponents | V3NodeScoreComponents
    version: NodeAnalysisVersion
    evidence_ids: list[str]
    evidence_summary: list[EvidenceSummary]
    canonical_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verdict: Literal["pending", "accepted", "rejected"] = "pending"


class ExportManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exported_at: datetime
    hash_algorithm: Literal["sha256"] = "sha256"
    node_count: int = Field(ge=0, le=MAXIMUM_REVIEW_NODES)
    node_ids: list[str]
    node_hashes: list[str]
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class NodeReviewPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["node-acceptance-v2", "node-acceptance-v3"]
    project_id: str
    analysis_run_id: str
    export_manifest: ExportManifest
    nodes: list[NodeReviewItem]

    @model_validator(mode="after")
    def validate_unique_nodes(self) -> NodeReviewPacket:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("审核包中的 node_id 不能重复")
        if len(node_ids) > MAXIMUM_REVIEW_NODES:
            raise ValueError(f"审核包最多包含 {MAXIMUM_REVIEW_NODES} 个节点")
        return self


class NodeAcceptanceResult(BaseModel):
    status: Literal["pass", "fail", "pending"]
    project_id: str
    analysis_run_id: str
    threshold: float
    precision: float | None
    total_nodes: int
    reviewed_nodes: int
    accepted_nodes: int
    rejected_nodes: int
    pending_nodes: int
    matched_nodes: int
    metric_scope: Literal["precision_only"] = "precision_only"
    recall: None = None
    f1: None = None
    metric_note: str
    reason: str


def build_review_packet(
    session: Session,
    *,
    project_id: str,
    run_id: str,
) -> NodeReviewPacket:
    """从指定运行的发布历史生成可复现、可校验的本地审核包。"""

    run = _validated_run(session, project_id=project_id, run_id=run_id)
    packet_schema_version: Literal[
        "node-acceptance-v2",
        "node-acceptance-v3",
    ] = "node-acceptance-v3" if run.analysis_version == "hybrid-v3" else "node-acceptance-v2"
    publish_reason = "V3 自动发布" if run.analysis_version == "hybrid-v3" else "V2 自动发布"
    rows = list(
        session.execute(
            select(AnalysisRevision, EventCandidate)
            .join(EventNode, EventNode.id == AnalysisRevision.event_id)
            .outerjoin(
                EventCandidate,
                and_(
                    EventCandidate.id == AnalysisRevision.candidate_id,
                    EventCandidate.run_id == run.id,
                ),
            )
            .where(
                AnalysisRevision.run_id == run.id,
                AnalysisRevision.action_reason == publish_reason,
                EventNode.project_id == project_id,
            )
            .order_by(
                cast(
                    func.json_extract(EventCandidate.scores, "$.total"),
                    Float,
                ).desc(),
                AnalysisRevision.created_at,
                AnalysisRevision.event_id,
            )
            .limit(MAXIMUM_REVIEW_NODES + 1)
        )
    )
    if len(rows) > MAXIMUM_REVIEW_NODES:
        raise NodeAcceptanceScopeError(
            f"该运行发布节点超过 {MAXIMUM_REVIEW_NODES} 个，拒绝截断验收"
        )

    prepared = [_prepare_revision(revision, candidate, run) for revision, candidate in rows]
    evidence_rows = _load_evidence_rows(
        session,
        import_id=run.import_id,
        evidence_ids={
            evidence_id
            for item in prepared
            for evidence_id in item["evidence_ids"]
            if isinstance(evidence_id, str)
        },
    )
    nodes = [_build_review_item(item, evidence_rows=evidence_rows) for item in prepared]
    exported_at = _stable_run_timestamp(run)
    node_ids = [node.node_id for node in nodes]
    node_hashes = [node.canonical_hash for node in nodes]
    manifest_payload = {
        "schema_version": packet_schema_version,
        "project_id": project_id,
        "analysis_run_id": run_id,
        "exported_at": exported_at.isoformat(),
        "hash_algorithm": _HASH_ALGORITHM,
        "node_count": len(nodes),
        "node_ids": node_ids,
        "node_hashes": node_hashes,
    }
    manifest = ExportManifest(
        exported_at=exported_at,
        hash_algorithm="sha256",
        node_count=len(nodes),
        node_ids=node_ids,
        node_hashes=node_hashes,
        manifest_hash=_canonical_hash(manifest_payload),
    )
    return NodeReviewPacket(
        schema_version=packet_schema_version,
        project_id=project_id,
        analysis_run_id=run_id,
        export_manifest=manifest,
        nodes=nodes,
    )


def evaluate_review_packet(
    session: Session,
    *,
    project_id: str,
    run_id: str,
    reviewed_packet: dict[str, object],
    threshold: float = DEFAULT_PRECISION_THRESHOLD,
) -> NodeAcceptanceResult:
    """校验审核包后，仅以人工节点审核结果判定预测精确率。"""

    if not 0 <= threshold <= 1:
        raise ValueError("threshold 必须在 0 到 1 之间")
    try:
        reviewed = NodeReviewPacket.model_validate(reviewed_packet)
    except ValidationError as error:
        raise NodeAcceptanceReviewError("人工审核包格式无效") from error
    if reviewed.project_id != project_id or reviewed.analysis_run_id != run_id:
        raise NodeAcceptanceReviewError("人工审核包的 project_id 或 analysis_run_id 不匹配")
    _verify_submitted_integrity(reviewed)
    expected = build_review_packet(
        session,
        project_id=project_id,
        run_id=run_id,
    )
    _verify_against_run(reviewed, expected)

    predicted = [NodeLabel(node.type, frozenset(node.evidence_ids)) for node in expected.nodes]
    gold = [
        NodeLabel(node.type, frozenset(node.evidence_ids))
        for node in reviewed.nodes
        if node.verdict == "accepted"
    ]
    metrics = evaluate_nodes(predicted, gold)
    accepted = len(gold)
    rejected = sum(node.verdict == "rejected" for node in reviewed.nodes)
    pending = sum(node.verdict == "pending" for node in reviewed.nodes)
    reviewed_count = accepted + rejected
    precision = metrics.precision if predicted else None
    if not predicted:
        status: Literal["pass", "fail", "pending"] = "pending"
        reason = "没有可审核节点，无法计算精确率"
    elif pending:
        status = "pending"
        reason = f"仍有 {pending} 个节点未完成人工审核"
    elif metrics.precision >= threshold:
        status = "pass"
        reason = "全部节点已审核且精确率达到阈值"
    else:
        status = "fail"
        reason = "全部节点已审核但精确率低于阈值"
    return NodeAcceptanceResult(
        status=status,
        project_id=project_id,
        analysis_run_id=run_id,
        threshold=threshold,
        precision=precision,
        total_nodes=len(predicted),
        reviewed_nodes=reviewed_count,
        accepted_nodes=accepted,
        rejected_nodes=rejected,
        pending_nodes=pending,
        matched_nodes=metrics.matched,
        metric_note=(
            "本结果只用于人工节点审核的 precision 门槛；"
            "审核包不是全量漏标语料，召回率与 F1 不构成真实召回结论。"
        ),
        reason=reason,
    )


def render_review_packet_markdown(packet: NodeReviewPacket) -> str:
    """以不执行 HTML、图片和自动链接的形式渲染审核内容。"""

    lines = [
        "# 节点质量人工审核包",
        "",
        "以下动态内容均按缩进代码文本展示，不执行 HTML、图片或链接。",
        "",
        "- 项目与运行：",
        f"    project_id: {_markdown_plain_text(packet.project_id)}",
        f"    analysis_run_id: {_markdown_plain_text(packet.analysis_run_id)}",
        f"    node_count: {packet.export_manifest.node_count}",
        "",
        "请只在 JSON 审核包中编辑 verdict：pending、accepted 或 rejected。",
    ]
    for index, node in enumerate(packet.nodes, start=1):
        lines.extend(
            [
                "",
                f"## 节点 {index}",
                "",
                f"    node_id: {_markdown_plain_text(node.node_id)}",
                (
                    "    类型: "
                    f"{_markdown_plain_text(node.type_label)} / "
                    f"{_markdown_plain_text(node.type)}"
                ),
                f"    通道: {_markdown_plain_text(node.lane)}",
                f"    标题: {_markdown_plain_text(node.title)}",
                f"    事件状态: {_markdown_plain_text(node.event_status)}",
                f"    摘要: {_markdown_plain_text(node.summary)}",
                (f"    前状态: {_markdown_plain_text(node.before_state or '不适用')}"),
                (f"    后状态: {_markdown_plain_text(node.after_state or '不适用')}"),
                f"    分数: {node.score:.6f}",
                f"    verdict: {node.verdict}",
                "    证据摘要:",
            ]
        )
        if not node.evidence_summary:
            lines.append("    支持证据内容：无可安全展示的文本")
        for evidence in node.evidence_summary:
            lines.extend(
                [
                    (
                        "    证据元数据: "
                        f"{_markdown_plain_text(evidence.message_id)} / "
                        f"{_markdown_plain_text(evidence.sender)} / "
                        f"{evidence.timestamp.isoformat()}"
                    ),
                    (f"    支持证据内容：{_markdown_plain_text(evidence.content)}"),
                ]
            )
    return "\n".join(lines) + "\n"


def _prepare_revision(
    revision: AnalysisRevision,
    candidate: EventCandidate | None,
    run: AnalysisRun,
) -> dict[str, Any]:
    if candidate is None or candidate.run_id != run.id:
        raise NodeAcceptanceScopeError("发布修订未绑定同一运行的候选")
    snapshot = revision.snapshot
    event_type = _snapshot_string(snapshot, "type")
    if event_type not in _TYPE_LABELS:
        raise NodeAcceptanceScopeError("发布修订包含未知节点类型")
    evidence_ids = _snapshot_string_list(snapshot, "evidence_ids")
    if len(evidence_ids) > MAXIMUM_EVIDENCE_IDS_PER_NODE:
        raise NodeAcceptanceScopeError(f"单个节点证据超过 {MAXIMUM_EVIDENCE_IDS_PER_NODE} 条")
    is_v3 = run.analysis_version == "hybrid-v3"
    score_source: Mapping[str, float]
    if is_v3:
        snapshot_scores = snapshot.get("scores")
        if not isinstance(snapshot_scores, Mapping):
            raise NodeAcceptanceScopeError("V3 发布修订缺少评分快照")
        score_source = type_cast(Mapping[str, float], snapshot_scores)
    else:
        score_source = candidate.scores
    score_components = _candidate_score_components(
        score_source,
        analysis_version=run.analysis_version,
    )
    lane = _snapshot_optional_string(snapshot, "lane") or "relationship"
    source_lanes = _snapshot_string_list(snapshot, "source_lanes") if is_v3 else ["relationship"]
    return {
        "node_id": revision.event_id,
        "lane": lane,
        "type": event_type,
        "type_label": _TYPE_LABELS[event_type],
        "title": (
            _safe_text(_snapshot_string(snapshot, "title")) if is_v3 else _safe_text(event_type)
        ),
        "event_status": (_snapshot_string(snapshot, "event_status") if is_v3 else "occurred"),
        "summary": (_safe_text(_snapshot_string(snapshot, "summary")) if is_v3 else ""),
        "source_lanes": source_lanes,
        "before_state": _safe_optional_text(_snapshot_optional_string(snapshot, "before_state")),
        "after_state": _safe_optional_text(_snapshot_optional_string(snapshot, "after_state")),
        "score": _score_value(score_source, "total"),
        "score_components": score_components,
        "version": NodeAnalysisVersion(
            analysis_version=revision.analysis_version,
            prompt_version=revision.prompt_version,
            model=revision.model or run.model,
        ),
        "evidence_ids": evidence_ids,
    }


def _load_evidence_rows(
    session: Session,
    *,
    import_id: str,
    evidence_ids: set[str],
) -> dict[str, EvidenceSummary]:
    ordered_ids = sorted(evidence_ids)
    rows_by_id: dict[str, EvidenceSummary] = {}
    for start in range(0, len(ordered_ids), EVIDENCE_QUERY_CHUNK_SIZE):
        chunk = ordered_ids[start : start + EVIDENCE_QUERY_CHUNK_SIZE]
        rows = session.execute(
            select(
                Message.source_id,
                Participant.name,
                Message.timestamp,
                Message.content,
            )
            .join(Participant, Participant.id == Message.participant_id)
            .where(
                Message.import_id == import_id,
                Message.source_id.in_(chunk),
            )
        )
        for source_id, sender, timestamp, content in rows:
            rows_by_id[source_id] = EvidenceSummary(
                message_id=source_id,
                sender=_safe_text(sender),
                timestamp=timestamp,
                content=_safe_text(content),
            )
    return rows_by_id


def _build_review_item(
    item: Mapping[str, Any],
    *,
    evidence_rows: Mapping[str, EvidenceSummary],
) -> NodeReviewItem:
    evidence_ids = list(item["evidence_ids"])
    payload = {
        **item,
        "evidence_summary": [
            evidence_rows[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_rows
        ],
    }
    canonical_hash = _canonical_hash(_json_ready(payload))
    return NodeReviewItem(
        **payload,
        canonical_hash=canonical_hash,
        verdict="pending",
    )


def _verify_submitted_integrity(packet: NodeReviewPacket) -> None:
    manifest = packet.export_manifest
    manifest_payload = {
        "schema_version": packet.schema_version,
        "project_id": packet.project_id,
        "analysis_run_id": packet.analysis_run_id,
        "exported_at": manifest.exported_at.isoformat(),
        "hash_algorithm": manifest.hash_algorithm,
        "node_count": manifest.node_count,
        "node_ids": manifest.node_ids,
        "node_hashes": manifest.node_hashes,
    }
    if _canonical_hash(manifest_payload) != manifest.manifest_hash:
        raise NodeAcceptanceReviewError("审核包 manifest 完整性校验失败")
    node_ids = [node.node_id for node in packet.nodes]
    node_hashes = [node.canonical_hash for node in packet.nodes]
    if (
        manifest.node_count != len(packet.nodes)
        or manifest.node_ids != node_ids
        or manifest.node_hashes != node_hashes
    ):
        raise NodeAcceptanceReviewError("审核包节点清单完整性校验失败")
    for node in packet.nodes:
        if _node_hash(node) != node.canonical_hash:
            raise NodeAcceptanceReviewError(f"节点 {node.node_id} 完整性校验失败")


def _verify_against_run(
    reviewed: NodeReviewPacket,
    expected: NodeReviewPacket,
) -> None:
    if reviewed.export_manifest != expected.export_manifest:
        raise NodeAcceptanceReviewError("审核包与分析运行的发布历史完整性不一致")


def _node_hash(node: NodeReviewItem) -> str:
    payload = node.model_dump(
        mode="json",
        exclude={"canonical_hash", "verdict"},
    )
    return _canonical_hash(payload)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _json_ready(payload: Mapping[str, Any]) -> dict[str, Any]:
    return type_cast(
        dict[str, Any],
        json.loads(json.dumps(payload, ensure_ascii=False, default=_json_default)),
    )


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"不支持的审核字段类型：{type(value).__name__}")


def _snapshot_string(snapshot: Mapping[str, object], key: str) -> str:
    value = snapshot.get(key)
    if not isinstance(value, str) or not value:
        raise NodeAcceptanceScopeError(f"发布修订缺少有效字段：{key}")
    return value


def _snapshot_optional_string(
    snapshot: Mapping[str, object],
    key: str,
) -> str | None:
    value = snapshot.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise NodeAcceptanceScopeError(f"发布修订字段类型无效：{key}")
    return value


def _snapshot_string_list(
    snapshot: Mapping[str, object],
    key: str,
) -> list[str]:
    value = snapshot.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise NodeAcceptanceScopeError(f"发布修订缺少有效字段：{key}")
    return list(value)


def _score_value(scores: Mapping[str, float], key: str) -> float:
    value = scores.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise NodeAcceptanceScopeError(f"候选分数缺少有效字段：{key}")
    normalized = float(value)
    if not 0 <= normalized <= 1:
        raise NodeAcceptanceScopeError(f"候选分数字段超出范围：{key}")
    return normalized


def _candidate_score_components(
    scores: Mapping[str, float],
    *,
    analysis_version: str,
) -> NodeScoreComponents | V3NodeScoreComponents:
    if analysis_version == "hybrid-v3":
        return V3NodeScoreComponents(
            event_significance=_score_value(scores, "event_significance"),
            relationship_impact=_score_value(scores, "relationship_impact"),
            evidence_quality=_score_value(scores, "evidence_quality"),
            persistence=_score_value(scores, "persistence"),
            type_support=_score_value(scores, "type_support"),
            model_confidence=_score_value(scores, "model_confidence"),
        )
    return NodeScoreComponents(
        state_change_strength=_score_value(scores, "state_change_strength"),
        persistence=_score_value(scores, "persistence"),
        evidence_quality=_score_value(scores, "evidence_quality"),
        model_confidence=_score_value(scores, "model_confidence"),
    )


def _validated_run(
    session: Session,
    *,
    project_id: str,
    run_id: str,
) -> AnalysisRun:
    if session.get(Project, project_id) is None:
        raise NodeAcceptanceScopeError("项目不存在")
    run = session.get(AnalysisRun, run_id)
    if run is None or run.project_id != project_id:
        raise NodeAcceptanceScopeError("分析运行不属于指定项目")
    if run.analysis_version not in {"hybrid-v2", "hybrid-v3"} or run.status != "succeeded":
        raise NodeAcceptanceScopeError("只能验收已成功完成的 hybrid-v2 或 hybrid-v3 分析运行")
    return run


def _stable_run_timestamp(run: AnalysisRun) -> datetime:
    if run.completed_at is None:
        raise NodeAcceptanceScopeError("成功运行缺少稳定完成时间")
    if run.completed_at.tzinfo is None:
        return run.completed_at.replace(tzinfo=UTC)
    return run.completed_at.astimezone(UTC)


def _safe_text(value: str) -> str:
    printable = "".join(character for character in value if character.isprintable()).strip()
    if "<" in printable or ">" in printable:
        return "内容已省略"
    return printable


def _safe_optional_text(value: str | None) -> str | None:
    return _safe_text(value) if value is not None else None


def _markdown_plain_text(value: str) -> str:
    without_images = _IMAGE_LINK.sub("图片链接已省略", value)
    without_urls = _URL.sub("链接已省略", without_images)
    return without_urls.replace("\r", " ").replace("\n", " ")


def _write_output(content: str, output: Path | None) -> None:
    if output is None:
        print(content, end="" if content.endswith("\n") else "\n")
        return
    if output.is_symlink():
        raise NodeAcceptanceOutputError("输出路径不能是符号链接")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(output, flags, 0o600)
    except FileExistsError as error:
        if output.is_symlink():
            raise NodeAcceptanceOutputError("输出路径不能是符号链接") from error
        raise NodeAcceptanceOutputError("输出文件已存在，拒绝覆盖") from error
    except OSError as error:
        raise NodeAcceptanceOutputError("无法创建审核包输出文件") from error

    opened_stat = os.fstat(descriptor)
    try:
        encoded = content.encode("utf-8")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("写入未前进")
            offset += written
        os.fsync(descriptor)
    except Exception as error:
        os.close(descriptor)
        _remove_partial_output(output, opened_stat.st_dev, opened_stat.st_ino)
        raise NodeAcceptanceOutputError("审核包写入失败，已清理部分文件") from error
    os.close(descriptor)


def _remove_partial_output(output: Path, device: int, inode: int) -> None:
    try:
        current = output.lstat()
        if current.st_dev == device and current.st_ino == inode:
            output.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _read_review_payload(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise NodeAcceptanceReviewError("审核包输入不能是符号链接")
    try:
        if path.stat().st_size > MAXIMUM_REVIEW_FILE_BYTES:
            raise NodeAcceptanceReviewError("审核包文件过大")
        decoded = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        raise NodeAcceptanceReviewError("审核包不是有效的 UTF-8 文件") from error
    except OSError as error:
        raise NodeAcceptanceReviewError("无法读取本地审核包") from error
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError as error:
        raise NodeAcceptanceReviewError("审核包 JSON 格式无效") from error
    if not isinstance(payload, dict):
        raise NodeAcceptanceReviewError("人工审核包必须是 JSON 对象")
    return payload


def _create_parser() -> _NodeAcceptanceArgumentParser:
    parser = _NodeAcceptanceArgumentParser(description="本地 V2 节点质量人工验收")
    parser.add_argument(
        "--database-url",
        default=os.getenv(
            "MOONLIGHTBOX_DATABASE_URL",
            "sqlite:///data/moonlightbox.db",
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="导出本地人工审核包")
    export.add_argument("--project", required=True)
    export.add_argument("--run", required=True)
    export.add_argument("--format", choices=("json", "markdown"), default="json")
    export.add_argument("--output", type=Path)

    evaluate = subparsers.add_parser("evaluate", help="读取审核后的 JSON 并计算精确率")
    evaluate.add_argument("--project", required=True)
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PRECISION_THRESHOLD,
    )
    return parser


def _sqlite_database_path(database_url: str) -> Path:
    try:
        url = make_url(database_url)
    except SQLAlchemyError as error:
        raise NodeAcceptanceCliInputError("SQLite 数据库地址无效") from error
    if url.get_backend_name() != "sqlite":
        raise NodeAcceptanceCliInputError("验收命令只支持本地 SQLite 数据库")
    if not url.database or url.database == ":memory:":
        raise NodeAcceptanceCliInputError("必须指定已有的 SQLite 数据库文件")
    path = Path(url.database)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise NodeAcceptanceCliInputError("SQLite 数据库文件不存在") from error
    if not stat.S_ISREG(mode):
        raise NodeAcceptanceCliInputError("SQLite 数据库路径必须是普通文件")
    return path


def _create_sqlite_snapshot_engine(database_url: str) -> tuple[Engine, Path]:
    source_path = _sqlite_database_path(database_url)
    descriptor, raw_snapshot_path = tempfile.mkstemp(
        prefix="moonlightbox-node-acceptance-",
        suffix=".db",
    )
    os.close(descriptor)
    snapshot_path = Path(raw_snapshot_path)
    source: sqlite3.Connection | None = None
    destination: sqlite3.Connection | None = None
    try:
        source_uri = f"file:{quote(str(source_path), safe='/')}?mode=ro"
        source = sqlite3.connect(
            source_uri,
            uri=True,
            check_same_thread=False,
        )
        destination = sqlite3.connect(snapshot_path)
        source.backup(destination)
        destination.close()
        destination = None
        source.close()
        source = None
    except Exception as error:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()
        _remove_snapshot_files(snapshot_path)
        raise NodeAcceptanceCliInputError("无法创建 SQLite 一致性只读快照") from error

    snapshot_uri = f"file:{quote(str(snapshot_path), safe='/')}?mode=ro&immutable=1"

    def connect_snapshot() -> sqlite3.Connection:
        return sqlite3.connect(
            snapshot_uri,
            uri=True,
            check_same_thread=False,
        )

    try:
        engine = create_engine(
            "sqlite+pysqlite://",
            creator=connect_snapshot,
            poolclass=NullPool,
        )
    except Exception:
        _remove_snapshot_files(snapshot_path)
        raise
    return engine, snapshot_path


def _remove_snapshot_files(snapshot_path: Path) -> None:
    for candidate in (
        snapshot_path,
        Path(f"{snapshot_path}-journal"),
        Path(f"{snapshot_path}-wal"),
        Path(f"{snapshot_path}-shm"),
    ):
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            continue


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_parser()
    engine: Engine | None = None
    snapshot_path: Path | None = None
    try:
        args = parser.parse_args(argv)
        engine, snapshot_path = _create_sqlite_snapshot_engine(str(args.database_url))
        with Session(engine) as session:
            if args.command == "export":
                packet = build_review_packet(
                    session,
                    project_id=str(args.project),
                    run_id=str(args.run),
                )
                content = (
                    json.dumps(
                        packet.model_dump(mode="json"),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                    if args.format == "json"
                    else render_review_packet_markdown(packet)
                )
                _write_output(content, args.output)
                return 0
            payload = _read_review_payload(args.input)
            result = evaluate_review_packet(
                session,
                project_id=str(args.project),
                run_id=str(args.run),
                reviewed_packet=payload,
                threshold=float(args.threshold),
            )
            print(
                json.dumps(
                    result.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return {"pass": 0, "fail": 1, "pending": 2}[result.status]
    except (
        NodeAcceptanceCliInputError,
        NodeAcceptanceReviewError,
        NodeAcceptanceScopeError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 3
    except NodeAcceptanceOutputError as error:
        print(str(error), file=sys.stderr)
        return 3
    except SQLAlchemyError:
        print("本地数据库不可用或尚未初始化", file=sys.stderr)
        return 3
    except (OSError, ValueError):
        print("本地验收输入或路径无效", file=sys.stderr)
        return 3
    except Exception:
        print("验收工具执行失败", file=sys.stderr)
        return 3
    finally:
        if engine is not None:
            engine.dispose()
        if snapshot_path is not None:
            _remove_snapshot_files(snapshot_path)


if __name__ == "__main__":
    raise SystemExit(main())
