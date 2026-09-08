import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.db import Database
from moonlightbox.evaluation.node_metrics import NodeLabel, NodeMetrics, evaluate_nodes
from moonlightbox.events.models import AnalysisRevision, EventNode
from moonlightbox.events.v3_types import (
    ALL_V3_EVENT_TYPES,
    EventLane,
    EventStatus,
    is_valid_event_type,
)
from moonlightbox.imports.analysis import analyze_import
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.imports.types import MessageKind
from moonlightbox.projects.models import Project

ALLOWED_NODE_TYPES = frozenset(
    {
        "relationship_started",
        "intimacy_increased",
        "commitment",
        "boundary_change",
        "conflict",
        "distancing",
        "reconciliation",
        "separation",
        "reconnection",
    }
)


class EventGoldMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    timestamp: datetime
    sender: str
    message_type: MessageKind
    content: str

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, value: object) -> object:
        if isinstance(value, datetime):
            return value
        if not isinstance(value, str):
            raise ValueError("timestamp 必须是可解析的 ISO 8601 时间")
        try:
            datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("timestamp 必须是可解析的 ISO 8601 时间") from error
        return value

    @field_validator("message_type", mode="before")
    @classmethod
    def validate_message_type(cls, value: object) -> object:
        if value not in {kind.value for kind in MessageKind}:
            raise ValueError("message_type 不合法")
        return value


class EventGoldNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    label: str
    start_message_id: str
    end_message_id: str
    evidence_ids: tuple[str, ...]

    @field_validator("type")
    @classmethod
    def validate_node_type(cls, value: str) -> str:
        if value not in ALLOWED_NODE_TYPES:
            raise ValueError("节点 type 不合法")
        return value

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def validate_evidence_ids(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("节点 evidence_ids 不能为空")
        return value


class EventGoldCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    description: str
    messages: tuple[EventGoldMessage, ...]
    expected_nodes: tuple[EventGoldNode, ...]
    excluded_from_display: frozenset[str]

    @model_validator(mode="after")
    def validate_message_references(self) -> Self:
        timezone_awareness = {
            message.timestamp.utcoffset() is not None for message in self.messages
        }
        if len(timezone_awareness) > 1:
            raise ValueError("timestamp 不得混用带时区和不带时区")
        order_keys = [(message.timestamp, message.source_id) for message in self.messages]
        if order_keys != sorted(order_keys):
            raise ValueError("消息必须按 (timestamp, source_id) 升序排列")

        message_ids = [message.source_id for message in self.messages]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("source_id 必须唯一")
        positions = {source_id: index for index, source_id in enumerate(message_ids)}

        missing_excluded_ids = self.excluded_from_display - positions.keys()
        if missing_excluded_ids:
            raise ValueError("excluded_from_display 引用了不存在的 source_id")

        for node in self.expected_nodes:
            missing_evidence = set(node.evidence_ids) - positions.keys()
            if missing_evidence:
                raise ValueError("节点 evidence_ids 引用了不存在的 source_id")
            if node.start_message_id not in positions or node.end_message_id not in positions:
                raise ValueError("节点起止范围引用了不存在的 source_id")
            start = positions[node.start_message_id]
            end = positions[node.end_message_id]
            if start > end:
                raise ValueError("节点起止范围无效")
            if any(not start <= positions[evidence_id] <= end for evidence_id in node.evidence_ids):
                raise ValueError("节点 evidence_ids 必须位于节点起止范围内")
        return self


class EventGoldDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str
    precision_gate: float = Field(ge=0.0, le=1.0)
    cases: tuple[EventGoldCase, ...] = Field(min_length=1)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != "important-event-gold-v1":
            raise ValueError("schema_version 必须是 important-event-gold-v1")
        return value

    @field_validator("precision_gate", mode="before")
    @classmethod
    def validate_precision_gate(cls, value: object) -> object:
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise ValueError("precision_gate 必须在 0 到 1 之间")
        return value

    @model_validator(mode="after")
    def validate_case_ids(self) -> Self:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("黄金样例的 case_id 必须唯一")
        source_ids = [message.source_id for case in self.cases for message in case.messages]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("整个黄金样例中的 source_id 必须唯一")
        return self


class V3EventGoldNode(BaseModel):
    """V3 黄金节点的最小可匹配合同。"""

    model_config = ConfigDict(frozen=True)

    lane: EventLane
    type: str
    title: str
    event_status: EventStatus
    start_message_id: str
    end_message_id: str
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lane_type(self) -> Self:
        if not is_valid_event_type(self.lane, self.type):
            raise ValueError("V3 节点 type 与 lane 不匹配")
        return self


class V3EventGoldCase(BaseModel):
    """同时描述应召回节点与明确不应出现类型的 V3 案例。"""

    model_config = ConfigDict(frozen=True)

    case_id: str
    description: str
    messages: tuple[EventGoldMessage, ...] = Field(min_length=1)
    expected_nodes: tuple[V3EventGoldNode, ...]
    expected_absent_types: tuple[str, ...] = ()
    excluded_from_display: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def validate_case_contract(self) -> Self:
        timezone_awareness = {
            message.timestamp.utcoffset() is not None for message in self.messages
        }
        if len(timezone_awareness) > 1:
            raise ValueError("timestamp 不得混用带时区和不带时区")
        order_keys = [(message.timestamp, message.source_id) for message in self.messages]
        if order_keys != sorted(order_keys):
            raise ValueError("消息必须按 (timestamp, source_id) 升序排列")
        message_ids = [message.source_id for message in self.messages]
        if len(message_ids) != len(set(message_ids)):
            raise ValueError("case 内 source_id 必须唯一")
        positions = {source_id: index for index, source_id in enumerate(message_ids)}
        if self.excluded_from_display - positions.keys():
            raise ValueError("excluded_from_display 引用了不存在的 source_id")
        if any(event_type not in ALL_V3_EVENT_TYPES for event_type in self.expected_absent_types):
            raise ValueError("expected_absent_types 包含未知 V3 类型")
        for node in self.expected_nodes:
            references = {
                node.start_message_id,
                node.end_message_id,
                *node.evidence_ids,
            }
            if references - positions.keys():
                raise ValueError("V3 节点引用了不存在的 source_id")
            start = positions[node.start_message_id]
            end = positions[node.end_message_id]
            if start > end:
                raise ValueError("V3 节点起止范围无效")
            if any(not start <= positions[evidence_id] <= end for evidence_id in node.evidence_ids):
                raise ValueError("V3 节点 evidence_ids 必须位于节点范围内")
        return self


class V3EventGoldDataset(BaseModel):
    """V3 双通道黄金数据集。"""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["important-event-gold-v3"]
    precision_gate: float = Field(ge=0.0, le=1.0)
    cases: tuple[V3EventGoldCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_global_ids(self) -> Self:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("V3 黄金样例的 case_id 必须唯一")
        source_ids = [message.source_id for case in self.cases for message in case.messages]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("V3 黄金样例的 source_id 必须全局唯一")
        return self


@dataclass(frozen=True)
class HeuristicV1Baseline:
    metrics: NodeMetrics
    predicted_types_by_case: dict[str, tuple[str, ...]]
    predicted_evidence_by_case: dict[str, tuple[frozenset[str], ...]]
    analysis_versions_by_case: dict[str, tuple[str, ...]]
    exposed_noise_source_ids: frozenset[str]


def load_event_gold_dataset(path: Path) -> EventGoldDataset:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"黄金样例 JSON 无法解析：第 {error.lineno} 行") from error
    try:
        return EventGoldDataset.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"黄金样例结构无效：{error}") from error


def load_v3_event_gold_dataset(path: Path) -> V3EventGoldDataset:
    """严格加载 V3 fixture，禁止把旧版数据隐式升级。"""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"V3 黄金样例 JSON 无法解析：第 {error.lineno} 行") from error
    try:
        return V3EventGoldDataset.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"V3 黄金样例结构无效：{error}") from error


def evaluate_heuristic_v1_baseline(
    dataset: EventGoldDataset,
    maximum_events: int = 30,
) -> HeuristicV1Baseline:
    if maximum_events < 0:
        raise ValueError("maximum_events 不能小于 0")

    predicted: list[NodeLabel] = []
    gold: list[NodeLabel] = []
    predicted_types_by_case: dict[str, tuple[str, ...]] = {}
    predicted_evidence_by_case: dict[str, tuple[frozenset[str], ...]] = {}
    analysis_versions_by_case: dict[str, tuple[str, ...]] = {}
    exposed_noise_source_ids: set[str] = set()

    with TemporaryDirectory(prefix="moonlightbox-golden-") as directory:
        database = Database(f"sqlite:///{Path(directory) / 'baseline.db'}")
        try:
            database.create_schema()
            with Session(database.engine) as session:
                for case in dataset.cases:
                    project_id, import_id = _persist_case(session, case)
                    analyze_import(
                        session,
                        project_id,
                        import_id,
                        maximum_events=maximum_events,
                    )
                    events = list(
                        session.scalars(
                            select(EventNode)
                            .where(EventNode.project_id == project_id)
                            .order_by(EventNode.created_at, EventNode.id)
                        )
                    )
                    predicted_types_by_case[case.case_id] = tuple(event.type for event in events)
                    predicted_evidence_by_case[case.case_id] = tuple(
                        frozenset(event.evidence_ids) for event in events
                    )
                    analysis_versions_by_case[case.case_id] = tuple(
                        session.scalars(
                            select(AnalysisRevision.analysis_version)
                            .join(EventNode, EventNode.id == AnalysisRevision.event_id)
                            .where(EventNode.project_id == project_id)
                            .order_by(AnalysisRevision.created_at, AnalysisRevision.id)
                        )
                    )
                    for event in events:
                        evidence_ids = frozenset(event.evidence_ids)
                        predicted.append(NodeLabel(event.type, evidence_ids))
                        exposed_noise_source_ids.update(evidence_ids & case.excluded_from_display)
                    gold.extend(
                        NodeLabel(node.type, frozenset(node.evidence_ids))
                        for node in case.expected_nodes
                    )
        finally:
            database.close()

    return HeuristicV1Baseline(
        metrics=evaluate_nodes(predicted, gold),
        predicted_types_by_case=predicted_types_by_case,
        predicted_evidence_by_case=predicted_evidence_by_case,
        analysis_versions_by_case=analysis_versions_by_case,
        exposed_noise_source_ids=frozenset(exposed_noise_source_ids),
    )


def _persist_case(
    session: Session,
    case: EventGoldCase,
) -> tuple[str, str]:
    project = Project(name=f"黄金基线-{case.case_id}")
    session.add(project)
    session.flush()

    source = ImportSource(
        project_id=project.id,
        preview_id=f"golden-{case.case_id}",
        source_path=f"{case.case_id}.json",
        message_count=len(case.messages),
        confirmed_at=datetime.now(UTC),
    )
    session.add(source)
    session.flush()

    participants: dict[str, Participant] = {}
    for sender in dict.fromkeys(message.sender for message in case.messages):
        participant = Participant(
            project_id=project.id,
            name=sender,
            role="participant",
        )
        session.add(participant)
        participants[sender] = participant
    session.flush()

    session.add_all(
        Message(
            project_id=project.id,
            import_id=source.id,
            participant_id=participants[message.sender].id,
            source_id=message.source_id,
            timestamp=message.timestamp,
            kind=message.message_type.value,
            content=message.content,
            raw={},
        )
        for message in case.messages
    )
    session.commit()
    return project.id, source.id
