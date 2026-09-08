import hashlib
import json
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from moonlightbox.branches.identity import (
    EvidenceBackedIdentityKernelBuilder,
    IdentityKernelService,
)
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.imports.types import ImportedMessage, MessageKind
from moonlightbox.personas.models import IdentityKernel
from moonlightbox.training.dataset_builder import DatasetBuilder
from moonlightbox.training.model_acceptance import AcceptanceReport
from moonlightbox.training.models import ModelVersion


class AcceptanceRunner(Protocol):
    def run(
        self,
        *,
        base_model: str,
        adapter_path: str,
        persona: str,
        cutoff: str,
        style_profile: dict[str, object] | None = None,
    ) -> AcceptanceReport: ...


class SubjectPersonaModelUpgrader:
    """通过 V2 验收后克隆旧模型版本，并生成证据化人格内核。"""

    def __init__(self, session: Session, acceptance_runner: AcceptanceRunner) -> None:
        self._session = session
        self._acceptance_runner = acceptance_runner

    def upgrade(self, project_id: str) -> ModelVersion:
        active = self._session.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.project_id == project_id,
                ModelVersion.active.is_(True),
            )
            .order_by(ModelVersion.created_at.desc())
        )
        if active is None:
            raise LookupError("项目没有活动模型")
        existing_kernel = self._session.scalar(
            select(IdentityKernel).where(IdentityKernel.model_version_id == active.id)
        )
        if (
            existing_kernel is not None
            and existing_kernel.schema_version == "subject-persona-v2"
            and existing_kernel.acceptance_report_id is not None
        ):
            return active
        source = self._session.scalar(
            select(ImportSource)
            .where(ImportSource.project_id == project_id)
            .order_by(ImportSource.confirmed_at.desc())
        )
        target_name = self._session.scalar(
            select(Participant.name).where(
                Participant.project_id == project_id,
                Participant.role == "target",
            )
        )
        if source is None or target_name is None:
            raise ValueError("缺少可追溯的训练来源或目标人物")
        rows = list(
            self._session.execute(
                select(Message, Participant.name)
                .join(Participant, Participant.id == Message.participant_id)
                .where(
                    Message.project_id == project_id,
                    Message.import_id == source.id,
                    Participant.role.in_(("self", "target")),
                )
                .order_by(Message.timestamp, Message.source_id)
            )
        )
        if not rows:
            raise ValueError("没有可用于主体人格升级的聊天记录")
        cutoff = max(message.timestamp for message, _sender in rows)
        imported = [
            ImportedMessage(
                source_id=message.source_id,
                timestamp=message.timestamp,
                sender=sender,
                kind=_message_kind(message.kind),
                content=message.content,
                raw={},
            )
            for message, sender in rows
        ]
        examples = DatasetBuilder().build(
            imported,
            target_sender=target_name,
            cutoff=cutoff,
        )
        if not examples:
            raise ValueError("聊天记录不足以生成主体人格训练样本")
        proposal = EvidenceBackedIdentityKernelBuilder().build(
            persona=target_name,
            examples=examples,
            event_contexts=[],
        )
        report = self._acceptance_runner.run(
            base_model=active.base_model,
            adapter_path=active.adapter_path,
            persona=target_name,
            cutoff=cutoff.isoformat(),
            style_profile=proposal.style_profile,
        )
        if not report.passed:
            raise ValueError("现有活动模型未通过主体人格 V2 验收")
        report_id = _report_id(report)
        upgraded = ModelVersion(
            project_id=project_id,
            base_model=active.base_model,
            adapter_path=active.adapter_path,
            dataset_hash=active.dataset_hash,
            status="ready",
            metrics={
                **active.metrics,
                "subject_v2_acceptance_pass_rate": (
                    report.passed_count / report.case_count if report.case_count else 0.0
                ),
            },
            recommended=True,
            active=False,
            timeline_confirmation_id=active.timeline_confirmation_id,
            training_config={
                **active.training_config,
                "subject_protocol_version": "subject-persona-v2",
                "upgraded_from_model_version_id": active.id,
            },
        )
        self._session.add(upgraded)
        self._session.flush()
        IdentityKernelService(self._session).create_and_lock(
            project_id=project_id,
            model_version_id=upgraded.id,
            proposal=proposal,
            evidence_message_ids=list(
                dict.fromkeys(source_id for example in examples for source_id in example.source_ids)
            ),
            acceptance_report_id=report_id,
            commit=False,
        )
        self._session.execute(
            update(ModelVersion)
            .where(ModelVersion.project_id == project_id)
            .values(active=False, recommended=False)
        )
        upgraded.active = True
        upgraded.recommended = True
        self._session.commit()
        self._session.refresh(upgraded)
        return upgraded


def _report_id(report: AcceptanceReport) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "case_count": report.case_count,
                "passed_count": report.passed_count,
                "structure_failures": report.structure_failures,
                "forbidden_fact_failures": report.forbidden_fact_failures,
                "failed_case_ids": report.failed_case_ids,
                "passed": report.passed,
                "raw_output_failures": report.raw_output_failures,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _message_kind(value: str) -> MessageKind:
    try:
        return MessageKind(value)
    except ValueError:
        # 媒体关联阶段可能产生更细的展示类型，训练时按未知非文本消息处理。
        return MessageKind.UNKNOWN
