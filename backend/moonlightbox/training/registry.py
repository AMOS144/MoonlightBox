import hashlib
import json
from typing import TYPE_CHECKING

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from moonlightbox.branches.identity import IdentityKernelService
from moonlightbox.evaluation.recommendation import ModelScore, recommend
from moonlightbox.training.models import ModelVersion

if TYPE_CHECKING:
    from moonlightbox.branches.identity import IdentityKernelProposal


class ConcurrentActivationError(RuntimeError):
    pass


class HumanBlindAcceptanceRequiredError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        project_id: str,
        base_model: str,
        adapter_path: str,
        dataset_hash: str,
        metrics: dict[str, float],
        *,
        timeline_confirmation_id: str | None = None,
        training_job_id: str | None = None,
        training_config: dict[str, object] | None = None,
        commit: bool = True,
    ) -> ModelVersion:
        version = ModelVersion(
            project_id=project_id,
            base_model=base_model,
            adapter_path=adapter_path,
            dataset_hash=dataset_hash,
            metrics=metrics,
            timeline_confirmation_id=timeline_confirmation_id,
            training_job_id=training_job_id,
            training_config=training_config or {},
        )
        self._session.add(version)
        if commit:
            self._session.commit()
            self._session.refresh(version)
        else:
            self._session.flush()
        return version

    def publish_and_activate(
        self,
        *,
        project_id: str,
        base_model: str,
        adapter_path: str,
        dataset_hash: str,
        metrics: dict[str, float],
        training_config: dict[str, object],
        kernel_proposal: "IdentityKernelProposal",
        evidence_message_ids: list[str],
        acceptance_report: dict[str, object],
        sticker_policy: dict[str, object],
        expected_active_model_id: str | None,
        timeline_confirmation_id: str | None = None,
        training_job_id: str | None = None,
        commit: bool = False,
    ) -> ModelVersion:
        """在一个事务保存点中发布全部产物，并以当前活动模型执行 CAS。"""

        current = self._session.scalar(
            select(ModelVersion)
            .where(
                ModelVersion.project_id == project_id,
                ModelVersion.active.is_(True),
            )
            .order_by(ModelVersion.created_at.desc())
            .with_for_update()
        )
        current_id = current.id if current is not None else None
        if current_id != expected_active_model_id:
            raise ConcurrentActivationError("活动模型已变化，拒绝覆盖并发激活")

        with self._session.begin_nested():
            human_blind_required = bool(training_config.get("human_blind_required"))
            report_id = hashlib.sha256(
                json.dumps(
                    acceptance_report,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            version = self.create(
                project_id=project_id,
                base_model=base_model,
                adapter_path=adapter_path,
                dataset_hash=dataset_hash,
                metrics=metrics,
                timeline_confirmation_id=timeline_confirmation_id,
                training_job_id=training_job_id,
                training_config={
                    **training_config,
                    "upgraded_from_model_version_id": current_id,
                    "acceptance_report": acceptance_report,
                    "acceptance_report_id": report_id,
                    "sticker_policy": sticker_policy,
                },
                commit=False,
            )
            IdentityKernelService(self._session).create_and_lock(
                project_id=project_id,
                model_version_id=version.id,
                proposal=kernel_proposal,
                evidence_message_ids=evidence_message_ids,
                acceptance_report_id=report_id,
                commit=False,
            )
            if human_blind_required:
                version.status = "awaiting_human_review"
                version.active = False
                version.recommended = False
            else:
                active_filter = [
                    ModelVersion.project_id == project_id,
                    ModelVersion.active.is_(True),
                ]
                if expected_active_model_id is not None:
                    active_filter.append(ModelVersion.id == expected_active_model_id)
                deactivated = self._session.execute(
                    update(ModelVersion).where(*active_filter).values(active=False)
                )
                expected_rows = 1 if expected_active_model_id is not None else 0
                if getattr(deactivated, "rowcount", None) != expected_rows:
                    raise ConcurrentActivationError("活动模型 CAS 失败，旧模型保持启用")
                version.active = True
            self._session.flush()

        if commit:
            self._session.commit()
            self._session.refresh(version)
        return version

    def list(self, project_id: str) -> list[ModelVersion]:
        return list(
            self._session.scalars(
                select(ModelVersion)
                .where(ModelVersion.project_id == project_id)
                .order_by(ModelVersion.created_at.desc())
            )
        )

    def activate(
        self,
        project_id: str,
        model_version_id: str,
        *,
        commit: bool = True,
    ) -> ModelVersion:
        selected = self._session.scalar(
            select(ModelVersion).where(
                ModelVersion.id == model_version_id,
                ModelVersion.project_id == project_id,
            )
        )
        if selected is None:
            raise LookupError("模型版本不存在")
        if _human_blind_pending(selected):
            raise HumanBlindAcceptanceRequiredError(
                "候选模型必须先完成并通过真实用户盲测"
            )
        self._session.execute(
            update(ModelVersion).where(ModelVersion.project_id == project_id).values(active=False)
        )
        selected.active = True
        if commit:
            self._session.commit()
            self._session.refresh(selected)
        return selected

    def recommend(self, project_id: str) -> ModelVersion:
        versions = self.list(project_id)
        eligible = [
            version
            for version in versions
            if version.status == "ready" and not _human_blind_pending(version)
        ]
        selected = recommend(
            [
                ModelScore(
                    model_version_id=version.id,
                    blind_win_rate=version.metrics.get("blind_win_rate", 0.0),
                    style_score=version.metrics.get("style_score", 0.0),
                    safety_score=version.metrics.get("safety_score", 0.0),
                )
                for version in eligible
            ]
        )
        for version in versions:
            version.recommended = version.id == selected.model_version_id
        self._session.commit()
        result = next(version for version in versions if version.id == selected.model_version_id)
        self._session.refresh(result)
        return result


def _human_blind_pending(version: ModelVersion) -> bool:
    if not bool(version.training_config.get("human_blind_required")):
        return False
    report = version.training_config.get("human_blind_report")
    return not isinstance(report, dict) or report.get("passed") is not True
