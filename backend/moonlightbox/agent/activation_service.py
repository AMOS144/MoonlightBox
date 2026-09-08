from sqlalchemy import select, update
from sqlalchemy.orm import Session

from moonlightbox.agent.acceptance import ReplayObservation, evaluate_acceptance
from moonlightbox.agent.models import SubjectAgentAcceptanceReport
from moonlightbox.branches.baseline_models import BranchBaselineManifest
from moonlightbox.branches.models import Branch
from moonlightbox.imports.models import ImportSource, Message
from moonlightbox.jobs.models import Job
from moonlightbox.training.models import ModelVersion

_BRANCH_FOREIGN_KEY_MODELS = (BranchBaselineManifest, ImportSource, Message, Job)


class BranchScopeError(LookupError):
    """项目范围内不存在指定分支。"""


class ModelScopeError(LookupError):
    """模型不属于指定项目或不是分支绑定模型。"""


class AcceptanceReportNotFoundError(LookupError):
    """指定范围内没有验收报告。"""


class AcceptanceRejectedError(RuntimeError):
    """最新验收报告未通过，不能激活。"""


class SubjectAgentActivationService:
    """保存影子验收证据，并控制分支级激活状态。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def evaluate_and_save(
        self,
        project_id: str,
        branch_id: str,
        model_version_id: str,
        observations: list[ReplayObservation],
        *,
        direct_lora_p95: float,
    ) -> SubjectAgentAcceptanceReport:
        self._require_branch_model(project_id, branch_id, model_version_id)
        evaluation = evaluate_acceptance(
            observations,
            direct_lora_p95=direct_lora_p95,
        )
        report = SubjectAgentAcceptanceReport(
            project_id=project_id,
            branch_id=branch_id,
            model_version_id=model_version_id,
            sample_count=evaluation.sample_count,
            structure_extraction_success_rate=evaluation.extraction_success_rate,
            fact_safety_rate=evaluation.fact_safety_rate,
            expression_decision_accuracy=evaluation.expression_decision_accuracy,
            p95_cognition_latency_ms=evaluation.agent_p95_latency_ms,
            direct_lora_p95=evaluation.direct_lora_p95_latency_ms,
            passed=evaluation.passed,
            failure_reasons=list(evaluation.failure_reasons),
            evidence={
                "observations": [
                    {
                        "source": observation.source,
                        "cutoff": observation.cutoff.isoformat(),
                        "expected_express": observation.expected_express,
                        "predicted_express": observation.predicted_express,
                        "extraction_succeeded": observation.extraction_succeeded,
                        "fact_safe": observation.fact_safe,
                        "latency_ms": observation.latency_ms,
                        "direct_latency_ms": observation.direct_latency_ms,
                        "context_continuity_score": observation.context_continuity_score,
                        "persona_style_score": observation.persona_style_score,
                        **dict(observation.evidence),
                    }
                    for observation in evaluation.observations
                ],
                "thresholds": {
                    "minimum_samples": 20,
                    "structure_extraction_success_rate": 0.95,
                    "fact_safety_rate": 1.0,
                    "expression_decision_accuracy": 0.80,
                    "context_continuity_score": 0.80,
                    "persona_style_score": 0.75,
                    "latency_multiplier": 1.20,
                },
                "quality_scores": {
                    "context_continuity_score": evaluation.context_continuity_score,
                    "persona_style_score": evaluation.persona_style_score,
                },
            },
        )
        self._session.add(report)
        self._session.commit()
        self._session.refresh(report)
        return report

    def latest_report(
        self,
        project_id: str,
        branch_id: str,
        model_version_id: str,
    ) -> SubjectAgentAcceptanceReport:
        self._require_branch_model(project_id, branch_id, model_version_id)
        report = self._latest_report(project_id, branch_id, model_version_id)
        if report is None:
            raise AcceptanceReportNotFoundError("当前分支和模型没有验收报告")
        return report

    def latest_bound_report(
        self,
        project_id: str,
        branch_id: str,
    ) -> SubjectAgentAcceptanceReport:
        branch = self._require_branch(project_id, branch_id)
        return self.latest_report(project_id, branch_id, branch.model_version_id)

    def activate_branch(
        self,
        project_id: str,
        branch_id: str,
        model_version_id: str,
    ) -> Branch:
        branch = self._require_branch_model(project_id, branch_id, model_version_id)
        if branch.lifecycle_status != "active" or branch.baseline_status != "ready":
            raise AcceptanceRejectedError("分支尚未处于可激活状态")
        active_model = self._session.scalar(
            select(ModelVersion.id).where(
                ModelVersion.project_id == project_id,
                ModelVersion.id == model_version_id,
                ModelVersion.active.is_(True),
            )
        )
        if active_model is None:
            raise ModelScopeError("分支绑定模型不是项目当前活动模型，请先升级分支")
        model = self._session.get(ModelVersion, model_version_id)
        human_report = (
            model.training_config.get("human_blind_report")
            if model is not None
            else None
        )
        if (
            model is not None
            and bool(model.training_config.get("human_blind_required"))
            and (not isinstance(human_report, dict) or human_report.get("passed") is not True)
        ):
            raise AcceptanceRejectedError("候选模型尚未通过真实用户盲测")
        report = self._latest_report(project_id, branch_id, model_version_id)
        if report is None or not report.passed:
            raise AcceptanceRejectedError("最新主体认知 Agent 验收报告未通过")
        self._session.execute(
            update(Branch)
            .where(
                Branch.project_id == project_id,
                Branch.id != branch_id,
                Branch.subject_agent_mode == "active",
            )
            .values(subject_agent_mode="shadow")
        )
        branch.subject_agent_mode = "active"
        self._session.commit()
        self._session.refresh(branch)
        return branch

    def activate_latest_passed_branch_for_model(
        self,
        project_id: str,
        model_version_id: str,
    ) -> Branch | None:
        """Activate the newest usable branch whose latest replay report passed."""

        branch = self.latest_passed_branch_for_model(project_id, model_version_id)
        if branch is None:
            return None
        return self.activate_branch(project_id, branch.id, model_version_id)

    def latest_passed_branch_for_model(
        self,
        project_id: str,
        model_version_id: str,
    ) -> Branch | None:
        """Return the newest replay-qualified branch without changing active state."""

        branches = self._session.scalars(
            select(Branch)
            .where(
                Branch.project_id == project_id,
                Branch.model_version_id == model_version_id,
                Branch.lifecycle_status == "active",
                Branch.baseline_status == "ready",
            )
            .order_by(Branch.created_at.desc(), Branch.id.desc())
        )
        for branch in branches:
            report = self._latest_report(project_id, branch.id, model_version_id)
            if report is not None and report.passed:
                return branch
        return None

    def rollback_branch(self, project_id: str, branch_id: str) -> Branch:
        branch = self._require_branch(project_id, branch_id)
        branch.subject_agent_mode = "shadow"
        self._session.commit()
        self._session.refresh(branch)
        return branch

    def _require_branch(self, project_id: str, branch_id: str) -> Branch:
        branch = self._session.scalar(
            select(Branch).where(
                Branch.project_id == project_id,
                Branch.id == branch_id,
            )
        )
        if branch is None:
            raise BranchScopeError("时间分支不存在")
        return branch

    def _require_branch_model(
        self,
        project_id: str,
        branch_id: str,
        model_version_id: str,
    ) -> Branch:
        branch = self._require_branch(project_id, branch_id)
        model = self._session.scalar(
            select(ModelVersion).where(
                ModelVersion.project_id == project_id,
                ModelVersion.id == model_version_id,
            )
        )
        if model is None or branch.model_version_id != model_version_id:
            raise ModelScopeError("模型不属于当前项目或不是分支绑定模型")
        return branch

    def _latest_report(
        self,
        project_id: str,
        branch_id: str,
        model_version_id: str,
    ) -> SubjectAgentAcceptanceReport | None:
        return self._session.scalar(
            select(SubjectAgentAcceptanceReport)
            .where(
                SubjectAgentAcceptanceReport.project_id == project_id,
                SubjectAgentAcceptanceReport.branch_id == branch_id,
                SubjectAgentAcceptanceReport.model_version_id == model_version_id,
            )
            .order_by(
                SubjectAgentAcceptanceReport.created_at.desc(),
                SubjectAgentAcceptanceReport.id.desc(),
            )
            .limit(1)
        )


def reconcile_subject_agent_modes(
    session: Session,
    *,
    project_id: str | None = None,
    commit: bool = True,
) -> int:
    """Demote stale active agents that are no longer bound to an active model."""

    active_models = select(ModelVersion.id).where(ModelVersion.active.is_(True))
    query = update(Branch).where(
        Branch.subject_agent_mode == "active",
        (
            (Branch.lifecycle_status != "active")
            | (~Branch.model_version_id.in_(active_models))
        ),
    )
    if project_id is not None:
        query = query.where(Branch.project_id == project_id)
    result = session.execute(query.values(subject_agent_mode="shadow"))
    count = int(getattr(result, "rowcount", 0) or 0)
    if commit:
        session.commit()
    return count
