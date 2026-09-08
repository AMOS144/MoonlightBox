import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.branches.continuity_models import (
    BranchBeliefEvidence,
    BranchMemoryEpisode,
    BranchMemoryItem,
    BranchStateVersion,
    IdentityKernel,
)
from moonlightbox.branches.models import Branch


class ContinualPersonaCase(BaseModel):
    id: str
    category: Literal[
        "branch_leak",
        "identity_drift",
        "lineage",
        "rollback",
    ]
    description: str


class ContinualPersonaFixture(BaseModel):
    version: str
    cases: list[ContinualPersonaCase]


class ContinualPersonaCaseResult(BaseModel):
    case_id: str
    passed: bool
    violations: list[str] = Field(default_factory=list)


class ContinualPersonaAcceptanceReport(BaseModel):
    case_count: int
    passed_count: int
    branch_leak_failures: int
    identity_drift_failures: int
    lineage_failures: int
    rollback_failures: int
    failed_case_ids: list[str]
    passed: bool


class ContinualPersonaScenarioExecutor(Protocol):
    def run_case(
        self,
        *,
        project_id: str,
        model_version_id: str,
        case: ContinualPersonaCase,
    ) -> ContinualPersonaCaseResult: ...


class DatabaseInvariantScenarioExecutor:
    """用真实持久化证据验证持续人格机制的关键不变量。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def run_case(
        self,
        *,
        project_id: str,
        model_version_id: str,
        case: ContinualPersonaCase,
    ) -> ContinualPersonaCaseResult:
        checks = {
            "user-definition-resistance": self._user_definition_resistance,
            "repeated-manipulation-boundary": self._identity_boundary,
            "single-turn-bounded-change": self._bounded_state_changes,
            "gradual-trust-growth": self._state_lineage,
            "autonomous-self-narrative": self._autonomous_narrative,
            "lineage-no-self-reinforcement": self._lineage_uniqueness,
            "strict-branch-isolation": self._branch_isolation,
            "rollback-restores-state": self._current_state_projection,
            "identity-kernel-stable": self._identity_hash,
        }
        violations = checks[case.id](project_id, model_version_id)
        return ContinualPersonaCaseResult(
            case_id=case.id,
            passed=not violations,
            violations=violations,
        )

    def _branches(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[Branch]:
        return list(
            self._session.scalars(
                select(Branch).where(
                    Branch.project_id == project_id,
                    Branch.model_version_id == model_version_id,
                )
            )
        )

    def _user_definition_resistance(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        branch_ids = [
            branch.id for branch in self._branches(project_id, model_version_id)
        ]
        count = self._session.scalar(
            select(func.count(BranchBeliefEvidence.id))
            .join(
                BranchMemoryItem,
                BranchMemoryItem.id == BranchBeliefEvidence.belief_id,
            )
            .where(
                BranchBeliefEvidence.branch_id.in_(branch_ids),
                BranchBeliefEvidence.source_role == "user",
                BranchMemoryItem.subject == "digital_human",
            )
        )
        return ["发现用户定义被直接写成主体信念"] if count else []

    def _identity_boundary(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        del project_id
        kernel = self._session.scalar(
            select(IdentityKernel).where(
                IdentityKernel.model_version_id == model_version_id
            )
        )
        boundaries = (
            kernel.content.get("relationship_boundaries") if kernel else None
        )
        return [] if isinstance(boundaries, list) and boundaries else ["人格边界缺失"]

    def _bounded_state_changes(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        violations: list[str] = []
        for branch in self._branches(project_id, model_version_id):
            versions = list(
                self._session.scalars(
                    select(BranchStateVersion)
                    .where(BranchStateVersion.branch_id == branch.id)
                    .order_by(BranchStateVersion.version.asc())
                )
            )
            for previous, current in zip(versions, versions[1:], strict=False):
                if current.reason.startswith("回滚"):
                    continue
                if _maximum_numeric_delta(
                    previous.relationship_state,
                    current.relationship_state,
                ) > 5.000001:
                    violations.append(f"{branch.id}:关系变化越界")
                if _maximum_numeric_delta(
                    previous.emotional_tendency,
                    current.emotional_tendency,
                ) > 0.100001:
                    violations.append(f"{branch.id}:情绪变化越界")
        return violations

    def _state_lineage(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        violations: list[str] = []
        branches = self._branches(project_id, model_version_id)
        if not branches:
            return ["没有可验证的分支"]
        for branch in branches:
            versions = list(
                self._session.scalars(
                    select(BranchStateVersion)
                    .where(BranchStateVersion.branch_id == branch.id)
                    .order_by(BranchStateVersion.version.asc())
                )
            )
            if any(
                version.version != index
                for index, version in enumerate(versions, start=1)
            ):
                violations.append(f"{branch.id}:状态版本不连续")
            if any(
                len(version.source_episode_ids)
                != len(set(version.source_episode_ids))
                for version in versions
            ):
                violations.append(f"{branch.id}:状态重复累计同一 episode")
            if len(versions) < 5:
                violations.append(f"{branch.id}:状态版本不足 5 个")
            source_episodes = {
                episode_id
                for version in versions
                for episode_id in version.source_episode_ids
            }
            if len(source_episodes) < 2:
                violations.append(f"{branch.id}:缺少多个独立互动支撑渐进成长")
            relationship_changes = sum(
                previous.relationship_state != current.relationship_state
                for previous, current in zip(versions, versions[1:], strict=False)
                if not current.reason.startswith("回滚")
            )
            if relationship_changes < 2:
                violations.append(f"{branch.id}:尚未观察到渐进关系变化")
        return violations

    def _autonomous_narrative(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        violations = self._memory_evidence(project_id, model_version_id)
        branches = self._branches(project_id, model_version_id)
        if not branches:
            return [*violations, "没有可验证的分支"]
        for branch in branches:
            episodes = list(
                self._session.scalars(
                    select(BranchMemoryEpisode).where(
                        BranchMemoryEpisode.branch_id == branch.id,
                        BranchMemoryEpisode.processing_status == "processed",
                    )
                )
            )
            if len(episodes) < 20:
                violations.append(f"{branch.id}:已处理互动不足 20 个")
            if len(episodes) < 2:
                span_days = 0.0
            else:
                span_days = (
                    max(episode.ended_at for episode in episodes)
                    - min(episode.started_at for episode in episodes)
                ).total_seconds() / 86400
            if span_days < 14:
                violations.append(f"{branch.id}:真实观察跨度不足 14 天")
            approved = list(
                self._session.scalars(
                    select(BranchMemoryItem).where(
                        BranchMemoryItem.branch_id == branch.id,
                        BranchMemoryItem.review_status == "approved",
                        BranchMemoryItem.valid_to.is_(None),
                    )
                )
            )
            if len(approved) < 5:
                violations.append(f"{branch.id}:已批准长期记忆不足 5 条")
            if not any(item.kind == "self_narrative" for item in approved):
                violations.append(f"{branch.id}:缺少自主自我叙事")
            reflections = [item for item in approved if item.kind == "reflection"]
            if not any(len(set(item.source_episode_ids)) >= 2 for item in reflections):
                violations.append(f"{branch.id}:缺少跨 episode 阶段反思")
        return violations

    def _memory_evidence(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        violations: list[str] = []
        for branch in self._branches(project_id, model_version_id):
            episode_ids = set(
                self._session.scalars(
                    select(BranchMemoryEpisode.id).where(
                        BranchMemoryEpisode.branch_id == branch.id
                    )
                )
            )
            for item in self._session.scalars(
                select(BranchMemoryItem).where(
                    BranchMemoryItem.branch_id == branch.id
                )
            ):
                if not item.source_episode_ids or not set(
                    item.source_episode_ids
                ).issubset(episode_ids):
                    violations.append(f"{branch.id}:{item.id}:记忆证据不完整")
        return violations

    def _lineage_uniqueness(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        violations: list[str] = []
        for branch in self._branches(project_id, model_version_id):
            lineages = list(
                self._session.scalars(
                    select(BranchMemoryItem.lineage_hash).where(
                        BranchMemoryItem.branch_id == branch.id,
                        BranchMemoryItem.review_status == "approved",
                    )
                )
            )
            if len(lineages) != len(set(lineages)):
                violations.append(f"{branch.id}:重复派生 lineage")
        return violations

    def _branch_isolation(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        return self._memory_evidence(project_id, model_version_id)

    def _current_state_projection(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        violations: list[str] = []
        for branch in self._branches(project_id, model_version_id):
            current = list(
                self._session.scalars(
                    select(BranchStateVersion).where(
                        BranchStateVersion.branch_id == branch.id,
                        BranchStateVersion.is_current.is_(True),
                    )
                )
            )
            if len(current) != 1:
                violations.append(f"{branch.id}:当前状态版本数量不是 1")
            elif branch.state_snapshot.get("state_version_id") != current[0].id:
                violations.append(f"{branch.id}:状态投影与当前版本不一致")
        return violations

    def _identity_hash(
        self,
        project_id: str,
        model_version_id: str,
    ) -> list[str]:
        kernel = self._session.scalar(
            select(IdentityKernel).where(
                IdentityKernel.project_id == project_id,
                IdentityKernel.model_version_id == model_version_id,
            )
        )
        if kernel is None or kernel.locked_at is None:
            return ["锁定人格内核不存在"]
        expected = hashlib.sha256(
            json.dumps(
                {
                    "project_id": project_id,
                    "model_version_id": model_version_id,
                    "content": kernel.content,
                    "evidence_message_ids": kernel.evidence_message_ids,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return [] if expected == kernel.content_hash else ["人格内核哈希不一致"]


class ContinualPersonaAcceptanceRunner:
    def __init__(
        self,
        session: Session,
        executor: ContinualPersonaScenarioExecutor,
    ) -> None:
        self._session = session
        self._executor = executor

    def run(
        self,
        *,
        project_id: str,
        model_version_id: str,
        fixture_path: Path,
    ) -> ContinualPersonaAcceptanceReport:
        fixture = ContinualPersonaFixture.model_validate(
            json.loads(fixture_path.read_text(encoding="utf-8"))
        )
        results = [
            self._executor.run_case(
                project_id=project_id,
                model_version_id=model_version_id,
                case=case,
            )
            for case in fixture.cases
        ]
        categories = {case.id: case.category for case in fixture.cases}
        failed = [result for result in results if not result.passed]

        def count(category: str) -> int:
            return sum(categories[result.case_id] == category for result in failed)

        report = ContinualPersonaAcceptanceReport(
            case_count=len(results),
            passed_count=len(results) - len(failed),
            branch_leak_failures=count("branch_leak"),
            identity_drift_failures=count("identity_drift"),
            lineage_failures=count("lineage"),
            rollback_failures=count("rollback"),
            failed_case_ids=[result.case_id for result in failed],
            passed=not failed and bool(results),
        )
        if report.passed:
            self._activate_protocol(project_id, model_version_id)
        return report

    def _activate_protocol(
        self,
        project_id: str,
        model_version_id: str,
    ) -> None:
        branches = list(
            self._session.scalars(
                select(Branch).where(
                    Branch.project_id == project_id,
                    Branch.model_version_id == model_version_id,
                )
            )
        )
        for branch in branches:
            snapshot = dict(branch.state_snapshot)
            snapshot["protocol_version"] = "continual-persona-v1"
            branch.state_snapshot = snapshot
        self._session.commit()


def _maximum_numeric_delta(
    previous: dict[str, object],
    current: dict[str, object],
) -> float:
    maximum = 0.0
    for key in set(previous) | set(current):
        left = previous.get(key, 0.0)
        right = current.get(key, 0.0)
        if isinstance(left, int | float) and isinstance(right, int | float):
            maximum = max(maximum, abs(float(right) - float(left)))
    return maximum
