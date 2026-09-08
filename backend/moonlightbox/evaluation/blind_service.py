import hashlib
import random
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from moonlightbox.agent.activation_service import SubjectAgentActivationService
from moonlightbox.evaluation.models import HumanBlindCase, HumanBlindRating, HumanBlindStudy
from moonlightbox.training.models import ModelVersion


class BlindStudyScopeError(LookupError):
    pass


class BlindStudyStateError(RuntimeError):
    pass


class HumanBlindStudyService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        project_id: str,
        model_version_id: str,
        cases: list[dict[str, object]],
        minimum_ratings: int = 20,
        minimum_preference: float = 0.5,
        seed: int = 0,
        commit: bool = True,
    ) -> HumanBlindStudy:
        model = self._session.scalar(
            select(ModelVersion).where(
                ModelVersion.id == model_version_id,
                ModelVersion.project_id == project_id,
            )
        )
        if model is None:
            raise BlindStudyScopeError("候选模型不属于当前项目")
        if len(cases) < minimum_ratings:
            raise ValueError("盲测案例数不能少于最低有效评审数")
        now = datetime.now(UTC)
        for stale in self._session.scalars(
            select(HumanBlindStudy).where(
                HumanBlindStudy.project_id == project_id,
                HumanBlindStudy.status == "open",
            )
        ):
            stale.status = "superseded"
            stale.completed_at = now
            stale.report = {
                "source": "superseded",
                "replacement_model_version_id": model_version_id,
                "passed": False,
            }
        study = HumanBlindStudy(
            project_id=project_id,
            model_version_id=model_version_id,
            minimum_ratings=minimum_ratings,
            minimum_preference=minimum_preference,
        )
        self._session.add(study)
        self._session.flush()
        rng = random.Random(f"{study.id}:{seed}")
        rows: list[HumanBlindCase] = []
        for position, payload in enumerate(cases):
            context = payload.get("context")
            human_reply = payload.get("human_reply")
            candidate_reply = payload.get("candidate_reply")
            if (
                not isinstance(context, list)
                or any(not isinstance(item, str) for item in context)
                or not isinstance(human_reply, str)
                or not human_reply.strip()
                or not isinstance(candidate_reply, str)
                or not candidate_reply.strip()
            ):
                raise ValueError("盲测案例必须包含上下文、真人回复和候选回复")
            candidate_option = "a" if rng.getrandbits(1) else "b"
            left = candidate_reply if candidate_option == "a" else human_reply
            right = human_reply if candidate_option == "a" else candidate_reply
            rows.append(
                HumanBlindCase(
                    study_id=study.id,
                    position=position,
                    context=context,
                    human_reply=human_reply,
                    candidate_reply=candidate_reply,
                    candidate_option=candidate_option,
                    order_hash=hashlib.sha256(f"{left}\0{right}".encode()).hexdigest(),
                )
            )
        self._session.add_all(rows)
        if commit:
            self._session.commit()
            self._session.refresh(study)
        else:
            self._session.flush()
        return study

    def public_cases(self, study_id: str) -> list[dict[str, object]]:
        study = self._study(study_id)
        if study.status != "open":
            raise BlindStudyStateError("盲测已经结束")
        return [self._public_case(case) for case in self._cases(study.id)]

    def list_for_project(self, project_id: str) -> list[HumanBlindStudy]:
        return list(
            self._session.scalars(
                select(HumanBlindStudy)
                .where(HumanBlindStudy.project_id == project_id)
                .order_by(HumanBlindStudy.created_at.desc())
            )
        )

    def rate(
        self,
        *,
        study_id: str,
        case_id: str,
        rater_key: str,
        choice: str,
    ) -> HumanBlindRating:
        study = self._study(study_id)
        if study.status != "open":
            raise BlindStudyStateError("盲测已经结束")
        case = self._session.scalar(
            select(HumanBlindCase).where(
                HumanBlindCase.id == case_id,
                HumanBlindCase.study_id == study.id,
            )
        )
        if case is None:
            raise BlindStudyScopeError("盲测案例不存在")
        existing = self._session.scalar(
            select(HumanBlindRating).where(
                HumanBlindRating.case_id == case.id,
                HumanBlindRating.rater_key == rater_key,
            )
        )
        if existing is not None:
            return existing
        rating = HumanBlindRating(
            study_id=study.id,
            case_id=case.id,
            rater_key=rater_key,
            choice=choice,
        )
        self._session.add(rating)
        self._session.commit()
        self._session.refresh(rating)
        return rating

    def finalize(self, *, project_id: str, model_version_id: str, study_id: str) -> HumanBlindStudy:
        study = self._session.scalar(
            select(HumanBlindStudy).where(
                HumanBlindStudy.id == study_id,
                HumanBlindStudy.project_id == project_id,
                HumanBlindStudy.model_version_id == model_version_id,
            )
        )
        if study is None:
            raise BlindStudyScopeError("盲测不属于当前项目和模型")
        if study.status != "open":
            return study
        model = self._session.get(ModelVersion, model_version_id)
        if model is None:
            raise BlindStudyScopeError("候选模型不存在")
        was_active = model.active
        expected_active_id = (
            model.id if was_active else model.training_config.get("upgraded_from_model_version_id")
        )
        current_active_id = self._session.scalar(
            select(ModelVersion.id)
            .where(
                ModelVersion.project_id == project_id,
                ModelVersion.active.is_(True),
            )
            .order_by(ModelVersion.created_at.desc())
        )
        if current_active_id != expected_active_id:
            raise BlindStudyStateError("活动模型已变化，旧候选盲测不能覆盖新版本")
        cases = {case.id: case for case in self._cases(study.id)}
        ratings = list(
            self._session.scalars(
                select(HumanBlindRating).where(HumanBlindRating.study_id == study.id)
            )
        )
        candidate_wins = 0
        ties = 0
        for rating in ratings:
            case = cases.get(rating.case_id)
            if case is None:
                continue
            ties += int(rating.choice == "tie")
            candidate_wins += int(rating.choice == case.candidate_option)
        valid = len(ratings)
        preference = (candidate_wins + 0.5 * ties) / valid if valid else 0.0
        passed = valid >= study.minimum_ratings and preference >= study.minimum_preference
        activation_service = SubjectAgentActivationService(self._session)
        replay_branch = None
        if passed:
            replay_branch = activation_service.latest_passed_branch_for_model(
                project_id,
                model_version_id,
            )
            if replay_branch is None:
                raise BlindStudyStateError("候选模型尚未通过历史回放，不能完成真人盲测")
        study.valid_rating_count = valid
        study.candidate_preference_rate = preference
        study.status = "passed" if passed else "failed"
        study.completed_at = datetime.now(UTC)
        report: dict[str, object] = {
            "source": "human_blind",
            "case_count": len(cases),
            "valid_rating_count": valid,
            "candidate_wins": candidate_wins,
            "ties": ties,
            "candidate_preference_rate": preference,
            "minimum_ratings": study.minimum_ratings,
            "minimum_preference": study.minimum_preference,
            "passed": passed,
            "case_order_hashes": [case.order_hash for case in cases.values()],
        }
        study.report = report
        model.metrics = {**model.metrics, "human_blind_preference_rate": preference}
        model.training_config = {**model.training_config, "human_blind_report": report}
        if passed:
            self._session.execute(
                update(ModelVersion)
                .where(ModelVersion.project_id == project_id)
                .values(active=False, recommended=False)
            )
            model.status = "ready"
            model.active = True
            model.recommended = True
            self._session.flush()
            activated = activation_service.activate_latest_passed_branch_for_model(
                project_id,
                model_version_id,
            )
            if activated is not None:
                report = {**report, "activated_branch_id": activated.id}
                study.report = report
                model.training_config = {
                    **model.training_config,
                    "human_blind_report": report,
                }
        else:
            model.status = "human_review_failed_active" if was_active else "human_review_failed"
            model.active = was_active
            model.recommended = False
        self._session.commit()
        self._session.refresh(study)
        return study

    def _study(self, study_id: str) -> HumanBlindStudy:
        study = self._session.get(HumanBlindStudy, study_id)
        if study is None:
            raise BlindStudyScopeError("盲测不存在")
        return study

    def _cases(self, study_id: str) -> list[HumanBlindCase]:
        return list(
            self._session.scalars(
                select(HumanBlindCase)
                .where(HumanBlindCase.study_id == study_id)
                .order_by(HumanBlindCase.position)
            )
        )

    @staticmethod
    def _public_case(case: HumanBlindCase) -> dict[str, object]:
        return {
            "id": case.id,
            "context": case.context,
            "options": {
                case.candidate_option: case.candidate_reply,
                ("b" if case.candidate_option == "a" else "a"): case.human_reply,
            },
        }
