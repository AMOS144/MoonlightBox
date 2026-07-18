from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.evaluation.recommendation import ModelScore, recommend
from moonlightbox.training.models import ModelVersion


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
    ) -> ModelVersion:
        version = ModelVersion(
            project_id=project_id,
            base_model=base_model,
            adapter_path=adapter_path,
            dataset_hash=dataset_hash,
            metrics=metrics,
        )
        self._session.add(version)
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

    def recommend(self, project_id: str) -> ModelVersion:
        versions = self.list(project_id)
        selected = recommend(
            [
                ModelScore(
                    model_version_id=version.id,
                    blind_win_rate=version.metrics.get("blind_win_rate", 0.0),
                    style_score=version.metrics.get("style_score", 0.0),
                    safety_score=version.metrics.get("safety_score", 0.0),
                )
                for version in versions
            ]
        )
        for version in versions:
            version.recommended = version.id == selected.model_version_id
        self._session.commit()
        result = next(
            version for version in versions if version.id == selected.model_version_id
        )
        self._session.refresh(result)
        return result
