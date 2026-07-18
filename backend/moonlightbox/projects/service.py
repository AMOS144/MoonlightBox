import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.projects.models import Project


class ProjectNotFoundError(LookupError):
    pass


class ProjectService:
    def __init__(
        self,
        session: Session,
        artifact_roots: tuple[Path, ...] = (),
    ) -> None:
        self._session = session
        self._artifact_roots = artifact_roots

    def create(self, name: str) -> Project:
        project = Project(name=name.strip())
        self._session.add(project)
        self._session.commit()
        return project

    def get(self, project_id: str) -> Project:
        project = self._session.get(Project, project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        return project

    def list(self) -> list[Project]:
        statement = select(Project).order_by(Project.created_at.desc())
        return list(self._session.scalars(statement))

    def delete(self, project_id: str) -> None:
        project = self.get(project_id)
        self._session.delete(project)
        self._session.commit()
        for root in self._artifact_roots:
            shutil.rmtree(root / "projects" / project_id, ignore_errors=True)
