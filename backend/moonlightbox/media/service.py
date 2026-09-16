import hashlib
import mimetypes
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.imports.models import Message, Participant
from moonlightbox.media.models import MediaAsset, MediaSemanticAnnotation

_BLOCKING_SAFETY_TAGS = frozenset(
    {
        "sensitive",
        "private_document",
        "identity_document",
        "financial",
        "medical",
        "explicit",
        "child",
    }
)


class MediaStore:
    def __init__(
        self,
        data_dir: Path,
        *,
        read_only_source_dir: Path | None = None,
    ) -> None:
        self._data_dir = data_dir
        # 仅在读取缺失的旧媒体时回退；save() 从不使用这个目录。
        self._read_only_source_dir = read_only_source_dir

    def save(
        self,
        session: Session,
        project_id: str,
        *,
        kind: str,
        filename: str,
        content: bytes,
        source_key: str | None = None,
    ) -> MediaAsset:
        digest = hashlib.sha256(content).hexdigest()
        existing = session.scalar(
            select(MediaAsset).where(
                MediaAsset.project_id == project_id,
                MediaAsset.sha256 == digest,
            )
        )
        if existing is not None:
            return existing
        suffix = Path(filename).suffix.lower()
        if len(suffix) > 10 or not suffix.startswith("."):
            suffix = ""
        relative_path = Path("projects") / project_id / "media" / f"{digest}{suffix}"
        absolute_path = self._data_dir / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_bytes(content)
        asset = MediaAsset(
            project_id=project_id,
            kind=kind,
            sha256=digest,
            relative_path=relative_path.as_posix(),
            mime_type=mimetypes.guess_type(filename)[0] or "application/octet-stream",
            source_key=source_key,
        )
        session.add(asset)
        session.flush()
        return asset

    def path_for(self, asset: MediaAsset) -> Path:
        candidate = self._resolve_media_path(self._data_dir, asset.relative_path)
        if candidate.is_file() or self._read_only_source_dir is None:
            return candidate

        # 隔离体验库通常只复制数据库；源目录只作为不可写的读取兜底，避免破图，
        # 同时保证新导入或新生成的媒体仍保存到当前 data_dir。
        fallback = self._resolve_media_path(
            self._read_only_source_dir,
            asset.relative_path,
        )
        return fallback if fallback.is_file() else candidate

    @staticmethod
    def _resolve_media_path(root: Path, relative_path: str) -> Path:
        candidate = (root / relative_path).resolve()
        resolved_root = root.resolve()
        if not candidate.is_relative_to(resolved_root):
            raise ValueError("媒体路径越界")
        return candidate


class MediaAnnotationService:
    """Persist semantic evidence and fail closed before historical media reuse."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_project(
        self,
        project_id: str,
        *,
        modality: str | None = None,
        status: str | None = None,
    ) -> list[MediaSemanticAnnotation]:
        query = select(MediaSemanticAnnotation).where(
            MediaSemanticAnnotation.project_id == project_id
        )
        if modality is not None:
            query = query.where(MediaSemanticAnnotation.modality == modality)
        if status is not None:
            query = query.where(MediaSemanticAnnotation.status == status)
        return list(
            self._session.scalars(
                query.order_by(MediaSemanticAnnotation.updated_at.desc())
            )
        )

    def summary(self, project_id: str) -> dict[str, int]:
        assets = list(
            self._session.scalars(
                select(MediaAsset)
                .join(Message, Message.media_asset_id == MediaAsset.id)
                .join(Participant, Participant.id == Message.participant_id)
                .where(
                    MediaAsset.project_id == project_id,
                    Message.project_id == project_id,
                    Participant.role == "target",
                )
                .distinct()
            )
        )
        annotations = self.list_for_project(project_id)
        eligible = [
            asset
            for asset in assets
            if asset.mime_type.casefold().startswith("audio/")
            or (
                asset.mime_type.casefold().startswith("image/")
                and asset.kind != "sticker"
            )
        ]
        return {
            "eligible": len(eligible),
            "annotated": len(annotations),
            "succeeded": sum(item.status == "succeeded" for item in annotations),
            "failed": sum(item.status == "failed" for item in annotations),
            "needs_review": sum(
                item.status == "needs_review" for item in annotations
            ),
            "approved": sum(item.reusable for item in annotations),
            "blocked": sum(
                item.reuse_decision == "blocked" for item in annotations
            ),
        }

    def annotate(
        self,
        project_id: str,
        asset_id: str,
        *,
        summary: str = "",
        transcript: str = "",
        ocr_text: str = "",
        safety_tags: list[str] | None = None,
        source_model: str,
        source_version: str,
        confidence: float,
        status: str = "succeeded",
        reuse_decision: str = "pending",
        failure_code: str | None = None,
        review_source: str | None = None,
        commit: bool = True,
    ) -> MediaSemanticAnnotation:
        asset = self._session.get(MediaAsset, asset_id)
        if asset is None or asset.project_id != project_id:
            raise LookupError("媒体资产不存在")
        modality = _asset_modality(asset)
        if status not in {"pending", "succeeded", "failed", "needs_review"}:
            raise ValueError("媒体语义状态无效")
        if reuse_decision not in {"pending", "approved", "blocked"}:
            raise ValueError("媒体复用决定无效")
        if not 0 <= confidence <= 1:
            raise ValueError("媒体语义置信度必须位于 0 到 1")
        normalized_tags = sorted(
            {
                tag.strip().casefold()
                for tag in safety_tags or []
                if isinstance(tag, str) and tag.strip()
            }
        )
        annotation = self._session.scalar(
            select(MediaSemanticAnnotation).where(
                MediaSemanticAnnotation.asset_id == asset.id
            )
        )
        if annotation is None:
            annotation = MediaSemanticAnnotation(
                project_id=project_id,
                asset_id=asset.id,
                modality=modality,
            )
            self._session.add(annotation)
        annotation.modality = modality
        annotation.status = status
        annotation.summary = summary.strip()
        annotation.transcript = transcript.strip()
        annotation.ocr_text = ocr_text.strip()
        annotation.safety_tags = normalized_tags
        annotation.source_model = source_model.strip()
        annotation.source_version = source_version.strip()
        annotation.confidence = confidence
        annotation.failure_code = failure_code
        annotation.reuse_decision = reuse_decision
        annotation.review_source = review_source.strip() if review_source else None
        annotation.reviewed_at = (
            datetime.now(UTC) if reuse_decision in {"approved", "blocked"} else None
        )
        annotation.reusable = _can_reuse(annotation)
        if reuse_decision == "approved" and not annotation.reusable:
            annotation.reuse_decision = "pending"
            annotation.review_source = None
            annotation.reviewed_at = None
            if status == "succeeded":
                annotation.status = "needs_review"
        annotation.updated_at = datetime.now(UTC)
        if commit:
            self._session.commit()
            self._session.refresh(annotation)
        else:
            self._session.flush()
        return annotation

    def reusable_assets(
        self,
        project_id: str,
        modality: str,
    ) -> list[MediaSemanticAnnotation]:
        return list(
            self._session.scalars(
                select(MediaSemanticAnnotation)
                .where(
                    MediaSemanticAnnotation.project_id == project_id,
                    MediaSemanticAnnotation.modality == modality,
                    MediaSemanticAnnotation.status == "succeeded",
                    MediaSemanticAnnotation.reuse_decision == "approved",
                    MediaSemanticAnnotation.reusable.is_(True),
                )
                .order_by(
                    MediaSemanticAnnotation.confidence.desc(),
                    MediaSemanticAnnotation.id.asc(),
                )
            )
        )


def _asset_modality(asset: MediaAsset) -> str:
    mime = asset.mime_type.casefold()
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("image/"):
        return "sticker" if asset.kind == "sticker" else "image"
    raise ValueError("媒体 MIME 类型不支持语义标注")


def _can_reuse(annotation: MediaSemanticAnnotation) -> bool:
    if annotation.status != "succeeded" or annotation.reuse_decision != "approved":
        return False
    if annotation.confidence < 0.75:
        return False
    if _BLOCKING_SAFETY_TAGS.intersection(annotation.safety_tags):
        return False
    if annotation.modality == "audio":
        return bool(annotation.transcript.strip())
    return bool(annotation.summary.strip())
