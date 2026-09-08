import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from moonlightbox.branches.baseline_models import BranchBaselineManifest
from moonlightbox.branches.embeddings import TextEmbedder
from moonlightbox.branches.models import Branch, BranchMessage
from moonlightbox.imports.models import Message, Participant
from moonlightbox.media.models import MediaAsset, MediaSemanticAnnotation
from moonlightbox.training.media_behavior_policy import MediaModality


@dataclass(frozen=True)
class MediaSelection:
    asset_id: str
    modality: MediaModality
    semantic_text: str
    similarity: float


def select_reusable_media(
    session: Session,
    *,
    branch: Branch,
    modality: MediaModality,
    query: str,
    embedder: TextEmbedder | None,
    minimum_similarity: float | None = None,
) -> MediaSelection | None:
    """Select approved media inside the branch's frozen historical boundary."""

    if embedder is None or not query.strip():
        return None
    manifest = session.scalar(
        select(BranchBaselineManifest).where(
            BranchBaselineManifest.branch_id == branch.id
        )
    )
    if manifest is None:
        return None
    expected_mime = f"{modality}/"
    rows = session.execute(
        select(MediaSemanticAnnotation, MediaAsset)
        .join(MediaAsset, MediaAsset.id == MediaSemanticAnnotation.asset_id)
        .join(Message, Message.media_asset_id == MediaAsset.id)
        .join(Participant, Participant.id == Message.participant_id)
        .where(
            MediaSemanticAnnotation.project_id == branch.project_id,
            MediaSemanticAnnotation.modality == modality,
            MediaSemanticAnnotation.status == "succeeded",
            MediaSemanticAnnotation.reuse_decision == "approved",
            MediaSemanticAnnotation.reusable.is_(True),
            MediaAsset.project_id == branch.project_id,
            MediaAsset.mime_type.startswith(expected_mime),
            Message.project_id == branch.project_id,
            Message.import_id == manifest.import_id,
            Participant.role == "target",
            or_(
                Message.timestamp < _aware(manifest.boundary_timestamp),
                and_(
                    Message.timestamp == _aware(manifest.boundary_timestamp),
                    Message.source_id < manifest.boundary_source_id,
                ),
                and_(
                    Message.timestamp == _aware(manifest.boundary_timestamp),
                    Message.source_id == manifest.boundary_source_id,
                    Message.id <= manifest.boundary_message_id,
                ),
            ),
        )
        .order_by(MediaSemanticAnnotation.confidence.desc(), MediaAsset.id.asc())
    )
    recent_asset_ids = set(
        session.scalars(
            select(BranchMessage.media_asset_id)
            .where(
                BranchMessage.branch_id == branch.id,
                BranchMessage.media_asset_id.is_not(None),
            )
            .order_by(BranchMessage.sequence.desc())
            .limit(20)
        )
    )
    candidates: list[tuple[MediaSemanticAnnotation, MediaAsset, str]] = []
    seen: set[str] = set()
    for annotation, asset in rows:
        if asset.id in seen or asset.id in recent_asset_ids:
            continue
        seen.add(asset.id)
        semantic_text = (
            annotation.transcript
            if modality == "audio"
            else "；".join(
                item for item in (annotation.summary, annotation.ocr_text) if item.strip()
            )
        ).strip()
        if semantic_text:
            candidates.append((annotation, asset, semantic_text))
    if not candidates:
        return None
    try:
        vectors = embedder.embed([query, *(item[2] for item in candidates)])
    except Exception:
        return None
    if len(vectors) != len(candidates) + 1:
        return None
    scored = [
        (_cosine(vectors[0], vectors[index + 1]), annotation, asset, semantic_text)
        for index, (annotation, asset, semantic_text) in enumerate(candidates)
    ]
    similarity, _annotation, asset, semantic_text = max(
        scored,
        key=lambda item: (item[0], item[1].confidence, item[2].id),
    )
    threshold = minimum_similarity if minimum_similarity is not None else (
        0.84 if modality == "audio" else 0.72
    )
    if similarity < threshold:
        return None
    if modality == "audio" and _bigram_overlap(query, semantic_text) < 0.5:
        return None
    return MediaSelection(
        asset_id=asset.id,
        modality=modality,
        semantic_text=semantic_text,
        similarity=similarity,
    )


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return numerator / (left_norm * right_norm)


def _bigram_overlap(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        compact = re.sub(r"\W+", "", value.casefold())
        if len(compact) < 2:
            return {compact} if compact else set()
        return {compact[index : index + 2] for index in range(len(compact) - 1)}

    left_grams = grams(left)
    right_grams = grams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(right_grams)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
