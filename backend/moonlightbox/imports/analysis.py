from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.events.models import Episode, EventNode
from moonlightbox.events.schemas import ReviewedEvent
from moonlightbox.events.segmentation import segment
from moonlightbox.events.service import EventService
from moonlightbox.imports.models import Message, Participant
from moonlightbox.imports.types import ImportedMessage, MessageKind

NEGATIVE_MARKERS = ("算了", "不要", "别联系", "分手", "失望", "生气", "讨厌", "拉黑")


@dataclass(frozen=True)
class AnalysisResult:
    event_count: int


def analyze_import(
    session: Session,
    project_id: str,
    import_id: str,
    maximum_events: int = 30,
) -> AnalysisResult:
    messages = _load_messages(session, import_id)
    episodes = segment(messages, gap=timedelta(hours=6))
    boundaries = sorted(
        _candidate_boundaries(episodes, messages),
        key=lambda item: item[0],
        reverse=True,
    )[:maximum_events]
    existing_boundaries = {
        (start_message_id, end_message_id)
        for start_message_id, end_message_id in session.execute(
            select(EventNode.start_message_id, EventNode.end_message_id).where(
                EventNode.project_id == project_id
            )
        )
    }
    service = EventService(session)
    created_count = 0
    for importance, left, right, left_message, right_message in reversed(boundaries):
        boundary = (left_message.source_id, right_message.source_id)
        if boundary in existing_boundaries:
            continue
        service.create(
            project_id,
            _reviewed_event(
                importance,
                left,
                right,
                left_message,
                right_message,
            ),
            analysis_version="heuristic-v1",
            prompt_version="none",
        )
        existing_boundaries.add(boundary)
        created_count += 1
    return AnalysisResult(event_count=created_count)


def _load_messages(session: Session, import_id: str) -> list[ImportedMessage]:
    rows = session.execute(
        select(Message, Participant.name)
        .join(Participant, Participant.id == Message.participant_id)
        .where(Message.import_id == import_id)
        .order_by(Message.timestamp)
    )
    result: list[ImportedMessage] = []
    for message, sender in rows:
        try:
            kind = MessageKind(message.kind)
        except ValueError:
            kind = MessageKind.UNKNOWN
        result.append(
            ImportedMessage(
                source_id=message.source_id,
                timestamp=message.timestamp,
                sender=sender,
                kind=kind,
                content=message.content,
                raw=message.raw,
            )
        )
    return result


def _candidate_boundaries(
    episodes: list[Episode],
    messages: list[ImportedMessage],
) -> list[tuple[float, Episode, Episode, ImportedMessage, ImportedMessage]]:
    by_id = {message.source_id: message for message in messages}
    candidates: list[tuple[float, Episode, Episode, ImportedMessage, ImportedMessage]] = []
    for left, right in zip(episodes, episodes[1:], strict=False):
        gap = right.started_at - left.ended_at
        if gap < timedelta(hours=6):
            continue
        left_message = by_id[left.message_ids[-1]]
        right_message = by_id[right.message_ids[0]]
        gap_days = gap.total_seconds() / 86400
        importance = min(1.0, 0.55 + gap_days / 30 * 0.45)
        candidates.append((importance, left, right, left_message, right_message))
    return candidates


def _reviewed_event(
    importance: float,
    left: Episode,
    right: Episode,
    left_message: ImportedMessage,
    right_message: ImportedMessage,
) -> ReviewedEvent:
    combined = f"{left_message.content} {right_message.content}"
    is_conflict = any(marker in combined for marker in NEGATIVE_MARKERS)
    gap = right.started_at - left.ended_at
    gap_hours = max(1, round(gap.total_seconds() / 3600))
    return ReviewedEvent(
        type="long_pause",
        start_message_id=left_message.source_id,
        end_message_id=right_message.source_id,
        before_state=left_message.content[:120] or "对话中断",
        after_state=right_message.content[:120] or "恢复对话",
        emotion_labels=["冲突"] if is_conflict else ["间隔"],
        topic="关系冲突" if is_conflict else "长时间中断后重新联系",
        conflict_level=4 if is_conflict else 2,
        importance=importance,
        reason=f"对话中断约 {gap_hours} 小时后出现新的交流",
        evidence_ids=[left_message.source_id, right_message.source_id],
    )
