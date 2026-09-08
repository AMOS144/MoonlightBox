import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.branches.baseline_boundary import (
    BaselineBoundaryResolver,
    ResolvedBaselineBoundary,
    semantic_message_digest,
)
from moonlightbox.branches.baseline_models import (
    BranchBaselineEventSnapshot,
    BranchBaselineManifest,
    BranchBaselineState,
)
from moonlightbox.branches.continuity_models import (
    BranchStateVersion,
)
from moonlightbox.branches.models import Branch
from moonlightbox.events.models import AnalysisRevision, EventNode
from moonlightbox.imports.models import Message


class BaselineIntegrityError(RuntimeError):
    pass


class BranchBaselineService:
    def __init__(self, session: Session) -> None:
        self._session = session

    def build(
        self,
        branch: Branch,
        boundary: ResolvedBaselineBoundary,
    ) -> tuple[BranchBaselineManifest, BranchBaselineState, BranchStateVersion]:
        existing = self._session.scalar(
            select(BranchBaselineManifest).where(BranchBaselineManifest.branch_id == branch.id)
        )
        if existing is not None:
            state = self._session.scalar(
                select(BranchBaselineState).where(BranchBaselineState.manifest_id == existing.id)
            )
            version = self._session.scalar(
                select(BranchStateVersion)
                .where(
                    BranchStateVersion.branch_id == branch.id,
                    BranchStateVersion.baseline_state_id == (state.id if state is not None else ""),
                )
                .order_by(BranchStateVersion.version.desc())
            )
            if state is not None and version is not None:
                return existing, state, version
        rows = BaselineBoundaryResolver(self._session).messages_before(boundary)
        message_ids = [message.id for message, _role in rows]
        digest = semantic_message_digest(rows)
        manifest = BranchBaselineManifest(
            branch_id=branch.id,
            project_id=branch.project_id,
            import_id=boundary.import_id,
            origin_event_id=branch.origin_event_id,
            boundary_message_id=boundary.message_id,
            boundary_timestamp=boundary.timestamp,
            boundary_source_id=boundary.source_id,
            boundary_inclusive=boundary.inclusive,
            message_count=len(rows),
            event_snapshot_count=0,
            message_digest=digest,
            event_digest=hashlib.sha256(b"").hexdigest(),
            index_fingerprint=digest,
            recent_tail_message_ids=_recent_tail_ids(rows),
            protocol_version="branch-baseline-v2",
            validated_at=datetime.now(UTC),
        )
        self._session.add(manifest)
        self._session.flush()
        snapshots = self._freeze_events(manifest, boundary, rows)
        manifest.event_snapshot_count = len(snapshots)
        manifest.event_digest = _event_digest(snapshots)
        baseline_state = self._create_state(branch, manifest, snapshots, message_ids)
        version = self._project_state(branch, manifest, baseline_state)
        self.validate(branch, manifest, boundary)
        branch.origin_import_id = boundary.import_id
        branch.origin_boundary_message_id = boundary.message_id
        branch.baseline_manifest_id = manifest.id
        return manifest, baseline_state, version

    def validate(
        self,
        branch: Branch,
        manifest: BranchBaselineManifest,
        boundary: ResolvedBaselineBoundary,
    ) -> None:
        rows = BaselineBoundaryResolver(self._session).messages_before(boundary)
        if len(rows) != manifest.message_count:
            raise BaselineIntegrityError("基础历史消息数量校验失败")
        if semantic_message_digest(rows) != manifest.message_digest:
            raise BaselineIntegrityError("基础历史语义摘要校验失败")
        contains_boundary = any(
            message.id == boundary.message_id for message, _role in rows
        )
        if contains_boundary != boundary.inclusive:
            raise BaselineIntegrityError("节点结束边界包含规则校验失败")
        if manifest.branch_id != branch.id or manifest.import_id != boundary.import_id:
            raise BaselineIntegrityError("基础历史归属校验失败")

    def _freeze_events(
        self,
        manifest: BranchBaselineManifest,
        boundary: ResolvedBaselineBoundary,
        rows: list[tuple[Message, str]],
    ) -> list[BranchBaselineEventSnapshot]:
        source_message_ids = {
            message.source_id: message.id for message, _role in rows
        }
        source_ids = set(source_message_ids)
        snapshots: list[BranchBaselineEventSnapshot] = []
        events = self._session.scalars(
            select(EventNode)
            .where(
                EventNode.project_id == manifest.project_id,
                EventNode.status == "active",
            )
            .order_by(
                EventNode.ended_at,
                EventNode.started_at,
                EventNode.created_at,
                EventNode.id,
            )
        )
        for event in events:
            if not set(event.evidence_ids).issubset(source_ids):
                continue
            ended_at = event.ended_at or event.started_at or event.created_at
            if (
                _aware(ended_at) > _aware(boundary.timestamp)
                or (
                    not boundary.inclusive
                    and _aware(ended_at) == _aware(boundary.timestamp)
                )
            ):
                continue
            revision = self._session.scalar(
                select(AnalysisRevision)
                .where(AnalysisRevision.event_id == event.id)
                .order_by(AnalysisRevision.revision_number.desc())
            )
            payload = {
                "title": event.title,
                "summary": event.summary,
                "type": event.type,
                "lane": event.lane,
                "event_status": event.event_status,
                "before_state": event.before_state,
                "after_state": event.after_state,
                "emotion_labels": event.emotion_labels,
                "topic": event.topic,
                "evidence_ids": event.evidence_ids,
            }
            content_hash = hashlib.sha256(
                json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            snapshot = BranchBaselineEventSnapshot(
                manifest_id=manifest.id,
                source_event_id=event.id,
                source_revision_id=revision.id if revision is not None else None,
                snapshot=payload,
                evidence_message_ids=[
                    source_message_ids[source_id]
                    for source_id in event.evidence_ids
                ],
                started_at=event.started_at or event.created_at,
                ended_at=ended_at,
                content_hash=content_hash,
            )
            self._session.add(snapshot)
            snapshots.append(snapshot)
        self._session.flush()
        return snapshots

    def _create_state(
        self,
        branch: Branch,
        manifest: BranchBaselineManifest,
        snapshots: list[BranchBaselineEventSnapshot],
        message_ids: list[str],
    ) -> BranchBaselineState:
        latest = max(
            snapshots,
            key=lambda item: (_aware(item.ended_at), item.source_event_id),
            default=None,
        )
        latest_snapshot = latest.snapshot if latest is not None else {}
        after_state = latest_snapshot.get("after_state")
        emotion_labels = latest_snapshot.get("emotion_labels")
        payload = {
            "persona_state": {},
            "relationship_state": (
                {"summary": after_state} if isinstance(after_state, str) else {}
            ),
            "emotional_tendency": (
                {"labels": emotion_labels}
                if isinstance(emotion_labels, list)
                else {}
            ),
            "user_model": {},
            "historical_belief_ids": [],
        }
        content_hash = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        state = BranchBaselineState(
            manifest_id=manifest.id,
            branch_id=branch.id,
            **payload,
            evidence_message_ids=message_ids[-32:],
            evidence_event_snapshot_ids=[item.id for item in snapshots],
            local_proposal=payload,
            review_result={
                "verdict": "approved_frozen_evidence",
                "reason": "仅依据节点前冻结事件和消息构建",
            },
            content_hash=content_hash,
        )
        self._session.add(state)
        self._session.flush()
        return state

    def _project_state(
        self,
        branch: Branch,
        manifest: BranchBaselineManifest,
        baseline: BranchBaselineState,
    ) -> BranchStateVersion:
        current = self._session.scalar(
            select(BranchStateVersion)
            .where(
                BranchStateVersion.branch_id == branch.id,
                BranchStateVersion.is_current.is_(True),
            )
            .order_by(BranchStateVersion.version.desc())
        )
        latest = self._session.scalar(
            select(func.max(BranchStateVersion.version)).where(
                BranchStateVersion.branch_id == branch.id
            )
        )
        if current is not None:
            current.is_current = False
        version = BranchStateVersion(
            branch_id=branch.id,
            version=int(latest or 0) + 1,
            previous_version_id=current.id if current is not None else None,
            persona_state=(
                dict(current.persona_state) if current is not None else dict(baseline.persona_state)
            ),
            relationship_state=(
                dict(current.relationship_state)
                if current is not None and current.relationship_state
                else dict(baseline.relationship_state)
            ),
            user_model=(
                dict(current.user_model)
                if current is not None and current.user_model
                else dict(baseline.user_model)
            ),
            emotional_tendency=(
                dict(current.emotional_tendency)
                if current is not None and current.emotional_tendency
                else dict(baseline.emotional_tendency)
            ),
            active_belief_ids=(
                list(current.active_belief_ids)
                if current is not None
                else list(baseline.historical_belief_ids)
            ),
            contested_belief_ids=(
                list(current.contested_belief_ids) if current is not None else []
            ),
            current_goals=(
                dict(current.current_goals) if current is not None else {}
            ),
            current_concerns=(
                dict(current.current_concerns) if current is not None else {}
            ),
            reason=("补建节点基础历史状态" if current is not None else "节点基础历史状态"),
            source_episode_ids=(list(current.source_episode_ids) if current is not None else []),
            is_current=True,
            baseline_manifest_id=manifest.id,
            baseline_state_id=baseline.id,
        )
        self._session.add(version)
        self._session.flush()
        branch.state_snapshot = {
            "protocol_version": "continual-persona-v1",
            "state_version_id": version.id,
            "version": version.version,
            "persona_state": version.persona_state,
            "relationship_state": version.relationship_state,
            "user_model": version.user_model,
            "emotional_tendency": version.emotional_tendency,
            "active_belief_ids": version.active_belief_ids,
            "contested_belief_ids": version.contested_belief_ids,
            "current_goals": version.current_goals,
            "current_concerns": version.current_concerns,
            "baseline_manifest_id": manifest.id,
            "baseline_state_id": baseline.id,
        }
        return version


def _event_digest(snapshots: list[BranchBaselineEventSnapshot]) -> str:
    digest = hashlib.sha256()
    for item in sorted(snapshots, key=lambda value: value.source_event_id):
        digest.update(item.content_hash.encode())
    return digest.hexdigest()


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _recent_tail_ids(
    rows: list[tuple[Message, str]],
    *,
    turn_limit: int = 8,
) -> list[str]:
    selected: list[tuple[Message, str]] = []
    turn_count = 0
    previous_role: str | None = None
    for message, role in reversed(rows):
        if role != previous_role:
            turn_count += 1
            previous_role = role
        if turn_count > turn_limit:
            break
        selected.append((message, role))
    return [message.id for message, _role in reversed(selected)]
