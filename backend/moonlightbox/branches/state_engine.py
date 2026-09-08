import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from moonlightbox.branches.attribution import SubjectAttributionGuard
from moonlightbox.branches.continuity_models import (
    BranchBeliefEvidence,
    BranchMemoryEpisode,
    BranchMemoryItem,
    BranchStateVersion,
    IdentityKernel,
)
from moonlightbox.branches.continuity_types import (
    MemoryCandidate,
    MemoryProposal,
    MemoryReviewResult,
)
from moonlightbox.branches.models import Branch


@dataclass(frozen=True)
class StateEngineConfig:
    max_relationship_delta: float = 5.0
    max_emotional_delta: float = 0.1
    belief_activation_threshold: float = 0.65
    belief_winner_margin: float = 0.15
    belief_decay_grace_days: float = 30.0
    belief_half_life_days: float = 365.0


class StateSnapshot(TypedDict):
    persona_state: dict[str, object]
    relationship_state: dict[str, object]
    user_model: dict[str, object]
    emotional_tendency: dict[str, object]
    active_belief_ids: list[str]
    contested_belief_ids: list[str]
    current_goals: dict[str, object]
    current_concerns: dict[str, object]


class BranchStateEngine:
    def __init__(
        self,
        session: Session,
        config: StateEngineConfig | None = None,
    ) -> None:
        self._session = session
        self._config = config or StateEngineConfig()

    def apply(
        self,
        *,
        branch: Branch,
        episode: BranchMemoryEpisode,
        proposal: MemoryProposal,
        review: MemoryReviewResult,
        identity_kernel: IdentityKernel,
    ) -> BranchStateVersion:
        if episode.branch_id != branch.id:
            raise ValueError("episode 不属于当前分支")
        if identity_kernel.model_version_id != branch.model_version_id:
            raise ValueError("人格内核与分支模型版本不一致")
        current = self._current(branch.id)
        if current is not None and episode.id in current.source_episode_ids:
            return current

        approved_indexes = (
            set(review.approved_candidate_indexes) if review.verdict == "approve" else set()
        )
        approved_items: list[BranchMemoryItem] = []
        for index, candidate in enumerate(proposal.candidates):
            if index not in approved_indexes:
                continue
            if not self._candidate_respects_subject(candidate):
                continue
            item = self._persist_candidate(branch, episode, candidate)
            approved_items.append(item)

        approved_delta = review.approved_state_delta
        if not approved_items:
            approved_delta = None
        durable_support = {
            index
            for index in approved_indexes
            if SubjectAttributionGuard().can_update_persistent_state(
                proposal.candidates[index]
            )
            and any(
                evidence_id in proposal.candidates[index].evidence_message_ids
                for evidence_id in self._user_evidence_ids(episode)
            )
        }
        if (
            approved_delta is not None
            and (
                not approved_delta.supporting_candidate_indexes
                or not set(approved_delta.supporting_candidate_indexes).issubset(
                    durable_support
                )
            )
        ):
            # Persist the utterance/narrative at its proper authority, but do
            # not let generated or reported content recursively rewrite state.
            approved_delta = None
        previous_snapshot = self._snapshot_from(branch, current)
        relationship_state = dict(previous_snapshot["relationship_state"])
        emotional_tendency = dict(previous_snapshot["emotional_tendency"])
        user_model = dict(previous_snapshot["user_model"])
        field_evidence = dict(current.field_evidence) if current is not None else {}
        field_confidence = dict(current.field_confidence) if current is not None else {}
        if approved_delta is not None:
            relationship_state = self._apply_numeric_delta(
                relationship_state,
                approved_delta.relationship_delta,
                maximum=self._config.max_relationship_delta,
                lower=0,
                upper=100,
            )
            emotional_tendency = self._apply_numeric_delta(
                emotional_tendency,
                approved_delta.emotional_delta,
                maximum=self._config.max_emotional_delta,
                lower=0,
                upper=1,
            )
            user_model.update(approved_delta.user_model_updates)
            support_confidence = max(
                (
                    proposal.candidates[index].confidence
                    for index in approved_delta.supporting_candidate_indexes
                ),
                default=0.0,
            )
            for namespace, values in (
                ("relationship_state", approved_delta.relationship_delta),
                ("emotional_tendency", approved_delta.emotional_delta),
            ):
                for key in values:
                    path = f"{namespace}.{key}"
                    field_evidence[path] = list(
                        dict.fromkeys([*field_evidence.get(path, []), episode.id])
                    )
                    field_confidence[path] = support_confidence
        active_beliefs, contested_beliefs = self._resolve_active_beliefs(
            branch_id=branch.id,
            effective_at=episode.ended_at,
        )
        if current is not None:
            current.is_current = False
        version = BranchStateVersion(
            branch_id=branch.id,
            version=self._next_version(branch.id),
            previous_version_id=current.id if current is not None else None,
            persona_state=dict(previous_snapshot["persona_state"]),
            relationship_state=relationship_state,
            user_model=user_model,
            emotional_tendency=emotional_tendency,
            active_belief_ids=active_beliefs,
            contested_belief_ids=contested_beliefs,
            current_goals=dict(previous_snapshot["current_goals"]),
            current_concerns=dict(previous_snapshot["current_concerns"]),
            reason=("本轮证据通过复核后更新" if approved_items else "本轮未产生可采纳的主体记忆"),
            source_episode_ids=[
                *(current.source_episode_ids if current is not None else []),
                episode.id,
            ],
            field_evidence=field_evidence,
            field_confidence=field_confidence,
            is_current=True,
        )
        self._session.add(version)
        self._session.flush()
        for item in approved_items:
            if item.state_version_id is None:
                item.state_version_id = version.id
        episode.processing_status = "processed"
        branch.state_snapshot = self._state_snapshot(
            version,
            situational_state=branch.state_snapshot.get("situational_state"),
        )
        self._session.flush()
        return version

    def rollback(self, branch: Branch, version_id: str) -> BranchStateVersion:
        target = self._session.get(BranchStateVersion, version_id)
        if target is None or target.branch_id != branch.id:
            raise LookupError("状态版本不存在")
        current = self._current(branch.id)
        if current is not None:
            current.is_current = False
        now = datetime.now(UTC)
        target.rolled_back_at = now
        invalidated_state_ids = list(
            self._session.scalars(
                select(BranchStateVersion.id).where(
                    BranchStateVersion.branch_id == branch.id,
                    BranchStateVersion.version > target.version,
                )
            )
        )
        if invalidated_state_ids:
            for item in self._session.scalars(
                select(BranchMemoryItem).where(
                    BranchMemoryItem.branch_id == branch.id,
                    BranchMemoryItem.state_version_id.in_(invalidated_state_ids),
                    BranchMemoryItem.valid_to.is_(None),
                )
            ):
                item.valid_to = now
                item.invalidated_at = now
        version = BranchStateVersion(
            branch_id=branch.id,
            version=self._next_version(branch.id),
            previous_version_id=current.id if current is not None else None,
            persona_state=dict(target.persona_state),
            relationship_state=dict(target.relationship_state),
            user_model=dict(target.user_model),
            emotional_tendency=dict(target.emotional_tendency),
            active_belief_ids=list(target.active_belief_ids),
            contested_belief_ids=list(target.contested_belief_ids),
            current_goals=dict(target.current_goals),
            current_concerns=dict(target.current_concerns),
            memory_cutoff_version=target.version,
            rollback_of_version_id=target.id,
            reason=f"回滚到版本 {target.version}",
            source_episode_ids=list(target.source_episode_ids),
            field_evidence=dict(target.field_evidence),
            field_confidence=dict(target.field_confidence),
            is_current=True,
        )
        self._session.add(version)
        self._session.flush()
        branch.state_snapshot = self._state_snapshot(
            version,
            situational_state=branch.state_snapshot.get("situational_state"),
        )
        self._session.flush()
        return version

    def _persist_candidate(
        self,
        branch: Branch,
        episode: BranchMemoryEpisode,
        candidate: MemoryCandidate,
    ) -> BranchMemoryItem:
        lineage_hash = self._lineage_hash(branch.id, episode.id, candidate)
        existing = self._session.scalar(
            select(BranchMemoryItem).where(
                BranchMemoryItem.branch_id == branch.id,
                BranchMemoryItem.lineage_hash == lineage_hash,
                BranchMemoryItem.review_status == "approved",
            )
        )
        if existing is not None:
            return existing
        item = BranchMemoryItem(
            branch_id=branch.id,
            kind=candidate.kind,
            content=candidate.content,
            subject=candidate.subject,
            predicate=candidate.predicate,
            object=candidate.object,
            confidence=candidate.confidence,
            importance=candidate.importance,
            valid_from=episode.ended_at,
            source_episode_ids=[episode.id],
            source_item_ids=[],
            lineage_hash=lineage_hash,
            review_status="approved",
            verification_status=SubjectAttributionGuard().verification_status(
                candidate
            ),
            claim_key=f"{candidate.subject}:{candidate.predicate}",
            stance=candidate.stance,
            root_episode_hashes=[episode.episode_hash],
        )
        self._session.add(item)
        self._session.flush()
        if candidate.kind == "belief":
            evidence_weight = SubjectAttributionGuard().evidence_weight(
                candidate.source_role
            )
            if (
                candidate.source_role == "interaction"
                and not any(
                    evidence_id in candidate.evidence_message_ids
                    for evidence_id in self._user_evidence_ids(episode)
                )
            ):
                evidence_weight = 0.0
            self._session.add(
                BranchBeliefEvidence(
                    branch_id=branch.id,
                    belief_id=item.id,
                    episode_id=episode.id,
                    stance=candidate.stance,
                    source_role=candidate.source_role,
                    evidence_type=(
                        "autonomous_statement"
                        if candidate.source_role == "digital_human"
                        else "observed_interaction"
                    ),
                    weight=evidence_weight,
                )
            )
            self._session.flush()
        return item

    def _candidate_respects_subject(self, candidate: MemoryCandidate) -> bool:
        return SubjectAttributionGuard().accepts(candidate)

    def _resolve_active_beliefs(
        self,
        *,
        branch_id: str,
        effective_at: datetime,
    ) -> tuple[list[str], list[str]]:
        beliefs = list(
            self._session.scalars(
                select(BranchMemoryItem).where(
                    BranchMemoryItem.branch_id == branch_id,
                    BranchMemoryItem.kind == "belief",
                    BranchMemoryItem.review_status == "approved",
                    BranchMemoryItem.valid_to.is_(None),
                    BranchMemoryItem.valid_from <= effective_at,
                )
            )
        )
        evidence_roles = {
            row.belief_id: row
            for row in self._session.scalars(
                select(BranchBeliefEvidence).where(
                    BranchBeliefEvidence.branch_id == branch_id,
                    BranchBeliefEvidence.source_role == "interaction",
                    BranchBeliefEvidence.weight > 0,
                )
            )
        }
        # One root episode contributes at most once to an alternative. Derived
        # summaries and repeated indexing therefore cannot manufacture support.
        alternatives: dict[
            tuple[str, str],
            dict[str, list[tuple[BranchMemoryItem, float]]],
        ] = {}
        for item in beliefs:
            evidence = evidence_roles.get(item.id)
            if evidence is None:
                continue
            age_days = max(
                0.0,
                (_aware(effective_at) - _aware(item.valid_from)).total_seconds()
                / 86400,
            )
            decay_age = max(0.0, age_days - self._config.belief_decay_grace_days)
            decay = 0.5 ** (decay_age / self._config.belief_half_life_days)
            signal = max(
                0.0,
                min(1.0, item.confidence * evidence.weight * decay),
            )
            alternatives.setdefault((item.subject, item.predicate), {}).setdefault(
                item.object, []
            ).append((item, signal))

        active: list[str] = []
        contested: list[str] = []
        for by_object in alternatives.values():
            scored: list[tuple[float, BranchMemoryItem]] = []
            for entries in by_object.values():
                roots: dict[str, tuple[BranchMemoryItem, float]] = {}
                for item, signal in entries:
                    root_keys = item.root_episode_hashes or item.source_episode_ids
                    for root in root_keys:
                        previous = roots.get(root)
                        if previous is None or signal > previous[1]:
                            roots[root] = (item, signal)
                support = 0.0
                oppose = 0.0
                for item, signal in roots.values():
                    if item.stance == "oppose":
                        oppose = 1 - (1 - oppose) * (1 - signal)
                    else:
                        support = 1 - (1 - support) * (1 - signal)
                score = support * (1 - oppose)
                representative = max(
                    (item for item, _signal in entries if item.stance == "support"),
                    key=lambda item: (_aware(item.valid_from), item.confidence),
                    default=max(entries, key=lambda entry: entry[1])[0],
                )
                scored.append((score, representative))
            scored.sort(key=lambda entry: entry[0], reverse=True)
            top_score, top_item = scored[0]
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            if (
                top_score >= self._config.belief_activation_threshold
                and top_score - second_score >= self._config.belief_winner_margin
            ):
                active.append(top_item.id)
                continue
            contested.extend(
                item.id
                for score, item in scored
                if score >= self._config.belief_activation_threshold / 2
            )
        return active, contested

    def _current(self, branch_id: str) -> BranchStateVersion | None:
        return self._session.scalar(
            select(BranchStateVersion)
            .where(
                BranchStateVersion.branch_id == branch_id,
                BranchStateVersion.is_current.is_(True),
            )
            .order_by(BranchStateVersion.version.desc())
        )

    def _next_version(self, branch_id: str) -> int:
        latest = self._session.scalar(
            select(func.max(BranchStateVersion.version)).where(
                BranchStateVersion.branch_id == branch_id
            )
        )
        return int(latest or 0) + 1

    def _snapshot_from(
        self,
        branch: Branch,
        current: BranchStateVersion | None,
    ) -> StateSnapshot:
        if current is not None:
            return {
                "persona_state": current.persona_state,
                "relationship_state": current.relationship_state,
                "user_model": current.user_model,
                "emotional_tendency": current.emotional_tendency,
                "active_belief_ids": current.active_belief_ids,
                "contested_belief_ids": current.contested_belief_ids,
                "current_goals": current.current_goals,
                "current_concerns": current.current_concerns,
            }
        source = branch.state_snapshot
        return {
            "persona_state": _object_dict(source.get("persona_state")),
            "relationship_state": _object_dict(source.get("relationship_state")),
            "user_model": _object_dict(source.get("user_model")),
            "emotional_tendency": _object_dict(source.get("emotional_tendency")),
            "active_belief_ids": _string_list(source.get("active_belief_ids")),
            "contested_belief_ids": _string_list(
                source.get("contested_belief_ids")
            ),
            "current_goals": _object_dict(source.get("current_goals")),
            "current_concerns": _object_dict(source.get("current_concerns")),
        }

    @staticmethod
    def _user_evidence_ids(episode: BranchMemoryEpisode) -> set[str]:
        ids = {episode.user_turn_id}
        for message in episode.user_messages or []:
            for field in ("message_id", "turn_id"):
                value = message.get(field)
                if isinstance(value, str) and value:
                    ids.add(value)
        return ids

    def _apply_numeric_delta(
        self,
        current: dict[str, object],
        delta: dict[str, float],
        *,
        maximum: float,
        lower: float,
        upper: float,
    ) -> dict[str, object]:
        result = dict(current)
        for key, requested in delta.items():
            bounded_delta = max(-maximum, min(maximum, requested))
            prior = result.get(key, 0.0)
            prior_value = float(prior) if isinstance(prior, int | float) else 0.0
            result[key] = round(
                max(lower, min(upper, prior_value + bounded_delta)),
                6,
            )
        return result

    def _state_snapshot(
        self,
        version: BranchStateVersion,
        *,
        situational_state: object | None = None,
    ) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "protocol_version": "continual-persona-v1-pending",
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
            "state_evidence": version.field_evidence,
            "state_confidence": version.field_confidence,
        }
        if isinstance(situational_state, dict):
            snapshot["situational_state"] = situational_state
        return snapshot

    def _lineage_hash(
        self,
        branch_id: str,
        episode_id: str,
        candidate: MemoryCandidate,
    ) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "branch_id": branch_id,
                    "root_episode_ids": [episode_id],
                    "kind": candidate.kind,
                    "subject": candidate.subject,
                    "predicate": candidate.predicate,
                    "object": candidate.object,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
