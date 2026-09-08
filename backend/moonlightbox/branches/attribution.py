from typing import Literal

from moonlightbox.branches.continuity_types import MemoryCandidate

VerificationStatus = Literal[
    "verified_interaction",
    "verified_history",
    "asserted_by_user",
    "self_claimed",
    "inferred",
    "disputed",
]


class SubjectAttributionGuard:
    """统一约束记忆候选的主体归属和真实性等级。"""

    def accepts(self, candidate: MemoryCandidate) -> bool:
        if candidate.subject != "digital_human":
            return True
        if candidate.source_role == "user":
            return candidate.kind == "experience"
        if candidate.kind == "fact" and candidate.source_role == "digital_human":
            return False
        return True

    def verification_status(self, candidate: MemoryCandidate) -> VerificationStatus:
        if candidate.kind == "experience":
            if candidate.source_role == "interaction":
                return "verified_interaction"
            if candidate.source_role == "user":
                return "asserted_by_user"
            return "self_claimed"
        if candidate.kind != "fact":
            return "inferred"
        if candidate.source_role == "interaction":
            return "verified_interaction"
        if candidate.source_role == "user":
            return "asserted_by_user"
        return "self_claimed"

    def evidence_weight(self, source_role: str) -> float:
        return {
            "interaction": 1.0,
            # A model utterance is evidence that the model said something, not
            # independent evidence that the corresponding belief is true.
            "digital_human": 0.0,
            "user": 0.15,
        }[source_role]

    def can_update_persistent_state(self, candidate: MemoryCandidate) -> bool:
        """Only independently observed interaction may alter durable state.

        User assertions remain reported claims and digital-human output remains
        a conversation record.  Neither is allowed to certify its own
        relationship, user-model, emotional-trait, or active-belief update.
        """

        return (
            candidate.source_role == "interaction"
            and candidate.kind in {"fact", "experience", "belief"}
        )
