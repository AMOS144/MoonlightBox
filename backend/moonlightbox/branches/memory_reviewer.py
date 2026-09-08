import json

import httpx
from pydantic import ValidationError

from moonlightbox.branches.continuity_models import BranchMemoryEpisode
from moonlightbox.branches.continuity_types import (
    MemoryProposal,
    MemoryReviewResult,
    StateDeltaProposal,
)
from moonlightbox.branches.memory_proposer import _episode_evidence_ids


class MemoryReviewFailedError(RuntimeError):
    pass


class ConservativeLocalMemoryReviewer:
    """Fail-closed evidence review used when no cloud reviewer is configured.

    The local path may retain grounded memories, but it never authorizes a
    persistent state delta. Higher-level reflections additionally require at
    least two already-approved source memories.
    """

    def review(
        self,
        episode: BranchMemoryEpisode,
        proposal: MemoryProposal,
        *,
        identity_kernel: dict[str, object] | None = None,
        current_state: dict[str, object] | None = None,
        competing_beliefs: tuple[str, ...] = (),
    ) -> MemoryReviewResult:
        del identity_kernel, competing_beliefs
        allowed_evidence = _episode_evidence_ids(episode)
        approved: list[int] = []
        reflection_inputs = (current_state or {}).get("reflection_inputs", [])
        for index, candidate in enumerate(proposal.candidates):
            evidence = set(candidate.evidence_message_ids)
            if not evidence or not evidence.issubset(allowed_evidence):
                continue
            if candidate.confidence > 0.9:
                continue
            if candidate.kind == "fact":
                if candidate.source_role != "user" or episode.user_turn_id not in evidence:
                    continue
            elif candidate.kind == "self_narrative":
                if (
                    candidate.source_role != "digital_human"
                    or episode.assistant_turn_id not in evidence
                ):
                    continue
            elif candidate.kind == "belief":
                if candidate.source_role not in {"user", "interaction"}:
                    continue
            elif candidate.kind == "experience":
                if candidate.source_role != "interaction":
                    continue
            elif candidate.kind == "reflection":
                if not isinstance(reflection_inputs, list) or len(reflection_inputs) < 2:
                    continue
            approved.append(index)
        return MemoryReviewResult(
            verdict="approve" if approved else "reject",
            approved_candidate_indexes=tuple(approved),
            approved_state_delta=None,
            rejected_reasons=(
                () if approved else ("本地保守复核未找到足够直接证据",)
            ),
        )


class DeepSeekMemoryReviewer:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 30,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.Client()

    def review(
        self,
        episode: BranchMemoryEpisode,
        proposal: MemoryProposal,
        *,
        identity_kernel: dict[str, object] | None = None,
        current_state: dict[str, object] | None = None,
        competing_beliefs: tuple[str, ...] = (),
    ) -> MemoryReviewResult:
        allowed_evidence = _episode_evidence_ids(episode)
        if any(
            not set(candidate.evidence_message_ids).issubset(allowed_evidence)
            for candidate in proposal.candidates
        ):
            raise MemoryReviewFailedError("候选记忆引用了本轮之外的证据")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是分支持续人格记忆的证据复核器。"
                        "只检查候选是否忠于给定本轮证据、角色归因是否正确、"
                        "是否越过稳定人格边界。"
                        "你只能删除候选或缩小状态变化，绝不能新增候选、"
                        "新增事实或扩大变化。只返回 JSON，格式必须是："
                        '{"verdict":"approve或reject",'
                        '"approved_candidate_indexes":[0],'
                        '"approved_state_delta":null,'
                        '"rejected_reasons":[]}。'
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "episode": {
                                "user_turn_id": episode.user_turn_id,
                                "user_messages": episode.user_messages,
                                "assistant_turn_id": episode.assistant_turn_id,
                                "user_content": episode.user_content,
                                "assistant_bubbles": episode.assistant_bubbles,
                            },
                            "proposal": proposal.model_dump(mode="json"),
                            "identity_kernel": identity_kernel or {},
                            "current_state": current_state or {},
                            "competing_beliefs": list(competing_beliefs),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        try:
            response = self._client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            result = MemoryReviewResult.model_validate(
                _normalize_review_payload(json.loads(content))
            )
            self._validate_review(proposal, result)
            return result
        except MemoryReviewFailedError:
            raise
        except (
            httpx.HTTPError,
            ValidationError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            raise MemoryReviewFailedError("云端记忆证据复核失败") from error

    def _validate_review(
        self,
        proposal: MemoryProposal,
        result: MemoryReviewResult,
    ) -> None:
        candidate_count = len(proposal.candidates)
        approved = set(result.approved_candidate_indexes)
        if any(index < 0 or index >= candidate_count for index in approved):
            raise MemoryReviewFailedError("复核器尝试新增候选记忆")
        if result.verdict == "reject":
            if approved or result.approved_state_delta is not None:
                raise MemoryReviewFailedError("拒绝结果不能包含已批准变化")
            return
        delta = result.approved_state_delta
        if delta is None:
            return
        if (
            delta.relationship_delta
            or delta.emotional_delta
            or delta.user_model_updates
        ) and not delta.supporting_candidate_indexes:
            raise MemoryReviewFailedError("状态变化必须引用已批准证据")
        if not set(delta.supporting_candidate_indexes).issubset(approved):
            raise MemoryReviewFailedError("状态变化引用了未批准候选")
        self._validate_smaller_delta(proposal.state_delta, delta)

    def _validate_smaller_delta(
        self,
        original: StateDeltaProposal,
        reviewed: StateDeltaProposal,
    ) -> None:
        for reviewed_values, original_values in (
            (reviewed.relationship_delta, original.relationship_delta),
            (reviewed.emotional_delta, original.emotional_delta),
        ):
            for key, value in reviewed_values.items():
                original_value = original_values.get(key)
                if (
                    original_value is None
                    or value * original_value < 0
                    or abs(value) > abs(original_value)
                ):
                    raise MemoryReviewFailedError("复核器扩大了状态变化")
        for key, state_value in reviewed.user_model_updates.items():
            if original.user_model_updates.get(key) != state_value:
                raise MemoryReviewFailedError("复核器新增了用户模型判断")


def _normalize_review_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("记忆复核结果必须是 JSON 对象")
    normalized = {str(key): item for key, item in value.items()}
    normalized.setdefault("approved_candidate_indexes", [])
    normalized.setdefault("approved_state_delta", None)
    normalized.setdefault("rejected_reasons", [])
    if "verdict" not in normalized:
        normalized["verdict"] = "approve"
    return normalized
