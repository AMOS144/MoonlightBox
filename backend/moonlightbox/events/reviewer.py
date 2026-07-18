import json
from typing import Protocol

from pydantic import ValidationError

from moonlightbox.events.change_points import CandidateBoundary
from moonlightbox.events.schemas import ReviewedEvent


class LlmClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class JsonEventReviewer:
    def __init__(self, client: LlmClient) -> None:
        self._client = client

    def review(
        self,
        candidate: CandidateBoundary,
        context: dict[str, str],
    ) -> ReviewedEvent | None:
        prompt = json.dumps(
            {
                "任务": "判断这是否为关系转折点，并只返回 JSON",
                "候选分数": candidate.score,
                "信号": candidate.signals,
                "对话": context,
            },
            ensure_ascii=False,
        )
        try:
            event = ReviewedEvent.model_validate_json(self._client.complete(prompt))
        except ValidationError:
            return None

        allowed_evidence = set(candidate.evidence_ids) & set(context)
        if not event.evidence_ids or not set(event.evidence_ids).issubset(allowed_evidence):
            return None
        if event.start_message_id not in context or event.end_message_id not in context:
            return None
        return event
