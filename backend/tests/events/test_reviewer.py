import json

from moonlightbox.events.change_points import CandidateBoundary


class FakeLlm:
    def complete(self, _prompt: str) -> str:
        return json.dumps(
            {
                "type": "cold_war",
                "start_message_id": "m1",
                "end_message_id": "m2",
                "before_state": "仍在沟通",
                "after_state": "停止回复",
                "emotion_labels": ["失望", "疏远"],
                "topic": "未解决冲突",
                "conflict_level": 4,
                "importance": 0.9,
                "reason": "回复突然中断",
                "evidence_ids": ["m1", "m2"],
            }
        )


def test_reviewer_returns_structured_event_with_valid_evidence() -> None:
    from moonlightbox.events.reviewer import JsonEventReviewer

    candidate = CandidateBoundary(0, 1, 0.8, {}, ["m1", "m2"])

    reviewed = JsonEventReviewer(FakeLlm()).review(candidate, {"m1": "争吵", "m2": "沉默"})

    assert reviewed is not None
    assert reviewed.type == "cold_war"
    assert reviewed.evidence_ids == ["m1", "m2"]


def test_reviewer_rejects_evidence_not_present_in_source() -> None:
    from moonlightbox.events.reviewer import JsonEventReviewer

    class HallucinatingLlm(FakeLlm):
        def complete(self, prompt: str) -> str:
            payload = json.loads(super().complete(prompt))
            payload["evidence_ids"] = ["m99"]
            return json.dumps(payload)

    candidate = CandidateBoundary(0, 1, 0.8, {}, ["m1", "m2"])

    reviewed = JsonEventReviewer(HallucinatingLlm()).review(candidate, {"m1": "争吵", "m2": "沉默"})

    assert reviewed is None
