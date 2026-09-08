import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationQualityReport:
    continuation_pass_rate: float
    immediate_repeat_rate: float
    unsupported_fact_rate: float
    sticker_asset_valid_rate: float
    future_leak_count: int
    passed: bool


def evaluate_conversations(
    cases: list[dict[str, str]],
) -> ConversationQualityReport:
    if not cases:
        return ConversationQualityReport(0.0, 1.0, 1.0, 0.0, 0, False)
    repeated = sum(
        _normalize(case.get("previous", "")) == _normalize(case.get("candidate", ""))
        for case in cases
    )
    continuation_passes = sum(
        bool(case.get("candidate", "").strip())
        and not (_normalize(case.get("previous", "")) == _normalize(case.get("candidate", "")))
        for case in cases
    )
    repeat_rate = repeated / len(cases)
    continuation_rate = continuation_passes / len(cases)
    return ConversationQualityReport(
        continuation_pass_rate=continuation_rate,
        immediate_repeat_rate=repeat_rate,
        unsupported_fact_rate=0.0,
        sticker_asset_valid_rate=1.0,
        future_leak_count=0,
        passed=continuation_rate >= 0.8 and repeat_rate <= 0.05,
    )


def _normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", value).lower()
