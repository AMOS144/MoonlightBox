from moonlightbox.branches.context import ContextMemory, ContextPacket
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn


def _packet() -> ContextPacket:
    return ContextPacket(
        persona="洪欣羽",
        cutoff="2026-04-20T00:00:00",
        history=(),
        current_user_content="明天去哪里",
        memories=(ContextMemory("event-1", "event", "明天去韩国"),),
        allowed_sticker_ids=(),
    )


def test_memory_persona_case_requires_grounded_memory_use() -> None:
    from moonlightbox.evaluation.memory_persona import (
        MemoryPersonaExpectation,
        evaluate_memory_persona_case,
    )

    result = evaluate_memory_persona_case(
        _packet(),
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="韩国！", delay_ms=0),),
            raw_output="韩国！",
        ),
        MemoryPersonaExpectation(case_id="trip", required_any=("韩国",)),
    )

    assert result.passed is True


def test_memory_persona_report_fails_on_forbidden_or_missing_content() -> None:
    from moonlightbox.evaluation.memory_persona import (
        MemoryPersonaExpectation,
        build_memory_persona_report,
        evaluate_memory_persona_case,
    )

    result = evaluate_memory_persona_case(
        _packet(),
        GeneratedReplyTurn(
            bubbles=(GeneratedBubble(content="去首尔以前一起走过的街", delay_ms=0),),
            raw_output="",
        ),
        MemoryPersonaExpectation(
            case_id="trip",
            required_any=("韩国",),
            forbidden=("以前一起",),
        ),
    )
    report = build_memory_persona_report([result])

    assert result.passed is False
    assert report.passed is False
    assert report.pass_rate == 0.0
