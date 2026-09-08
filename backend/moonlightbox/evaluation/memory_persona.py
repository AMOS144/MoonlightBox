from dataclasses import dataclass

from moonlightbox.branches.context import ContextPacket
from moonlightbox.branches.replies import GeneratedReplyTurn
from moonlightbox.branches.understanding import validate_grounded_reply


@dataclass(frozen=True)
class MemoryPersonaExpectation:
    case_id: str
    required_any: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryPersonaCaseResult:
    case_id: str
    passed: bool
    reply: str
    failures: tuple[str, ...]


@dataclass(frozen=True)
class MemoryPersonaReport:
    results: tuple[MemoryPersonaCaseResult, ...]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.passed for result in self.results) / len(self.results)

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)


def evaluate_memory_persona_case(
    packet: ContextPacket,
    reply: GeneratedReplyTurn,
    expectation: MemoryPersonaExpectation,
) -> MemoryPersonaCaseResult:
    text = "\n".join(bubble.content or "" for bubble in reply.bubbles)
    failures: list[str] = []
    try:
        validate_grounded_reply(packet, reply)
    except ValueError as error:
        failures.append(str(error))
    if expectation.required_any and not any(
        marker in text for marker in expectation.required_any
    ):
        failures.append("未使用本题要求的可信记忆或状态")
    used_forbidden = tuple(marker for marker in expectation.forbidden if marker in text)
    if used_forbidden:
        failures.append("使用了禁止内容：" + "、".join(used_forbidden))
    return MemoryPersonaCaseResult(
        case_id=expectation.case_id,
        passed=not failures,
        reply=text,
        failures=tuple(failures),
    )


def build_memory_persona_report(
    results: list[MemoryPersonaCaseResult],
) -> MemoryPersonaReport:
    return MemoryPersonaReport(results=tuple(results))
