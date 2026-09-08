from datetime import datetime

from moonlightbox.graph.models import TemporalFact
from moonlightbox.imports.types import ImportedMessage, MessageKind


def test_snapshot_and_context_use_only_information_valid_at_node_time() -> None:
    from moonlightbox.timeline.state import StateBuilder, StateTransition

    transitions = [
        StateTransition(
            at=datetime(2026, 1, 1),
            relationship_status="朋友",
            emotions={"她": "开心"},
            open_loops=["下次见面"],
            evidence_ids=["m1"],
        ),
        StateTransition(
            at=datetime(2026, 3, 1),
            relationship_status="恋人",
            emotions={"她": "依恋"},
            open_loops=[],
            evidence_ids=["m99"],
        ),
    ]

    snapshot = StateBuilder().build(transitions, datetime(2026, 2, 1))

    assert snapshot.relationship_status == "朋友"
    assert snapshot.evidence_ids == ["m1"]


def test_context_applies_message_and_fact_time_filters() -> None:
    from moonlightbox.timeline.context import TemporalContextBuilder

    messages = [
        ImportedMessage(
            "m1",
            datetime(2026, 1, 1),
            "她",
            MessageKind.TEXT,
            "过去消息",
            {},
        ),
        ImportedMessage(
            "m2",
            datetime(2026, 3, 1),
            "她",
            MessageKind.TEXT,
            "未来消息",
            {},
        ),
    ]
    facts = [
        TemporalFact("f1", "她", "关系", "朋友", datetime(2026, 1, 1), None, ["m1"]),
        TemporalFact("f2", "她", "关系", "恋人", datetime(2026, 3, 1), None, ["m2"]),
    ]

    context = TemporalContextBuilder().build(messages, facts, datetime(2026, 2, 1))

    assert [message.source_id for message in context.messages] == ["m1"]
    assert [fact.id for fact in context.facts] == ["f1"]


def test_prompt_uses_snapshot_as_historical_persona_state() -> None:
    from moonlightbox.timeline.prompt import build_persona_prompt
    from moonlightbox.timeline.state import StateSnapshot

    snapshot = StateSnapshot(
        at=datetime(2026, 2, 1),
        relationship_status="朋友",
        emotions={"她": "期待"},
        open_loops=["下次见面"],
        evidence_ids=["m1"],
    )

    prompt = build_persona_prompt(snapshot, style_instruction="保持原有口头禅")

    assert "朋友" in prompt
    assert "期待" in prompt
    assert "保持原有口头禅" in prompt
