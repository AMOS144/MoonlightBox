import json
from pathlib import Path

import pytest
from moonlightbox.evaluation.golden import load_v3_event_gold_dataset

FIXTURE = Path(__file__).parents[1] / "fixtures" / "events" / "important_event_detection_v3.json"


def test_v3_golden_fixture_covers_both_lanes_and_negative_cases() -> None:
    dataset = load_v3_event_gold_dataset(FIXTURE)
    nodes = [node for case in dataset.cases for node in case.expected_nodes]

    assert dataset.schema_version == "important-event-gold-v3"
    assert dataset.precision_gate == 0.8
    assert {node.lane for node in nodes} == {
        "relationship",
        "shared_experience",
    }
    assert {"travel", "date", "celebration", "family_social", "support_care"} <= {
        node.type for node in nodes
    }
    assert any(not case.expected_nodes for case in dataset.cases)
    assert any(case.excluded_from_display for case in dataset.cases)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update({"schema_version": "important-event-gold-v1"}),
            "important-event-gold-v3",
        ),
        (
            lambda payload: payload["cases"][0]["expected_nodes"][0].update(
                {"lane": "relationship", "type": "travel"}
            ),
            "type 与 lane",
        ),
        (
            lambda payload: payload["cases"][0]["messages"].reverse(),
            "升序",
        ),
        (
            lambda payload: payload["cases"][2]["messages"][0].update({"source_id": "travel-1"}),
            "全局唯一",
        ),
    ],
)
def test_v3_golden_rejects_invalid_contract(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutation(payload)  # type: ignore[operator]
    path = tmp_path / "invalid-v3.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_v3_event_gold_dataset(path)
