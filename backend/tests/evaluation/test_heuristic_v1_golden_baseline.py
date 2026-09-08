import json
from copy import deepcopy
from pathlib import Path

import pytest
from moonlightbox.evaluation.golden import (
    evaluate_heuristic_v1_baseline,
    load_event_gold_dataset,
)
from moonlightbox.events.validation import ALLOWED_EVENT_TYPES

FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "events" / "important_event_detection_v2.json"
)


def test_gold_dataset_covers_required_relationship_changes_and_noise() -> None:
    dataset = load_event_gold_dataset(FIXTURE_PATH)

    assert {
        "ordinary-long-gap",
        "brief-quarrel",
        "sustained-conflict",
        "growing-alienation",
        "relationship-upgrade",
        "reconciliation",
        "protocol-noise",
    } <= {case.case_id for case in dataset.cases}
    assert {
        "持续争吵",
        "持续疏远",
        "关系升级",
        "和解",
    } <= {node.label for case in dataset.cases for node in case.expected_nodes}
    assert all(
        node.evidence_ids and node.start_message_id and node.end_message_id
        for case in dataset.cases
        for node in case.expected_nodes
    )
    assert {
        node.type for case in dataset.cases for node in case.expected_nodes
    } <= ALLOWED_EVENT_TYPES


def test_heuristic_v1_baseline_explicitly_stays_below_v2_acceptance_target() -> None:
    dataset = load_event_gold_dataset(FIXTURE_PATH)

    baseline = evaluate_heuristic_v1_baseline(dataset)

    assert baseline.metrics.precision < dataset.precision_gate
    assert baseline.predicted_types_by_case["ordinary-long-gap"] == ("long_pause",)
    assert baseline.predicted_evidence_by_case["ordinary-long-gap"] == (
        frozenset({"ordinary-001", "ordinary-002"}),
    )
    assert baseline.exposed_noise_source_ids == frozenset({"noise-001", "noise-002"})


def test_heuristic_v1_baseline_uses_production_limit_and_event_writes() -> None:
    dataset = load_event_gold_dataset(FIXTURE_PATH)

    baseline = evaluate_heuristic_v1_baseline(dataset, maximum_events=1)

    assert baseline.predicted_evidence_by_case["sustained-conflict"] == (
        frozenset({"conflict-004", "conflict-005"}),
    )
    assert all(
        versions == ("heuristic-v1",) * len(baseline.predicted_types_by_case[case_id])
        for case_id, versions in baseline.analysis_versions_by_case.items()
    )


def test_gold_dataset_rejects_invalid_runtime_contract(tmp_path: Path) -> None:
    source = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    invalid_samples: list[tuple[dict[str, object], str]] = []

    wrong_schema = deepcopy(source)
    wrong_schema["schema_version"] = "unknown"
    invalid_samples.append((wrong_schema, "schema_version 必须是 important-event-gold-v1"))

    invalid_gate = deepcopy(source)
    invalid_gate["precision_gate"] = 1.1
    invalid_samples.append((invalid_gate, "precision_gate 必须在 0 到 1 之间"))

    invalid_timestamp = deepcopy(source)
    invalid_timestamp["cases"][0]["messages"][0]["timestamp"] = "不是时间"
    invalid_samples.append((invalid_timestamp, "timestamp 必须是可解析的 ISO 8601 时间"))

    unordered = deepcopy(source)
    unordered["cases"][0]["messages"].reverse()
    invalid_samples.append((unordered, "消息必须按 \\(timestamp, source_id\\) 升序排列"))

    duplicate_source_id = deepcopy(source)
    duplicate_source_id["cases"][0]["messages"][1]["source_id"] = "ordinary-001"
    invalid_samples.append((duplicate_source_id, "source_id 必须唯一"))

    duplicate_global_source_id = deepcopy(source)
    duplicate_global_source_id["cases"][1]["messages"][0]["source_id"] = "ordinary-001"
    invalid_samples.append((duplicate_global_source_id, "整个黄金样例中的 source_id 必须唯一"))

    invalid_kind = deepcopy(source)
    invalid_kind["cases"][0]["messages"][0]["message_type"] = "location"
    invalid_samples.append((invalid_kind, "message_type 不合法"))

    invalid_node_type = deepcopy(source)
    invalid_node_type["cases"][2]["expected_nodes"][0]["type"] = "long_pause"
    invalid_samples.append((invalid_node_type, "节点 type 不合法"))

    empty_evidence = deepcopy(source)
    empty_evidence["cases"][2]["expected_nodes"][0]["evidence_ids"] = []
    invalid_samples.append((empty_evidence, "节点 evidence_ids 不能为空"))

    missing_evidence = deepcopy(source)
    missing_evidence["cases"][2]["expected_nodes"][0]["evidence_ids"] = ["missing"]
    invalid_samples.append((missing_evidence, "节点 evidence_ids 引用了不存在的 source_id"))

    invalid_range = deepcopy(source)
    invalid_range["cases"][2]["expected_nodes"][0]["start_message_id"] = "conflict-005"
    invalid_range["cases"][2]["expected_nodes"][0]["end_message_id"] = "conflict-001"
    invalid_samples.append((invalid_range, "节点起止范围无效"))

    for index, (payload, expected_error) in enumerate(invalid_samples):
        fixture = tmp_path / f"invalid-{index}.json"
        fixture.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        with pytest.raises(ValueError, match=expected_error):
            load_event_gold_dataset(fixture)


def test_gold_dataset_reports_malformed_json_in_chinese(tmp_path: Path) -> None:
    fixture = tmp_path / "broken.json"
    fixture.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="黄金样例 JSON 无法解析"):
        load_event_gold_dataset(fixture)


def test_gold_dataset_rejects_reverse_source_ids_at_same_timestamp(
    tmp_path: Path,
) -> None:
    source = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    messages = source["cases"][0]["messages"]
    messages[0]["timestamp"] = "2026-01-01T21:00:00"
    messages[0]["source_id"] = "same-time-002"
    messages[1]["timestamp"] = "2026-01-01T21:00:00"
    messages[1]["source_id"] = "same-time-001"
    fixture = tmp_path / "reverse-source-id.json"
    fixture.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="消息必须按 \\(timestamp, source_id\\) 升序排列"):
        load_event_gold_dataset(fixture)


def test_gold_dataset_rejects_mixed_timezone_awareness_in_chinese(
    tmp_path: Path,
) -> None:
    source = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    source["cases"][0]["messages"][0]["timestamp"] = "2026-01-01T21:00:00+08:00"
    fixture = tmp_path / "mixed-timezone.json"
    fixture.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="timestamp 不得混用带时区和不带时区"):
        load_event_gold_dataset(fixture)
