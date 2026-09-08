import copy
import json
import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from moonlightbox.db import Database
from moonlightbox.evaluation import node_acceptance
from moonlightbox.evaluation.node_acceptance import (
    NodeAcceptanceOutputError,
    NodeAcceptanceReviewError,
    NodeAcceptanceScopeError,
    _write_output,
    build_review_packet,
    evaluate_review_packet,
    main,
    render_review_packet_markdown,
)
from moonlightbox.evaluation.node_metrics import evaluate_nodes
from moonlightbox.events.config import config_fingerprint, window_manifest_fingerprint
from moonlightbox.events.models import (
    AnalysisRevision,
    AnalysisRun,
    EventCandidate,
    EventNode,
)
from moonlightbox.events.service import EventService
from moonlightbox.imports.analysis_job import ANALYSIS_PROMPT_VERSION
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from sqlalchemy import event, insert
from sqlalchemy.orm import Session


def _event_snapshot(
    *,
    index: int,
    source_id: str,
) -> dict[str, object]:
    return {
        "after_state": f"发生冲突 {index}",
        "before_state": f"关系稳定 {index}",
        "conflict_level": 4,
        "emotion_labels": ["失望"],
        "end_message_id": source_id,
        "evidence_ids": [source_id],
        "importance": 0.9 - index / 1000,
        "reason": "关系状态发生变化",
        "start_message_id": source_id,
        "status": "active",
        "topic": "联系频率",
        "type": "conflict",
    }


def _seed_run(
    tmp_path: Path,
    *,
    node_count: int,
    evidence_contents: list[str] | None = None,
    unrelated_node_count: int = 0,
) -> tuple[Database, str, str]:
    database = Database(f"sqlite:///{tmp_path / 'acceptance.db'}")
    database.create_schema()
    now = datetime.now(UTC)
    with Session(database.engine) as session:
        project = Project(name="真实项目")
        session.add(project)
        session.flush()
        imported = ImportSource(
            project_id=project.id,
            preview_id="preview-1",
            source_path="/本地/聊天.csv",
            message_count=node_count,
            confirmed_at=now,
        )
        participant = Participant(project_id=project.id, name="甲", role="target")
        session.add_all([imported, participant])
        session.flush()
        window_ids = ["window-1"]
        config = {"maximum_nodes": 25}
        run = AnalysisRun(
            project_id=project.id,
            import_id=imported.id,
            analysis_version="hybrid-v2",
            prompt_version=ANALYSIS_PROMPT_VERSION,
            model="test-model",
            config=config,
            config_fingerprint=config_fingerprint(config),
            window_ids=window_ids,
            window_manifest_fingerprint=window_manifest_fingerprint(window_ids),
            status="succeeded",
            total_windows=1,
            completed_windows=1,
            checkpoint=1,
            completed_at=now,
        )
        session.add(run)
        session.flush()
        for index in range(node_count):
            source_id = f"message-{index}"
            content = (
                evidence_contents[index]
                if evidence_contents is not None and index < len(evidence_contents)
                else f"支持节点的消息 {index}"
            )
            session.add(
                Message(
                    project_id=project.id,
                    import_id=imported.id,
                    participant_id=participant.id,
                    source_id=source_id,
                    timestamp=now,
                    kind="text",
                    content=content,
                    raw={
                        "xml": "<msg>raw_secret</msg>",
                        "api_key": "never-export",
                    },
                )
            )
            scores = {
                "state_change_strength": 0.9,
                "persistence": 0.8,
                "evidence_quality": 1.0,
                "model_confidence": 0.9,
                "total": 0.9 - index / 1000,
            }
            candidate = EventCandidate(
                run_id=run.id,
                window_id="window-1",
                candidate_key=f"candidate-{index}",
                raw_payload={"secret": "<raw>never export</raw>"},
                review_payload={"accepted": True},
                status="accepted",
                rejection_reason=None,
                scores=scores,
            )
            session.add(candidate)
            session.flush()
            snapshot = _event_snapshot(index=index, source_id=source_id)
            event_node = EventNode(
                project_id=project.id,
                type=str(snapshot["type"]),
                start_message_id=source_id,
                end_message_id=source_id,
                before_state=str(snapshot["before_state"]),
                after_state=str(snapshot["after_state"]),
                emotion_labels=["失望"],
                topic="联系频率",
                conflict_level=4,
                importance=float(snapshot["importance"]),
                reason="关系状态发生变化",
                evidence_ids=[source_id],
                status="active",
            )
            session.add(event_node)
            session.flush()
            session.add(
                AnalysisRevision(
                    event_id=event_node.id,
                    revision_number=1,
                    snapshot=snapshot,
                    action_reason="V2 自动发布",
                    analysis_version="hybrid-v2",
                    prompt_version=ANALYSIS_PROMPT_VERSION,
                    model="test-model",
                    run_id=run.id,
                    candidate_id=candidate.id,
                )
            )
        if unrelated_node_count:
            session.execute(
                insert(EventNode),
                [
                    {
                        "id": f"unrelated-{index}",
                        "project_id": project.id,
                        "type": "conflict",
                        "start_message_id": f"other-{index}",
                        "end_message_id": f"other-{index}",
                        "before_state": "其他",
                        "after_state": "其他",
                        "emotion_labels": [],
                        "topic": "其他",
                        "conflict_level": 0,
                        "importance": 0.1,
                        "reason": "其他",
                        "evidence_ids": [],
                        "status": "active",
                        "created_at": now,
                    }
                    for index in range(unrelated_node_count)
                ],
            )
        session.commit()
        return database, project.id, run.id


def _set_verdicts(
    packet: dict[str, Any],
    *,
    accepted: int,
    rejected: int,
) -> dict[str, Any]:
    for index, node in enumerate(packet["nodes"]):
        if index < accepted:
            node["verdict"] = "accepted"
        elif index < accepted + rejected:
            node["verdict"] = "rejected"
        else:
            node["verdict"] = "pending"
    return packet


def test_packet_uses_independent_pending_verdict_and_v2_manifest(tmp_path: Path) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=2)
    with Session(database.engine) as session:
        packet = build_review_packet(session, project_id=project_id, run_id=run_id)

    payload = packet.model_dump(mode="json")
    assert packet.schema_version == "node-acceptance-v2"
    assert packet.analysis_run_id == run_id
    assert packet.export_manifest.node_count == 2
    assert packet.export_manifest.node_ids == [node.node_id for node in packet.nodes]
    assert packet.export_manifest.hash_algorithm == "sha256"
    assert all(node.verdict == "pending" for node in packet.nodes)
    assert all("status" not in node for node in payload["nodes"])
    assert all(len(node.canonical_hash) == 64 for node in packet.nodes)
    assert "threshold" not in payload
    database.close()


@pytest.mark.parametrize(
    ("accepted", "rejected", "expected_status", "expected_precision"),
    [(16, 4, "pass", 0.8), (15, 5, "fail", 0.75)],
)
def test_precision_gate_calls_node_metrics_with_all_predicted_and_accepted_gold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted: int,
    rejected: int,
    expected_status: str,
    expected_precision: float,
) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=20)
    observed: dict[str, object] = {}

    def spy(predicted: list[Any], gold: list[Any]) -> Any:
        observed["predicted"] = predicted
        observed["gold"] = gold
        return evaluate_nodes(predicted, gold)

    monkeypatch.setattr(node_acceptance, "evaluate_nodes", spy)
    with Session(database.engine) as session:
        packet = build_review_packet(session, project_id=project_id, run_id=run_id)
        result = evaluate_review_packet(
            session,
            project_id=project_id,
            run_id=run_id,
            reviewed_packet=_set_verdicts(
                packet.model_dump(mode="json"),
                accepted=accepted,
                rejected=rejected,
            ),
        )

    assert len(observed["predicted"]) == 20  # type: ignore[arg-type]
    assert len(observed["gold"]) == accepted  # type: ignore[arg-type]
    assert result.status == expected_status
    assert result.precision == expected_precision
    assert result.threshold == 0.8
    assert result.metric_scope == "precision_only"
    assert result.recall is None
    assert result.f1 is None
    assert "召回率" in result.metric_note
    database.close()


def test_incomplete_verdicts_are_pending(tmp_path: Path) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=20)
    with Session(database.engine) as session:
        packet = build_review_packet(session, project_id=project_id, run_id=run_id)
        result = evaluate_review_packet(
            session,
            project_id=project_id,
            run_id=run_id,
            reviewed_packet=_set_verdicts(
                packet.model_dump(mode="json"),
                accepted=19,
                rejected=0,
            ),
        )

    assert result.status == "pending"
    assert result.pending_nodes == 1
    database.close()


def test_zero_node_run_is_pending_without_precision(tmp_path: Path) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=0)
    with Session(database.engine) as session:
        packet = build_review_packet(session, project_id=project_id, run_id=run_id)
        result = evaluate_review_packet(
            session,
            project_id=project_id,
            run_id=run_id,
            reviewed_packet=packet.model_dump(mode="json"),
        )

    assert packet.nodes == []
    assert result.status == "pending"
    assert result.precision is None
    database.close()


def test_export_uses_run_revision_snapshot_after_current_event_is_revised(
    tmp_path: Path,
) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=1)
    with Session(database.engine) as session:
        original = build_review_packet(session, project_id=project_id, run_id=run_id)
        EventService(session).revise(
            original.nodes[0].node_id,
            {
                "type": "reconciliation",
                "before_state": "已被人工修改",
                "after_state": "当前状态已变化",
                "evidence_ids": ["tampered-current-id"],
                "importance": 0.1,
            },
            "人工修改当前节点",
            project_id=project_id,
        )
        replayed = build_review_packet(session, project_id=project_id, run_id=run_id)

    assert replayed.nodes == original.nodes
    assert replayed.export_manifest.node_ids == original.export_manifest.node_ids
    database.close()


def test_packet_reads_score_components_and_versions_from_bound_candidate(
    tmp_path: Path,
) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=1)
    with Session(database.engine) as session:
        node = build_review_packet(
            session,
            project_id=project_id,
            run_id=run_id,
        ).nodes[0]

    assert node.score == 0.9
    assert node.score_components.model_dump() == {
        "state_change_strength": 0.9,
        "persistence": 0.8,
        "evidence_quality": 1.0,
        "model_confidence": 0.9,
    }
    assert node.version.model_dump() == {
        "analysis_version": "hybrid-v2",
        "prompt_version": ANALYSIS_PROMPT_VERSION,
        "model": "test-model",
    }
    database.close()


def test_any_node_field_tampering_is_rejected_by_canonical_hash(tmp_path: Path) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=2)
    with Session(database.engine) as session:
        original = build_review_packet(
            session,
            project_id=project_id,
            run_id=run_id,
        ).model_dump(mode="json")
        mutations: list[dict[str, Any]] = []
        for field, value in (
            ("type", "reconciliation"),
            ("before_state", "篡改前状态"),
            ("after_state", "篡改后状态"),
            ("score", 0.1),
            ("evidence_ids", ["other-id"]),
        ):
            changed = copy.deepcopy(original)
            changed["nodes"][0][field] = value
            mutations.append(changed)
        changed_components = copy.deepcopy(original)
        changed_components["nodes"][0]["score_components"]["persistence"] = 0.1
        mutations.append(changed_components)
        changed_version = copy.deepcopy(original)
        changed_version["nodes"][0]["version"]["model"] = "other-model"
        mutations.append(changed_version)
        changed_summary = copy.deepcopy(original)
        changed_summary["nodes"][0]["evidence_summary"][0]["content"] = "伪造证据"
        mutations.append(changed_summary)

        for reviewed in mutations:
            reviewed["nodes"][0]["verdict"] = "accepted"
            with pytest.raises(NodeAcceptanceReviewError, match="完整性"):
                evaluate_review_packet(
                    session,
                    project_id=project_id,
                    run_id=run_id,
                    reviewed_packet=reviewed,
                )
    database.close()


def test_node_deletion_addition_duplication_and_reordering_are_rejected(
    tmp_path: Path,
) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=2)
    with Session(database.engine) as session:
        original = build_review_packet(
            session,
            project_id=project_id,
            run_id=run_id,
        ).model_dump(mode="json")
        tampered_packets: list[dict[str, Any]] = []
        deleted = copy.deepcopy(original)
        deleted["nodes"].pop()
        tampered_packets.append(deleted)
        added = copy.deepcopy(original)
        added["nodes"].append(copy.deepcopy(added["nodes"][0]))
        added["nodes"][-1]["node_id"] = "added-node"
        tampered_packets.append(added)
        duplicated = copy.deepcopy(original)
        duplicated["nodes"].append(copy.deepcopy(duplicated["nodes"][0]))
        tampered_packets.append(duplicated)
        reordered = copy.deepcopy(original)
        reordered["nodes"].reverse()
        tampered_packets.append(reordered)

        for reviewed in tampered_packets:
            with pytest.raises(NodeAcceptanceReviewError, match="完整性|格式"):
                evaluate_review_packet(
                    session,
                    project_id=project_id,
                    run_id=run_id,
                    reviewed_packet=reviewed,
                )
    database.close()


def test_threshold_only_comes_from_evaluator_argument(tmp_path: Path) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=20)
    with Session(database.engine) as session:
        packet = build_review_packet(session, project_id=project_id, run_id=run_id)
        reviewed = _set_verdicts(
            packet.model_dump(mode="json"),
            accepted=16,
            rejected=4,
        )
        reviewed["threshold"] = 0.1
        with pytest.raises(NodeAcceptanceReviewError, match="格式"):
            evaluate_review_packet(
                session,
                project_id=project_id,
                run_id=run_id,
                reviewed_packet=reviewed,
                threshold=0.9,
            )
        reviewed.pop("threshold")
        result = evaluate_review_packet(
            session,
            project_id=project_id,
            run_id=run_id,
            reviewed_packet=reviewed,
            threshold=0.9,
        )

    assert result.status == "fail"
    assert result.threshold == 0.9
    database.close()


def test_malformed_protocol_is_redacted_and_markdown_links_cannot_execute(
    tmp_path: Path,
) -> None:
    database, project_id, run_id = _seed_run(
        tmp_path,
        node_count=3,
        evidence_contents=[
            "<msg truncated protocol",
            "broken protocol > tail",
            "![x](https://evil.example/tracker.png)",
        ],
    )
    with Session(database.engine) as session:
        packet = build_review_packet(session, project_id=project_id, run_id=run_id)
        markdown = render_review_packet_markdown(packet)

    contents = [summary.content for node in packet.nodes for summary in node.evidence_summary]
    assert contents[:2] == ["内容已省略", "内容已省略"]
    assert "<msg" not in markdown
    assert "![x]" not in markdown
    assert "https://evil.example" not in markdown
    assert "图片链接已省略" in markdown
    assert all(
        line.startswith("    ")
        for line in markdown.splitlines()
        if "支持证据内容：" in line or "图片链接已省略" in line
    )
    database.close()


def test_export_queries_only_bound_run_without_loading_project_nodes(
    tmp_path: Path,
) -> None:
    database, project_id, run_id = _seed_run(
        tmp_path,
        node_count=25,
        unrelated_node_count=500,
    )
    statements: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        statements.append(statement)

    event.listen(database.engine, "before_cursor_execute", record_statement)
    try:
        with Session(database.engine) as session:
            packet = build_review_packet(
                session,
                project_id=project_id,
                run_id=run_id,
            )
            assert not any(
                isinstance(instance, EventNode) for instance in session.identity_map.values()
            )
    finally:
        event.remove(database.engine, "before_cursor_execute", record_statement)

    assert len(packet.nodes) == 25
    assert len(statements) <= 5
    assert not any(
        "FROM event_nodes" in statement and "analysis_revisions" not in statement
        for statement in statements
    )
    database.close()


def test_output_is_exclusive_utf8_and_rejects_symlink(tmp_path: Path) -> None:
    output = tmp_path / "审核包.json"
    _write_output('{"中文":"月光"}\n', output)
    assert output.read_bytes().decode("utf-8") == '{"中文":"月光"}\n'
    with pytest.raises(NodeAcceptanceOutputError, match="已存在"):
        _write_output("不能覆盖", output)
    assert output.read_text(encoding="utf-8") == '{"中文":"月光"}\n'

    target = tmp_path / "target.json"
    target.write_text("原文件", encoding="utf-8")
    symlink = tmp_path / "review-link.json"
    symlink.symlink_to(target)
    with pytest.raises(NodeAcceptanceOutputError, match="符号链接"):
        _write_output("不能跟随", symlink)
    assert target.read_text(encoding="utf-8") == "原文件"


def test_partial_output_is_removed_when_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "partial.json"
    real_write = os.write
    calls = 0

    def fail_after_partial_write(fd: int, data: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(fd, data[:2])
        raise OSError("磁盘写入失败")

    monkeypatch.setattr(node_acceptance.os, "write", fail_after_partial_write)
    with pytest.raises(NodeAcceptanceOutputError, match="写入失败"):
        _write_output("需要完整写入的内容", output)
    assert not output.exists()


def test_cli_exit_codes_and_safe_error_messages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=1)
    database_url = str(database.engine.url)
    database.close()
    output = tmp_path / "node-review.json"
    base = [
        "--database-url",
        database_url,
    ]
    export_args = base + [
        "export",
        "--project",
        project_id,
        "--run",
        run_id,
        "--output",
        str(output),
    ]
    evaluate_args = base + [
        "evaluate",
        "--project",
        project_id,
        "--run",
        run_id,
        "--input",
        str(output),
    ]
    assert main(export_args) == 0

    assert main(evaluate_args) == 2
    packet = json.loads(output.read_text(encoding="utf-8"))
    packet["nodes"][0]["verdict"] = "rejected"
    output.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
    assert main(evaluate_args) == 1
    packet["nodes"][0]["verdict"] = "accepted"
    output.write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
    assert main(evaluate_args) == 0

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{broken", encoding="utf-8")
    malformed_args = evaluate_args[:-1] + [str(malformed)]
    assert main(malformed_args) == 3
    missing_schema_url = f"sqlite:///{tmp_path / 'empty.db'}"
    assert (
        main(
            [
                "--database-url",
                missing_schema_url,
                "export",
                "--project",
                project_id,
                "--run",
                run_id,
            ]
        )
        == 3
    )
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "JSON" in captured.err
    assert "数据库" in captured.err


def test_wrong_project_or_run_is_rejected(tmp_path: Path) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=1)
    with Session(database.engine) as session:
        with pytest.raises(NodeAcceptanceScopeError, match="项目不存在"):
            build_review_packet(session, project_id="missing-project", run_id=run_id)
        with pytest.raises(NodeAcceptanceScopeError, match="分析运行不属于指定项目"):
            build_review_packet(session, project_id=project_id, run_id="missing-run")
    database.close()


def _database_artifacts(path: Path) -> dict[str, tuple[bool, int | None, int | None]]:
    artifacts = [path, Path(f"{path}-wal"), Path(f"{path}-shm")]
    return {
        artifact.name: (
            artifact.exists(),
            artifact.stat().st_mtime_ns if artifact.exists() else None,
            artifact.stat().st_size if artifact.exists() else None,
        )
        for artifact in artifacts
    }


def test_cli_missing_sqlite_path_is_not_created(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing.db"
    exit_code = main(
        [
            "--database-url",
            f"sqlite:///{missing}",
            "export",
            "--project",
            "project",
            "--run",
            "run",
        ]
    )

    assert exit_code == 3
    assert not missing.exists()
    assert not Path(f"{missing}-wal").exists()
    assert not Path(f"{missing}-shm").exists()
    assert "数据库" in capsys.readouterr().err


def test_cli_reads_existing_sqlite_without_touching_database_artifacts(
    tmp_path: Path,
) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=1)
    raw_path = database.engine.url.database
    assert raw_path is not None
    database_path = Path(raw_path)
    database.close()
    maintenance = sqlite3.connect(database_path)
    assert maintenance.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
    maintenance.close()
    before = _database_artifacts(database_path)
    output = tmp_path / "readonly-review.json"

    assert (
        main(
            [
                "--database-url",
                f"sqlite:///{database_path}",
                "export",
                "--project",
                project_id,
                "--run",
                run_id,
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert output.exists()
    assert _database_artifacts(database_path) == before


def test_cli_rejects_non_regular_or_symlink_database_paths(
    tmp_path: Path,
) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=1)
    raw_path = database.engine.url.database
    assert raw_path is not None
    database_path = Path(raw_path)
    database.close()
    symlink = tmp_path / "linked.db"
    symlink.symlink_to(database_path)

    for invalid_path in (tmp_path, symlink):
        output = tmp_path / f"{invalid_path.name}-review.json"
        assert (
            main(
                [
                    "--database-url",
                    f"sqlite:///{invalid_path}",
                    "export",
                    "--project",
                    project_id,
                    "--run",
                    run_id,
                    "--output",
                    str(output),
                ]
            )
            == 3
        )
        assert not output.exists()


def test_cli_snapshots_active_wal_without_touching_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, source_project_id, source_run_id = _seed_run(tmp_path, node_count=1)
    raw_path = database.engine.url.database
    assert raw_path is not None
    database_path = Path(raw_path)
    database.close()
    writer = sqlite3.connect(database_path)
    writer.execute("PRAGMA wal_autocheckpoint=0")
    wal_project_id = "wal-project"
    wal_import_id = "wal-import"
    wal_participant_id = "wal-participant"
    wal_message_id = "wal-message"
    wal_run_id = "wal-run"
    wal_candidate_id = "wal-candidate"
    wal_event_id = "wal-event"
    wal_revision_id = "wal-revision"
    writer.execute(
        "INSERT INTO projects "
        "SELECT ?, name, status, created_at, updated_at FROM projects WHERE id = ?",
        (wal_project_id, source_project_id),
    )
    writer.execute(
        "INSERT INTO import_sources "
        "SELECT ?, ?, 'wal-preview', source_path, message_count, confirmed_at "
        "FROM import_sources WHERE project_id = ?",
        (wal_import_id, wal_project_id, source_project_id),
    )
    writer.execute(
        "INSERT INTO participants "
        "(id, project_id, name, role, avatar_asset_id) "
        "SELECT ?, ?, name, role, avatar_asset_id "
        "FROM participants WHERE project_id = ?",
        (wal_participant_id, wal_project_id, source_project_id),
    )
    writer.execute(
        "INSERT INTO messages "
        "(id, project_id, import_id, participant_id, source_id, timestamp, "
        "kind, content, raw, media_asset_id) "
        "SELECT ?, ?, ?, ?, source_id, timestamp, kind, content, raw, media_asset_id "
        "FROM messages WHERE project_id = ?",
        (
            wal_message_id,
            wal_project_id,
            wal_import_id,
            wal_participant_id,
            source_project_id,
        ),
    )
    writer.execute(
        "INSERT INTO analysis_runs "
        "SELECT ?, ?, ?, analysis_version, prompt_version, model, config, "
        "config_fingerprint, window_ids, window_manifest_fingerprint, status, "
        "total_windows, completed_windows, checkpoint, error_category, "
        "error_message, lease_owner, lease_token, lease_expires_at, created_at, "
        "updated_at, completed_at FROM analysis_runs WHERE id = ?",
        (wal_run_id, wal_project_id, wal_import_id, source_run_id),
    )
    writer.execute(
        "INSERT INTO event_candidates "
        "SELECT ?, ?, window_id, candidate_key, raw_payload, review_payload, "
        "status, rejection_reason, scores, created_at, updated_at "
        "FROM event_candidates WHERE run_id = ?",
        (wal_candidate_id, wal_run_id, source_run_id),
    )
    writer.execute(
        "INSERT INTO event_nodes ("
        "id, project_id, type, start_message_id, end_message_id, before_state, "
        "after_state, emotion_labels, topic, conflict_level, importance, reason, "
        "evidence_ids, status, created_at, lane, event_status, title, summary, "
            "started_at, ended_at, source_lanes, display_summary, summary_status, "
            "summary_model"
        ") "
        "SELECT ?, ?, type, start_message_id, end_message_id, before_state, "
        "after_state, emotion_labels, topic, conflict_level, importance, reason, "
        "evidence_ids, status, created_at, lane, event_status, title, summary, "
            "started_at, ended_at, source_lanes, display_summary, summary_status, "
            "summary_model "
        "FROM event_nodes WHERE project_id = ?",
        (wal_event_id, wal_project_id, source_project_id),
    )
    writer.execute(
        "INSERT INTO analysis_revisions "
        "SELECT ?, ?, revision_number, snapshot, action_reason, analysis_version, "
        "prompt_version, model, ?, ?, created_at FROM analysis_revisions "
        "WHERE run_id = ?",
        (
            wal_revision_id,
            wal_event_id,
            wal_run_id,
            wal_candidate_id,
            source_run_id,
        ),
    )
    writer.commit()
    writer.execute("SELECT 1")
    wal_path = Path(f"{database_path}-wal")
    shm_path = Path(f"{database_path}-shm")
    assert wal_path.exists()
    assert shm_path.exists()

    immutable_reader = sqlite3.connect(
        f"file:{database_path}?mode=ro&immutable=1",
        uri=True,
    )
    assert immutable_reader.execute(
        "SELECT count(*) FROM projects WHERE id = ?",
        (wal_project_id,),
    ).fetchone() == (0,)
    immutable_reader.close()
    before = _database_artifacts(database_path)
    output = tmp_path / "active-wal-review.json"
    temporary_paths: list[Path] = []
    traced_source_statements: list[str] = []
    real_mkstemp = tempfile.mkstemp
    real_connect = sqlite3.connect

    def tracked_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        descriptor, path = real_mkstemp(*args, **kwargs)
        temporary_paths.append(Path(path))
        return descriptor, path

    def traced_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection = real_connect(*args, **kwargs)
        if args and isinstance(args[0], str) and "mode=ro" in args[0]:
            connection.set_trace_callback(traced_source_statements.append)
        return connection

    monkeypatch.setattr(node_acceptance.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(node_acceptance.sqlite3, "connect", traced_connect)
    try:
        assert (
            main(
                [
                    "--database-url",
                    f"sqlite:///{database_path}",
                    "export",
                    "--project",
                    wal_project_id,
                    "--run",
                    wal_run_id,
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        packet = json.loads(output.read_text(encoding="utf-8"))
        assert packet["project_id"] == wal_project_id
        assert packet["analysis_run_id"] == wal_run_id
        assert _database_artifacts(database_path) == before
        assert temporary_paths
        assert all(not path.exists() for path in temporary_paths)
        assert not any(
            "journal_mode" in statement.lower() or "wal_checkpoint" in statement.lower()
            for statement in traced_source_statements
        )
    finally:
        writer.close()


def test_repeat_export_uses_stable_run_completion_time(tmp_path: Path) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=1)
    with Session(database.engine) as session:
        first = build_review_packet(session, project_id=project_id, run_id=run_id)
        second = build_review_packet(session, project_id=project_id, run_id=run_id)
        run = session.get(AnalysisRun, run_id)
        assert run is not None
        assert run.completed_at is not None

    assert first == second
    expected_time = run.completed_at.replace(tzinfo=run.completed_at.tzinfo or UTC)
    assert first.export_manifest.exported_at == expected_time
    database.close()


def test_exported_at_tampering_is_rejected_even_with_recomputed_manifest_hash(
    tmp_path: Path,
) -> None:
    database, project_id, run_id = _seed_run(tmp_path, node_count=1)
    with Session(database.engine) as session:
        packet = build_review_packet(
            session,
            project_id=project_id,
            run_id=run_id,
        ).model_dump(mode="json")
        packet["export_manifest"]["exported_at"] = "2000-01-01T00:00:00+00:00"
        manifest = packet["export_manifest"]
        manifest_payload = {
            "schema_version": packet["schema_version"],
            "project_id": packet["project_id"],
            "analysis_run_id": packet["analysis_run_id"],
            "exported_at": manifest["exported_at"],
            "hash_algorithm": manifest["hash_algorithm"],
            "node_count": manifest["node_count"],
            "node_ids": manifest["node_ids"],
            "node_hashes": manifest["node_hashes"],
        }
        serialized = json.dumps(
            manifest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        manifest["manifest_hash"] = sha256(serialized.encode("utf-8")).hexdigest()

        with pytest.raises(NodeAcceptanceReviewError, match="发布历史"):
            evaluate_review_packet(
                session,
                project_id=project_id,
                run_id=run_id,
                reviewed_packet=packet,
            )
    database.close()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["unknown-command"],
        ["export", "--project", "project"],
        ["export", "--unknown-option"],
    ],
)
def test_argparse_errors_return_three_without_usage_or_system_exit(
    arguments: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(arguments) == 3
    captured = capsys.readouterr()
    assert "命令参数" in captured.err
    assert "usage:" not in captured.err
    assert "Traceback" not in captured.err
