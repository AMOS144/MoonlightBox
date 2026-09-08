from datetime import UTC, datetime, timedelta
from pathlib import Path

from moonlightbox.db import Database
from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventCandidate, EventNode
from moonlightbox.events.runs import AnalysisRunService
from moonlightbox.events.service import EventService
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from sqlalchemy import event, select
from sqlalchemy.orm import Session


def _create_enriched_events(tmp_path: Path) -> tuple[Database, Session]:
    database = Database(f"sqlite:///{tmp_path / 'event-read.db'}")
    AnalysisRun.metadata.create_all(database.engine)
    session = Session(database.engine)
    session.add(Project(id="project-1", name="节点解释测试"))
    session.flush()
    session.add(
        ImportSource(
            id="import-1",
            project_id="project-1",
            preview_id="preview-1",
            source_path="/tmp/chat.json",
            message_count=3,
            confirmed_at=datetime.now(UTC),
        )
    )
    participant = Participant(
        id="participant-1",
        project_id="project-1",
        name="甲",
        role="self",
    )
    session.add(participant)
    session.flush()
    for source_id, content, minute in [
        ("m1", "  我想先冷静一下  ", 1),
        ("m2", "<msg><appmsg><title>不应泄漏</title></appmsg></msg>", 2),
        ("m3", "后续仍然没有解决", 3),
    ]:
        session.add(
            Message(
                id=f"row-{source_id}",
                project_id="project-1",
                import_id="import-1",
                participant_id=participant.id,
                source_id=source_id,
                timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minute),
                kind="text",
                content=content,
                raw={"api_key": "绝不能返回"},
            )
        )
    session.commit()
    run = AnalysisRunService(session).get_or_create(
        project_id="project-1",
        import_id="import-1",
        analysis_version="hybrid-v2",
        prompt_version="prompt-v2",
        model="model-v2",
        config={"threshold": 0.72},
        window_ids=["window-1", "window-2"],
    )
    for index in range(2):
        start_id = "m1"
        node = EventNode(
            id=f"event-{index + 1}",
            project_id="project-1",
            type="cold_war",
            start_message_id=start_id,
            end_message_id="m2",
            before_state="仍在沟通",
            after_state="停止回复",
            emotion_labels=["失望"],
            topic="冲突",
            conflict_level=4,
            importance=0.9,
            reason="回复突然中断",
            evidence_ids=[start_id, "m2"],
            status="active",
        )
        session.add(node)
        candidate = EventCandidate(
            run_id=run.id,
            window_id=f"window-{index + 1}",
            candidate_key="same-key",
            raw_payload={
                "type": "cold_war",
                "start_message_id": start_id,
                "end_message_id": "m2",
                "before_state": "仍在沟通",
                "after_state": "停止回复",
                "topic": "冲突",
                "evidence_ids": [start_id, "m2"],
            },
            review_payload={},
            status="accepted",
            rejection_reason=None,
            scores={
                "state_change_strength": 0.88 - index * 0.44,
                "persistence": 0.77,
                "evidence_quality": 0.66,
                "model_confidence": 0.55,
                "total": 0.76,
            },
        )
        session.add(candidate)
        session.flush()
        session.add(
            AnalysisRevision(
                event_id=node.id,
                revision_number=1,
                snapshot={},
                action_reason="V2 自动发布",
                analysis_version="hybrid-v2",
                prompt_version="prompt-v2",
                model="model-v2",
                run_id=run.id,
                candidate_id=candidate.id,
            )
        )
    session.commit()
    return database, session


def test_list_read_batches_enrichment_and_never_exposes_xml(tmp_path: Path) -> None:
    database, session = _create_enriched_events(tmp_path)
    statements: list[str] = []

    def capture_select(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(database.engine, "before_cursor_execute", capture_select)
    try:
        result = EventService(session).list_read("project-1")

        assert len(result) == 2
        assert len(statements) <= 4
        first = result[0]
        assert first.score_components is not None
        assert first.score_components.model_dump() == {
            "state_change_strength": 0.88,
            "persistence": 0.77,
            "evidence_quality": 0.66,
            "model_confidence": 0.55,
        }
        assert first.analysis_version == "hybrid-v2"
        assert first.prompt_version == "prompt-v2"
        assert first.model == "model-v2"
        assert first.revision_number == 1
        assert first.analysis_run_id is not None
        assert [summary.message_id for summary in first.evidence_summaries] == ["m1"]
        assert first.evidence_summaries[0].sender == "甲"
        assert first.evidence_summaries[0].content == "我想先冷静一下"
        serialized = first.model_dump_json()
        assert "<msg>" not in serialized
        assert "不应泄漏" not in serialized
        assert "api_key" not in serialized
        assert "绝不能返回" not in serialized
        second = result[1]
        assert second.score_components is not None
        assert second.score_components.state_change_strength == 0.44
    finally:
        event.remove(database.engine, "before_cursor_execute", capture_select)
        session.close()
        database.close()


def test_explicit_run_never_falls_back_to_another_import_with_same_source_id(
    tmp_path: Path,
) -> None:
    database, session = _create_enriched_events(tmp_path)
    try:
        session.add(
            ImportSource(
                id="import-2",
                project_id="project-1",
                preview_id="preview-2",
                source_path="/tmp/chat-2.json",
                message_count=1,
                confirmed_at=datetime.now(UTC),
            )
        )
        session.add(
            Message(
                id="row-foreign",
                project_id="project-1",
                import_id="import-2",
                participant_id="participant-1",
                source_id="foreign-only",
                timestamp=datetime(2026, 1, 2, tzinfo=UTC),
                kind="text",
                content="另一导入中的同源编号内容",
                raw={},
            )
        )
        run_id = session.scalar(
            select(AnalysisRevision.run_id).where(AnalysisRevision.event_id == "event-1")
        )
        assert run_id is not None
        node = EventNode(
            id="event-explicit-import",
            project_id="project-1",
            type="conflict",
            start_message_id="foreign-only",
            end_message_id="foreign-only",
            before_state="平静",
            after_state="冲突",
            emotion_labels=["失望"],
            topic="导入隔离",
            conflict_level=3,
            importance=0.7,
            reason="测试导入隔离",
            evidence_ids=["foreign-only"],
            status="active",
        )
        session.add(node)
        session.flush()
        session.add(
            AnalysisRevision(
                event_id=node.id,
                revision_number=1,
                snapshot={},
                action_reason="V2 自动发布",
                analysis_version="hybrid-v2",
                prompt_version="prompt-v2",
                model="model-v2",
                run_id=run_id,
                candidate_id=None,
            )
        )
        session.commit()

        listed = {item.id: item for item in EventService(session).list_read("project-1")}
        assert listed["event-explicit-import"].evidence_summaries == []
    finally:
        session.close()
        database.close()


def test_legacy_event_only_falls_back_when_source_id_is_unambiguous(
    tmp_path: Path,
) -> None:
    database, session = _create_enriched_events(tmp_path)
    try:
        session.add(
            ImportSource(
                id="import-2",
                project_id="project-1",
                preview_id="preview-2",
                source_path="/tmp/chat-2.json",
                message_count=1,
                confirmed_at=datetime.now(UTC),
            )
        )
        for import_id, row_id in [("import-1", "legacy-row-1"), ("import-2", "legacy-row-2")]:
            session.add(
                Message(
                    id=row_id,
                    project_id="project-1",
                    import_id=import_id,
                    participant_id="participant-1",
                    source_id="legacy-duplicate",
                    timestamp=datetime(2026, 1, 2, tzinfo=UTC),
                    kind="text",
                    content=f"{import_id} 的内容",
                    raw={},
                )
            )
        node = EventNode(
            id="event-legacy",
            project_id="project-1",
            type="conflict",
            start_message_id="legacy-duplicate",
            end_message_id="legacy-duplicate",
            before_state="平静",
            after_state="冲突",
            emotion_labels=["失望"],
            topic="旧节点",
            conflict_level=3,
            importance=0.7,
            reason="旧节点没有导入关联",
            evidence_ids=["legacy-duplicate"],
            status="active",
        )
        session.add(node)
        session.flush()
        session.add(
            AnalysisRevision(
                event_id=node.id,
                revision_number=1,
                snapshot={},
                action_reason="自动分析",
                analysis_version="heuristic-v1",
                prompt_version="none",
                model=None,
                run_id=None,
                candidate_id=None,
            )
        )
        session.commit()

        listed = {item.id: item for item in EventService(session).list_read("project-1")}
        assert listed["event-legacy"].evidence_summaries == []
    finally:
        session.close()
        database.close()
