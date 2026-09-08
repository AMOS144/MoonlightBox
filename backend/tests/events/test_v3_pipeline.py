from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event as ThreadEvent

import pytest
from moonlightbox.db import Database
from moonlightbox.evaluation.node_acceptance import build_review_packet
from moonlightbox.events.models import AnalysisRevision, AnalysisRun, EventNode
from moonlightbox.events.service import EventService
from moonlightbox.events.v3_reviewer import (
    V3CandidateReview,
    V3EventCandidate,
    V3ExtractionResult,
    stable_v3_candidate_key,
)
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from sqlalchemy import select
from sqlalchemy.orm import Session


def _candidate(lane: str) -> V3EventCandidate:
    payload: dict[str, object] = {
        "candidate_key": "pending",
        "lane": lane,
        "type": "intimacy_increased" if lane == "relationship" else "travel",
        "title": "共同旅行推动关系升温",
        "event_status": "occurred",
        "start_message_id": "m1",
        "end_message_id": "m2",
        "summary": "双方完成了一次共同旅行",
        "before_state": "普通亲密关系" if lane == "relationship" else None,
        "after_state": "关系更加亲密" if lane == "relationship" else None,
        "emotion_labels": ["开心"],
        "topic": "旅行",
        "conflict_level": 0,
        "event_significance": 0.9,
        "relationship_impact": 0.9,
        "model_confidence": 0.9,
        "reason": "旅行结束后双方明确表达了积极体验",
        "evidence_ids": ["m1", "m2"],
    }
    candidate = V3EventCandidate.model_validate(payload)
    return candidate.model_copy(update={"candidate_key": stable_v3_candidate_key(candidate)})


def _review() -> V3CandidateReview:
    return V3CandidateReview(
        facts_supported=True,
        occurrence_supported=True,
        bilateral_confirmation=True,
        evidence_alignment=0.95,
        persistence=0.4,
        type_support=0.9,
        relationship_impact=0.9,
        event_significance=0.9,
        model_confidence=0.9,
        evidence_ids=["m3"],
        reason="事后回顾证明活动已经发生",
    )


class FakeReviewer:
    def __init__(self, *, fail_lane: str | None = None) -> None:
        self.fail_lane = fail_lane
        self.extraction_calls: list[str] = []

    def extract_candidates(
        self,
        window: object,
        *,
        lane: str,
        run_id: str | None = None,
    ) -> V3ExtractionResult:
        del window, run_id
        self.extraction_calls.append(lane)
        if lane == self.fail_lane:
            raise RuntimeError("测试通道失败")
        return V3ExtractionResult(
            candidates=(_candidate(lane),),
            rejected_candidates=(),
        )

    def review_candidate(
        self,
        candidate: V3EventCandidate,
        window: object,
        *,
        run_id: str | None = None,
    ) -> V3CandidateReview:
        del candidate, window, run_id
        return _review()


class FakeGlobalSelector:
    def __init__(self) -> None:
        self.candidates: list[dict[str, object]] = []

    def select_globally(
        self,
        *,
        candidates: list[dict[str, object]],
        rejected_examples: list[dict[str, object]],
        maximum_nodes: int,
        run_id: str | None = None,
    ) -> object:
        from moonlightbox.events.v3_reviewer import V3GlobalSelectionResult

        del rejected_examples, maximum_nodes, run_id
        self.candidates = candidates
        return V3GlobalSelectionResult.model_validate(
            {
                "selected": [
                    {
                        "candidate_key": candidates[0]["candidate_key"],
                        "relative_importance": 0.73,
                        "reason": "全局比较后仍会改变后续共同选择",
                    }
                ]
            }
        )


class FakeNarrativeSummarizer:
    def summarize(self, candidate: object, messages: object) -> str:
        del candidate, messages
        return "那次旅行让两个人真正开始期待下一段共同经历。"


def _session(tmp_path: Path) -> tuple[Database, Session]:
    database = Database(f"sqlite:///{tmp_path / 'v3-pipeline.db'}")
    AnalysisRun.metadata.create_all(database.engine)
    session = Session(database.engine)
    session.add(Project(id="project-1", name="V3 流水线测试"))
    session.flush()
    participant = Participant(
        id="participant-1",
        project_id="project-1",
        name="甲",
        role="self",
    )
    session.add(participant)
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
    session.flush()
    for index, content in enumerate(
        ["我们到酒店了", "这次旅行真的很开心", "下次还想一起去"],
        start=1,
    ):
        session.add(
            Message(
                id=f"row-{index}",
                project_id="project-1",
                import_id="import-1",
                participant_id=participant.id,
                source_id=f"m{index}",
                timestamp=datetime(2026, 5, 1, tzinfo=UTC) + timedelta(minutes=index),
                kind="text",
                content=content,
                raw={},
            )
        )
    session.commit()
    return database, session


def test_v3_pipeline_processes_both_lanes_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.v3_pipeline import EventV3Pipeline, V3PipelineConfig

    database, session = _session(tmp_path)
    reviewer = FakeReviewer()
    try:
        result = EventV3Pipeline(
            session,
            reviewer,  # type: ignore[arg-type]
            config=V3PipelineConfig(character_budget=1000),
        ).run(
            project_id="project-1",
            import_id="import-1",
            relationship_prompt_version="relationship-v1",
            shared_experience_prompt_version="experience-v1",
            model="cloud-v3",
        )

        run = session.get(AnalysisRun, result.run_id)
        nodes = list(session.scalars(select(EventNode)))
        revision = session.scalar(select(AnalysisRevision))
        assert reviewer.extraction_calls == ["relationship", "shared_experience"]
        assert run is not None and run.status == "succeeded"
        assert run.total_windows == 2
        assert run.completed_windows == 2
        assert result.event_count == 1
        assert len(nodes) == 1
        assert set(nodes[0].source_lanes) == {
            "relationship",
            "shared_experience",
        }
        assert revision is not None
        assert revision.analysis_version == "hybrid-v3"
        assert revision.snapshot["scores"]["total"] == pytest.approx(nodes[0].importance)
        assert len(revision.snapshot["source_scores"]) == 2
        read = EventService(session).list_read(
            "project-1",
            lane="shared_experience",
        )
        assert len(read) == 1
        assert read[0].score_components is not None
        assert read[0].score_components.model_dump() == {
            "event_significance": 0.9,
            "relationship_impact": 0.9,
            "evidence_quality": 0.95,
            "persistence": 0.4,
            "type_support": 0.9,
            "model_confidence": 0.9,
        }
        assert (
            len(
                EventService(session).list_read(
                    "project-1",
                    lane="relationship",
                )
            )
            == 1
        )
        packet = build_review_packet(
            session,
            project_id="project-1",
            run_id=result.run_id,
        )
        assert packet.schema_version == "node-acceptance-v3"
        assert packet.nodes[0].source_lanes == [
            "relationship",
            "shared_experience",
        ]
        assert packet.nodes[0].score_components.model_dump() == {
            "event_significance": 0.9,
            "relationship_impact": 0.9,
            "evidence_quality": 0.95,
            "persistence": 0.4,
            "type_support": 0.9,
            "model_confidence": 0.9,
        }
    finally:
        session.close()
        database.close()


def test_v3_pipeline_uses_global_relative_importance_for_published_node(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.v3_pipeline import EventV3Pipeline, V3PipelineConfig

    database, session = _session(tmp_path)
    try:
        selector = FakeGlobalSelector()
        result = EventV3Pipeline(
            session,
            FakeReviewer(),  # type: ignore[arg-type]
            global_selector=selector,  # type: ignore[arg-type]
            narrative_summarizer=FakeNarrativeSummarizer(),  # type: ignore[arg-type]
            config=V3PipelineConfig(character_budget=1000),
        ).run(
            project_id="project-1",
            import_id="import-1",
            relationship_prompt_version="relationship-v1",
            shared_experience_prompt_version="experience-v1",
            global_selection_prompt_version="global-v1",
            model="cloud-v3",
        )

        node = session.scalar(select(EventNode))
        revision = session.scalar(select(AnalysisRevision))
        assert result.event_count == 1
        assert node is not None and node.importance == pytest.approx(0.73)
        assert node.reason == "全局比较后仍会改变后续共同选择"
        assert node.display_summary == "那次旅行让两个人真正开始期待下一段共同经历。"
        assert node.summary_status == "ready"
        assert revision is not None
        assert revision.snapshot["global_selection"]["relative_importance"] == pytest.approx(0.73)
        assert selector.candidates
        global_payload = selector.candidates[0]
        assert global_payload["type"] in {"travel", "intimacy_increased"}
        assert set(global_payload["source_lanes"]) == {  # type: ignore[arg-type]
            "relationship",
            "shared_experience",
        }
        assert global_payload["event_status"] == "occurred"
        assert global_payload["scores"]["total"] > 0.6  # type: ignore[index]
        assert global_payload["review"]["facts_supported"] is True  # type: ignore[index]
    finally:
        session.close()
        database.close()


def test_v3_pipeline_failure_keeps_existing_nodes_and_resumes_failed_slot(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.v3_pipeline import (
        EventV3Pipeline,
        PipelineExecutionError,
        V3PipelineConfig,
    )

    database, session = _session(tmp_path)
    old = EventNode(
        project_id="project-1",
        type="conflict",
        start_message_id="old-1",
        end_message_id="old-2",
        before_state="平静",
        after_state="冲突",
        emotion_labels=[],
        topic="旧节点",
        conflict_level=3,
        importance=0.8,
        reason="旧节点",
        evidence_ids=["old-1", "old-2"],
        status="active",
    )
    session.add(old)
    session.flush()
    session.add(
        AnalysisRevision(
            event_id=old.id,
            revision_number=1,
            snapshot={"status": "active"},
            action_reason="V2 自动发布",
            analysis_version="hybrid-v2",
            prompt_version="v2",
        )
    )
    session.commit()
    config = V3PipelineConfig(character_budget=1000)
    try:
        with pytest.raises(PipelineExecutionError):
            EventV3Pipeline(
                session,
                FakeReviewer(fail_lane="shared_experience"),  # type: ignore[arg-type]
                config=config,
            ).run(
                project_id="project-1",
                import_id="import-1",
                relationship_prompt_version="relationship-v1",
                shared_experience_prompt_version="experience-v1",
                model="cloud-v3",
            )

        session.refresh(old)
        run = session.scalar(select(AnalysisRun).where(AnalysisRun.analysis_version == "hybrid-v3"))
        assert old.status == "active"
        assert run is not None and run.status == "failed"
        assert run.completed_windows == 1

        reviewer = FakeReviewer()
        result = EventV3Pipeline(
            session,
            reviewer,  # type: ignore[arg-type]
            config=config,
        ).run(
            project_id="project-1",
            import_id="import-1",
            relationship_prompt_version="relationship-v1",
            shared_experience_prompt_version="experience-v1",
            model="cloud-v3",
        )

        assert reviewer.extraction_calls == ["shared_experience"]
        assert result.run_id == run.id
        session.refresh(old)
        assert old.status == "superseded"
    finally:
        session.close()
        database.close()


def test_v3_pipeline_cancellation_interrupts_without_partial_publish(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.v3_pipeline import (
        EventV3Pipeline,
        PipelineCancelledError,
        V3PipelineConfig,
    )

    database, session = _session(tmp_path)
    cancelled = ThreadEvent()

    class CancellingReviewer(FakeReviewer):
        def extract_candidates(
            self,
            window: object,
            *,
            lane: str,
            run_id: str | None = None,
        ) -> V3ExtractionResult:
            result = super().extract_candidates(
                window,
                lane=lane,
                run_id=run_id,
            )
            cancelled.set()
            return result

    try:
        with pytest.raises(PipelineCancelledError):
            EventV3Pipeline(
                session,
                CancellingReviewer(),  # type: ignore[arg-type]
                config=V3PipelineConfig(character_budget=1000),
                should_cancel=cancelled.is_set,
            ).run(
                project_id="project-1",
                import_id="import-1",
                relationship_prompt_version="relationship-v1",
                shared_experience_prompt_version="experience-v1",
                model="cloud-v3",
            )

        run = session.scalar(select(AnalysisRun))
        assert run is not None and run.status == "interrupted"
        assert run.completed_windows == 0
        assert session.scalar(select(EventNode)) is None
    finally:
        session.close()
        database.close()


def test_v3_pipeline_isolates_all_invalid_extraction_batch(
    tmp_path: Path,
) -> None:
    from moonlightbox.events.v3_pipeline import EventV3Pipeline, V3PipelineConfig
    from moonlightbox.events.v3_reviewer import (
        V3CandidateBatchStructureError,
        V3RejectedCandidate,
    )

    class StructurallyInvalidRelationshipReviewer(FakeReviewer):
        def extract_candidates(
            self,
            window: object,
            *,
            lane: str,
            run_id: str | None = None,
        ) -> V3ExtractionResult:
            if lane == "relationship":
                raise V3CandidateBatchStructureError(
                    (
                        V3RejectedCandidate(
                            index=0,
                            error_code="candidate_validation_failed",
                            field_locations=(("type",),),
                        ),
                    )
                )
            return super().extract_candidates(
                window,
                lane=lane,
                run_id=run_id,
            )

    database, session = _session(tmp_path)
    checkpoints: list[dict[str, object]] = []
    try:
        result = EventV3Pipeline(
            session,
            StructurallyInvalidRelationshipReviewer(),  # type: ignore[arg-type]
            config=V3PipelineConfig(character_budget=1000),
            progress_callback=checkpoints.append,
        ).run(
            project_id="project-1",
            import_id="import-1",
            relationship_prompt_version="relationship-v1",
            shared_experience_prompt_version="experience-v1",
            model="cloud-v3",
        )

        run = session.get(AnalysisRun, result.run_id)
        assert run is not None and run.status == "succeeded"
        assert result.event_count == 1
        assert any(
            checkpoint.get("stage") == "extraction_batch_rejected"
            and checkpoint.get("rejected_candidates") == 1
            for checkpoint in checkpoints
        )
    finally:
        session.close()
        database.close()


def test_v3_pipeline_isolates_malformed_candidate_review(tmp_path: Path) -> None:
    from moonlightbox.events.v3_pipeline import EventV3Pipeline, V3PipelineConfig
    from moonlightbox.events.v3_reviewer import V3ReviewStructureError

    class MalformedRelationshipReviewReviewer(FakeReviewer):
        def review_candidate(
            self,
            candidate: V3EventCandidate,
            window: object,
            *,
            run_id: str | None = None,
        ) -> V3CandidateReview:
            if candidate.lane == "relationship":
                raise V3ReviewStructureError
            return super().review_candidate(
                candidate,
                window,
                run_id=run_id,
            )

    database, session = _session(tmp_path)
    checkpoints: list[dict[str, object]] = []
    try:
        result = EventV3Pipeline(
            session,
            MalformedRelationshipReviewReviewer(),  # type: ignore[arg-type]
            config=V3PipelineConfig(character_budget=1000),
            progress_callback=checkpoints.append,
        ).run(
            project_id="project-1",
            import_id="import-1",
            relationship_prompt_version="relationship-v1",
            shared_experience_prompt_version="experience-v1",
            model="cloud-v3",
        )

        assert result.event_count == 1
        assert any(
            checkpoint.get("stage") == "candidate_review_rejected"
            and checkpoint.get("error_type") == "V3ReviewStructureError"
            for checkpoint in checkpoints
        )
    finally:
        session.close()
        database.close()
