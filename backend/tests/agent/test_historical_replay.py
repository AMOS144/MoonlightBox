from datetime import UTC, datetime, timedelta
from pathlib import Path

from moonlightbox.agent.replay import (
    FactValidation,
    GroundedReplayFactValidator,
    HistoricalReplayRunner,
    HistoricalReplaySample,
    ReplayMessage,
)
from moonlightbox.agent.types import CognitionDraft, ExpressionDecision, FusedAgentTurn
from moonlightbox.branches.models import Branch
from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn
from moonlightbox.branches.reviewer import FactSafetyResult
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.imports.models import ImportSource, Message, Participant
from moonlightbox.projects.models import Project
from moonlightbox.training.models import ModelVersion
from sqlalchemy.orm import Session

PROJECT_ID = "project-1"
BRANCH_ID = "branch-1"
MODEL_ID = "model-1"
START = datetime(2026, 5, 10, 21, 0)
END = datetime(2026, 5, 11, 12, 0)


class FakeInferenceClient:
    def __init__(self, *, express: bool = True) -> None:
        self.express = express
        self.direct_calls: list[list[dict[str, str]]] = []
        self.direct_prompts: list[str] = []
        self.fused_calls: list[list[dict[str, str]]] = []
        self.fused_prompts: list[str] = []
        self.cognition_calls: list[object] = []
        self.rewrite_calls: list[str] = []

    def generate(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        assert model_version_id == MODEL_ID
        assert system_prompt
        self.direct_prompts.append(system_prompt)
        self.direct_calls.append(messages)
        return _reply("直接回复")

    def generate_fused(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        allowed_sticker_ids: tuple[str, ...],
    ) -> FusedAgentTurn:
        assert model_version_id == MODEL_ID
        assert system_prompt
        assert allowed_sticker_ids == ()
        self.fused_prompts.append(system_prompt)
        self.fused_calls.append(messages)
        decision = ExpressionDecision(
            express=self.express,
            content="融合回复" if self.express else None,
            reason="测试决定",
        )
        return FusedAgentTurn(
            cognition=CognitionDraft(
                private_content="私密判断",
                subjective_feelings={},
                attention_target={},
                desired_actions=(),
                expression_decision=decision,
                suggested_next_wakeup=None,
                structured_changes={},
                confidence=1,
            ),
            expression_decision=decision,
            reply=_reply("融合回复") if self.express else None,
        )

    def generate_cognition(
        self,
        request: object,
        *,
        model_version_id: str | None = None,
    ) -> CognitionDraft:
        assert model_version_id == MODEL_ID
        self.cognition_calls.append(request)
        decision = ExpressionDecision(
            express=self.express,
            content="直接回复" if self.express else None,
            reason="测试决定",
        )
        return CognitionDraft(
            private_content="私密判断",
            subjective_feelings={},
            attention_target={},
            desired_actions=(),
            expression_decision=decision,
            suggested_next_wakeup=None,
            structured_changes={},
            confidence=1,
        )

    def rewrite_content_draft(
        self,
        model_version_id: str,
        draft: str,
    ) -> GeneratedReplyTurn | None:
        assert model_version_id == MODEL_ID
        self.rewrite_calls.append(draft)
        return _reply(draft)


class SafeFactValidator:
    def validate(self, sample: object, turn: FusedAgentTurn) -> FactValidation:
        del sample, turn
        return FactValidation(safe=True, reasons=("grounded_reviewer_approved",))


class RejectingDirectValidator:
    def validate(self, sample: object, turn: FusedAgentTurn) -> FactValidation:
        del sample
        content = "\n".join(
            bubble.content or "" for bubble in turn.reply.bubbles
        ) if turn.reply is not None else ""
        return FactValidation(
            safe=content != "直接回复",
            reasons=("direct_reply_rejected",) if content == "直接回复" else ("safe",),
        )


class SafeOnGroundedRetryClient(FakeInferenceClient):
    def generate(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        super().generate(model_version_id, system_prompt, messages)
        if "上一版回复没有通过事实校验" in system_prompt:
            return _reply("不知道呢")
        return _reply("直接回复")


class ApprovingFactReviewer:
    def review_fact_safety(
        self,
        packet: object,
        draft: GeneratedReplyTurn,
        *,
        private_content: str,
    ) -> FactSafetyResult:
        del packet, draft, private_content
        return FactSafetyResult(
            no_future_leak=True,
            grounded_claims=True,
            no_private_leak=True,
            reasons=("这是规划后的自然公开回应",),
        )


def test_local_replay_validator_works_without_cloud_reviewer() -> None:
    sample = HistoricalReplaySample(
        cutoff=datetime(2026, 5, 10, 21, 0, tzinfo=UTC),
        expected_express=True,
        context=(
            ReplayMessage(
                id="message-1",
                source_id="source-1",
                timestamp=datetime(2026, 5, 10, 21, 0, tzinfo=UTC),
                role="self",
                content="在吗",
            ),
        ),
        response_message_id="response-1",
        evidence={},
    )
    turn = _fused_reply("干嘛")

    result = GroundedReplayFactValidator(None, persona="目标").validate(sample, turn)

    assert result.safe is True
    assert "local_grounding_approved" in result.reasons


def test_local_replay_validator_rejects_unsupported_current_state() -> None:
    sample = HistoricalReplaySample(
        cutoff=datetime(2026, 5, 10, 21, 0, tzinfo=UTC),
        expected_express=True,
        context=(
            ReplayMessage(
                id="message-1",
                source_id="source-1",
                timestamp=datetime(2026, 5, 10, 21, 0, tzinfo=UTC),
                role="self",
                content="你在干嘛",
            ),
        ),
        response_message_id="response-1",
        evidence={},
    )
    turn = _fused_reply("我正在开会")

    result = GroundedReplayFactValidator(None, persona="目标").validate(sample, turn)

    assert result.safe is False
    assert result.reasons[0] == "local_grounding_rejected"


class OneCognitionFailureClient(FakeInferenceClient):
    def __init__(self) -> None:
        super().__init__()
        self.cognition_attempts = 0

    def generate_cognition(
        self,
        request: object,
        *,
        model_version_id: str | None = None,
    ) -> CognitionDraft:
        self.cognition_attempts += 1
        if self.cognition_attempts == 1:
            raise RuntimeError("临时协议失败")
        return super().generate_cognition(
            request,
            model_version_id=model_version_id,
        )


def _reply(content: str) -> GeneratedReplyTurn:
    return GeneratedReplyTurn(
        bubbles=(GeneratedBubble(content=content, delay_ms=0),),
        raw_output=f"<bubble>{content}</bubble>",
    )


def _fused_reply(content: str) -> FusedAgentTurn:
    decision = ExpressionDecision(express=True, content=content, reason="测试")
    return FusedAgentTurn(
        cognition=CognitionDraft(
            private_content="私密判断",
            subjective_feelings={},
            attention_target={},
            desired_actions=(),
            expression_decision=decision,
            suggested_next_wakeup=None,
            structured_changes={},
            confidence=1,
        ),
        expression_decision=decision,
        reply=_reply(content),
    )


def _database(tmp_path: Path, *, positive_count: int = 20) -> Database:
    database = Database(f"sqlite:///{tmp_path / 'replay.db'}")
    Project.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        session.add(Project(id=PROJECT_ID, name="回放项目"))
        session.flush()
        session.add(
            ImportSource(
                id="import-1",
                project_id=PROJECT_ID,
                preview_id="preview-1",
                source_path="/tmp/source",
                message_count=positive_count * 2 + 3,
                confirmed_at=START,
            )
        )
        self_participant = Participant(
            id="self-1",
            project_id=PROJECT_ID,
            name="我",
            role="self",
        )
        target_participant = Participant(
            id="target-1",
            project_id=PROJECT_ID,
            name="目标",
            role="target",
        )
        session.add_all([self_participant, target_participant])
        session.flush()
        session.add_all(
            [
                _message("before", START - timedelta(seconds=1), self_participant),
                _message("after", END + timedelta(seconds=1), target_participant),
            ]
        )
        for index in range(positive_count):
            cutoff = START + timedelta(minutes=index * 10)
            session.add(_message(f"self-{index}", cutoff, self_participant))
            session.add(
                _message(
                    f"target-{index}",
                    cutoff + timedelta(minutes=1),
                    target_participant,
                )
            )
        session.add(
            _message(
                "self-negative",
                START + timedelta(hours=8),
                self_participant,
            )
        )
        session.add(
            ModelVersion(
                id=MODEL_ID,
                project_id=PROJECT_ID,
                base_model="/models/base",
                adapter_path="/models/adapter",
                dataset_hash="hash",
                metrics={},
                active=True,
                training_config={
                    "reply_protocol_version": "persona-text-v1",
                    "data_manifest": {
                        "target_sender": "目标",
                        "split_time_boundaries": {
                            "test": {
                                "start": START.isoformat(),
                                "end": END.isoformat(),
                            }
                        },
                    }
                },
            )
        )
        session.add(
            EventNode(
                id="event-1",
                project_id=PROJECT_ID,
                type="origin",
                start_message_id="self-0",
                end_message_id="self-0",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        session.flush()
        session.add(
            Branch(
                id=BRANCH_ID,
                project_id=PROJECT_ID,
                origin_event_id="event-1",
                model_version_id=MODEL_ID,
                title="回放分支",
                origin_time=START.replace(tzinfo=UTC),
                state_snapshot={},
                subject_agent_mode="shadow",
            )
        )
        session.commit()
    return database


def _message(
    source_id: str,
    timestamp: datetime,
    participant: Participant,
) -> Message:
    return Message(
        project_id=PROJECT_ID,
        import_id="import-1",
        participant_id=participant.id,
        source_id=source_id,
        timestamp=timestamp,
        kind="text",
        content=source_id,
        raw={},
    )


def test_build_samples_uses_strict_test_window_without_future_input(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with Session(database.engine) as session:
        samples = HistoricalReplayRunner(
            session,
            FakeInferenceClient(),
            SafeFactValidator(),
        ).build_samples(PROJECT_ID, BRANCH_ID, MODEL_ID, sample_count=20)

    assert len(samples) == 20
    assert any(sample.expected_express for sample in samples)
    assert any(not sample.expected_express for sample in samples)
    for sample in samples:
        assert START.replace(tzinfo=UTC) <= sample.cutoff <= END.replace(tzinfo=UTC)
        assert len(sample.context) <= 12
        assert all(message.timestamp <= sample.cutoff for message in sample.context)
        assert "before" not in sample.context_message_ids
        assert "after" not in sample.context_message_ids
        assert sample.response_message_id not in sample.context_message_ids
        assert sample.evidence["label_rule_version"] == "historical-expression-v1"
        assert sample.evidence["cutoff"] == sample.cutoff.isoformat()
        assert sample.evidence["context_message_ids"] == list(sample.context_message_ids)
    database.close()


def test_positive_and_negative_labels_have_frozen_audit_derivation(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with Session(database.engine) as session:
        samples = HistoricalReplayRunner(
            session,
            FakeInferenceClient(),
            SafeFactValidator(),
        ).build_samples(PROJECT_ID, BRANCH_ID, MODEL_ID, sample_count=20)

    positive = next(sample for sample in samples if sample.expected_express)
    negative = next(sample for sample in samples if not sample.expected_express)
    assert positive.evidence["label_derivation"] == "target_replied_after_self_contribution"
    assert positive.response_message_id is not None
    assert negative.evidence["label_derivation"] == "no_target_reply_before_session_boundary"
    assert negative.response_message_id is None
    assert negative.evidence["silence_window_seconds"] == 1800
    database.close()


def test_run_excludes_warmup_and_persists_real_observation_evidence(tmp_path: Path) -> None:
    database = _database(tmp_path)
    client = FakeInferenceClient()
    ticks = iter(float(value) for value in range(100))
    with Session(database.engine) as session:
        result = HistoricalReplayRunner(
            session,
            client,
            SafeFactValidator(),
            timer=lambda: next(ticks),
        ).run(
            PROJECT_ID,
            BRANCH_ID,
            MODEL_ID,
            sample_count=20,
            activate_if_passed=True,
        )

        assert result.report.sample_count == 20
        assert result.report.direct_lora_p95 == 1000
        assert result.report.p95_cognition_latency_ms == 1000
        assert result.report.passed is True
        assert result.activated is True
        assert session.get(Branch, BRANCH_ID).subject_agent_mode == "active"
        assert result.summary() == {
            "report_id": result.report.id,
            "sample_count": 20,
            "structure_extraction_success_rate": 1,
            "fact_safety_rate": 1,
            "expression_decision_accuracy": 0.95,
            "fused_p95_latency_ms": 1000,
            "direct_p95_latency_ms": 1000,
            "passed": True,
            "failure_reasons": [],
            "activated": True,
        }
        evidence = result.report.evidence["observations"][0]
        assert evidence["direct_latency_ms"] == 1000
        assert evidence["context_message_ids"]
        assert evidence["label_derivation"]
        assert evidence["fact_validation_reasons"] == ["grounded_reviewer_approved"]
        assert evidence["direct_fact_validation_reasons"] == [
            "grounded_reviewer_approved"
        ]
        assert evidence["production_pipeline"] == "base-cognition-persona-style-v2"
    assert len(client.cognition_calls) == 20
    assert len(client.rewrite_calls) == 20
    assert client.direct_calls == []
    assert client.fused_calls == []
    database.close()


def test_legacy_compact_model_keeps_compact_direct_replay_prompt(tmp_path: Path) -> None:
    database = _database(tmp_path, positive_count=2)
    client = FakeInferenceClient()
    with Session(database.engine) as session:
        model = session.get(ModelVersion, MODEL_ID)
        assert model is not None
        model.training_config = {
            **model.training_config,
            "reply_protocol_version": "high-fidelity-compact-bubble-v2",
        }
        session.commit()
        HistoricalReplayRunner(session, client, SafeFactValidator()).run(
            PROJECT_ID,
            BRANCH_ID,
            MODEL_ID,
            sample_count=3,
        )

    assert "紧凑气泡协议" in client.direct_prompts[1]
    assert "聊天文字本身" not in client.direct_prompts[1]
    database.close()


def test_direct_persona_reply_is_part_of_fail_closed_replay_gate(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with Session(database.engine) as session:
        result = HistoricalReplayRunner(
            session,
            FakeInferenceClient(),
            RejectingDirectValidator(),
        ).run(PROJECT_ID, BRANCH_ID, MODEL_ID, sample_count=20)

        assert result.report.fact_safety_rate == 0
        assert result.report.passed is False
        observation = result.report.evidence["observations"][0]
        assert observation["direct_fact_validation_reasons"] == [
            "direct_reply_rejected"
        ]
        model = session.get(ModelVersion, MODEL_ID)
        assert model is not None
        assert model.recommended is False
        assert model.metrics["latest_subject_replay_passed"] == 0
        assert model.training_config["latest_subject_replay"]["report_id"] == (
            result.report.id
        )
    database.close()


def test_two_stage_persona_reply_does_not_retry_with_unplanned_direct_content(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    client = SafeOnGroundedRetryClient()
    with Session(database.engine) as session:
        result = HistoricalReplayRunner(
            session,
            client,
            RejectingDirectValidator(),
        ).run(PROJECT_ID, BRANCH_ID, MODEL_ID, sample_count=20)

        assert result.report.fact_safety_rate == 0
        retried = [
            observation
            for observation in result.report.evidence["observations"]
            if observation["direct_grounded_retry_attempted"]
        ]
        assert retried == []
        assert client.direct_calls == []
    database.close()


def test_expressed_reply_without_provable_validator_fails_closed(tmp_path: Path) -> None:
    database = _database(tmp_path)
    with Session(database.engine) as session:
        result = HistoricalReplayRunner(
            session,
            FakeInferenceClient(),
            fact_validator=None,
        ).run(PROJECT_ID, BRANCH_ID, MODEL_ID, sample_count=20)

        assert result.report.fact_safety_rate == 0
        assert result.report.passed is False
        assert "fact_safety_rate_below_threshold" in result.report.failure_reasons
        reasons = result.report.evidence["observations"][0]["fact_validation_reasons"]
        assert reasons == ["fact_validator_unavailable"]
        assert session.get(Branch, BRANCH_ID).subject_agent_mode == "shadow"
    database.close()


def test_extraction_failure_is_not_double_counted_as_an_unsafe_public_claim(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    ticks = iter(float(value) for value in range(100))
    with Session(database.engine) as session:
        result = HistoricalReplayRunner(
            session,
            OneCognitionFailureClient(),
            SafeFactValidator(),
            timer=lambda: next(ticks),
        ).run(PROJECT_ID, BRANCH_ID, MODEL_ID, sample_count=20)

        assert result.report.structure_extraction_success_rate == 0.95
        assert result.report.fact_safety_rate == 1
        observation = result.report.evidence["observations"][0]
        assert observation["extraction_succeeded"] is False
        assert observation["fact_safe"] is True
        assert observation["fact_validation_reasons"] == [
            "no_public_output_due_to_extraction_failure"
        ]
    database.close()


def test_planned_public_reply_matching_private_wording_is_not_automatically_a_leak() -> None:
    cutoff = START.replace(tzinfo=UTC)
    sample = HistoricalReplaySample(
        cutoff=cutoff,
        expected_express=True,
        context=(
            ReplayMessage(
                id="message-1",
                source_id="source-1",
                timestamp=cutoff,
                role="self",
                content="在吗",
            ),
        ),
        response_message_id="source-2",
        evidence={},
    )
    decision = ExpressionDecision(express=True, content="我也在呢", reason="自然回应")
    turn = FusedAgentTurn(
        cognition=CognitionDraft(
            private_content="我也在呢",
            subjective_feelings={},
            attention_target={},
            desired_actions=(),
            expression_decision=decision,
            suggested_next_wakeup=None,
            structured_changes={},
            confidence=1,
        ),
        expression_decision=decision,
        reply=_reply("我也在呢"),
    )

    result = GroundedReplayFactValidator(
        ApprovingFactReviewer(),
        persona="目标",
    ).validate(sample, turn)

    assert result.safe is True
    assert "grounded_fact_reviewer_approved" in result.reasons


def test_cognitive_silence_needs_no_fact_reviewer_but_cannot_pass_expression_gate(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, positive_count=2)
    with Session(database.engine) as session:
        result = HistoricalReplayRunner(
            session,
            FakeInferenceClient(express=False),
            fact_validator=None,
        ).run(PROJECT_ID, BRANCH_ID, MODEL_ID, sample_count=3, activate_if_passed=True)

        assert result.report.sample_count == 3
        assert result.report.fact_safety_rate == 1
        assert result.report.passed is False
        assert result.activated is False
        assert "insufficient_samples" in result.report.failure_reasons
        assert session.get(Branch, BRANCH_ID).subject_agent_mode == "shadow"
    database.close()
