"""纯 Worker、失败事务与模拟计划的冒烟回归，不调用外部服务。"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime

import pytest
from langchain_core.messages import ToolMessage
from moonlightbox.db import Database
from moonlightbox.events.models import EventNode
from moonlightbox.projects.models import Project
from moonlightbox.runtime_v1.branch_models import Branch, BranchMessage
from moonlightbox.runtime_v1.day_planner import DayPlanAgent
from moonlightbox.runtime_v1.db_models import (
    RuntimeCycleTraceRow,
    RuntimeDayPlanRow,
    RuntimeEventRow,
    RuntimeSnapshotRow,
)
from moonlightbox.runtime_v1.schemas import DayPlanProposalBlock
from moonlightbox.runtime_v1.service import RuntimeService
from moonlightbox.training.models import ModelVersion
from sqlalchemy import select
from sqlalchemy.orm import Session
from submission_helpers import submission_message
from test_harness_repairs import context


def test_worker_registers_all_foreign_keys_without_api_import():
    code = """
import sys
from moonlightbox.db import Database, Base
d = Database('sqlite:///:memory:')
assert 'moonlightbox.api' not in sys.modules
for table in Base.metadata.tables.values():
    for fk in table.foreign_keys:
        assert fk.column is not None
d.create_schema()
"""
    subprocess.run([sys.executable, "-c", code], env=os.environ.copy(), check=True, timeout=30)


def test_simulated_arrangement_needs_explanation_not_fabricated_evidence():
    data = dict(
        start="08:00",
        end="08:30",
        activity="早餐",
        basis="simulation_assumption",
        confidence="inferred",
        evidence_ids=[],
    )
    with pytest.raises(ValueError, match="assumption"):
        DayPlanProposalBlock(**data)
    block = DayPlanProposalBlock(**data, assumption="为模拟当天早餐预留半小时，并非历史事实")
    assert block.evidence_ids == []


def test_final_planner_draft_gets_targeted_repair():
    ctx = context()
    ctx.evidence_requirements["required_topics"] = []

    class Model:
        def __init__(self):
            self.inputs = []

        def invoke(self, messages):
            self.inputs.append(messages)
            if len(self.inputs) <= 3:
                return submission_message(content='{"plan_date":"2026-05-09","blocks":[]}')
            return submission_message(
                content=json.dumps(
                    dict(
                        plan_date="2026-05-09",
                        blocks=[
                            dict(
                                start="00:00",
                                end="24:00",
                                activity="离线测试安排",
                                basis="simulation_assumption",
                                confidence="inferred",
                                assumption="纯测试模拟",
                                evidence_ids=[],
                            )
                        ],
                    )
                )
            )

    model = Model()
    result = DayPlanAgent(model).run_with_trace(ctx, tools=[])
    assert result.proposal is not None
    errors = [m.content for m in model.inputs[-1] if isinstance(m, ToolMessage)]
    assert errors and "blocks" in str(errors)


def test_reply_flush_failure_preserves_initial_plan_and_restores_event(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'persistence.db'}")
    database.create_schema()
    now = datetime.now(UTC)
    with Session(database.engine) as s:
        s.add(Project(id="p", name="测试"))
        s.flush()
        s.add(
            ModelVersion(
                id="m",
                project_id="p",
                base_model="test",
                adapter_path="none",
                dataset_hash="h",
                metrics={},
            )
        )
        s.flush()
        s.add(
            EventNode(
                id="e",
                project_id="p",
                type="origin",
                start_message_id="1",
                end_message_id="1",
                emotion_labels=[],
                topic="",
                conflict_level=0,
                importance=0,
                reason="",
                evidence_ids=[],
            )
        )
        s.flush()
        s.add(
            Branch(
                id="b",
                project_id="p",
                origin_event_id="e",
                model_version_id="m",
                title="测试",
                origin_time=now,
            )
        )
        s.flush()
        s.add(
            RuntimeSnapshotRow(
                id="snap",
                branch_id="b",
                cutoff_at=now,
                timezone="UTC",
                snapshot_mode="latest_profile",
                source_message_ids=[],
                profile={},
                routine_profile={},
                compiler_version="test",
            )
        )
        s.commit()

        class Model:
            def __init__(self, role):
                self.role = role

            def invoke(self, messages):
                if self.role == "planner":
                    result = dict(
                        plan_date=now.date().isoformat(),
                        blocks=[
                            dict(
                                start="00:00",
                                end="24:00",
                                activity="测试安排",
                                basis="fallback",
                                confidence="fallback",
                                evidence_ids=[],
                            )
                        ],
                    )
                elif self.role == "actor":
                    result = dict(text="你好", bubbles=["你好"])
                else:
                    result = dict(
                        action="speak",
                        speech_mode="reply",
                        reply={"messages": [{"kind": "text", "text": "你好"}]},
                    )
                from test_peer_collaboration import with_pending_inputs

                return submission_message(content=json.dumps(with_pending_inputs(result, messages)))

        service = RuntimeService(
            s,
            director_model=Model("director"),
            actor_model=Model("actor"),
            planner_model=Model("planner"),
        )
        service.process_next(project_id="p", branch_id="b", prepare_branch=True)
        service.submit_user_message(
            project_id="p", branch_id="b", content="你好", idempotency_key="one"
        )
        # 真实数据库触发器制造 flush 异常，不能只用普通 Python 异常替代失效事务。
        s.connection().exec_driver_sql("""CREATE TRIGGER reject_reply
        BEFORE INSERT ON branch_messages
        WHEN NEW.role = 'assistant' BEGIN SELECT RAISE(ABORT, 'test_reply_write_failure'); END""")
        s.commit()
        with pytest.raises(Exception, match="test_reply_write_failure"):
            service.process_next(project_id="p", branch_id="b")
        plan = s.scalar(select(RuntimeDayPlanRow))
        assert plan.generation_metadata["status"] == "agent" and len(plan.blocks) == 1
        assert s.scalar(select(RuntimeEventRow)).status == "queued"
        trace = s.scalar(
            select(RuntimeCycleTraceRow).where(RuntimeCycleTraceRow.status == "failed")
        )
        assert trace.status == "failed" and trace.error_code == "IntegrityError"
        assert "test_reply_write_failure" in trace.error_message
        assert not list(s.scalars(select(BranchMessage).where(BranchMessage.role == "assistant")))
        s.connection().exec_driver_sql("DROP TRIGGER reject_reply")
        s.commit()
        result = service.process_next(project_id="p", branch_id="b")
        assert result["message"].content == "你好"
        assert (
            len(list(s.scalars(select(BranchMessage).where(BranchMessage.role == "assistant"))))
            == 1
        )
