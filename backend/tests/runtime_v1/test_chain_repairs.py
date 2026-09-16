"""链路回归：真实事务与提交工具，不调用真实模型。"""

# ruff: noqa: F811
from datetime import UTC, datetime, timedelta

import pytest
from moonlightbox.agent_runtime.submission import submission_context
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.runtime_recovery import recovery, settle_failure
from moonlightbox.jobs.service import JobService
from moonlightbox.runtime_v1.collaboration.persistence import branch_writer
from moonlightbox.runtime_v1.collaboration.planner_tasks import run_planner_task
from moonlightbox.runtime_v1.executor import RuntimeExecutor
from moonlightbox.runtime_v1.jobs import enqueue_input_tick
from moonlightbox.runtime_v1.life_events.contracts import LifeAdvanceDecision
from moonlightbox.runtime_v1.tools.submit_life_result import build_submit_life_result_tool
from sqlalchemy import select
from test_life_events import candidate, prepare
from test_peer_collaboration import session  # noqa: F401


def test_planner_lock_contention_remains_retryable(session):
    runtime, now = prepare(session)
    job = Job(
        kind="runtime-v1-cycle",
        status="running",
        worker_token="test",
        lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        payload={"branch_id": "b", "planner_task": {}},
    )
    session.add(job)
    session.commit()
    with branch_writer(session, "planner:b"):
        with pytest.raises(RuntimeError) as failed:
            run_planner_task(runtime, job)
    job.status, job.worker_token, job.lease_expires_at = "failed", None, None
    job.error_code = failed.value.code
    settle_failure(session, job, now)
    assert recovery(job)["status"] == "waiting"
    assert recovery(job)["retry_at"] and recovery(job)["retries"] == 0


def test_project_job_scope_reads_old_and_new_runtime_tasks(session):
    prepare(session)
    old = Job(kind="runtime-v1-cycle", payload={"branch_id": "b"})
    wrong = Job(kind="runtime-v1-cycle", payload={"project_id": "other", "branch_id": "b"})
    session.add_all([old, wrong])
    session.commit()
    enqueue_input_tick(session, "b", "input")
    session.commit()
    jobs = JobService(session).list_for_project("p")
    assert old.id in {job.id for job in jobs}
    assert wrong.id not in {job.id for job in jobs}
    assert JobService(session).list_for_project("missing") == []
    # 老的 queued input 已占位，单独验证已存在任务可读，不靠新增 Job 避开恢复预算。
    old.status = "succeeded"
    wrong.status = "succeeded"
    session.commit()
    enqueue_input_tick(session, "b", "next")
    session.commit()
    newest = session.scalar(select(Job).where(Job.payload["input_tick"].as_boolean().is_(True)))
    assert newest is not None and newest.payload["project_id"] == "p"


def test_life_parameters_rejected_inside_submission_tool(session):
    _, now = prepare(session)
    data = candidate(
        plan_proposals=[
            {
                "plan_date": (now + timedelta(days=1)).date().isoformat(),
                "blocks": [
                    {
                        "start": "00:00",
                        "end": "10:07",
                        "activity": "休息",
                        "basis": "fallback",
                        "confidence": "fallback",
                        "evidence_ids": [],
                    },
                    {
                        "start": "10:07",
                        "end": "24:00",
                        "activity": "整理",
                        "basis": "fallback",
                        "confidence": "fallback",
                        "evidence_ids": [],
                    },
                ],
            }
        ]
    )
    tool = build_submit_life_result_tool(
        LifeAdvanceDecision,
        context={"virtual_now": now.isoformat()},
        validate_plan_sources=lambda refs: RuntimeExecutor(session)._validate_plan_evidence_sources(
            "b", refs
        ),
    )
    with submission_context(None) as receipt:
        result = tool.tool.invoke({"result": data})
        assert result["status"] == "rejected" and "15 分钟" in result["message"]
        assert receipt.value is None
        blocks = data["plan_proposals"][0]["blocks"]
        blocks[0]["end"] = blocks[1]["start"] = "10:15"
        blocks[0]["basis"] = "profile_inference"
        blocks[0]["confidence"] = "inferred"
        blocks[0]["evidence_ids"] = ["other-branch-source"]
        result = tool.tool.invoke({"result": data})
        assert result["status"] == "rejected" and "证据" in result["message"]
        blocks[0]["basis"] = "fallback"
        blocks[0]["confidence"] = "fallback"
        blocks[0]["evidence_ids"] = []
        assert tool.tool.invoke({"result": data})["status"] == "accepted"
