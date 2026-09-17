"""整任务只有 Job 决定恢复；终态释放占位但不会偷偷重建旧输入。"""

# ruff: noqa: F811

from datetime import UTC, datetime, timedelta

import pytest
from moonlightbox.jobs.runtime_recovery import recovery, resume_job, scan_recovery
from moonlightbox.jobs.service import JobService
from moonlightbox.runtime_v1.db_models import RuntimeClockRow, RuntimeEventRow
from moonlightbox.runtime_v1.jobs import enqueue_input_tick
from test_peer_collaboration import session  # noqa: F401


def setup_job(session):
    now = datetime.now(UTC)
    session.add(RuntimeClockRow(branch_id="b", virtual_anchor=now, wall_anchor=now))
    for key in ("old", "read_during_run", "new"):
        session.add(
            RuntimeEventRow(
                id=key,
                project_id="p",
                branch_id="b",
                event_type="user_message",
                idempotency_key=key,
            )
        )
    service = JobService(session)
    job = service.enqueue("runtime-v1-cycle", {"branch_id": "b", "input_tick": True})
    job.checkpoint = {
        "runtime_cycle": {
            "key": "stable",
            "trigger_ids": ["old"],
            "input_event_ids": ["old", "read_during_run"],
        },
        "work": {"draft": "保留", "usage": 12345},
    }
    session.commit()
    service.claim_next_queued(worker_token="test", lease_duration=timedelta(minutes=2))
    return service, job


def test_transient_recovery_preserves_identity_and_work(session):
    service, job = setup_job(session)
    service.fail(job.id, "network", "暂时断网", token="test")
    assert recovery(job)["status"] == "waiting"
    assert not enqueue_input_tick(session, "b", "new")
    assert not resume_job(session, job, datetime.now(UTC))
    scan_recovery(session, datetime.now(UTC) + timedelta(minutes=1))
    assert job.status == "queued"
    assert job.checkpoint["runtime_retry_attempt"] == 1
    assert job.checkpoint["work"] == {"draft": "保留", "usage": 12345}
    assert job.checkpoint["runtime_cycle"]["key"] == "stable"


@pytest.mark.parametrize(
    "code,retries", [("authentication", 0), ("collaboration_budget_exhausted", 0), ("network", 3)]
)
def test_terminal_releases_slot_without_replaying_old_input(session, code, retries):
    service, job = setup_job(session)
    job.checkpoint = {**job.checkpoint, "runtime_retry_attempt": retries}
    session.commit()
    service.fail(job.id, code, "失败", token="test")
    assert recovery(job)["status"] == "terminal"
    for key in ("old", "read_during_run"):
        assert session.get(RuntimeEventRow, key).status == "failed"
    assert session.get(RuntimeEventRow, "new").status == "queued"
    scan_recovery(session, datetime.now(UTC) + timedelta(days=1))
    assert job.status == "failed"
    # terminal 只阻止自动恢复；用户修复配额/密钥等外部原因后显式重试应当放行，
    # 恢复后重新占位，新输入不会绕过它另起任务。
    service.resume(job.id)
    assert job.status == "queued"
    assert recovery(job) == {}
    assert job.checkpoint["runtime_retry_attempt"] == retries + 1
    assert not enqueue_input_tick(session, "b", "new")


def test_expired_worker_uses_same_recovery_policy(session):
    service, job = setup_job(session)
    before = dict(job.checkpoint)
    future = datetime.now(UTC) + timedelta(hours=1)
    assert service.recover_expired_running(now=future) == 1
    assert job.status == "interrupted"
    assert recovery(job)["status"] == "waiting"
    scan_recovery(session, future + timedelta(minutes=1))
    assert job.status == "queued"
    assert job.checkpoint["runtime_retry_attempt"] == 1
    assert job.checkpoint["work"] == before["work"]
