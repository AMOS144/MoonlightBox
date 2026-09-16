"""真实 SQLite、LangGraph 与原生工具替身；无供应商请求。"""

# ruff: noqa: F811
from datetime import UTC, datetime, timedelta

import pytest
from moonlightbox.jobs.models import Job
from moonlightbox.runtime_v1.branch_models import Branch
from moonlightbox.runtime_v1.db_models import (
    RuntimeClockRow,
    RuntimeEventRow,
    RuntimeLifeEventRow,
    RuntimeWakeupRow,
)
from moonlightbox.runtime_v1.life_events.commit import validate_plan_change
from moonlightbox.runtime_v1.life_events.contracts import EventOpportunity
from moonlightbox.runtime_v1.life_events.models import LifeOpportunityRow, LifeScheduleCursor
from moonlightbox.runtime_v1.life_events.policy import LifeEventPolicy, load_policy
from moonlightbox.runtime_v1.life_events.scheduling import scan_branch
from moonlightbox.runtime_v1.life_events.service import process_life_step
from moonlightbox.runtime_v1.service import RuntimeService
from sqlalchemy import select
from test_peer_collaboration import Model, session  # noqa: F401


def candidate(**changes):
    return {
        "outcome": "submit_event",
        "reason": "结合目前工作情境自然展开",
        "event": {
            "occurrence": {"summary": "整理资料时发现有用的旧笔记"},
            "situation": "工作中",
            "characteristics": {"disruption": "无"},
            "impact": {"estimated_added_minutes": 0},
            "handling": {"available_options": ["先标记"], "chosen_intent": "先标记"},
        },
        **changes,
    }


def prepare(session, outputs=None):
    runtime = RuntimeService(
        session, director_model=Model([]), planner_model=Model(outputs or [candidate()])
    )
    bootstrap = runtime.bootstrap("p", "b")
    clock = session.get(RuntimeClockRow, "b")
    now = datetime.now(UTC).replace(hour=10, minute=0, second=0, microsecond=0)
    clock.virtual_anchor, clock.wall_anchor, clock.status = now, datetime.now(UTC), "running"
    plan = bootstrap["plan"]
    plan.plan_date = now.date().isoformat()
    plan.blocks = [
        {
            "id": "block",
            "start": "00:00",
            "end": "24:00",
            "activity": "工作",
            "location_role": "work",
            "default_availability": "busy",
            "basis": "fallback",
            "confidence": "fallback",
            "evidence_ids": [],
        }
    ]
    plan.generation_metadata = {"status": "agent"}
    session.get(Branch, "b").lifecycle_status = "active"
    session.commit()
    return runtime, now


def opportunity(session, now, *, kind="opportunity", work=None):
    row = LifeOpportunityRow(
        branch_id="b",
        status="active",
        kind=kind,
        cursor_at=now,
        brief={"direction": "competence", "intensity": 0.3} if kind == "opportunity" else {},
        policy=load_policy().model_dump(),
        work=work or {},
    )
    session.add(row)
    session.commit()
    return row


def step(session, runtime, row):
    job = Job(
        kind="runtime-v1-cycle",
        status="running",
        worker_token="test",
        lease_expires_at=datetime.now(UTC) + timedelta(hours=1),
        payload={"branch_id": "b", "life_opportunity_id": row.id},
    )
    session.add(job)
    session.flush()
    row.job_id = job.id
    session.commit()
    process_life_step(runtime, job)
    job.status, job.worker_token, job.lease_expires_at = "succeeded", None, None
    session.commit()
    return job


def certain_policy(p=1):
    data = load_policy().model_dump()
    data["trigger_probability"]["anchors"] = {"00:00": p, "12:00": p}
    return LifeEventPolicy.model_validate(data)


def test_curve_seed_and_minimal_contract():
    policy = load_policy()
    for key, probability in policy.trigger_probability.anchors.items():
        at = datetime.fromisoformat("2026-05-09T" + key + ":00+08:00")
        assert policy.trigger_probability.at(at) == pytest.approx(probability)
    midnight = datetime(2026, 5, 9, tzinfo=UTC)
    assert (
        abs(
            policy.trigger_probability.at(midnight - timedelta(seconds=1))
            - policy.trigger_probability.at(midnight + timedelta(seconds=1))
        )
        < 1e-5
    )
    assert policy.sample("same", midnight) == policy.sample("same", midnight)
    hit = certain_policy().sample("same", midnight)[1]
    assert set(hit.model_dump()) == {"direction", "intensity"}
    assert 0 <= hit.intensity <= 1
    with pytest.raises(ValueError):
        EventOpportunity(direction="relatedness", intensity=float("nan"))


def test_no_hit_advances_cursor_no_job(session):
    runtime, now = prepare(session)
    clock = session.get(RuntimeClockRow, "b")
    wall = clock.wall_anchor
    branch = session.get(Branch, "b")
    policy = certain_policy(0)
    scan_branch(session, branch, wall, policy=policy)
    # 心跳连续但虚拟加速：不是离线，不把大段虚拟推进误当停机。
    clock.time_scale = 1200
    session.commit()
    assert scan_branch(session, branch, wall + timedelta(seconds=1), policy=policy) == 0
    session.commit()
    cursor = session.get(LifeScheduleCursor, "b")
    assert cursor.sequence == 1 and not cursor.last_check["hit"]
    assert not list(session.scalars(select(LifeOpportunityRow)))
    assert not list(session.scalars(select(Job)))
    scan_branch(session, branch, wall + timedelta(seconds=1), policy=policy)
    assert cursor.sequence == 1


def test_busy_accumulates_without_cooldown_and_restart_redraw(session):
    runtime, now = prepare(session)
    clock = session.get(RuntimeClockRow, "b")
    wall = clock.wall_anchor
    branch = session.get(Branch, "b")
    policy = certain_policy()
    scan_branch(session, branch, wall, policy=policy)
    clock.time_scale = 1200
    scan_branch(session, branch, wall + timedelta(seconds=1), policy=policy)
    session.commit()
    first = session.scalar(select(LifeOpportunityRow))
    seed = first.seed
    for second in range(2, 6):
        scan_branch(session, branch, wall + timedelta(seconds=second), policy=policy)
        session.commit()
    assert len(list(session.scalars(select(LifeOpportunityRow)))) == 5
    assert len(list(session.scalars(select(Job)))) == 1
    assert first.seed == seed  # 运行中仍抽后续机会；新消息不取消当前任务。
    session.expire_all()
    scan_branch(session, branch, wall + timedelta(seconds=5), policy=policy)
    assert len(list(session.scalars(select(LifeOpportunityRow)))) == 5


def test_pause_and_offline_recovery_not_replay_lottery(session):
    runtime, now = prepare(session)
    clock = session.get(RuntimeClockRow, "b")
    wall = clock.wall_anchor
    branch = session.get(Branch, "b")
    scan_branch(session, branch, wall, policy=certain_policy())
    clock.status = "paused"
    scan_branch(session, branch, wall + timedelta(hours=3), policy=certain_policy())
    assert session.get(LifeScheduleCursor, "b").sequence == 0
    clock.status = "running"
    scan_branch(session, branch, wall + timedelta(hours=3, seconds=1), policy=certain_policy())
    session.commit()
    scan_branch(session, branch, wall + timedelta(hours=7), policy=certain_policy())
    session.commit()
    rows = list(session.scalars(select(LifeOpportunityRow)))
    assert len(rows) == 1 and rows[0].kind == "recovery"
    assert rows[0].brief == {} and "recovery" in rows[0].work


def test_planner_decides_without_director_approval_and_replay(session):
    runtime, now = prepare(session)
    row = opportunity(session, now)
    job = step(session, runtime, row)
    assert row.status == "committed"
    assert not runtime.director.model.inputs
    assert len(runtime.planner.model.inputs) == 1
    # 已提交任务直接重放，不需要再次运行模型。
    job.status, job.worker_token = "running", "test"
    job.lease_expires_at = datetime.now(UTC) + timedelta(hours=1)
    session.commit()
    process_life_step(runtime, job)
    events = list(
        session.scalars(
            select(RuntimeLifeEventRow).where(RuntimeLifeEventRow.event_type == "simulated_life")
        )
    )
    notices = [
        e
        for e in session.scalars(select(RuntimeEventRow))
        if e.payload.get("reason") == "simulated_life_committed"
    ]
    assert len(events) == len(notices) == 1


def test_no_event_no_notifications(session):
    runtime, now = prepare(session, [{"outcome": "no_event", "reason": "现在不需要变化"}])
    row = opportunity(session, now)
    step(session, runtime, row)
    assert row.status == "no_event"
    assert not list(session.scalars(select(RuntimeEventRow)))


def test_public_reply_resumes_original_work(session):
    from moonlightbox.runtime_v1.collaboration.transport import schedule_planner_inbox
    from moonlightbox.runtime_v1.tools.send_agent_message import build_send_agent_message_tool

    runtime, now = prepare(
        session,
        [
            {
                "outcome": "discuss",
                "reason": "想调整安排",
                "question": "今天更想休息还是继续整理？",
            },
            candidate(),
        ],
    )
    row = opportunity(session, now)
    step(session, runtime, row)
    assert row.work["awaiting_director_event_id"]
    assert not runtime.director.model.inputs
    sent = build_send_agent_message_tool(
        session, branch_id="b", sender="director", task_id="answer"
    ).invoke({"recipient": "day_planner", "content": "先休息一下，再整理；不用问我批准每个细节"})
    schedule_planner_inbox(session, "b", runtime.get_clock("p", "b").now())
    session.commit()
    assert row.work.get("awaiting_director_event_id") is None
    step(session, runtime, row)
    assert row.status == "committed"
    assert "先休息一下" in str(runtime.planner.model.inputs[-1])
    assert session.get(RuntimeEventRow, sent["message_id"]).status == "completed"


def test_old_limits_do_not_reject_long_or_cross_day_effect(session):
    runtime, now = prepare(session)
    value = candidate()
    value["event"]["impact"] = {
        "estimated_added_minutes": 180,
        "scope": "后续数天",
        "expected_long_term_consequences": ["可能重新安排整理习惯"],
    }
    runtime.planner.model.outputs = [value]
    row = opportunity(session, now)
    # 时间已延迟超过旧 TTL、旧块，也不会自动取消。
    row.cursor_at = now - timedelta(hours=3)
    for n in range(4):
        session.add(
            RuntimeLifeEventRow(
                branch_id="b",
                event_type="simulated_life",
                occurred_at=now,
                payload={"budget_minutes": 40},
                idempotency_key=f"old-{n}",
            )
        )
    session.commit()
    step(session, runtime, row)
    assert row.status == "committed"


def test_cross_block_prefix_protection():
    now = datetime(2026, 5, 9, 10, tzinfo=UTC)
    old = [
        {"start": "00:00", "end": "12:00", "activity": "工作"},
        {"start": "12:00", "end": "24:00", "activity": "原安排"},
    ]
    new = [
        {"start": "00:00", "end": "12:00", "activity": "工作"},
        {"start": "12:00", "end": "24:00", "activity": "新安排"},
    ]
    validate_plan_change(old, new, now, now.date())
    new[0]["activity"] = "改写过去"
    with pytest.raises(ValueError, match="前缀"):
        validate_plan_change(old, new, now, now.date())


def test_followup_and_recovery_are_not_future_facts(session):
    runtime, now = prepare(
        session, [candidate(follow_up_at=(datetime.now(UTC) + timedelta(days=1)).isoformat())]
    )
    row = opportunity(session, now)
    step(session, runtime, row)
    wake = session.scalar(
        select(RuntimeWakeupRow).where(RuntimeWakeupRow.trigger_type == "life_followup")
    )
    assert wake and wake.reason == row.event_id
    recovery = opportunity(
        session,
        now,
        kind="recovery",
        work={"recovery": {"from": (now - timedelta(hours=2)).isoformat(), "to": now.isoformat()}},
    )
    value = candidate()
    value["event"]["occurrence"]["occurred_at"] = (now - timedelta(hours=1)).isoformat()
    runtime.planner.model.outputs = [value]
    step(session, runtime, recovery)
    event = session.get(RuntimeLifeEventRow, recovery.event_id)
    assert event.payload["generated_during_recovery"]


def test_configuration_has_no_enable_flag():
    from moonlightbox.config import Settings

    assert "runtime_life_events_enabled" not in Settings.model_fields


def test_future_plan_commit_and_batch_completion(session):
    from moonlightbox.runtime_v1.db_models import RuntimeDayPlanRow

    runtime, now = prepare(session)
    tomorrow = (now + timedelta(days=2)).date().isoformat()
    runtime.planner.model.outputs = [
        candidate(
            plan_proposals=[
                {
                    "plan_date": tomorrow,
                    "blocks": [
                        {
                            "start": "00:00",
                            "end": "24:00",
                            "activity": "整理旧资料",
                            "basis": "fallback",
                            "confidence": "fallback",
                            "evidence_ids": [],
                        }
                    ],
                }
            ]
        )
    ]
    row = opportunity(session, now)
    member = opportunity(session, now + timedelta(seconds=1))
    member.status = "batched"
    row.work = {"batch_ids": [row.id, member.id]}
    session.commit()
    step(session, runtime, row)
    plan = session.scalar(
        select(RuntimeDayPlanRow).where(
            RuntimeDayPlanRow.branch_id == "b",
            RuntimeDayPlanRow.plan_date == tomorrow,
        )
    )
    assert plan and plan.blocks[0]["activity"] == "整理旧资料"
    assert member.status == "committed" and member.event_id == row.event_id


def test_policy_errors_are_not_silent_fallback(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("version: first\nversion: second\n")
    with pytest.raises(ValueError, match="重复"):
        load_policy(str(path))


def test_followup_is_delivered_without_probability_hit(session):
    runtime, now = prepare(session)
    clock = session.get(RuntimeClockRow, "b")
    wall = clock.wall_anchor
    branch = session.get(Branch, "b")
    scan_branch(session, branch, wall, policy=certain_policy(0))
    wake = RuntimeWakeupRow(
        branch_id="b",
        wake_at=now + timedelta(seconds=1),
        trigger_type="life_followup",
        reason="existing-event",
        idempotency_key="followup-test",
    )
    session.add(wake)
    session.commit()
    scan_branch(session, branch, wall + timedelta(seconds=2), policy=certain_policy(0))
    session.commit()
    row = session.scalar(select(LifeOpportunityRow))
    assert row.kind == "followup" and row.work["parent_event_id"] == "existing-event"
    assert wake.status == "completed" and session.get(LifeScheduleCursor, "b").sequence == 0
