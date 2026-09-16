"""Send 编排烟测：有界并发、阶段快照、协调线程写入及取消回收。"""

import asyncio
from copy import deepcopy
from threading import Event, Lock, get_ident
from time import sleep

import pytest
from moonlightbox.agent_runtime.async_execution import run_sync_owned
from moonlightbox.agent_runtime.cancellation import cancellation_requested, cancellation_scope
from moonlightbox.db import Base
from moonlightbox.world.models import WorldGraphVersion
from moonlightbox.world.person_world.coordinator_v3 import PersonWorldCoordinatorV3
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture
def coordinator(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/world.db")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        graph = WorldGraphVersion(
            id="graph",
            project_id="project",
            trigger_import_id="import",
            workspace_key="test",
            status="compiling_profile",
            source_fingerprint="s",
            config_fingerprint="c",
            source_import_ids=[],
            compiler_version="v3",
        )
        session.add(graph)
        session.commit()
        value = PersonWorldCoordinatorV3(
            session=session,
            graph=graph,
            lightrag=None,
            compiler=None,
            subject_name="目标",
            user_name="用户",
            target_participant_id="target",
            user_participant_id="user",
            section_concurrency=2,
        )
        value._initialize(mode="initial_compile", resume_key="test")
        yield value
    engine.dispose()


def test_send_concurrency_frozen_inputs_and_coordinator_writes(coordinator, monkeypatch):
    owner = get_ident()
    lock = Lock()
    running = peak = 0
    seen = {}
    accepted = []
    coordinator.baseline = {"life_context": {"summary": "已发布背景"}}

    def worker(section, phase, snapshot, context):
        nonlocal running, peak
        assert get_ident() != owner
        with lock:
            running += 1
            peak = max(peak, running)
            seen[section] = deepcopy(context)
        sleep(0.04)
        with lock:
            running -= 1
        return {"summary": section}

    def accept(section, phase, result):
        assert get_ident() == owner
        accepted.append(section)
        coordinator.results[section] = result

    monkeypatch.setattr(coordinator, "_worker", worker)
    monkeypatch.setattr(coordinator, "_accept", accept)

    async def exercise():
        ticked = False

        async def ticker():
            nonlocal ticked
            await asyncio.sleep(0.01)
            ticked = True
            assert running > 0

        await asyncio.gather(
            coordinator._batch(["agency", "practices", "identity"], "draft"),
            ticker(),
        )
        assert ticked
        assert peak == 2
        assert set(accepted) == {"agency", "practices", "identity"}
        assert all(item["profile_snapshot"] == coordinator.baseline for item in seen.values())
        # identity 后补全才读到其他栏目已提交结果，而不是碰运气读取同批次输出。
        await coordinator._batch(["identity"], "enrichment")
        assert seen["identity"]["profile_snapshot"]["agency"]["summary"] == "agency"

    asyncio.run(exercise())


def test_one_failed_send_does_not_discard_sibling_result(coordinator, monkeypatch):
    accepted, failed = [], []

    def worker(section, *args):
        if section == "agency":
            raise ValueError("模拟单栏失败")
        return {}

    monkeypatch.setattr(coordinator, "_worker", worker)
    monkeypatch.setattr(coordinator, "_accept", lambda section, *args: accepted.append(section))
    monkeypatch.setattr(coordinator, "_failed", lambda section, *args: failed.append(section))
    asyncio.run(coordinator._batch(["agency", "practices"], "draft"))
    assert accepted == ["practices"]
    assert failed == ["agency"]


def test_async_cancellation_waits_for_owned_worker_exit():
    started, closed = Event(), Event()

    def operation():
        started.set()
        try:
            while not cancellation_requested():
                sleep(0.005)
            sleep(0.01)  # 模拟关闭 Session／同步客户端，退出前不能报告已停止。
        finally:
            closed.set()

    async def exercise():
        task = asyncio.create_task(run_sync_owned(operation))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()

    asyncio.run(exercise())


def test_parent_cancellation_context_is_preserved():
    async def exercise():
        with cancellation_scope(lambda: True):
            assert await run_sync_owned(cancellation_requested)

    asyncio.run(exercise())


def test_sync_entry_rejects_nested_loop(coordinator):
    async def exercise():
        with pytest.raises(RuntimeError, match="await coordinator.arun"):
            coordinator.run()

    asyncio.run(exercise())
