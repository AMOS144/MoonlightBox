from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from time import sleep

from moonlightbox.db import Database
from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobRegistry
from moonlightbox.jobs.service import JobService
from sqlalchemy.orm import Session


def test_worker_roles_isolate_realtime_cognition_and_background_kinds() -> None:
    from moonlightbox.agent.extraction import COGNITIVE_EXTRACTION_JOB_KIND
    from moonlightbox.agent.jobs import COGNITIVE_CYCLE_JOB_KIND
    from moonlightbox.branches.actor import BRANCH_CONVERSATION_JOB_KIND
    from moonlightbox.runtime_v1.jobs import RUNTIME_CYCLE_JOB_KIND
    from moonlightbox.training.confirmation import TRAINING_JOB_KIND
    from moonlightbox.worker_main import (
        BACKGROUND_JOB_KINDS,
        allowed_job_kinds_for_role,
    )

    assert allowed_job_kinds_for_role("realtime") == {BRANCH_CONVERSATION_JOB_KIND}
    assert allowed_job_kinds_for_role("cognition") == {
        COGNITIVE_CYCLE_JOB_KIND,
        RUNTIME_CYCLE_JOB_KIND,
    }
    assert COGNITIVE_CYCLE_JOB_KIND not in BACKGROUND_JOB_KINDS
    assert RUNTIME_CYCLE_JOB_KIND in BACKGROUND_JOB_KINDS
    assert COGNITIVE_EXTRACTION_JOB_KIND in BACKGROUND_JOB_KINDS
    assert COGNITIVE_EXTRACTION_JOB_KIND not in allowed_job_kinds_for_role("realtime")
    assert COGNITIVE_EXTRACTION_JOB_KIND not in allowed_job_kinds_for_role("cognition")
    assert allowed_job_kinds_for_role("background") == BACKGROUND_JOB_KINDS
    assert TRAINING_JOB_KIND not in BACKGROUND_JOB_KINDS
    assert allowed_job_kinds_for_role("training") == {TRAINING_JOB_KIND}
    assert allowed_job_kinds_for_role("all") is None


def test_background_consumers_reuse_same_json_generator(tmp_path: Path) -> None:
    from moonlightbox.worker_main import create_background_json_consumers

    database = Database(f"sqlite:///{tmp_path / 'shared-generator.db'}")
    generated: list[object] = []

    class JsonGenerator:
        def generate_json(self, **_kwargs: object) -> dict[str, object]:
            return {}

    def factory(_database: Database) -> JsonGenerator:
        generator = JsonGenerator()
        generated.append(generator)
        return generator

    proposer, extractor = create_background_json_consumers(
        database,
        generator_factory=factory,
    )

    assert len(generated) == 1
    assert proposer._generator is generated[0]
    assert extractor._generator is generated[0]


def test_unknown_job_kind_is_marked_failed(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'worker.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job = JobService(session).enqueue("unknown", {})
        job_id = job.id

    from moonlightbox.worker import Worker

    Worker(database, JobRegistry()).run_once()

    with Session(database.engine) as session:
        failed = JobService(session).get(job_id)
        assert failed.status == "failed"
        assert failed.error_code == "unknown_job_kind"


def test_registered_job_handler_succeeds(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'registered.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job = JobService(session).enqueue("sample", {"value": 1})
        job_id = job.id

    handled: list[str] = []
    registry = JobRegistry()
    registry.register("sample", lambda _service, current: handled.append(current.id))

    from moonlightbox.worker import Worker

    Worker(database, registry).run_once()

    with Session(database.engine) as session:
        succeeded = JobService(session).get(job_id)
        assert succeeded.status == "succeeded"
    assert handled == [job_id]


def test_worker_lane_claims_only_allowed_job_kinds(tmp_path: Path) -> None:
    from moonlightbox.worker import Worker

    database = Database(f"sqlite:///{tmp_path / 'worker-lanes.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        background_id = JobService(session).enqueue("memory", {}).id
        realtime_id = JobService(session).enqueue("conversation", {}).id

    handled: list[str] = []
    registry = JobRegistry()
    registry.register(
        "conversation",
        lambda _service, current: handled.append(current.id),
    )

    assert (
        Worker(
            database,
            registry,
            allowed_kinds={"conversation"},
        ).run_once()
        is True
    )

    with Session(database.engine) as session:
        assert JobService(session).get(realtime_id).status == "succeeded"
        assert JobService(session).get(background_id).status == "queued"
    assert handled == [realtime_id]


def test_handler_exception_is_safely_marked_failed(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'handler-failure.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job_id = JobService(session).enqueue("sample", {}).id

    registry = JobRegistry()

    def fail_handler(_service: JobService, _job: Job) -> None:
        raise RuntimeError("不应写入数据库的敏感详情")

    registry.register("sample", fail_handler)

    from moonlightbox.worker import Worker

    assert Worker(database, registry).run_once() is True

    with Session(database.engine) as session:
        failed = JobService(session).get(job_id)
        assert failed.status == "failed"
        assert failed.error_code == "job_handler_failed"
        assert failed.error_message == "任务处理失败"


def test_two_workers_cannot_execute_the_same_job(tmp_path: Path) -> None:
    from moonlightbox.worker import Worker

    database_url = f"sqlite:///{tmp_path / 'worker-race.db'}"
    setup_database = Database(database_url)
    Job.metadata.create_all(setup_database.engine)
    with Session(setup_database.engine) as session:
        job_id = JobService(session).enqueue("sample", {}).id
    setup_database.close()

    barrier = Barrier(2)
    handled: list[str] = []
    handled_lock = Lock()
    results: list[bool] = []

    def run_worker() -> None:
        database = Database(database_url)
        registry = JobRegistry()

        def handler(_service: JobService, current: Job) -> None:
            with handled_lock:
                handled.append(current.id)

        registry.register("sample", handler)
        barrier.wait()
        results.append(Worker(database, registry).run_once())
        database.close()

    threads = [Thread(target=run_worker), Thread(target=run_worker)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert handled == [job_id]
    assert sorted(results) == [False, True]


def test_recover_running_jobs_preserves_checkpoint_and_requeues(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'recovery.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        service = JobService(session)
        job = service.enqueue("sample", {})
        running = service.start(
            job.id,
            now=datetime.now(UTC) - timedelta(minutes=5),
            lease_duration=timedelta(seconds=1),
        )
        assert running.worker_token is not None
        service.checkpoint(
            job.id,
            {"stage": "windows", "window": 2, "progress": 0.5},
            token=running.worker_token,
        )

    from moonlightbox.worker import recover_interrupted_jobs

    assert recover_interrupted_jobs(database) == 1

    with Session(database.engine) as session:
        recovered = JobService(session).get(job.id)
        assert recovered.status == "queued"
        assert recovered.checkpoint == {
            "stage": "windows",
            "window": 2,
            "progress": 0.5,
        }


def test_running_cancel_keeps_job_cancelled_when_handler_returns(tmp_path: Path) -> None:
    from moonlightbox.worker import Worker

    database = Database(f"sqlite:///{tmp_path / 'cancel.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job_id = JobService(session).enqueue("sample", {}).id

    registry = JobRegistry()
    registry.register("sample", lambda service, job: service.cancel(job.id))

    assert Worker(database, registry).run_once() is True
    with Session(database.engine) as session:
        assert JobService(session).get(job_id).status == "cancelled"


def test_cancelled_handler_exception_does_not_escape_or_overwrite_status(
    tmp_path: Path,
) -> None:
    from moonlightbox.worker import Worker

    database = Database(f"sqlite:///{tmp_path / 'cancel-error.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job_id = JobService(session).enqueue("sample", {}).id

    def handler(service: JobService, job: Job) -> None:
        service.cancel(job.id)
        raise RuntimeError("取消后的异常")

    registry = JobRegistry()
    registry.register("sample", handler)

    assert Worker(database, registry).run_once() is True
    with Session(database.engine) as session:
        assert JobService(session).get(job_id).status == "cancelled"


def test_worker_does_not_claim_after_stop_requested(tmp_path: Path) -> None:
    from moonlightbox.worker import Worker

    database = Database(f"sqlite:///{tmp_path / 'stopped.db'}")
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job_id = JobService(session).enqueue("sample", {}).id
    stop_event = Event()
    stop_event.set()

    assert Worker(database, JobRegistry(), stop_event=stop_event).run_once() is False
    with Session(database.engine) as session:
        assert JobService(session).get(job_id).status == "queued"


def test_slow_handler_heartbeat_prevents_startup_takeover(tmp_path: Path) -> None:
    from moonlightbox.worker import Worker, recover_interrupted_jobs

    database_url = f"sqlite:///{tmp_path / 'slow-handler.db'}"
    database = Database(database_url)
    Job.metadata.create_all(database.engine)
    with Session(database.engine) as session:
        job_id = JobService(session).enqueue("sample", {}).id
    entered = Event()
    release = Event()
    registry = JobRegistry()

    def handler(_service: JobService, _job: Job) -> None:
        entered.set()
        assert release.wait(timeout=2)

    registry.register("sample", handler)
    worker = Worker(
        database,
        registry,
        lease_duration=timedelta(milliseconds=90),
        heartbeat_interval=timedelta(milliseconds=20),
    )
    thread = Thread(target=worker.run_once)
    thread.start()
    assert entered.wait(timeout=1)
    sleep(0.15)

    second_database = Database(database_url)
    assert recover_interrupted_jobs(second_database) == 0
    with Session(second_database.engine) as session:
        running = JobService(session).get(job_id)
        assert running.status == "running"
    release.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    with Session(second_database.engine) as session:
        assert JobService(session).get(job_id).status == "succeeded"
    second_database.close()
    database.close()
