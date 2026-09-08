import logging
import os
import signal
from threading import Event
from time import monotonic

from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.embeddings import LocalChineseEmbedder
from moonlightbox.imports.analysis_job import (
    ANALYSIS_JOB_KIND,
    V3_ANALYSIS_JOB_KIND,
    create_event_analysis_v2_handler,
    create_event_analysis_v3_handler,
)
from moonlightbox.jobs.registry import JobHandler, JobRegistry
from moonlightbox.runtime_v1.inference_client import RuntimeInferenceClient
from moonlightbox.runtime_v1.jobs import (
    RUNTIME_CYCLE_JOB_KIND,
    create_runtime_cycle_handler,
    enqueue_due_runtime_cycles,
)
from moonlightbox.spatial.jobs import (
    SPATIAL_ANALYSIS_JOB_KIND,
    create_spatial_analysis_handler,
)
from moonlightbox.training.confirmation import TRAINING_JOB_KIND
from moonlightbox.worker import Worker, recover_interrupted_jobs
from moonlightbox.world.jobs import WORLD_BUILD_JOB_KIND, create_world_build_handler

IDLE_SLEEP_SECONDS = 0.25
# Worker 在租约尚未到期时重启是正常情况（例如 systemd 自动重启）。
# 不能只在启动时恢复一次，否则等租约过期后任务会一直卡在 running。
INTERRUPTED_JOB_RECOVERY_INTERVAL_SECONDS = 5.0
LOGGER = logging.getLogger(__name__)
BACKGROUND_JOB_KINDS = {
    ANALYSIS_JOB_KIND,
    V3_ANALYSIS_JOB_KIND,
    SPATIAL_ANALYSIS_JOB_KIND,
    WORLD_BUILD_JOB_KIND,
    RUNTIME_CYCLE_JOB_KIND,
}


def allowed_job_kinds_for_role(role: str) -> set[str] | None:
    if role == "realtime":
        return {RUNTIME_CYCLE_JOB_KIND}
    if role == "cognition":
        return {RUNTIME_CYCLE_JOB_KIND}
    if role == "background":
        return BACKGROUND_JOB_KINDS
    if role == "training":
        # 训练进程独占 GPU：避免普通后台 Worker 在没有 CUDA 权限的容器中误领
        # QLoRA 任务，也避免与在线 PersonaActor 同时加载模型。
        return {TRAINING_JOB_KIND}
    if role == "all":
        return None
    raise RuntimeError(
        "MOONLIGHTBOX_WORKER_ROLE 必须是 all、realtime、cognition、background 或 training"
    )


def role_scans_due_wakeups(role: str) -> bool:
    """Runtime cycle 与全功能 Worker 扫描虚拟时钟唤醒。"""

    return role in {"cognition", "all"}


def scan_due_runtime_wakeups(database: Database) -> int:
    """把 Runtime v1 到期 Wakeup 转为通用 Worker 任务。"""
    try:
        return enqueue_due_runtime_cycles(database)
    except Exception:
        LOGGER.exception("Runtime v1 唤醒扫描失败")
        return 0


def scan_interrupted_jobs(database: Database) -> int:
    """周期性收回已过期租约，确保自动重启后的任务可以继续执行。"""

    try:
        return recover_interrupted_jobs(database)
    except Exception:
        # 恢复扫描不能影响正在运行的 Worker；下一轮会再次尝试收回租约。
        LOGGER.exception("中断任务恢复扫描失败")
        return 0


def create_platform_training_handler(
    settings: Settings,
    embedder: LocalChineseEmbedder | None,
) -> JobHandler:
    """创建 Linux QLoRA 训练处理器。

    具体依赖在这里延迟导入，缺少 ``linux-ml`` 时普通后台任务仍可运行；训练任务
    会给出可操作的依赖错误，而不再根据 macOS 平台拒绝 Linux。
    """

    from moonlightbox.training.jobs import create_digital_human_training_handler
    from moonlightbox.training.model_acceptance import (
        LocalAcceptanceReviewer,
        ModelAcceptanceRunner,
        default_acceptance_fixture,
    )
    from moonlightbox.training.peft_adapter import PeftLmAdapter
    from moonlightbox.training.peft_generation import PeftPathReplyGenerator

    acceptance_runner = ModelAcceptanceRunner(
        PeftPathReplyGenerator(
            device=settings.persona_device,
            load_in_4bit=settings.persona_load_in_4bit,
        ),
        LocalAcceptanceReviewer(),
        default_acceptance_fixture(),
        semantic_embedder=embedder,
    )
    return create_digital_human_training_handler(
        PeftLmAdapter(
            device=settings.persona_device,
            load_in_4bit=settings.persona_load_in_4bit,
        ),
        data_dir=settings.data_dir,
        model_dir=settings.model_dir,
        acceptance_runner=acceptance_runner,
        required_base_model=settings.training_base_model,
    )


def main() -> None:
    """启动独立任务 Worker，并在收到退出信号时释放资源。"""

    settings = Settings()
    worker_role = os.environ.get("MOONLIGHTBOX_WORKER_ROLE", "all").strip().lower()
    allowed_kinds = allowed_job_kinds_for_role(worker_role)
    database = Database(settings.database_url)
    stop_event = Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    registry = JobRegistry()
    embedding_root = settings.model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
    embedder = LocalChineseEmbedder(embedding_root) if embedding_root.is_dir() else None
    registry.register(
        ANALYSIS_JOB_KIND,
        create_event_analysis_v2_handler(
            settings,
            should_stop=stop_event.is_set,
        ),
    )
    registry.register(
        V3_ANALYSIS_JOB_KIND,
        create_event_analysis_v3_handler(
            settings,
            should_stop=stop_event.is_set,
        ),
    )
    registry.register(
        SPATIAL_ANALYSIS_JOB_KIND,
        create_spatial_analysis_handler(settings),
    )
    registry.register(
        WORLD_BUILD_JOB_KIND,
        create_world_build_handler(settings),
    )
    registry.register(
        TRAINING_JOB_KIND,
        create_platform_training_handler(settings, embedder),
    )
    persona_client = RuntimeInferenceClient(
        settings.persona_inference_url,
        settings.persona_inference_token.get_secret_value(),
        timeout=settings.persona_inference_timeout_seconds,
    )
    registry.register(
        RUNTIME_CYCLE_JOB_KIND,
        create_runtime_cycle_handler(persona_client=persona_client),
    )
    worker = Worker(
        database,
        registry,
        stop_event=stop_event,
        allowed_kinds=allowed_kinds,
    )
    next_runtime_wakeup_scan = monotonic()
    next_interrupted_job_recovery_scan = monotonic()
    try:
        recover_interrupted_jobs(database)
        while not stop_event.is_set():
            if role_scans_due_wakeups(worker_role) and monotonic() >= next_runtime_wakeup_scan:
                scan_due_runtime_wakeups(database)
                next_runtime_wakeup_scan = monotonic() + 1.0
            if monotonic() >= next_interrupted_job_recovery_scan:
                scan_interrupted_jobs(database)
                next_interrupted_job_recovery_scan = (
                    monotonic() + INTERRUPTED_JOB_RECOVERY_INTERVAL_SECONDS
                )
            if not worker.run_once():
                stop_event.wait(IDLE_SLEEP_SECONDS)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        database.close()


if __name__ == "__main__":
    main()
