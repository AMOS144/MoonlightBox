import logging
import os
import signal
from threading import Event, Thread
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
from moonlightbox.model_settings import apply_saved_agent_settings
from moonlightbox.node_investigation.jobs import create_investigation_handler
from moonlightbox.node_investigation.store import JOB_KIND as NODE_INVESTIGATION_JOB_KIND
from moonlightbox.observability import initialize_phoenix, shutdown_phoenix
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
from moonlightbox.world.person_world.jobs import (
    PERSON_WORLD_PROFILE_RECOMPILE_JOB_KIND,
    WORLD_GRAPH_PATCH_JOB_KIND,
    create_graph_patch_handler,
    create_profile_recompile_handler,
)
from moonlightbox.world.person_world.node_jobs import (
    NODE_PROFILE_JOB_KIND,
    create_node_compilation_handler,
)
from moonlightbox.world.person_world.review import (
    REVISION_TURN_JOB_KIND,
    create_revision_turn_handler,
)
from moonlightbox.world.person_world.review.profile_preview_jobs import (
    PROFILE_PREVIEW_JOB_KIND,
    create_profile_preview_handler,
)
from moonlightbox.world.person_world.section_retry import (
    PERSON_WORLD_SECTION_RETRY_JOB_KIND,
    create_section_retry_handler,
)

IDLE_SLEEP_SECONDS = 0.25
# Worker 在租约尚未到期时重启是正常情况（例如 systemd 自动重启）。
# 不能只在启动时恢复一次，否则等租约过期后任务会一直卡在 running。
INTERRUPTED_JOB_RECOVERY_INTERVAL_SECONDS = 5.0
LOGGER = logging.getLogger(__name__)
BACKGROUND_JOB_KINDS = {
    NODE_PROFILE_JOB_KIND,
    NODE_INVESTIGATION_JOB_KIND,
    ANALYSIS_JOB_KIND,
    V3_ANALYSIS_JOB_KIND,
    SPATIAL_ANALYSIS_JOB_KIND,
    WORLD_BUILD_JOB_KIND,
    WORLD_GRAPH_PATCH_JOB_KIND,
    PERSON_WORLD_PROFILE_RECOMPILE_JOB_KIND,
    PERSON_WORLD_SECTION_RETRY_JOB_KIND,
    REVISION_TURN_JOB_KIND,
    PROFILE_PREVIEW_JOB_KIND,
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

    return role in {"realtime", "cognition", "background", "all"}


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
    # PersonWorld、Revision 与 Runtime Cycle 都在独立 Worker 中执行；这里的初始化
    # 让它们与 API 共享同一 Phoenix project，又以 role 作为可筛选的 service 标签。
    initialize_phoenix(settings, service_name=f"worker:{worker_role}")
    allowed_kinds = allowed_job_kinds_for_role(worker_role)
    database = Database(settings.database_url)
    stop_event = Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    registry = JobRegistry()
    # 普通 Runtime / PersonWorld Worker 不做模型验收，不应仅因注册训练
    # handler 就加载本地 embedding。否则每个开发 Worker 都会白占一份内存。
    training_enabled_for_role = allowed_kinds is None or TRAINING_JOB_KIND in allowed_kinds
    embedding_root = settings.model_dir / "embeddings" / "fastembed-bge-small-zh-v1.5"
    embedder = (
        LocalChineseEmbedder(embedding_root)
        if training_enabled_for_role and embedding_root.is_dir()
        else None
    )
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
        WORLD_GRAPH_PATCH_JOB_KIND,
        create_graph_patch_handler(settings),
    )
    registry.register(
        PERSON_WORLD_PROFILE_RECOMPILE_JOB_KIND,
        create_profile_recompile_handler(settings),
    )
    registry.register(
        PERSON_WORLD_SECTION_RETRY_JOB_KIND,
        create_section_retry_handler(settings),
    )
    registry.register(
        REVISION_TURN_JOB_KIND,
        create_revision_turn_handler(settings),
    )
    registry.register(PROFILE_PREVIEW_JOB_KIND, create_profile_preview_handler(settings))
    registry.register(NODE_INVESTIGATION_JOB_KIND, create_investigation_handler(settings))
    registry.register(NODE_PROFILE_JOB_KIND, create_node_compilation_handler(settings))
    if training_enabled_for_role:
        registry.register(
            TRAINING_JOB_KIND,
            create_platform_training_handler(settings, embedder),
        )
    registry.register(
        RUNTIME_CYCLE_JOB_KIND,
        create_runtime_cycle_handler(settings=settings),
    )
    worker = Worker(
        database,
        registry,
        stop_event=stop_event,
        allowed_kinds=allowed_kinds,
    )

    def scan_runtime_loop():
        """扫描独立于耗时 Agent，使用自己的数据库会话，不执行模型或训练。"""
        next_tick = monotonic()
        while not stop_event.is_set():
            scan_due_runtime_wakeups(database)
            # 固定节拍；扫描跨拍则跳过过期拍，不补跑积压的扫描任务。
            next_tick += 3.0
            now = monotonic()
            if next_tick <= now:
                next_tick += (int((now - next_tick) // 3.0) + 1) * 3.0
            stop_event.wait(max(0.0, next_tick - now))

    runtime_scanner = None
    if role_scans_due_wakeups(worker_role):
        runtime_scanner = Thread(target=scan_runtime_loop, name="runtime-scheduler", daemon=True)
        runtime_scanner.start()
    def scan_recovery_loop():
        while not stop_event.is_set():
            scan_interrupted_jobs(database)
            stop_event.wait(INTERRUPTED_JOB_RECOVERY_INTERVAL_SECONDS)

    recovery_scanner = Thread(target=scan_recovery_loop, name="job-recovery", daemon=True)
    recovery_scanner.start()
    try:
        recover_interrupted_jobs(database)
        settings_path = settings.data_dir / "agent-model-settings.json"
        settings_mtime_ns = -1
        while not stop_event.is_set():
            current_mtime_ns = settings_path.stat().st_mtime_ns if settings_path.exists() else 0
            if current_mtime_ns != settings_mtime_ns:
                apply_saved_agent_settings(settings)
                settings_mtime_ns = current_mtime_ns
            if not worker.run_once():
                stop_event.wait(IDLE_SLEEP_SECONDS)
    finally:
        stop_event.set()
        recovery_scanner.join(timeout=5)
        if runtime_scanner is not None:
            runtime_scanner.join(timeout=5)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        shutdown_phoenix()
        database.close()


if __name__ == "__main__":
    main()
