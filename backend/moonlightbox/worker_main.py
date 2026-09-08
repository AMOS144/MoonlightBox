import logging
import os
import signal
from collections.abc import Callable
from threading import Event
from time import monotonic

from sqlalchemy.orm import Session

from moonlightbox.agent.activation_service import reconcile_subject_agent_modes
from moonlightbox.agent.cloud_cognition import DeepSeekCognitionGenerator
from moonlightbox.agent.extraction import (
    COGNITIVE_EXTRACTION_JOB_KIND,
    CognitiveStructureExtractor,
    create_cognitive_extraction_handler,
)
from moonlightbox.agent.inference_client import PersonaInferenceClient
from moonlightbox.agent.jobs import (
    COGNITIVE_CYCLE_JOB_KIND,
    CognitionGenerator,
    create_cognitive_cycle_handler,
    resume_retryable_cognitive_jobs,
)
from moonlightbox.agent.service import process_due_wakeups
from moonlightbox.branches.actor import (
    BRANCH_CONVERSATION_JOB_KIND,
    create_conversation_actor_handler,
    deliver_due_pending_bubbles,
    enqueue_due_actor_reviews,
)
from moonlightbox.branches.baseline_jobs import (
    BRANCH_BASELINE_JOB_KIND,
    create_branch_baseline_handler,
)
from moonlightbox.branches.continuity_index import BranchContinuityRepository
from moonlightbox.branches.continuity_jobs import (
    BRANCH_REFLECTION_JOB_KIND,
    MemoryReviewer,
    create_branch_reflection_handler,
    create_continual_memory_handler,
    resume_retryable_continuity_jobs,
)
from moonlightbox.branches.embeddings import LocalChineseEmbedder
from moonlightbox.branches.episodes import CONTINUAL_MEMORY_JOB_KIND
from moonlightbox.branches.linux_memory import DatabaseLinuxMemoryJsonGenerator
from moonlightbox.branches.memory_jobs import ProjectMemoryRepository
from moonlightbox.branches.memory_proposer import (
    LocalMemoryProposer,
    MemoryJsonGenerator,
)
from moonlightbox.branches.memory_reviewer import (
    ConservativeLocalMemoryReviewer,
    DeepSeekMemoryReviewer,
)
from moonlightbox.branches.reviewer import DeepSeekReplyReviewer
from moonlightbox.config import Settings
from moonlightbox.db import Database
from moonlightbox.events.cloud_client import NodeAnalysisCloudClient
from moonlightbox.imports.analysis_job import (
    ANALYSIS_JOB_KIND,
    V3_ANALYSIS_JOB_KIND,
    create_event_analysis_v2_handler,
    create_event_analysis_v3_handler,
)
from moonlightbox.jobs.registry import JobHandler, JobRegistry
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
    CONTINUAL_MEMORY_JOB_KIND,
    BRANCH_REFLECTION_JOB_KIND,
    BRANCH_BASELINE_JOB_KIND,
    COGNITIVE_EXTRACTION_JOB_KIND,
    SPATIAL_ANALYSIS_JOB_KIND,
    WORLD_BUILD_JOB_KIND,
    RUNTIME_CYCLE_JOB_KIND,
}


def create_background_json_consumers(
    database: Database,
    *,
    generator_factory: Callable[[Database], MemoryJsonGenerator] = (
        DatabaseLinuxMemoryJsonGenerator
    ),
) -> tuple[LocalMemoryProposer, CognitiveStructureExtractor]:
    """创建共享同一干净基座生成器的后台消费者。"""

    generator = generator_factory(database)
    return LocalMemoryProposer(generator), CognitiveStructureExtractor(generator)


def allowed_job_kinds_for_role(role: str) -> set[str] | None:
    if role == "realtime":
        return {BRANCH_CONVERSATION_JOB_KIND}
    if role == "cognition":
        return {COGNITIVE_CYCLE_JOB_KIND, RUNTIME_CYCLE_JOB_KIND}
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
    """仅认知通道和全功能 Worker 扫描动态唤醒。"""

    return role in {"cognition", "all"}


def scan_due_wakeups(database: Database) -> int:
    """隔离扫描异常，避免单次数据库故障终止 Worker。"""

    try:
        with Session(database.engine) as session:
            return process_due_wakeups(session)
    except Exception:
        LOGGER.exception("动态唤醒扫描失败")
        return 0


def scan_due_runtime_wakeups(database: Database) -> int:
    """把 Runtime v1 到期 Wakeup 转为通用 Worker 任务。"""
    try:
        return enqueue_due_runtime_cycles(database)
    except Exception:
        LOGGER.exception("Runtime v1 唤醒扫描失败")
        return 0


def scan_retryable_cognition(database: Database) -> int:
    """Retry transient cognition failures without requiring manual API calls."""

    try:
        with Session(database.engine) as session:
            return resume_retryable_cognitive_jobs(session)
    except Exception:
        LOGGER.exception("认知失败重试扫描失败")
        return 0


def scan_retryable_continuity(database: Database) -> int:
    """恢复瞬时失败的长期记忆与反思任务。"""

    try:
        with Session(database.engine) as session:
            return resume_retryable_continuity_jobs(session)
    except Exception:
        LOGGER.exception("长期记忆失败重试扫描失败")
        return 0


def scan_interrupted_jobs(database: Database) -> int:
    """周期性收回已过期租约，确保自动重启后的任务可以继续执行。"""

    try:
        return recover_interrupted_jobs(database)
    except Exception:
        # 恢复扫描不能影响正在运行的 Worker；下一轮会再次尝试收回租约。
        LOGGER.exception("中断任务恢复扫描失败")
        return 0


def create_cognition_generator(
    settings: Settings,
    local_persona_client: PersonaInferenceClient,
) -> CognitionGenerator:
    """Select the semantic decision brain independently from persona wording."""

    if settings.cognition_backend == "local":
        return local_persona_client
    api_key = settings.resolved_cognition_api_key()
    if api_key is None or not api_key.get_secret_value().strip():
        raise RuntimeError("云端认知已启用，但没有配置 DeepSeek API key")
    cloud_client = NodeAnalysisCloudClient(
        enabled=True,
        endpoint=settings.cognition_endpoint,
        model=settings.cognition_model,
        api_key=api_key,
        timeout_seconds=settings.cognition_timeout_seconds,
        max_retries=settings.cognition_max_retries,
        response_format=settings.cognition_response_format,
        thinking_mode=settings.cognition_thinking_mode,
        max_output_tokens=settings.cognition_max_output_tokens,
    )
    return DeepSeekCognitionGenerator(
        cloud_client,
        adaptive_deliberation=settings.cognition_thinking_mode == "default",
    )


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
    reply_reviewer = None
    if settings.reply_review_enabled and settings.node_analysis_api_key is not None:
        reply_reviewer = DeepSeekReplyReviewer(
            endpoint=settings.node_analysis_endpoint,
            model=settings.node_analysis_model,
            api_key=settings.node_analysis_api_key.get_secret_value(),
            timeout_seconds=settings.node_analysis_timeout_seconds,
        )
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
    memory_proposer, cognitive_extractor = create_background_json_consumers(database)
    continuity_repository = (
        BranchContinuityRepository(
            str(settings.chroma_dir),
            embedder,
        )
        if embedder is not None
        else None
    )
    project_memory_repository = (
        ProjectMemoryRepository(str(settings.chroma_dir), embedder)
        if embedder is not None
        else None
    )
    memory_reviewer: MemoryReviewer = ConservativeLocalMemoryReviewer()
    if settings.node_analysis_api_key is not None:
        memory_reviewer = DeepSeekMemoryReviewer(
            endpoint=settings.node_analysis_endpoint,
            model=settings.node_analysis_model,
            api_key=settings.node_analysis_api_key.get_secret_value(),
            timeout_seconds=settings.node_analysis_timeout_seconds,
        )
    registry.register(
        CONTINUAL_MEMORY_JOB_KIND,
        create_continual_memory_handler(
            memory_proposer,
            memory_reviewer,
            continuity_repository,
        ),
    )
    registry.register(
        BRANCH_REFLECTION_JOB_KIND,
        create_branch_reflection_handler(
            memory_proposer,
            memory_reviewer,
            continuity_repository,
        ),
    )
    registry.register(
        BRANCH_BASELINE_JOB_KIND,
        create_branch_baseline_handler(project_memory_repository),
    )
    registry.register(
        COGNITIVE_EXTRACTION_JOB_KIND,
        create_cognitive_extraction_handler(cognitive_extractor),
    )
    persona_client = PersonaInferenceClient(
        settings.persona_inference_url,
        settings.persona_inference_token.get_secret_value(),
        timeout=settings.persona_inference_timeout_seconds,
    )
    cognition_generator = create_cognition_generator(settings, persona_client)
    registry.register(
        BRANCH_CONVERSATION_JOB_KIND,
        create_conversation_actor_handler(
            persona_client,
            reply_reviewer,
            project_memory_repository,
            continuity_repository,
        ),
    )
    registry.register(
        COGNITIVE_CYCLE_JOB_KIND,
        create_cognitive_cycle_handler(cognition_generator),
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
    next_actor_scan = monotonic()
    next_wakeup_scan = monotonic()
    next_runtime_wakeup_scan = monotonic()
    next_cognition_retry_scan = monotonic()
    next_continuity_retry_scan = monotonic()
    next_interrupted_job_recovery_scan = monotonic()
    try:
        recover_interrupted_jobs(database)
        with Session(database.engine) as session:
            reconcile_subject_agent_modes(session)
        while not stop_event.is_set():
            if worker_role == "realtime" and monotonic() >= next_actor_scan:
                with Session(database.engine) as session:
                    deliver_due_pending_bubbles(session)
                    enqueue_due_actor_reviews(session)
                next_actor_scan = monotonic() + 1.0
            if role_scans_due_wakeups(worker_role) and monotonic() >= next_wakeup_scan:
                scan_due_wakeups(database)
                next_wakeup_scan = monotonic() + 1.0
            if role_scans_due_wakeups(worker_role) and monotonic() >= next_runtime_wakeup_scan:
                scan_due_runtime_wakeups(database)
                next_runtime_wakeup_scan = monotonic() + 1.0
            if role_scans_due_wakeups(worker_role) and monotonic() >= next_cognition_retry_scan:
                scan_retryable_cognition(database)
                next_cognition_retry_scan = monotonic() + 30.0
            if worker_role in {"background", "all"} and monotonic() >= next_continuity_retry_scan:
                scan_retryable_continuity(database)
                next_continuity_retry_scan = monotonic() + 30.0
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
        close_cognition = getattr(cognition_generator, "close", None)
        if callable(close_cognition):
            close_cognition()
        database.close()


if __name__ == "__main__":
    main()
