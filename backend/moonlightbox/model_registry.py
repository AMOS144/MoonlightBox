"""所有进程共用的 ORM 注册入口；只加载表定义，不加载推理模型或创建数据表。"""

from importlib import import_module


def register_models() -> None:
    # 显式列出持久化模块，避免 API 路由的偶然导入掩盖 Worker 缺失外键目标。
    for module in (
        "projects.models",
        "agent_runtime.sensitive_mask",
        "imports.models",
        "events.models",
        "node_investigation.models",
        "jobs.models",
        "media.models",
        "personas.models",
        "training.models",
        "evaluation.models",
        "spatial.models",
        "world.models",
        "runtime_v1.branch_models",
        "runtime_v1.db_models",
        "runtime_v1.life_events.models",
        "runtime_v1.conversation_index",
    ):
        import_module(f"moonlightbox.{module}")
