"""可选的原生 LangGraph 检查点作用域，不创建 AgentRun/Step 调用账本。"""

import fcntl
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

_scope = ContextVar("agent_checkpoint_scope", default=None)
_usage = ContextVar("agent_budget_sink", default=None)
_job_recovery = ContextVar("agent_job_recovery", default=False)


@contextmanager
def job_recovery_scope(enabled=True):
    """Job 已掌管整任务恢复时，Agent 不再以第二套尝试次数拒绝恢复。"""
    token = _job_recovery.set(enabled)
    try:
        yield
    finally:
        _job_recovery.reset(token)


def job_manages_recovery():
    return _job_recovery.get()


def checkpoint_path(session):
    """所有 Agent 共用存档位置；保留现有文件名，普通重启不迁移旧存档。"""
    bind = session.get_bind()
    database = bind.url.database
    if bind.dialect.name != "sqlite":
        raise ValueError("当前 Agent 存档仅支持 SQLite，不能静默降级为内存存档")
    if not database or database == ":memory:":
        return None
    return str(Path(database).resolve().with_name("runtime-collaboration.sqlite"))


@contextmanager
def open_checkpointer(path):
    """每个调用线程使用独立连接，不跨栏目共享 SQLite 连接。"""
    if path is None:
        yield InMemorySaver()
        return
    with SqliteSaver.from_conn_string(path) as saver:
        saver.conn.execute("PRAGMA busy_timeout=30000")
        # 多栏目可同时第一次启动；仅串行初始化 WAL/表结构，不串行模型和工具执行。
        # Linux 文件锁同时覆盖线程和进程，退出后由内核释放。
        with Path(str(path) + ".setup.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                saver.setup()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        yield saver


@contextmanager
def request_checkpoint(request):
    """已有 Runtime 作用域保持不变；独立 Agent 按 owner 和输入版本接入同一机制。"""
    if current_checkpoint() is not None or request.checkpoint_path is None:
        yield
        return
    import hashlib
    import json

    scope = request.scope.model_dump(mode="json")
    scope["permissions"] = sorted(scope["permissions"])
    identity = json.dumps(
        [request.owner_type, request.owner_id, request.input_revision, scope],
        sort_keys=True,
        ensure_ascii=False,
    )
    thread = "agent:" + hashlib.sha256(identity.encode()).hexdigest()
    with open_checkpointer(request.checkpoint_path) as saver, checkpoint_scope(saver, thread):
        yield


@contextmanager
def checkpoint_scope(saver, thread_id):
    token = _scope.set((saver, thread_id))
    usage = {}
    usage_token = _usage.set(usage)
    try:
        yield usage
    finally:
        _scope.reset(token)
        _usage.reset(usage_token)


def current_checkpoint():
    return _scope.get()


def report_budget(budget):
    sink = _usage.get()
    if sink is not None:
        sink.update(
            model_steps=budget.model_steps,
            tool_calls=sum(budget.tool_calls_by_name.values()),
            provider_tokens=(budget.provider_input_tokens or 0)
            + (budget.provider_output_tokens or 0),
        )
