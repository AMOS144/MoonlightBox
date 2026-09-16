"""使用官方 SqliteSaver，操作状态放在业务数据库旁，不放进 worktree。"""

import fcntl
import hashlib
from contextlib import contextmanager
from pathlib import Path

from moonlightbox.agent_runtime.persistence import checkpoint_path, open_checkpointer


@contextmanager
def collaboration_checkpointer(session):
    with open_checkpointer(checkpoint_path(session)) as saver:
        yield saver


@contextmanager
def branch_writer(session, branch_id):
    """Linux 本机开发单写者锁；进程退出由内核释放，不会五分钟后偷走活跃任务。"""
    database = session.get_bind().url.database
    if not database or database == ":memory:":
        yield True
        return
    directory = Path(database).resolve().parent / "runtime-locks"
    directory.mkdir(exist_ok=True)
    name = hashlib.sha256(branch_id.encode()).hexdigest() + ".lock"
    with (directory / name).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
