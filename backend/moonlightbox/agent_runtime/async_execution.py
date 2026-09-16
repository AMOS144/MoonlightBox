"""异步编排与同步资源的安全边界，不另建模型循环或重试策略。"""

import asyncio
from threading import Event

from .cancellation import cancellation_scope, current_cancellation_signal


async def run_sync_owned(operation, /, *args):
    """将完整同步工作单元隔离执行，资源须在 operation 内创建和关闭。

    to_thread 保留追踪及 Job 取消的 ContextVar。协程取消后通知同步 Controller，
    并等待它退出；不能杀死的同步 I/O 不伪装成已经停止，也不允许迟到结果交付。
    """
    stopped = Event()
    parent_cancelled = current_cancellation_signal()

    def execute():
        with cancellation_scope(
            lambda: stopped.is_set() or bool(parent_cancelled and parent_cancelled())
        ):
            return operation(*args)

    task = asyncio.create_task(asyncio.to_thread(execute))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        stopped.set()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                stopped.set()
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise
