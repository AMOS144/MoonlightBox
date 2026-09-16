"""模型 HTTP 的协作式取消：只取消拥有的异步请求，不遗留后台执行线程。"""

import asyncio
from contextlib import suppress

import httpx

from .resilience import check_interruption, emit_status, managed


async def _post(endpoint, kwargs):
    async with httpx.AsyncClient() as client:
        task = asyncio.create_task(client.post(endpoint, **kwargs))
        try:
            while not task.done():
                check_interruption()
                await asyncio.wait({task}, timeout=0.1)
            check_interruption()
            return await task
        finally:
            if not task.done():
                emit_status({"status": "cancelling_io", "operation": "model_http"})
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            elif not task.cancelled():
                task.exception()  # 即使取消与网络失败同时发生，也回收已完成任务的异常。


def post(client, endpoint, *, cancellable=False, **kwargs):
    check_interruption()
    if managed() and cancellable:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_post(endpoint, kwargs))
    # 注入的同步客户端保持原 transport/TLS/测试配置；绝不偷偷换成不兼容客户端。
    emit_status({"status": "requesting", "cancellation": "wait_for_sync_exit"})
    response = client.post(endpoint, **kwargs)
    check_interruption()
    return response
