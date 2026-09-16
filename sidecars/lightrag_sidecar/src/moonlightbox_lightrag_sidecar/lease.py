"""跨进程 workspace 索引租约：原子 mkdir + 心跳 + 崩溃回收。"""
import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4


class IndexLeaseBusy(RuntimeError):
    """另一个 Sidecar 进程持有 workspace 索引租约。"""


class WorkspaceIndexLease:
    def __init__(self, root: Path, workspace: str, ttl_seconds: float = 120.0) -> None:
        self.path = root / workspace / ".moonlightbox-index-lease"
        self.ttl_seconds = ttl_seconds
        self.owner = uuid4().hex
        self._heartbeat: asyncio.Task[None] | None = None
        self._held = False

    async def __aenter__(self) -> "WorkspaceIndexLease":
        await asyncio.to_thread(self._acquire)
        self._held = True
        self._heartbeat = asyncio.create_task(self._heartbeat_loop())
        return self

    async def __aexit__(self, *_: object) -> None:
        self._held = False
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
        await asyncio.to_thread(self._release)

    def _acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError:
            if not self._is_stale():
                raise IndexLeaseBusy("LightRAG workspace 正在由其他进程构建") from None
            shutil.rmtree(self.path, ignore_errors=True)
            try:
                self.path.mkdir()
            except FileExistsError as error:
                raise IndexLeaseBusy("LightRAG workspace 正在由其他进程构建") from error
        self._write_owner()

    def _write_owner(self) -> None:
        payload = {"owner": self.owner, "pid": os.getpid(), "updated_at": time.time()}
        temporary = self.path / f"owner.{self.owner}.tmp"
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, self.path / "owner.json")

    def _is_stale(self) -> bool:
        try:
            value = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
            return time.time() - float(value.get("updated_at", 0)) > self.ttl_seconds
        except (OSError, ValueError, TypeError):
            try:
                return time.time() - self.path.stat().st_mtime > self.ttl_seconds
            except OSError:
                return False

    async def _heartbeat_loop(self) -> None:
        try:
            while self._held:
                await asyncio.sleep(max(1.0, self.ttl_seconds / 3))
                if self._held:
                    await asyncio.to_thread(self._write_owner)
        except asyncio.CancelledError:
            return

    def _release(self) -> None:
        try:
            value = json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            value = {}
        if value.get("owner") == self.owner:
            shutil.rmtree(self.path, ignore_errors=True)
