import fcntl
import json
import os
import time
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4


class TrainingFingerprintMismatch(RuntimeError):
    pass


class AllCandidatesFailedError(RuntimeError):
    pass


class SearchLockTimeout(RuntimeError):
    pass


class FencingTokenError(RuntimeError):
    pass


class CandidateValidationError(RuntimeError):
    pass


class CandidateResourceError(RuntimeError):
    pass


class SearchOutputLock(AbstractContextManager[dict[str, object]]):
    def __init__(self, output_dir: Path, *, timeout_seconds: float = 30.0) -> None:
        self._output_dir = output_dir
        self._timeout_seconds = timeout_seconds
        self._stream: BinaryIO | None = None
        self.owner = {
            "pid": os.getpid(),
            "run_generation": str(uuid4()),
            "acquired_at": datetime.now().astimezone().isoformat(),
        }

    def __enter__(self) -> dict[str, object]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        stream = (self._output_dir / ".search.lock").open("a+b")
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    stream.close()
                    raise SearchLockTimeout("等待搜索目录独占锁超时") from None
                time.sleep(0.01)
        self._stream = stream
        _atomic_write_json(self._output_dir / ".search-lock-owner.json", self.owner)
        return self.owner

    def __exit__(self, *_args: object) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


class SearchStateStore:
    def __init__(self, path: Path, *, run_generation: str) -> None:
        self.path = path
        self.run_generation = run_generation

    def write(self, state: dict[str, Any]) -> None:
        if self.path.is_file():
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(current, dict) and current.get("run_generation") != self.run_generation:
                raise FencingTokenError("旧 worker 的 fencing token 已失效")
        state["run_generation"] = self.run_generation
        _atomic_write_json(self.path, state)

    def takeover(self, state: dict[str, Any]) -> None:
        state["run_generation"] = self.run_generation
        _atomic_write_json(self.path, state)


def _atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
