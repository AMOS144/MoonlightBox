import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from moonlightbox.training.search.lock import (
    TrainingFingerprintMismatch,
    _atomic_write_json,
)


def _resolve_base_model_identity(
    base_model: str,
    *,
    cache_path: Path,
) -> dict[str, object]:
    """解析不可变基座，并为全部加载文件计算流式摘要。"""

    path = Path(base_model)
    if path.is_dir():
        return build_local_model_identity(
            path,
            cache_path=cache_path,
            requested=base_model,
        )
    remote_match = re.fullmatch(
        r"(?:hf://)?(?P<repo>[^@]+)@(?P<commit>[0-9a-fA-F]{40})",
        base_model,
    )
    if remote_match is None:
        raise TrainingFingerprintMismatch(
            "远程 Hugging Face 基座必须显式指定 40 位不可变 commit revision"
        )
    repo_id = remote_match.group("repo")
    commit = remote_match.group("commit").lower()
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(repo_id=repo_id, revision=commit))
    except Exception as error:
        raise TrainingFingerprintMismatch(
            "无法解析并固定远程 Hugging Face 基座 revision"
        ) from error
    if snapshot.name.lower() != commit:
        raise TrainingFingerprintMismatch("Hugging Face 缓存未解析到请求的不可变 commit")
    return {
        "source": "huggingface",
        "requested": base_model,
        "resolved_path": str(snapshot.resolve()),
        "repo_id": repo_id,
        "commit": commit,
        "digest": commit,
        "files": [],
    }


def build_local_model_identity(
    path: Path,
    *,
    cache_path: Path,
    requested: str | None = None,
    hash_file: Callable[[Path], str] | None = None,
) -> dict[str, object]:
    canonical = path.resolve()
    items = sorted(candidate for candidate in canonical.rglob("*") if candidate.is_file())
    stats = [
        {
            "path": str(item.relative_to(canonical)),
            "size": item.stat().st_size,
            "mtime_ns": item.stat().st_mtime_ns,
            "ctime_ns": item.stat().st_ctime_ns,
            "inode": item.stat().st_ino,
            "mode": item.stat().st_mode,
            "device": item.stat().st_dev,
            # mtime/ctime/inode 都可能在同一文件系统时钟粒度内保持不变，或被
            # 显式恢复。轻量内容指纹让缓存不会把“同尺寸替换的模型权重”误判为
            # 同一基座；完整 SHA-256 仍只在签名变化时重新计算。
            "content_probe": _file_content_probe(item),
        }
        for item in items
    ]
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and cached.get("schema_version") == "base-digest-cache-v2"
            and cached.get("canonical_path") == str(canonical)
            and cached.get("stats") == stats
            and isinstance(cached.get("identity"), dict)
        ):
            return cast(dict[str, object], cached["identity"])
    digest_file = hash_file or _stream_sha256
    files: list[dict[str, object]] = []
    for item in items:
        files.append(
            {
                "path": str(item.relative_to(canonical)),
                "size": item.stat().st_size,
                "sha256": digest_file(item),
            }
        )
    if not files:
        raise TrainingFingerprintMismatch("本地基础模型目录为空")
    identity: dict[str, object] = {
        "source": "local",
        "requested": requested or str(path),
        "resolved_path": str(canonical),
        "files": files,
        "digest": _canonical_digest(files),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        cache_path,
        {
            "schema_version": "base-digest-cache-v2",
            "cache_semantics": "performance-only-content-sha-authoritative",
            "canonical_path": str(canonical),
            "stats": stats,
            "identity": identity,
        },
    )
    return identity


def _file_content_probe(path: Path, *, window_bytes: int = 4096) -> str:
    """返回用于缓存失效判断的首尾内容摘要，不替代完整权重摘要。"""

    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(window_bytes))
        if size > window_bytes:
            stream.seek(max(0, size - window_bytes))
            digest.update(stream.read(window_bytes))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def _training_data_identity(data_dir: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for name in ("manifest.json", "train.jsonl", "valid.jsonl", "test.jsonl"):
        path = data_dir / name
        if not path.is_file():
            if name == "test.jsonl":
                continue
            raise TrainingFingerprintMismatch(f"训练数据缺少必需文件：{name}")
        # manifest.json is enriched with artifact metadata after training has
        # started. Its pre-training digest is already fenced separately by
        # declared_data_manifest_digest, so hashing the mutable copy here makes
        # an interrupted job reject its own otherwise identical dataset.
        if name == "manifest.json":
            continue
        files.append(
            {
                "name": name,
                "size": path.stat().st_size,
                "sha256": _stream_sha256(path),
                "used_for_selection": name in {"train.jsonl", "valid.jsonl"},
            }
        )
    return {
        "files": files,
        "digest": _canonical_digest(files),
        "test_used_for_tuning": False,
    }


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _training_environment() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
    }


def _package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _canonical_digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
