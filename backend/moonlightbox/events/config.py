import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import cast


class AnalysisConfigIntegrityError(ValueError):
    pass


class AnalysisWindowManifestIntegrityError(ValueError):
    pass


class InvalidWindowManifestError(ValueError):
    pass


def _serialize_config(config: Mapping[str, object]) -> str:
    return json.dumps(
        config,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def normalize_config(config: Mapping[str, object]) -> dict[str, object]:
    """生成与调用方对象完全隔离的规范化配置快照。"""
    return cast(dict[str, object], json.loads(_serialize_config(config)))


def config_fingerprint(config: Mapping[str, object]) -> str:
    """对配置进行规范化序列化，生成稳定指纹。"""
    serialized = _serialize_config(config)
    return hashlib.sha256(serialized.encode()).hexdigest()


def ensure_config_integrity(
    config: Mapping[str, object],
    fingerprint: str,
) -> None:
    """阻止配置快照与其指纹不一致地进入持久化层。"""
    if config_fingerprint(config) != fingerprint:
        raise AnalysisConfigIntegrityError("配置快照与配置指纹不一致")


def normalize_window_ids(window_ids: Sequence[str]) -> list[str]:
    """复制并校验有序窗口清单。"""
    normalized = list(window_ids)
    if any(not window_id.strip() for window_id in normalized):
        raise InvalidWindowManifestError("window_ids 不能包含空字符串")
    if len(normalized) != len(set(normalized)):
        raise InvalidWindowManifestError("window_ids 不能包含重复值")
    return normalized


def window_manifest_fingerprint(window_ids: Sequence[str]) -> str:
    serialized = json.dumps(
        list(window_ids),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def ensure_window_manifest_integrity(
    window_ids: Sequence[str],
    fingerprint: str,
    total_windows: int,
) -> None:
    """阻止窗口清单被修改或与窗口总数不一致。"""
    if window_manifest_fingerprint(window_ids) != fingerprint or len(window_ids) != total_windows:
        raise AnalysisWindowManifestIntegrityError("窗口清单完整性校验失败")
