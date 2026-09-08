import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any


@dataclass(frozen=True, slots=True)
class SpatialPipelineConfig:
    segmentation_version: str = "spatial-episode-v1"
    query_bank_version: str = "spatial-query-bank-v1"
    analyzer_version: str = "spatial-bundle-v1"
    resolver_version: str = "spatial-resolver-v1"
    graph_version: str = "personal-place-graph-v1"
    episode_gap: timedelta = timedelta(minutes=90)
    episode_character_budget: int = 8_000
    episode_message_limit: int = 160
    neighbor_gap: timedelta = timedelta(hours=6)
    query_top_k_per_month: int = 4
    query_minimum_similarity: float = 0.42
    automatic_link_threshold: float = 0.78
    automatic_link_margin: float = 0.15
    provisional_link_threshold: float = 0.55
    provisional_link_margin: float = 0.08
    enable_alias_backfill: bool = False
    enable_event_recall_hint: bool = False

    def __post_init__(self) -> None:
        if self.episode_gap <= timedelta(0):
            raise ValueError("episode_gap 必须大于零")
        if self.episode_character_budget <= 0 or self.episode_message_limit <= 0:
            raise ValueError("Episode 预算必须大于零")
        if self.neighbor_gap <= timedelta(0):
            raise ValueError("neighbor_gap 必须大于零")
        if self.query_top_k_per_month <= 0:
            raise ValueError("query_top_k_per_month 必须大于零")
        for value in (
            self.query_minimum_similarity,
            self.automatic_link_threshold,
            self.automatic_link_margin,
            self.provisional_link_threshold,
            self.provisional_link_margin,
        ):
            if not 0 <= value <= 1:
                raise ValueError("空间分析阈值必须在 0 到 1 之间")

    def snapshot(self) -> dict[str, object]:
        value = asdict(self)
        value["episode_gap_seconds"] = self.episode_gap.total_seconds()
        value["neighbor_gap_seconds"] = self.neighbor_gap.total_seconds()
        del value["episode_gap"]
        del value["neighbor_gap"]
        return value

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.snapshot(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_snapshot(cls, value: dict[str, Any]) -> "SpatialPipelineConfig":
        data = dict(value)
        episode_seconds = data.pop("episode_gap_seconds", None)
        neighbor_seconds = data.pop("neighbor_gap_seconds", None)
        if not isinstance(episode_seconds, int | float) or not isinstance(
            neighbor_seconds,
            int | float,
        ):
            raise ValueError("空间流水线快照缺少时间间隔")
        return cls(
            **data,
            episode_gap=timedelta(seconds=float(episode_seconds)),
            neighbor_gap=timedelta(seconds=float(neighbor_seconds)),
        )
