"""按本地时刻平滑抽取当下机会，不规定剧情、情绪或影响预算。"""

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import EventOpportunity


class PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ProbabilityCurve(PolicyModel):
    interpolation: Literal["periodic_smoothstep"] = "periodic_smoothstep"
    anchors: dict[str, float]

    @model_validator(mode="after")
    def valid_anchors(self):
        if len(self.anchors) < 2:
            raise ValueError("概率曲线至少需要两个时间锚点")
        for key, value in self.anchors.items():
            parts = key.split(":")
            if (
                len(parts) != 2
                or not all(p.isdigit() and len(p) == 2 for p in parts)
                or not 0 <= int(parts[0]) < 24
                or not 0 <= int(parts[1]) < 60
            ):
                raise ValueError("锚点必须为 HH:MM，范围 00:00～23:59")
            if not 0 <= value <= 1:
                raise ValueError("概率必须在 0～1 之间")
        return self

    def at(self, local_now):
        minute = local_now.hour * 60 + local_now.minute + local_now.second / 60
        points = sorted((int(k[:2]) * 60 + int(k[3:]), v) for k, v in self.anchors.items())
        points = [
            (points[-1][0] - 1440, points[-1][1]),
            *points,
            (points[0][0] + 1440, points[0][1]),
        ]
        for (start, low), (end, high) in zip(points, points[1:], strict=False):
            if start <= minute <= end:
                u = (minute - start) / (end - start)
                return low + (high - low) * u * u * (3 - 2 * u)
        raise ValueError("无法定位本地时间")


class IntensityDistribution(PolicyModel):
    family: Literal["beta"] = "beta"
    alpha: float = Field(default=2, gt=0)
    beta: float = Field(default=5, gt=0)


class LifeEventPolicy(PolicyModel):
    version: str = Field(min_length=1)
    check_interval_minutes: int = Field(default=20, gt=0)
    trigger_probability: ProbabilityCurve
    direction_weights: dict[Literal["autonomy", "competence", "relatedness"], float]
    intensity_distribution: IntensityDistribution = Field(default_factory=IntensityDistribution)

    @model_validator(mode="after")
    def valid_weights(self):
        values = self.direction_weights.values()
        if (
            not values
            or any(not 0 <= v < float("inf") for v in values)
            or not 0 < sum(values) < float("inf")
        ):
            raise ValueError("方向权重必须非负、有限，且总和大于零")
        return self

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()

    def sample(self, seed, local_now):
        rng = random.Random(seed)
        probability = self.trigger_probability.at(local_now)
        if rng.random() >= probability:
            return probability, None
        names = sorted(self.direction_weights)
        direction = rng.choices(names, weights=[self.direction_weights[n] for n in names])[0]
        distribution = self.intensity_distribution
        return probability, EventOpportunity(
            direction=direction, intensity=rng.betavariate(distribution.alpha, distribution.beta)
        )


class UniqueKeyLoader(yaml.SafeLoader):
    """重复键直接报错，不能让 YAML 悄悄覆盖时间锚点。"""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"生活策略存在重复配置键：{key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


@lru_cache(maxsize=8)
def load_policy(path=None):
    """进程内冻结配置；外部 YAML 修改后重启生效，错误不静默回退。"""
    file = Path(path) if path else Path(__file__).with_name("policies") / "default.yaml"
    return LifeEventPolicy.model_validate(yaml.load(file.read_text(), Loader=UniqueKeyLoader))
