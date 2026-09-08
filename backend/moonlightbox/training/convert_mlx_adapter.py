# ruff: noqa: E501

"""把历史 MLX LoRA 权重显式转换为标准 PEFT adapter。

转换不会覆盖原 adapter。它只改权重布局和配置文件，不会也不能把 8B adapter
变成 1.7B adapter：转换后仍必须搭配训练时完全相同的 Hugging Face 基座。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MlxAdapterConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    output: Path
    converted_tensor_count: int
    base_model: str


def convert_mlx_adapter(
    source: Path,
    output: Path,
    *,
    base_model: str,
) -> ConversionResult:
    """转换一个 MLX adapter 目录，保留可审计转换清单。"""

    config_path = source / "adapter_config.json"
    weights_path = source / "adapters.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise MlxAdapterConversionError(
            "输入目录必须包含 MLX adapter_config.json 与 adapters.safetensors"
        )
    if output.exists() and any(output.iterdir()):
        raise MlxAdapterConversionError(f"输出目录已存在且非空，拒绝覆盖：{output}")
    try:
        from safetensors.torch import load_file, save_file  # type: ignore[import-not-found]
    except ImportError as error:
        raise MlxAdapterConversionError(
            "转换需要 torch 与 safetensors；请执行 uv sync --extra linux-ml"
        ) from error
    raw_config = _load_json_object(config_path)
    rank, alpha, dropout, target_modules, layers_to_transform = _lora_parameters(raw_config)
    source_tensors = load_file(str(weights_path), device="cpu")
    converted: dict[str, Any] = {}
    paired: set[str] = set()
    for key, tensor in source_tensors.items():
        mapped = _map_weight_key(key)
        if mapped is None:
            raise MlxAdapterConversionError(f"不支持的 MLX LoRA 权重键：{key}")
        pair_key = key.rsplit(".", 1)[0]
        paired.add(pair_key)
        # MLX Linear 的 LoRA 矩阵为 [in, rank] / [rank, out]，而 PEFT 的
        # Linear 保存为 [rank, in] / [out, rank]，因此两者都需要转置。
        converted[mapped] = tensor.transpose(0, 1).contiguous()
    _validate_pairs(source_tensors, paired)
    output.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output / "adapter_model.safetensors"))
    peft_config: dict[str, object] = {
        "base_model_name_or_path": base_model,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
    }
    if layers_to_transform is not None:
        peft_config["layers_to_transform"] = layers_to_transform
    (output / "adapter_config.json").write_text(
        json.dumps(peft_config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "moonlightbox_mlx_conversion.json").write_text(
        json.dumps(
            {
                "schema_version": "moonlightbox-mlx-to-peft-v1",
                "source_adapter": str(source),
                "source_weights": str(weights_path),
                "base_model": base_model,
                "converted_tensor_count": len(converted),
                "weight_layout": "mlx_linear_lora_transposed_to_peft",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ConversionResult(source, output, len(converted), base_model)


def _map_weight_key(key: str) -> str | None:
    if key.endswith(".lora_a"):
        base = key.removesuffix(".lora_a")
        suffix = "lora_A.default.weight"
    elif key.endswith(".lora_b"):
        base = key.removesuffix(".lora_b")
        suffix = "lora_B.default.weight"
    else:
        return None
    if not base.startswith("model.layers."):
        return None
    return "base_model.model." + base + "." + suffix


def _validate_pairs(tensors: dict[str, Any], prefixes: set[str]) -> None:
    for prefix in prefixes:
        a = prefix + ".lora_a"
        b = prefix + ".lora_b"
        if a not in tensors or b not in tensors:
            raise MlxAdapterConversionError(f"LoRA 层缺少成对矩阵：{prefix}")
        if len(tensors[a].shape) != 2 or len(tensors[b].shape) != 2:
            raise MlxAdapterConversionError(f"LoRA 矩阵必须是二维：{prefix}")
        if tensors[a].shape[1] != tensors[b].shape[0]:
            raise MlxAdapterConversionError(f"LoRA rank 不匹配：{prefix}")


def _lora_parameters(
    config: dict[str, object],
) -> tuple[int, int, float, tuple[str, ...], list[int] | None]:
    parameters = config.get("lora_parameters")
    if not isinstance(parameters, dict):
        raise MlxAdapterConversionError("MLX adapter_config.json 缺少 lora_parameters")
    rank = parameters.get("rank")
    scale = parameters.get("scale", 1.0)
    dropout = parameters.get("dropout", 0.0)
    if (
        not isinstance(rank, int)
        or rank < 1
        or not isinstance(scale, (int, float))
        or not isinstance(dropout, (int, float))
    ):
        raise MlxAdapterConversionError("MLX LoRA 参数无效")
    keys = parameters.get("keys")
    if not isinstance(keys, list):
        keys = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    target_modules = tuple(str(key).rsplit(".", 1)[-1] for key in keys)
    num_layers = config.get("num_layers", -1)
    layers = list(range(num_layers)) if isinstance(num_layers, int) and num_layers > 0 else None
    # MLX 的 scale 等价于 alpha / rank。
    return rank, max(1, round(rank * float(scale))), float(dropout), target_modules, layers


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MlxAdapterConversionError("无法读取 MLX adapter 配置") from error
    if not isinstance(value, dict):
        raise MlxAdapterConversionError("MLX adapter 配置必须是对象")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="将 MoonlightBox 历史 MLX LoRA 转为 PEFT")
    parser.add_argument("source", type=Path, help="包含 adapters.safetensors 的 MLX adapter 目录")
    parser.add_argument("output", type=Path, help="新的、空的 PEFT adapter 目录")
    parser.add_argument(
        "--base-model", required=True, help="与原 LoRA 完全一致的 Hugging Face 基座"
    )
    args = parser.parse_args()
    result = convert_mlx_adapter(args.source, args.output, base_model=args.base_model)
    print(f"已转换 {result.converted_tensor_count} 个张量：{result.output}")


if __name__ == "__main__":
    main()
