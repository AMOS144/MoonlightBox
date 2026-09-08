# ruff: noqa: E501

"""Linux 上的 PyTorch + PEFT QLoRA 训练引擎。"""

from __future__ import annotations

import gc
import json
import math
import random
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ProgressCallback = Callable[[dict[str, object]], None]


class PeftEnvironmentError(RuntimeError):
    pass


class PeftTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PeftLoraConfig:
    model: str
    data_dir: Path
    adapter_dir: Path
    iterations: int | None = 600
    train_example_count: int = 0
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 4
    validation_batches: int = 10
    steps_per_evaluation: int = 50
    save_every: int = 100
    # GTX 1650 4GB 的保守上限；更长上下文应由更大 Linux GPU 显式配置。
    max_seq_length: int = 512
    seed: int = 42
    warmup_steps: int = 10
    weight_decay: float = 0.01
    grad_checkpoint: bool = True
    num_layers: int = 16
    target_modules: tuple[str, ...] = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    )
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    early_stopping_patience: int = 0
    preserve_checkpoints: bool = False
    resume_adapter_file: Path | None = None

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def steps_per_epoch(self) -> int:
        if self.train_example_count < 1:
            raise ValueError("计算有效 epoch 需要正数训练样本量")
        return max(1, math.ceil(self.train_example_count / self.effective_batch_size))

    @property
    def resolved_iterations(self) -> int:
        """返回训练循环需要消费的 micro-batch 数。

        训练循环每次迭代只前向/反向一个 ``batch_size`` 的样本，并在
        ``gradient_accumulation_steps`` 次后才执行一次 optimizer step。因此这里
        不能使用 ``steps_per_epoch``（它表示 optimizer step 数），否则会再次按
        梯度累积除以样本数，导致一次“完整 epoch”只训练约四分之一数据。
        """

        if self.train_example_count > 0:
            return math.ceil(self.train_example_count / self.batch_size) * self.epochs
        if self.iterations is None or self.iterations < 1:
            raise ValueError("训练步数必须由正数 iterations 或训练集大小推导")
        return self.iterations


@dataclass(frozen=True)
class PeftCapabilities:
    version: str
    torch_version: str
    transformers_version: str


@dataclass(frozen=True)
class LoraCandidate:
    candidate_id: str
    search_space_version: str
    rank: int
    alpha: int
    num_layers: int
    target_modules: tuple[str, ...]
    dropout: float
    learning_rate: float
    seed: int
    resume_adapter_file: Path | None = None


def default_lora_candidates() -> tuple[LoraCandidate, ...]:
    attention = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
    query_value = ("self_attn.q_proj", "self_attn.v_proj")
    return (
        LoraCandidate(
            "rank16-attn16", "moonlightbox-lora-search-v1", 16, 32, 16, attention, 0.05, 1e-5, 17
        ),
        LoraCandidate(
            "rank32-attn24", "moonlightbox-lora-search-v1", 32, 64, 24, attention, 0.1, 5e-6, 29
        ),
        LoraCandidate(
            "rank32-qv-all", "moonlightbox-lora-search-v1", 32, 64, -1, query_value, 0.05, 1e-5, 43
        ),
    )


def high_capacity_persona_candidates() -> tuple[LoraCandidate, ...]:
    attention = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
    all_linear = (*attention, "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    return (
        LoraCandidate(
            "rank32-qkvo-all",
            "moonlightbox-persona-capacity-search-v4",
            32,
            64,
            -1,
            attention,
            0.05,
            7.5e-6,
            47,
        ),
        LoraCandidate(
            "rank32-all-linear-12",
            "moonlightbox-persona-capacity-search-v4",
            32,
            64,
            12,
            all_linear,
            0.1,
            5e-6,
            53,
        ),
        LoraCandidate(
            "rank32-all-linear-18",
            "moonlightbox-persona-capacity-search-v4",
            32,
            64,
            18,
            all_linear,
            0.1,
            4e-6,
            61,
        ),
    )


def compact_persona_candidates() -> tuple[LoraCandidate, ...]:
    """4GB Linux GPU 的单候选配置，避免高容量搜索反复触发显存不足。

    Q/V 覆盖全部层足以学习人称、用词和句长等表达差异，rank 16 则把可训练参数
    与 optimizer 状态控制在小范围内。若迁移到更大显存机器，可改用高容量搜索。
    """

    return (
        LoraCandidate(
            "compact-r16-qv-all",
            "moonlightbox-persona-4gb-v1",
            16,
            32,
            -1,
            ("self_attn.q_proj", "self_attn.v_proj"),
            0.05,
            1e-5,
            47,
        ),
    )


@dataclass(frozen=True)
class BestCheckpointSelection:
    validation_loss: float
    validation_iteration: int
    checkpoint_iteration: int
    checkpoint: Path
    approximation_steps: int = 0
    mapping_rule: str = "peft-validation-at-save-step"


@dataclass(frozen=True)
class TrainingResult:
    adapter_dir: Path
    iterations: int
    config_path: Path | None = None
    peft_version: str = "unknown"
    validation_curve: tuple[dict[str, float | int], ...] = ()
    best_checkpoint_selection: BestCheckpointSelection | None = None
    command: tuple[str, ...] = ()
    early_stopped: bool = False
    stopped_iteration: int | None = None

    @property
    def best_checkpoint(self) -> Path | None:
        selection = self.best_checkpoint_selection
        return selection.checkpoint if selection is not None else None

    @property
    def best_validation_iteration(self) -> int | None:
        selection = self.best_checkpoint_selection
        return selection.validation_iteration if selection is not None else None

    @property
    def best_checkpoint_iteration(self) -> int | None:
        selection = self.best_checkpoint_selection
        return selection.checkpoint_iteration if selection is not None else None

    @property
    def checkpoint_mapping_rule(self) -> str:
        return (
            self.best_checkpoint_selection.mapping_rule
            if self.best_checkpoint_selection
            else "peft-validation-at-save-step"
        )


def normalized_peft_config(
    config: PeftLoraConfig, capabilities: PeftCapabilities | None = None
) -> str:
    """把实际训练参数写成可审计 JSON；不再伪造 MLX YAML。"""

    payload = asdict(config)
    payload["data_dir"] = str(config.data_dir)
    payload["adapter_dir"] = str(config.adapter_dir)
    payload["resume_adapter_file"] = (
        str(config.resume_adapter_file) if config.resume_adapter_file else None
    )
    payload["target_modules"] = list(config.target_modules)
    if capabilities is not None:
        payload["backend"] = {
            "name": "pytorch-transformers-peft",
            "peft_version": capabilities.version,
            "torch_version": capabilities.torch_version,
            "transformers_version": capabilities.transformers_version,
        }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


class PeftLmAdapter:
    """用标准 PEFT adapter 训练与验证人格 LoRA。"""

    def __init__(self, *, device: str = "auto", load_in_4bit: bool = True) -> None:
        self._device = device
        self._load_in_4bit = load_in_4bit

    @property
    def capabilities(self) -> PeftCapabilities:
        self._dependencies()
        return PeftCapabilities(
            _package_version("peft"),
            _package_version("torch"),
            _package_version("transformers"),
        )

    def train(
        self, config: PeftLoraConfig, on_progress: ProgressCallback | None = None
    ) -> TrainingResult:
        torch, transformers, peft = self._dependencies()
        self._validate_config(config)
        callback = on_progress or (lambda _: None)
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        config_path = config.adapter_dir / "training.json"
        config_path.write_text(normalized_peft_config(config, self.capabilities), encoding="utf-8")
        tokenizer, model = self._build_trainable_model(config, torch, transformers, peft)
        try:
            return self._run_training_loop(
                config=config,
                config_path=config_path,
                tokenizer=tokenizer,
                model=model,
                torch=torch,
                callback=callback,
            )
        finally:
            # 候选失败、任务取消或 OOM 后也必须立即释放量化基座；否则下一个候选会
            # 把上一次残留的显存误判为自身 OOM。
            self._release(torch, model)

    def _run_training_loop(
        self,
        *,
        config: PeftLoraConfig,
        config_path: Path,
        tokenizer: Any,
        model: Any,
        torch: Any,
        callback: ProgressCallback,
    ) -> TrainingResult:
        train_rows = _load_examples(
            config.data_dir / "train.jsonl", tokenizer, config.max_seq_length
        )
        valid_rows = _load_examples(
            config.data_dir / "valid.jsonl", tokenizer, config.max_seq_length
        )
        if not train_rows or not valid_rows:
            raise PeftTrainingError("训练或验证集没有可监督的 assistant 目标 token")
        device = self._model_device(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        randomizer = random.Random(config.seed)
        validation_curve: list[dict[str, float | int]] = []
        best: BestCheckpointSelection | None = None
        stale = 0
        stopped_iteration: int | None = None
        model.train()
        optimizer.zero_grad()
        for iteration in range(1, config.resolved_iterations + 1):
            row = train_rows[(iteration - 1) % len(train_rows)]
            batch = _to_device(row, device)
            loss = model(**batch).loss / config.gradient_accumulation_steps
            loss.backward()
            if (
                iteration % config.gradient_accumulation_steps == 0
                or iteration == config.resolved_iterations
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            callback(
                {"stage": "training", "iteration": iteration, "loss": float(loss.detach().cpu())}
            )
            should_evaluate = (
                iteration % config.steps_per_evaluation == 0
                or iteration == config.resolved_iterations
            )
            if not should_evaluate:
                continue
            validation_loss = self._evaluate_rows(
                model, valid_rows, config.validation_batches, torch, device
            )
            point: dict[str, float | int] = {
                "iteration": iteration,
                "validation_loss": validation_loss,
            }
            validation_curve.append(point)
            callback({"stage": "training", **point})
            self._save_adapter(model, config.adapter_dir)
            checkpoint = config.adapter_dir / f"{iteration:07d}_adapter_model.safetensors"
            shutil.copyfile(config.adapter_dir / "adapter_model.safetensors", checkpoint)
            callback({"stage": "saved", "iteration": iteration})
            selection = BestCheckpointSelection(validation_loss, iteration, iteration, checkpoint)
            if best is None or validation_loss < best.validation_loss:
                best = selection
                stale = 0
            else:
                stale += 1
            if not config.preserve_checkpoints and best is not None:
                _prune_adapter_checkpoints(config.adapter_dir, keep=best.checkpoint)
            if config.early_stopping_patience > 0 and stale >= config.early_stopping_patience:
                stopped_iteration = iteration
                break
            randomizer.random()  # 固定 PRNG 演进，保证未来加入抽样时可复现。
        self._save_adapter(model, config.adapter_dir)
        return TrainingResult(
            config.adapter_dir,
            stopped_iteration or config.resolved_iterations,
            config_path,
            self.capabilities.version,
            tuple(validation_curve),
            best,
            ("python", "-m", "moonlightbox.training.peft_adapter"),
            stopped_iteration is not None,
            stopped_iteration,
        )

    def evaluate_validation_loss(self, config: PeftLoraConfig) -> float:
        return self.evaluate_split_loss(config, config.adapter_dir, split="valid")

    def evaluate_loss(self, config: PeftLoraConfig, checkpoint: Path) -> float:
        """在从未参与训练选择的 test 集上复验一个 adapter。"""

        return self.evaluate_split_loss(config, checkpoint, split="test")

    def evaluate_split_loss(
        self,
        config: PeftLoraConfig,
        checkpoint: Path,
        *,
        split: str,
    ) -> float:
        """在指定的数据切分上评估已保存的 adapter，并始终释放量化基座。

        完整训练在所有权重都已落盘、但尚未来得及写入汇总状态时可能被外部
        Worker 中断。恢复这种边界状态只能重新计算 valid loss；绝不能误用
        独立 test 集，也不能为此重新执行数小时的 QLoRA 训练。
        """

        if split not in {"valid", "test"}:
            raise ValueError("评估切分只能是 valid 或 test")
        torch, transformers, peft = self._dependencies()
        adapter_directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
        if not (adapter_directory / "adapter_model.safetensors").is_file():
            raise PeftTrainingError("PEFT adapter 缺少 adapter_model.safetensors")
        tokenizer, model = self._build_trainable_model(
            config, torch, transformers, peft, adapter_directory
        )
        try:
            rows = _load_examples(
                config.data_dir / f"{split}.jsonl", tokenizer, config.max_seq_length
            )
            if not rows:
                raise PeftTrainingError(f"{split} 集没有可监督的 assistant 目标 token")
            return self._evaluate_rows(
                model, rows, config.validation_batches, torch, self._model_device(model)
            )
        finally:
            self._release(torch, model)

    def _build_trainable_model(
        self,
        config: PeftLoraConfig,
        torch: Any,
        transformers: Any,
        peft: Any,
        adapter_directory: Path | None = None,
    ) -> tuple[Any, Any]:
        tokenizer = transformers.AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs: dict[str, object] = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        device_kind = self._resolved_device(torch)
        self._validate_gpu_capacity(torch, config.model, device_kind)
        if device_kind == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            if self._load_in_4bit:
                try:
                    model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        # NF4 与双重量化是 PEFT 推荐的 QLoRA 组合；GTX 1650 不支持
                        # BF16，因此计算精度固定为 FP16。
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except (AttributeError, ImportError) as error:
                    raise PeftEnvironmentError("4-bit QLoRA 需要 bitsandbytes") from error
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = {"": "cuda:0"}
        else:
            model_kwargs["torch_dtype"] = torch.float32
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
            if device_kind == "cpu":
                model.to("cpu")
            # 训练阶段的 KV cache 只增加显存占用，并与梯度检查点不兼容。
            if getattr(model, "config", None) is not None:
                model.config.use_cache = False
            adapter_directory = adapter_directory or (
                config.resume_adapter_file.parent
                if config.resume_adapter_file is not None
                else None
            )
            # ``prepare_model_for_kbit_training`` 不只是新建 LoRA 时的辅助函数：
            # 它会冻结量化基座、处理输入梯度并与梯度检查点协作。断点恢复若跳过它，
            # 会在同一份 adapter 上得到不同的可训练参数集合。
            if self._load_in_4bit and device_kind == "cuda":
                model = peft.prepare_model_for_kbit_training(model)
            if adapter_directory is not None:
                model = peft.PeftModel.from_pretrained(
                    model, str(adapter_directory), is_trainable=True
                )
            else:
                target_modules = tuple(
                    sorted({value.rsplit(".", 1)[-1] for value in config.target_modules})
                )
                lora_kwargs: dict[str, object] = {
                    "task_type": peft.TaskType.CAUSAL_LM,
                    "r": config.rank,
                    "lora_alpha": config.alpha,
                    "lora_dropout": config.dropout,
                    "target_modules": target_modules,
                }
                if config.num_layers > 0:
                    lora_kwargs["layers_to_transform"] = list(range(config.num_layers))
                model = peft.get_peft_model(model, peft.LoraConfig(**lora_kwargs))
            if config.grad_checkpoint:
                enable = getattr(model, "gradient_checkpointing_enable", None)
                if callable(enable):
                    enable()
        except (OSError, RuntimeError, ValueError) as error:
            raise PeftTrainingError(
                "无法初始化 QLoRA 训练模型；请检查基础模型、显存和 target_modules"
            ) from error
        return tokenizer, model

    @staticmethod
    def _save_adapter(model: Any, adapter_dir: Path) -> None:
        model.save_pretrained(adapter_dir, safe_serialization=True)

    @staticmethod
    def _evaluate_rows(
        model: Any, rows: list[dict[str, Any]], limit: int, torch: Any, device: Any
    ) -> float:
        model.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for row in rows[:limit]:
                losses.append(float(model(**_to_device(row, device)).loss.detach().cpu()))
        model.train()
        if not losses:
            raise PeftTrainingError("验证集为空")
        return sum(losses) / len(losses)

    def _dependencies(self) -> tuple[Any, Any, Any]:
        try:
            import peft  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
            import transformers  # type: ignore[import-not-found]
        except ImportError as error:
            raise PeftEnvironmentError(
                "请执行 `uv sync --extra linux-ml` 安装 Linux QLoRA 依赖"
            ) from error
        return torch, transformers, peft

    def _resolved_device(self, torch: Any) -> str:
        if self._device == "cpu":
            return "cpu"
        if self._device == "cuda":
            if not torch.cuda.is_available():
                raise PeftEnvironmentError("已配置 CUDA，但未检测到 Linux NVIDIA GPU")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _model_device(model: Any) -> Any:
        return next(model.parameters()).device

    @staticmethod
    def _validate_gpu_capacity(torch: Any, base_model: str, device: str) -> None:
        if device != "cuda" or "qwen3-8b" not in base_model.lower():
            return
        memory_bytes = torch.cuda.get_device_properties(0).total_memory
        if memory_bytes < 10 * 1024**3:
            raise PeftEnvironmentError(
                "Qwen3-8B 的 QLoRA 训练至少需要约 10GB GPU 显存；"
                "请改用 Qwen3-1.7B 或更大的 Linux GPU。"
            )

    @staticmethod
    def _release(torch: Any, model: Any) -> None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _validate_config(config: PeftLoraConfig) -> None:
        if config.rank < 1 or config.alpha < 1:
            raise PeftTrainingError("LoRA rank 与 alpha 必须为正数")
        if (
            not (config.data_dir / "train.jsonl").is_file()
            or not (config.data_dir / "valid.jsonl").is_file()
        ):
            raise PeftTrainingError("训练数据必须包含 train.jsonl 与 valid.jsonl")


def _load_examples(path: Path, tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        messages = raw.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) < 2
            or messages[-1].get("role") != "assistant"
        ):
            continue
        normalized = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in messages
            if isinstance(item, dict)
            and isinstance(item.get("role"), str)
            and isinstance(item.get("content"), str)
        ]
        if len(normalized) != len(messages):
            continue
        full = _apply_template(tokenizer, normalized)
        prompt = _apply_template(tokenizer, normalized[:-1])
        input_ids = full["input_ids"][0][:max_length]
        attention_mask = full["attention_mask"][0][:max_length]
        labels = input_ids.clone()
        labels[: min(len(prompt["input_ids"][0]), len(labels))] = -100
        if bool((labels != -100).any()):
            rows.append(
                {
                    "input_ids": input_ids.unsqueeze(0),
                    "attention_mask": attention_mask.unsqueeze(0),
                    "labels": labels.unsqueeze(0),
                }
            )
    return rows


def _apply_template(tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return dict(encoded)


def _to_device(row: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in row.items()}


def _prune_adapter_checkpoints(directory: Path, *, keep: Path) -> None:
    for checkpoint in directory.glob("[0-9]" * 7 + "_adapter_model.safetensors"):
        if checkpoint.resolve() != keep.resolve():
            checkpoint.unlink()


def cleanup_generated_checkpoints(
    output_dir: Path, *, include_latest: bool = False, keep: tuple[Path, ...] = ()
) -> None:
    retained = {path.resolve() for path in keep if path.exists()}
    for checkpoint in output_dir.rglob("[0-9]" * 7 + "_adapter_model.safetensors"):
        if checkpoint.resolve() not in retained:
            checkpoint.unlink()
    if include_latest:
        for checkpoint in output_dir.rglob("adapter_model.safetensors"):
            if checkpoint.resolve() not in retained:
                checkpoint.unlink()


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"
