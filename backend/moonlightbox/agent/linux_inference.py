"""Linux 上的人格推理实现。

这里是本项目唯一直接接触 PyTorch、Transformers 与 PEFT 的人格推理层。业务层只
依赖 ``PersonaInferenceBackend`` 和 HTTP 客户端，因此既不会感知 CUDA，也不会把
平台判断散落到 Director、Actor 或分支聊天代码中。
"""

from __future__ import annotations

import gc
import importlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from moonlightbox.agent.inference import (
    InferencePreemptedError,
    PersonaInferenceRequest,
    PersonaInferenceResult,
)
from moonlightbox.db import Database
from moonlightbox.training.models import ModelVersion


class RuntimeInferenceError(RuntimeError):
    """Linux 人格服务无法满足 Runtime 请求。"""


@dataclass(frozen=True, slots=True)
class PersonaSamplingConfig:
    """人格表达的默认采样参数，供推理与验收共用。"""

    temperature: float = 0.65
    top_p: float = 0.9
    min_p: float = 0.02
    repetition_penalty: float = 1.02
    repetition_context_size: int = 128


class LinuxModelFormatError(RuntimeInferenceError):
    """模型或 LoRA 文件不是 Linux/PEFT 可加载格式。"""


def _split_pinned_hf_reference(value: str) -> tuple[str, dict[str, str]]:
    """把 ``仓库@commit`` 转成 Transformers 所需的 repo 与 revision 参数。

    训练记录保存不可变 Hugging Face commit，推理时不能把 ``@commit`` 当作普通
    repo 名传给 Hub。未固定的仓库名和本地路径保持原样，以兼容开发期推理。
    """

    matched = re.fullmatch(r"(?:hf://)?(?P<repo>[^@]+)@(?P<commit>[0-9a-fA-F]{40})", value)
    if matched is None:
        return value, {}
    return matched.group("repo"), {"revision": matched.group("commit").lower()}


class SharedLinuxModelRuntime:
    """按需常驻一组 Transformers 基座模型与 PEFT LoRA。

    同一时刻只保留一组基础模型权重。Director 和 PersonaActor 串行调用同一基座，
    避免体验阶段因加载 LoRA adapter 而额外占用有限显存。
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        load_in_4bit: bool = True,
        sampling: PersonaSamplingConfig | None = None,
        import_module: Callable[[str], object] = importlib.import_module,
    ) -> None:
        self.loaded_key: tuple[str, str | None] | None = None
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._tokenizer_cache: dict[str, Any] = {}
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._sampling = sampling or PersonaSamplingConfig()
        self._import_module = import_module
        self._torch: Any | None = None

    def generate_raw(
        self,
        *,
        base_model: str,
        adapter_path: str | None,
        messages: list[dict[str, str]],
        max_tokens: int,
        should_cancel: Callable[[], bool] | None = None,
        deadline: datetime | None = None,
    ) -> str:
        self._raise_if_stopped(should_cancel, deadline)
        loaded_key = (base_model, adapter_path)
        if self.loaded_key != loaded_key:
            self.release()
            self.model, self.tokenizer = self._load(base_model, adapter_path)
            self.loaded_key = loaded_key
        if self.model is None or self.tokenizer is None:
            raise RuntimeInferenceError("Linux 人格模型加载失败")

        torch = self._dependencies()[0]
        inputs = self._tokenize(messages)
        input_ids = inputs["input_ids"]
        model_device = self._input_device()
        inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        generation_kwargs: dict[str, object] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": self._sampling.temperature,
            "top_p": self._sampling.top_p,
            "repetition_penalty": self._sampling.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        try:
            with torch.inference_mode():
                generated = self.model.generate(**generation_kwargs)
        except RuntimeError as error:
            # OOM 后必须释放权重和 allocator cache，避免下一次推理继续受残留缓存影响。
            # 其他 RuntimeError 保留原始语义上抛。
            if "out of memory" in str(error).lower():
                self.release()
                raise RuntimeInferenceError(
                    "Runtime 推理显存不足，已释放模型等待下次请求"
                ) from error
            raise
        self._raise_if_stopped(should_cancel, deadline)
        return str(
            self.tokenizer.decode(
                generated[0][input_ids.shape[-1] :],
                skip_special_tokens=True,
            )
        ).strip()

    def set_seed(self, seed: int) -> None:
        torch, transformers, _ = self._dependencies()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        set_seed = getattr(transformers, "set_seed", None)
        if callable(set_seed):
            set_seed(seed)

    def count_text_tokens(self, *, base_model: str, text: str) -> int:
        """使用目标 Director 基座的真实 tokenizer 计数，不以字符数代替。"""

        tokenizer = self._tokenizer_for(base_model)
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else encoded.input_ids
        return len(input_ids)

    def release(self) -> None:
        """切换版本或关闭服务时释放 GPU/CPU 上的旧权重。"""

        had_model = self.model is not None or self.tokenizer is not None
        self.model = None
        self.tokenizer = None
        self.loaded_key = None
        if not had_model:
            return
        gc.collect()
        torch = self._torch
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load(self, base_model: str, adapter_path: str | None) -> tuple[Any, Any]:
        torch, transformers, peft = self._dependencies()
        self._validate_model_reference(base_model, "基础模型")
        if adapter_path is not None:
            self._validate_adapter_reference(adapter_path)
        model_reference, revision_kwargs = _split_pinned_hf_reference(base_model)
        tokenizer = self._tokenizer_for(base_model, transformers=transformers)

        model_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            **revision_kwargs,
        }
        resolved_device = self._resolved_device(torch)
        self._validate_gpu_capacity(torch, base_model, resolved_device)
        if resolved_device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            if self._load_in_4bit:
                try:
                    quantization = transformers.BitsAndBytesConfig(
                        load_in_4bit=True,
                        # 与训练端保持 NF4 + 双重量化，避免 4GB GPU 在推理服务与
                        # 训练服务间出现不同的显存行为。
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except (AttributeError, ImportError) as error:
                    raise RuntimeInferenceError(
                        "4-bit Linux 推理需要安装 bitsandbytes；请执行 uv sync --extra linux-ml"
                    ) from error
                model_kwargs["quantization_config"] = quantization
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = {"": "cuda:0"}
        else:
            # CPU 是明确的开发回退；不假装它适合交互式人格推理。
            model_kwargs["torch_dtype"] = torch.float32

        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_reference, **model_kwargs
            )
            if resolved_device == "cpu":
                model.to("cpu")
            if adapter_path is not None:
                model = peft.PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
            model.eval()
        except (OSError, ValueError, RuntimeError) as error:
            self.release()
            raise RuntimeInferenceError(
                "无法加载 Linux 人格模型；请确认基础模型、PEFT adapter 与显存配置匹配"
            ) from error
        return model, tokenizer

    def _tokenizer_for(self, base_model: str, *, transformers: Any | None = None) -> Any:
        cached = self._tokenizer_cache.get(base_model)
        if cached is not None:
            return cached
        resolved_transformers = transformers or self._dependencies()[1]
        model_reference, revision_kwargs = _split_pinned_hf_reference(base_model)
        tokenizer = resolved_transformers.AutoTokenizer.from_pretrained(
            model_reference,
            trust_remote_code=True,
            **revision_kwargs,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer_cache[base_model] = tokenizer
        return tokenizer

    def _tokenize(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.tokenizer is None:
            raise RuntimeInferenceError("tokenizer 未加载")
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        except TypeError:
            # 并非每个 HF 模型都认识 Qwen 的 enable_thinking 参数。
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        if hasattr(encoded, "items"):
            return dict(encoded)
        return {"input_ids": encoded, "attention_mask": encoded.ne(self.tokenizer.pad_token_id)}

    def _input_device(self) -> Any:
        if self.model is None:
            raise RuntimeInferenceError("模型未加载")
        try:
            return next(self.model.parameters()).device
        except StopIteration as error:
            raise RuntimeInferenceError("模型没有可用参数") from error

    def _resolved_device(self, torch: Any) -> str:
        if self._device == "cpu":
            return "cpu"
        if self._device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeInferenceError("已配置 CUDA，但 Linux 中未检测到可用 NVIDIA GPU")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _validate_model_reference(value: str, label: str) -> None:
        path = Path(value)
        if path.exists() and path.is_file():
            raise LinuxModelFormatError(
                f"{label} {value!r} 是文件而不是 Hugging Face 模型目录；"
                "请设置为 Hugging Face 仓库名或完整模型目录"
            )

    @staticmethod
    def _validate_adapter_reference(adapter_path: str) -> None:
        path = Path(adapter_path)
        if not path.exists():
            return
        if path.is_file() or not (path / "adapter_config.json").is_file():
            raise LinuxModelFormatError(
                "adapter 不是完整的 Linux PEFT 目录；需要 adapter_config.json 与 "
                "adapter_model.safetensors"
            )
        if not (path / "adapter_model.safetensors").is_file():
            raise LinuxModelFormatError("PEFT adapter 目录缺少 adapter_model.safetensors")

    @staticmethod
    def _validate_gpu_capacity(torch: Any, base_model: str, device: str) -> None:
        """在下载/加载前拦截已知不可能的 8B + 4GB 组合。"""

        if device != "cuda" or "qwen3-8b" not in base_model.lower():
            return
        memory_bytes = torch.cuda.get_device_properties(0).total_memory
        if memory_bytes < 10 * 1024**3:
            memory_gib = memory_bytes / 1024**3
            raise RuntimeInferenceError(
                "历史 Qwen3-8B LoRA 需要至少约 10GB GPU 显存；"
                f"当前仅检测到 {memory_gib:.1f}GB。请使用更大 Linux GPU，"
                "或为 1.7B 基座重新训练 LoRA。"
            )

    def _dependencies(self) -> tuple[Any, Any, Any]:
        if self._torch is not None:
            torch = self._torch
        else:
            try:
                torch = self._import_module("torch")
            except ImportError as error:
                raise RuntimeInferenceError(
                    "未安装 Linux 模型依赖；请执行 uv sync --extra linux-ml"
                ) from error
            self._torch = torch
        try:
            transformers = self._import_module("transformers")
            peft = self._import_module("peft")
        except ImportError as error:
            raise RuntimeInferenceError(
                "未安装 Transformers/PEFT；请执行 uv sync --extra linux-ml"
            ) from error
        return torch, transformers, peft

    @staticmethod
    def _raise_if_stopped(
        should_cancel: Callable[[], bool] | None,
        deadline: datetime | None,
    ) -> None:
        if should_cancel is not None and should_cancel():
            raise InferencePreemptedError("推理已被更高优先级请求抢占")
        if deadline is not None and datetime.now(UTC) >= deadline:
            raise InferencePreemptedError("推理已超过截止时间")


class DatabaseLinuxRuntimeInferenceBackend:
    """仅实现 Runtime Director、PersonaActor 与 token 计数三种 Linux 请求。"""

    def __init__(
        self,
        database: Database,
        *,
        device: str = "auto",
        load_in_4bit: bool = True,
    ) -> None:
        self._database = database
        self._runtime = SharedLinuxModelRuntime(device=device, load_in_4bit=load_in_4bit)

    def infer(
        self,
        request: PersonaInferenceRequest,
        should_cancel: Callable[[], bool] | None = None,
    ) -> PersonaInferenceResult:
        if request.request_type not in {
            "runtime_director",
            "runtime_actor",
            "runtime_token_count",
        }:
            raise RuntimeInferenceError("不支持的 Runtime 推理请求类型")
        base_model, _adapter_path = self._model_paths(request.model_version_id)
        system_prompt = request.payload.get("system_prompt")
        if request.request_type == "runtime_token_count":
            text = request.payload.get("text")
            if not isinstance(text, str):
                raise RuntimeInferenceError("Runtime token 计数请求结构无效")
            output: Mapping[str, object] = {
                "count": self._runtime.count_text_tokens(base_model=base_model, text=text)
            }
        else:
            raw_messages = request.payload.get("messages")
            if not isinstance(system_prompt, str) or not isinstance(raw_messages, list):
                raise RuntimeInferenceError("Runtime 推理请求结构无效")
            messages = self._validate_messages(raw_messages)
            output = {
                "content": self._runtime.generate_raw(
                    base_model=base_model,
                    # 体验阶段保留 PersonaActor 的 Agent 与工具链，但不加载 LoRA。
                    # 模型版本仍用于选择同一受控基础模型；adapter 路径不参与推理。
                    adapter_path=None,
                    messages=[{"role": "system", "content": system_prompt}, *messages],
                    max_tokens=512,
                    should_cancel=should_cancel,
                    deadline=request.deadline,
                )
            }
        return PersonaInferenceResult(
            request_id=request.request_id,
            request_type=request.request_type,
            output=output,
        )

    def _model_paths(self, model_version_id: str) -> tuple[str, str]:
        from sqlalchemy.orm import Session

        with Session(self._database.engine) as session:
            version = session.get(ModelVersion, model_version_id)
            if version is None:
                raise RuntimeInferenceError("模型版本不存在")
            return version.base_model, version.adapter_path

    @staticmethod
    def _validate_messages(raw_messages: list[object]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                raise RuntimeInferenceError("Runtime 消息结构无效")
            role, content = item.get("role"), item.get("content")
            if role not in {"user", "assistant", "tool"} or not isinstance(content, str):
                raise RuntimeInferenceError("Runtime 消息结构无效")
            messages.append({"role": "user" if role == "tool" else role, "content": content})
        return messages

    def close(self) -> None:
        self._runtime.release()
