"""Linux 本地模型的结构化记忆提取器。"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from moonlightbox.agent.linux_inference import SharedLinuxModelRuntime
from moonlightbox.branches.generation import GenerationFailedError, GeneratorUnavailableError
from moonlightbox.branches.memory_proposer import _extract_json_object
from moonlightbox.db import Database
from moonlightbox.training.models import ModelVersion


class DatabaseLinuxMemoryJsonGenerator:
    """以干净基座生成 JSON，避免人格 LoRA 干扰结构化协议。"""

    def __init__(
        self,
        database: Database,
        *,
        device: str = "auto",
        load_in_4bit: bool = True,
    ) -> None:
        self._database = database
        self._runtime = SharedLinuxModelRuntime(device=device, load_in_4bit=load_in_4bit)

    def generate_json(
        self,
        *,
        model_version_id: str,
        system_prompt: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        with Session(self._database.engine) as session:
            version = session.get(ModelVersion, model_version_id)
            if version is None:
                raise GeneratorUnavailableError("模型版本不存在")
            base_model = version.base_model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for attempt in range(2):
            raw = self._runtime.generate_raw(
                base_model=base_model,
                adapter_path=None,
                messages=messages,
                max_tokens=512,
            )
            try:
                return _extract_json_object(raw)
            except (json.JSONDecodeError, ValueError):
                if attempt == 0:
                    messages.extend(
                        [
                            {"role": "assistant", "content": raw},
                            {
                                "role": "user",
                                "content": "格式无效。只重新输出完整 JSON，不要解释。",
                            },
                        ]
                    )
        raise GenerationFailedError("Linux 本地模型未能生成有效记忆 JSON")
