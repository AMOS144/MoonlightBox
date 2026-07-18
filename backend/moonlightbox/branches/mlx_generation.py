import importlib
import json
from typing import Protocol, cast

from sqlalchemy.orm import Session

from moonlightbox.branches.generation import GeneratorUnavailableError
from moonlightbox.db import Database
from moonlightbox.training.models import ModelVersion


class MlxRuntime(Protocol):
    def load(
        self,
        model_path: str,
        *,
        adapter_path: str,
    ) -> tuple[object, object]: ...

    def generate(
        self,
        model: object,
        tokenizer: object,
        *,
        prompt: str,
        max_tokens: int,
        verbose: bool,
    ) -> str: ...


class DatabaseMlxGenerator:
    def __init__(self, database: Database) -> None:
        self._database = database

    def generate(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        with Session(self._database.engine) as session:
            version = session.get(ModelVersion, model_version_id)
            if version is None:
                raise GeneratorUnavailableError("模型版本不存在")
            base_model = version.base_model
            adapter_path = version.adapter_path

        try:
            runtime = cast(MlxRuntime, importlib.import_module("mlx_lm"))
        except ImportError as error:
            raise GeneratorUnavailableError("未安装官方 mlx-lm 依赖") from error
        model, tokenizer = runtime.load(base_model, adapter_path=adapter_path)
        prompt = json.dumps(
            {
                "system": system_prompt,
                "messages": messages,
            },
            ensure_ascii=False,
        )
        return runtime.generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False,
        )
