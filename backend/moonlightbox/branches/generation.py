from typing import Protocol


class BranchGenerator(Protocol):
    def generate(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str: ...


class GeneratorUnavailableError(RuntimeError):
    pass


class UnavailableGenerator:
    def generate(
        self,
        model_version_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> str:
        del model_version_id, system_prompt, messages
        raise GeneratorUnavailableError("本地推理生成器尚未配置")
