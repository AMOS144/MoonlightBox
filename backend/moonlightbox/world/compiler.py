from dataclasses import dataclass
from typing import Any, Protocol

from moonlightbox.world.schemas import WorldProfileDraft

COMPILER_VERSION = "person-world-lived-world-v3"


class AgentCompilerClient(Protocol):
    """人物世界模型入口只提供原生工具模型；结构化批处理有自己的客户端接口。"""

    def create_agent_chat_model(self) -> Any: ...


@dataclass(frozen=True)
class RetrievalManifestItem:
    question: str
    mode: str
    document_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompiledWorldProfile:
    draft: WorldProfileDraft
    source_message_ids: tuple[str, ...]
    retrieval_manifest: tuple[RetrievalManifestItem, ...]
