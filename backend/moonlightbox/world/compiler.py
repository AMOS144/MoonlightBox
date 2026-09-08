from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

from moonlightbox.world.bundles import match_bundle_document_reference
from moonlightbox.world.client import LightRAGRetrieval, LightRAGSidecarClient
from moonlightbox.world.schemas import WorldProfileDraft

COMPILER_VERSION = "person-world-profile-v1"
PROFILE_QUESTIONS = (
    "目标人物“{subject_name}”是谁？她怎样称呼自己，别人怎样称呼她？",
    "目标人物“{subject_name}”现在和过去分别处于什么工作、学习和生活状态？",
    "目标人物“{subject_name}”认识哪些人？她如何理解自己与这些人的关系？",
    "哪些地方对目标人物“{subject_name}”的生活有意义？她在哪里居住、工作、学习或经常活动？",
    "目标人物“{subject_name}”明确表达过哪些喜欢、排斥和有条件的偏好？",
    "目标人物“{subject_name}”有哪些被自己或他人描述过的生活规律？这些规律适用于什么时间和情境？",
    "哪些事件改变了目标人物“{subject_name}”的生活、工作、地点或人际关系？",
    "目标人物“{subject_name}”和用户“{user_name}”的关系在不同时间如何变化？",
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class StructuredCompilerClient(Protocol):
    def create_structured_completion(
        self,
        *,
        system_content: str,
        user_content: str,
        response_model: type[ResponseModel],
        operation_id: str | None = None,
        max_prompt_chars: int | None = None,
    ) -> ResponseModel: ...


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


class BackgroundCompiler:
    def __init__(
        self,
        lightrag: LightRAGSidecarClient,
        compiler: StructuredCompilerClient,
        *,
        top_k: int = 30,
        chunk_top_k: int = 12,
        max_total_tokens: int = 16000,
    ) -> None:
        self._lightrag = lightrag
        self._compiler = compiler
        self._top_k = top_k
        self._chunk_top_k = chunk_top_k
        self._max_total_tokens = max_total_tokens

    def compile(
        self,
        *,
        workspace: str,
        project_id: str,
        subject_name: str,
        user_name: str,
        source_messages_by_document: dict[str, list[str]],
        progress: Callable[[int, int], None] | None = None,
    ) -> CompiledWorldProfile:
        retrievals: list[tuple[str, LightRAGRetrieval]] = []
        for index, question_template in enumerate(PROFILE_QUESTIONS):
            question = question_template.format(subject_name=subject_name, user_name=user_name)
            retrievals.append(
                (
                    question,
                    self._lightrag.query(
                        workspace,
                        question,
                        mode="mix",
                        top_k=self._top_k,
                        chunk_top_k=self._chunk_top_k,
                        max_total_tokens=self._max_total_tokens,
                    ),
                )
            )
            if progress is not None:
                progress(index + 1, len(PROFILE_QUESTIONS))

        allowed_sources = set(source_messages_by_document)
        manifests: list[RetrievalManifestItem] = []
        context_sections: list[str] = []
        referenced_sources: list[str] = []
        for index, (question, retrieval) in enumerate(retrievals, start=1):
            document_ids = tuple(dict.fromkeys(
                source_id
                for reference in retrieval.references
                for source_id in (
                    match_bundle_document_reference(reference.file_path, allowed_sources),
                )
                if source_id is not None
            ))
            referenced_sources.extend(document_ids)
            manifests.append(
                RetrievalManifestItem(
                    question=question,
                    mode="mix",
                    document_ids=document_ids,
                )
            )
            context_sections.append(
                f"## 检索问题 {index}\n{question}\n\n"
                f"命中文档：{', '.join(document_ids) or '未返回来源'}\n\n"
                f"检索上下文：\n{retrieval.context[:12000]}"
            )

        prompt = "\n\n".join(context_sections)
        generated = self._compiler.create_structured_completion(
            system_content=_compiler_system_prompt(subject_name, user_name),
            user_content=prompt,
            response_model=WorldProfileDraft,
            operation_id=f"world-profile:{project_id}",
            max_prompt_chars=110_000,
        )
        draft = _sanitize_sources(generated, allowed_sources)
        message_ids = tuple(
            dict.fromkeys(
                message_id
                for source in referenced_sources
                for message_id in source_messages_by_document.get(source, [])
            )
        )
        return CompiledWorldProfile(
            draft=draft,
            source_message_ids=message_ids,
            retrieval_manifest=tuple(manifests),
        )


def _compiler_system_prompt(subject_name: str, user_name: str) -> str:
    return f"""
你正在为数字人项目编译人物世界档案。目标人物是“{subject_name}”，用户是“{user_name}”。
输入是 LightRAG 从真实聊天知识图谱和原始文本块检索出的上下文，不是最终回答。

这不是客观人物百科，而是“{subject_name}”眼中的自己、他人和生活。请把检索到的内容整理进给定 JSON 结构：
身份、工作与教育、对她有意义的地点、她与他人的社会关系、偏好、重复活动、工作日/周末规律、
生活阶段、与用户的关系、重要事件及尚未消歧的候选。

不要设置证据分数，不要因为信息不完整而整条删除。每条内容必须填写对象、陈述类型、时间状态、消息来源和证据摘录。每条内容只区分：
- direct：原始对话直接表达；
- summarized：由多段历史归纳；
- inferred：根据上下文合理推导；
- superseded：旧信息已被后来的信息替代。

source_document_ids 只能填写输入“命中文档”中真实出现的文档名；不确定时留空。
不得把文档号、消息号或时间戳当成人物、地点或事件。不要把推断写成确定的实时事实。
不要把“我”“她”“他”“对方”等聊天代词当成目标人物或用户的身份依据；本任务中目标人物固定是“{subject_name}”，用户固定是“{user_name}”。
不要把 LightRAG 的实体类型或关系标签当成最终档案栏目；它们只是检索辅助。
关系必须读取完整上下文。“关系很好”不能单独归一为 friend_of；“搬到”应记录为地点变化事件，
并在语义明确时更新居住状态。保留关系的原始描述、对象、时间、状态和限定信息。
不确定、矛盾、转述和计划必须分别标记。发言者不等于陈述对象；行业规则、他人经历、引用、玩笑和反问不能写成目标人物事实。
一次性活动不能写成规律；计划不能写成当前事实；无法确定对象或时间时使用 unknown。每条陈述必须带真实消息 ID 和短证据摘录，不得为了填满字段而猜测。
没有内容的分类返回空数组，不要虚构信息。
""".strip()


def _sanitize_sources(
    profile: WorldProfileDraft,
    allowed_sources: set[str],
) -> WorldProfileDraft:
    payload = profile.model_dump(mode="python")

    def walk(value: object) -> None:
        if isinstance(value, dict):
            sources = value.get("source_document_ids")
            if isinstance(sources, list):
                value["source_document_ids"] = [
                    item
                    for item in dict.fromkeys(sources)
                    if isinstance(item, str) and item in allowed_sources
                ]
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(payload)
    return WorldProfileDraft.model_validate(payload)
