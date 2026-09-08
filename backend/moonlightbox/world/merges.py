"""LightRAG 建图后的别名归一化 workflow。

这个模块只生成待人工审核的提案。真正修改 LightRAG 的 ``amerge_entities``
仍然只由审核接口在用户批准后调用。候选生成使用一个有界 LangGraph 节点，
并把原文定位、图节点详情和 LightRAG 局部检索封装成只读工具。
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from moonlightbox.world.client import LightRAGEntity, LightRAGSidecarClient
from moonlightbox.world.compiler import StructuredCompilerClient


class MergeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    entity: str = Field(min_length=1, max_length=500)
    basis: str = Field(min_length=1, max_length=1000)
    source_document_id: str | None = Field(default=None, max_length=180)
    message_id: str | None = Field(default=None, max_length=80)
    quote: str | None = Field(default=None, max_length=1200)


class MergeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    source_entities: list[str] = Field(min_length=1, max_length=20)
    target_entity: str = Field(min_length=1, max_length=500)
    reason: str = Field(default="", max_length=2000)
    evidence: list[MergeEvidence] = Field(default_factory=list, max_length=20)


class MergeCandidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    proposals: list[MergeCandidate] = Field(default_factory=list, max_length=100)


_TECHNICAL_ENTITY_ID = re.compile(
    r"^(?:wxid_[A-Za-z0-9_-]+|[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*)$", re.IGNORECASE
)


def is_technical_entity_name(name: str) -> bool:
    value = name.strip()
    return bool(_TECHNICAL_ENTITY_ID.fullmatch(value)) and (
        value.lower().startswith("wxid_") or len(value) >= 8
    )


def person_entities(entities: Iterable[LightRAGEntity]) -> list[LightRAGEntity]:
    """只保留 LightRAG 标为人物的节点，不从普通正文猜测人物。"""
    return [
        item
        for item in entities
        if not is_technical_entity_name(item.entity_name)
        if _is_person_graph_data(item.graph_data)
    ]


def _is_person_graph_data(graph_data: dict[str, object] | None) -> bool:
    if not isinstance(graph_data, dict):
        return False
    for key in ("entity_type", "type", "node_type", "entityType"):
        value = graph_data.get(key)
        if isinstance(value, str) and value.strip().casefold() in {
            "person",
            "human",
            "people",
            "人物",
            "人",
        }:
            return True
    return False


_ALIAS_DESCRIPTION_CUES = (
    "昵称",
    "暱稱",
    "别称",
    "简称",
    "小名",
    "称呼",
    "改名",
    "原本叫",
    "原名",
    "被叫作",
    "被称为",
    "称为",
    "叫作",
    "叫做",
)
_ROLE_PLACEHOLDER_CUES = (
    "第一人称",
    "对话中的用户",
    "对话的一方用户",
    "说话者",
    "聊天参与者",
    "项目用户",
)
_IDENTITY_DESCRIPTION_CUES = ("名字", "姓名", "实名", "全名", "原名", "昵称为", "叫作", "叫做")


def _description_marks_alias(node: dict[str, object]) -> bool:
    """只依据 LightRAG 节点描述识别“这是昵称节点”，不维护人名黑名单。"""
    description = str(node.get("description", ""))
    return any(cue in description for cue in _ALIAS_DESCRIPTION_CUES)


def _description_marks_role_placeholder(node: dict[str, object]) -> bool:
    description = str(node.get("description", ""))
    if any(cue in description for cue in _ROLE_PLACEHOLDER_CUES):
        return True
    return "用户" in description and any(
        cue in description for cue in ("对话", "聊天", "说话", "自称", "发送者")
    )


def _description_has_identity_signal(node: dict[str, object]) -> bool:
    description = str(node.get("description", ""))
    return any(cue in description for cue in _IDENTITY_DESCRIPTION_CUES)


@dataclass(frozen=True)
class AliasAgentMessage:
    message_id: str
    document_id: str | None
    timestamp: str
    participant: str
    content: str
    participant_role: str = "other"


class AliasResolutionTools:
    """Agent 使用的只读工具集合。工具不能写图谱或改变审核状态。"""

    def __init__(
        self,
        *,
        sidecar: LightRAGSidecarClient,
        workspace: str,
        entities: list[LightRAGEntity],
        original_messages: Iterable[AliasAgentMessage] = (),
        max_lightrag_queries: int = 32,
    ) -> None:
        self.sidecar = sidecar
        self.workspace = workspace
        self.entities = tuple(entities)
        self.original_messages = tuple(original_messages)
        self.max_lightrag_queries = max(0, max_lightrag_queries)

    def list_person_nodes(self) -> list[dict[str, object]]:
        return [
            {
                "entity": item.entity_name,
                "description": (
                    str(item.graph_data.get("description", ""))[:450]
                    if isinstance(item.graph_data, dict)
                    else ""
                ),
                "graph_data": dict(item.graph_data or {}),
            }
            for item in self.entities
        ]

    def locate_original_mentions(
        self, names: list[str], limit: int = 80
    ) -> list[dict[str, object]]:
        needles = tuple(dict.fromkeys(item.strip() for item in names if item.strip()))
        matches = [
            {
                "message_id": item.message_id,
                "document_id": item.document_id,
                "timestamp": item.timestamp,
                "participant": item.participant,
                "content": item.content[:400],
            }
            for item in self.original_messages
            if any(needle in item.content or needle == item.participant for needle in needles)
        ]
        return matches[: max(0, limit)]

    def query_lightrag(self, name: str) -> dict[str, object]:
        try:
            result = self.sidecar.query(
                self.workspace,
                f"请返回人物节点“{name}”的描述、关系网络、出现过的称呼及相关原文证据。",
                mode="local",
                top_k=12,
                chunk_top_k=8,
                max_total_tokens=6000,
            )
            return {"entity": name, "context": result.context[:1200]}
        except Exception as error:
            return {"entity": name, "context": "", "error": type(error).__name__}

    def as_langchain_tools(self) -> dict[str, StructuredTool]:
        return {
            "list_person_nodes": StructuredTool.from_function(
                func=lambda: self.list_person_nodes(),
                name="list_person_nodes",
                description="读取已由 LightRAG 标记为人物的全部图节点及描述。",
            ),
            "locate_original_mentions": StructuredTool.from_function(
                func=self.locate_original_mentions,
                name="locate_original_mentions",
                description="在原始聊天消息中定位名称，返回 message_id、Bundle 文档和原文摘录。",
            ),
            "query_lightrag": StructuredTool.from_function(
                func=self.query_lightrag,
                name="query_lightrag",
                description="查询一个人物节点的 LightRAG 局部关系和上下文。",
            ),
        }


class _AliasAgentState(TypedDict, total=False):
    nodes: list[dict[str, object]]
    original: list[dict[str, object]]
    lightrag: list[dict[str, object]]
    proposals: list[MergeCandidate]


class AliasResolutionAgent:
    """有界 LangGraph agent：工具读取 → 结构化候选输出。"""

    def __init__(
        self,
        *,
        tools: AliasResolutionTools,
        compiler: StructuredCompilerClient,
        subject_name: str,
        user_name: str,
    ) -> None:
        self.tools = tools
        self.compiler = compiler
        self.subject_name = subject_name
        self.user_name = user_name
        self._toolset = tools.as_langchain_tools()

    def run(self) -> list[MergeCandidate]:
        graph = StateGraph(_AliasAgentState)
        graph.add_node("read_person_nodes", self._read_person_nodes)
        graph.add_node("locate_original_sources", self._locate_original_sources)
        graph.add_node("query_lightrag_context", self._query_lightrag_context)
        graph.add_node("propose_aliases", self._propose_aliases)
        graph.add_edge(START, "read_person_nodes")
        graph.add_edge("read_person_nodes", "locate_original_sources")
        graph.add_edge("locate_original_sources", "query_lightrag_context")
        graph.add_edge("query_lightrag_context", "propose_aliases")
        graph.add_edge("propose_aliases", END)
        result = graph.compile().invoke({})
        return list(result.get("proposals", []))

    def _read_person_nodes(self, _: _AliasAgentState) -> _AliasAgentState:
        return {"nodes": self._toolset["list_person_nodes"].invoke({})}

    def _locate_original_sources(self, state: _AliasAgentState) -> _AliasAgentState:
        # 原文定位优先覆盖疑似别称和项目主体/用户，避免“我、你”等高频
        # 角色词占满前 20 条结果，让 Agent 看不到 jxy、羽、胖笨笨等实际称呼。
        nodes = state.get("nodes", [])
        names = [
            str(item["entity"])
            for item in nodes
            if _description_marks_alias(item)
            or str(item["entity"]) in {self.subject_name, self.user_name}
        ]
        if not names:
            names = [str(item["entity"]) for item in nodes]
        return {
            "original": self._toolset["locate_original_mentions"].invoke(
                {"names": names, "limit": 80}
            )
        }

    def _query_lightrag_context(self, state: _AliasAgentState) -> _AliasAgentState:
        nodes = state.get("nodes", [])
        # 优先查询描述中明确出现“昵称/改名/称呼”的节点；这样工具预算不会
        # 被普通人物节点消耗，模型能直接看到“胖笨笨/笨胖胖”的关系证据。
        aliases = [
            str(item["entity"])
            for item in nodes
            if _description_marks_alias(item)
        ]
        identity_names = [
            str(item["entity"])
            for item in nodes
            if _description_has_identity_signal(item)
            and str(item["entity"]) not in aliases
        ]
        names = aliases + identity_names + [
            str(item["entity"])
            for item in nodes
            if str(item["entity"]) not in aliases
            and str(item["entity"]) not in identity_names
        ]
        contexts = [
            self._toolset["query_lightrag"].invoke({"name": name})
            for name in names[: self.tools.max_lightrag_queries]
        ]
        return {"lightrag": contexts}

    def _propose_aliases(self, state: _AliasAgentState) -> _AliasAgentState:
        nodes = state.get("nodes", [])
        allowed = {str(item["entity"]) for item in nodes}
        explicit = {self.subject_name.strip(), self.user_name.strip()}
        # 规范姓名不依赖“小/胖/笨”等词表。只把节点描述明确标成昵称/称呼
        # 的节点排除为 target；像“余熠”这种被正文直接当作姓名提及的节点
        # 仍然可以作为第三方人物的规范名。
        alias_names = {
            name
            for name in allowed
            if _description_marks_alias(next(item for item in nodes if item["entity"] == name))
        }
        role_names = {
            str(item["entity"])
            for item in nodes
            if _description_marks_role_placeholder(item)
            and str(item["entity"]) != self.subject_name.strip()
            and not (
                str(item["entity"]) == self.user_name.strip()
                and _description_has_identity_signal(item)
            )
        }
        identity_names = {
            str(item["entity"])
            for item in nodes
            if _description_has_identity_signal(item)
        }
        canonical_names = (identity_names - alias_names - role_names) | (
            (allowed & explicit) - role_names
        )
        node_lines = [
            f"- {item['entity']}: {item['description']}\n  graph_data={str(item['graph_data'])[:300]}"
            for item in nodes
        ]
        original_lines = [
            f"- message_id={item.get('message_id')} document={item.get('document_id') or 'unknown'} "
            f"time={item.get('timestamp')} {item.get('participant')}：{str(item.get('content', ''))[:400]}"
            for item in state.get("original", [])
        ]
        lightrag_lines = [
            f"### {item.get('entity')}\n{item.get('context', '')}"
            for item in state.get("lightrag", [])
            if item.get("context")
        ]
        prompt = f"""你是人物身份归一化 workflow 中的 AliasResolutionAgent。请提出待人工审核的“别称 → 真实姓名”候选。

绝对规则：
1. source_entities 只能是昵称、简称、英文名、拼音或拼音首字母等别称；target_entity 必须是列表中同一现实人物的稳定实名节点。
2. 禁止输出“昵称 → 昵称”。例如不能输出“胖笨笨 → 笨胖胖”或“入 → 此入”；如果两个都是别称，必须一起放入 source_entities，并归入列表中的真实姓名（例如余熠、洪欣羽）。
3. target_entity 必须逐字来自人物节点列表，并且最好能在原始消息的参与者字段中找到。 “我、用户、说话者、聊天参与者”等只是角色占位符，不能作为规范姓名；有实名节点时必须归到实名。不能创造姓名，也不能把仅出现在正文中的称呼当 target_entity。
4. “A 的恋人/母亲/朋友/同事”只证明 source 与 A 的关系，绝不证明 source 就是 A；这种描述不能把 source 合并到 A。只有节点描述、LightRAG 查询或原文定位能证明同一人时才输出。名称相似、同时出现、关系相同都不够；证据不足就不输出。
5. evidence 必须引用工具返回的 message_id/document 或明确的 LightRAG 依据，不能编造来源。为保证 JSON 有效，basis/reason 中不要使用双引号；quote 可以留空。

输出 JSON：{{"proposals":[{{"source_entities":["别称1","别称2"],"target_entity":"实名","reason":"...","evidence":[{{"entity":"...","basis":"...","source_document_id":"...","message_id":"...","quote":"..."}}]}}]}}

项目目标人物实名：{self.subject_name}
项目用户实名：{self.user_name}

可用的规范姓名候选（target_entity 只能从这里选择）：
{", ".join(sorted(canonical_names))}

人物节点列表：
{chr(10).join(node_lines)}

原文定位工具结果：
{chr(10).join(original_lines) or "（没有定位到名称原文）"}

LightRAG 查询工具结果：
{chr(10).join(lightrag_lines) or "（没有额外检索结果）"}
"""
        try:
            response = self.compiler.create_structured_completion(
                system_content=(
                    "你是一个受 LangGraph 工具约束的实体归一化节点。只生成候选，不执行合并；"
                    "必须把昵称归入真实姓名，绝不生成昵称到昵称的映射。"
                ),
                user_content=prompt,
                response_model=MergeCandidateResponse,
                operation_id=f"entity-alias-agent:{self.tools.workspace}",
                max_prompt_chars=110_000,
            )
        except Exception:
            # 供应商偶尔返回不合法 JSON 时，仍用已读取的图描述和 LightRAG
            # 上下文生成保守候选；这一步同样只写 pending，不执行合并。
            return {
                "proposals": _attach_original_evidence(
                    _dedupe_candidates(
                        _heuristic_candidates(
                            nodes,
                            state.get("lightrag", []),
                            allowed=allowed,
                            canonical_names=canonical_names,
                        )
                    + _explicit_identity_candidates(
                        nodes,
                        canonical_names=canonical_names,
                        subject_name=self.subject_name,
                        excluded_names={self.user_name},
                    )
                    ),
                    state.get("original", []),
                )
            }
        proposals = _sanitize_candidates(
            response.proposals,
            allowed,
            # 只接受描述中出现身份信号的稳定姓名节点，以及项目主体/用户；
            # 关系角色（女同事、他、母亲）即使是 person 节点，也不能成为
            # 别名归并目标。先允许角色占位符走到下一步，再依据工具证据换成实名。
            canonical_names=canonical_names | (allowed & explicit),
        )
        proposals = _remap_role_targets(
            proposals,
            nodes,
            state.get("lightrag", []),
            role_names=role_names,
            canonical_names=canonical_names,
            excluded_names={self.subject_name.strip()},
        )
        # 对节点描述中的明确身份表达做一个可解释的补充。模型有时只返回
        # 一个高置信候选，但“江昕岳的昵称为 jxy”“洪欣羽的暱稱是羽”
        # 已经足够形成审核卡片，不能因为模型输出截断而丢失。
        explicit_proposals = _explicit_identity_candidates(
            nodes,
            canonical_names=canonical_names,
            subject_name=self.subject_name,
            excluded_names={self.user_name},
        )
        merged = _dedupe_candidates(
            _filter_relation_target_conflicts(
                proposals,
                nodes,
                state.get("lightrag", []),
            )
            + explicit_proposals
        )
        return {
            "proposals": _attach_original_evidence(merged, state.get("original", []))
        }


def _heuristic_candidates(
    nodes: list[dict[str, object]],
    contexts: list[dict[str, object]],
    *,
    allowed: set[str],
    canonical_names: set[str],
) -> list[MergeCandidate]:
    """模型响应不可解析时的保守降级：只使用工具文本中的显式节点共现。"""
    context_by_name = {
        str(item.get("entity")): str(item.get("context", ""))
        for item in contexts
    }
    alias_names = {
        str(item["entity"])
        for item in nodes
        if _description_marks_alias(item)
    }
    grouped: dict[str, list[str]] = {}
    for node in nodes:
        alias = str(node["entity"])
        if alias not in alias_names:
            continue
        text = f"{node.get('description', '')} {context_by_name.get(alias, '')}"
        targets = [
            name
            for name in canonical_names
            if name != alias and name in text
        ]
        for target in targets:
            grouped.setdefault(target, []).append(alias)
    return [
        MergeCandidate(
            source_entities=list(dict.fromkeys(sources)),
            target_entity=target,
            reason="图节点描述或 LightRAG 工具上下文显式共现，提交人工确认。",
            evidence=[{"entity": source, "basis": "tool_context_cooccurrence"} for source in sources],
        )
        for target, sources in grouped.items()
        if sources
    ]


def _explicit_identity_candidates(
    nodes: list[dict[str, object]],
    *,
    canonical_names: set[str],
    subject_name: str,
    excluded_names: set[str] | None = None,
) -> list[MergeCandidate]:
    """从节点描述中的明确“昵称/简称为 X”表达补充高置信提案。"""
    result: list[MergeCandidate] = []
    for source_node in nodes:
        source = str(source_node.get("entity", ""))
        if not source or source == subject_name or source in (excluded_names or set()):
            continue
        source_description = str(source_node.get("description", ""))
        source_graph = source_node.get("graph_data")
        source_document = (
            str(source_graph.get("source_id"))[:180]
            if isinstance(source_graph, dict) and source_graph.get("source_id")
            else None
        )
        for target_node in nodes:
            target = str(target_node.get("entity", ""))
            if not target or target == source or target not in canonical_names:
                continue
            target_description = str(target_node.get("description", ""))
            target_graph = target_node.get("graph_data")
            target_document = (
                str(target_graph.get("source_id"))[:180]
                if isinstance(target_graph, dict) and target_graph.get("source_id")
                else None
            )
            # 目标节点直接声明“昵称为 source / 简称 source”。
            target_declares_source = bool(
                re.search(
                    rf"(?:昵称|暱稱|简称|簡稱|小名|称呼)\s*(?:为|是|叫作|叫做)\s*[“\"']?{re.escape(source)}",
                    target_description,
                    flags=re.IGNORECASE,
                )
            )
            # 别称节点直接声明“target 的昵称/暱稱”。仅对项目主体放宽，
            # 避免把“某人的昵称”这种关系描述误当成身份归并。
            source_declares_subject = (
                target == subject_name
                and bool(
                    re.search(
                        rf"{re.escape(target)}的(?:昵称|暱稱|别称|简称|小名)"
                        rf".{{0,36}}(?:使用|称呼|称为|叫作|叫做)"
                        rf".{{0,16}}(?:这个|這個|此|該)?(?:昵称|暱稱|别称|簡稱|简称|小名|称呼)",
                        source_description,
                    )
                )
            )
            if not (target_declares_source or source_declares_subject):
                continue
            document_id = target_document or source_document
            result.append(
                MergeCandidate(
                    source_entities=[source],
                    target_entity=target,
                    reason="节点描述直接声明该名称是目标人物的昵称或简称，提交人工确认。",
                    evidence=[
                        MergeEvidence(
                            entity=source,
                            basis="explicit_identity_statement",
                            source_document_id=document_id,
                        )
                    ],
                )
            )
    return result


def _dedupe_candidates(candidates: Iterable[MergeCandidate]) -> list[MergeCandidate]:
    result: list[MergeCandidate] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for candidate in candidates:
        key = (candidate.target_entity, tuple(sorted(candidate.source_entities)))
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _attach_original_evidence(
    candidates: Iterable[MergeCandidate],
    original: Iterable[dict[str, object]],
) -> list[MergeCandidate]:
    """用定位工具的真实消息补齐模型遗漏的 message_id/document_id。"""
    matches: dict[str, dict[str, object]] = {}
    for item in original:
        content = str(item.get("content", ""))
        participant = str(item.get("participant", ""))
        for key in {content, participant}:
            if key and key not in matches:
                matches[key] = item
    result: list[MergeCandidate] = []
    for candidate in candidates:
        evidence = list(candidate.evidence)
        covered = {item.entity for item in evidence if item.message_id}
        for source in candidate.source_entities:
            if source in covered:
                continue
            match = next(
                (
                    item
                    for key, item in matches.items()
                    if source in key
                ),
                None,
            )
            if match is None:
                continue
            evidence.append(
                MergeEvidence(
                    entity=source,
                    basis="original_message_match",
                    source_document_id=(
                        str(match.get("document_id"))[:180]
                        if match.get("document_id")
                        else None
                    ),
                    message_id=(
                        str(match.get("message_id"))[:80]
                        if match.get("message_id")
                        else None
                    ),
                    quote=str(match.get("content", ""))[:1200],
                )
            )
        result.append(candidate.model_copy(update={"evidence": evidence[:20]}))
    return result


def _filter_relation_target_conflicts(
    candidates: Iterable[MergeCandidate],
    nodes: Iterable[dict[str, object]],
    contexts: Iterable[dict[str, object]],
) -> list[MergeCandidate]:
    """拒绝把“某人的恋人/朋友”等关系对象错归到该某人名下的候选。"""
    descriptions = {
        str(item["entity"]): str(item.get("description", ""))
        for item in nodes
    }
    contexts_by_name = {
        str(item.get("entity")): str(item.get("context", ""))
        for item in contexts
    }
    accepted: list[MergeCandidate] = []
    for candidate in candidates:
        source_text = "\n".join(
            f"{descriptions.get(source, '')}\n{contexts_by_name.get(source, '')}"
            for source in candidate.source_entities
        )
        target = re.escape(candidate.target_entity)
        direct_identity = any(
            re.search(pattern, source_text)
            for source in candidate.source_entities
            for pattern in (
                rf"{target}.{{0,12}}(?:昵称|别称|简称|小名|叫作|叫做).{{0,24}}{re.escape(source)}",
                rf"{re.escape(source)}.{{0,24}}(?:是|叫作|叫做|称为).{{0,24}}{target}",
            )
        )
        relation_only = bool(re.search(rf"{target}的[^，。；\n]{{1,12}}", source_text))
        if relation_only and not direct_identity:
            continue
        accepted.append(candidate)
    return accepted


def _remap_role_targets(
    candidates: Iterable[MergeCandidate],
    nodes: Iterable[dict[str, object]],
    contexts: Iterable[dict[str, object]],
    *,
    role_names: set[str],
    canonical_names: set[str],
    excluded_names: set[str] | None = None,
) -> list[MergeCandidate]:
    """把模型返回的“我/用户”等角色节点，按关系上下文换成实名节点。"""
    if not role_names:
        return list(candidates)
    descriptions = {
        str(item["entity"]): str(item.get("description", "")) for item in nodes
    }
    contexts_by_name = {
        str(item.get("entity")): str(item.get("context", "")) for item in contexts
    }
    result: list[MergeCandidate] = []
    for candidate in candidates:
        if candidate.target_entity not in role_names:
            result.append(candidate)
            continue
        source_text = " ".join(
            f"{descriptions.get(source, '')} {contexts_by_name.get(source, '')}"
            for source in candidate.source_entities
        )
        # 关系词来自 LightRAG 描述，不依赖具体人名黑名单；只在有唯一最高
        # 匹配时替换，无法区分则保留原候选交由用户处理。
        chinese = "".join(re.findall(r"[一-鿿]", source_text))
        relation_terms = [
            chinese[index : index + 2]
            for index in range(max(0, len(chinese) - 1))
        ]
        scored: list[tuple[int, str]] = []
        for name in canonical_names - role_names - (excluded_names or set()):
            target_text = f"{descriptions.get(name, '')} {contexts_by_name.get(name, '')}"
            score = sum(1 for term in relation_terms if len(term) >= 2 and term in target_text)
            if score:
                scored.append((score, name))
        scored.sort(reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            target = scored[0][1]
            result.append(candidate.model_copy(update={"target_entity": target}))
        else:
            # 角色占位符不应进入最终审核卡片；无法解析时宁可不出候选。
            continue
    return result


def generate_merge_candidates(
    *,
    sidecar: LightRAGSidecarClient,
    compiler: StructuredCompilerClient,
    workspace: str,
    subject_name: str,
    user_name: str,
    original_messages: Iterable[AliasAgentMessage] = (),
    max_entities: int = 160,
) -> list[MergeCandidate]:
    entities = person_entities(sidecar.list_entities(workspace))[:max_entities]
    if len(entities) < 2:
        return []
    tools = AliasResolutionTools(
        sidecar=sidecar,
        workspace=workspace,
        entities=entities,
        original_messages=original_messages,
    )
    try:
        return AliasResolutionAgent(
            tools=tools,
            compiler=compiler,
            subject_name=subject_name,
            user_name=user_name,
        ).run()
    except Exception:
        explicit = {subject_name.strip(), user_name.strip()}
        allowed = {item.entity_name for item in entities}
        descriptions = {
            item.entity_name: item.graph_data or {} for item in entities
        }
        alias_names = {
            name for name, data in descriptions.items() if _description_marks_alias(data)
        }
        role_names = {
            name
            for name, data in descriptions.items()
            if _description_marks_role_placeholder(data)
            and name != subject_name.strip()
            and not (
                name == user_name.strip()
                and _description_has_identity_signal({"description": data.get("description", "")})
            )
        }
        identity_names = {
            name for name, data in descriptions.items() if _description_has_identity_signal(data)
        }
        canonical = (identity_names - alias_names - role_names) | (
            (allowed & explicit) - role_names
        )
        return _deterministic_format_candidates(entities, canonical_names=canonical)


def _sanitize_candidates(
    candidates: Iterable[MergeCandidate],
    allowed: set[str],
    *,
    canonical_names: set[str] | None = None,
) -> list[MergeCandidate]:
    result: list[MergeCandidate] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for candidate in candidates:
        sources = list(dict.fromkeys(item for item in candidate.source_entities if item in allowed))
        target = candidate.target_entity if candidate.target_entity in allowed else ""
        sources = [item for item in sources if item != target]
        if canonical_names is not None and target not in canonical_names:
            continue
        if not target or not sources:
            continue
        key = (target, tuple(sorted(sources)))
        if key in seen:
            continue
        seen.add(key)
        result.append(
            MergeCandidate(
                source_entities=sources,
                target_entity=target,
                reason=candidate.reason.strip(),
                evidence=[item.model_dump(mode="json") for item in candidate.evidence],
            )
        )
    return result


def _deterministic_format_candidates(
    entities: Iterable[LightRAGEntity], *, canonical_names: set[str] | None = None
) -> list[MergeCandidate]:
    groups: dict[str, list[str]] = {}
    for item in entities:
        key = re.sub(r"[\s_\-]+", "", item.entity_name).casefold()
        if key:
            groups.setdefault(key, []).append(item.entity_name)
    result: list[MergeCandidate] = []
    for names in groups.values():
        unique = list(dict.fromkeys(names))
        if len(unique) < 2:
            continue
        target = next(
            (name for name in unique if canonical_names is None or name in canonical_names),
            None,
        )
        if target is None:
            continue
        sources = [name for name in unique if name != target]
        result.append(
            MergeCandidate(
                source_entities=sources,
                target_entity=target,
                reason="名称仅存在大小写、空格、下划线或连字符格式差异，提交人工确认。",
                evidence=[{"entity": name, "basis": "format_variant"} for name in unique],
            )
        )
    return result
