import importlib
import json
from typing import Protocol, cast

from pydantic import ValidationError
from sqlalchemy.orm import Session

from moonlightbox.branches.attribution import SubjectAttributionGuard
from moonlightbox.branches.context import runtime_identity_kernel
from moonlightbox.branches.continuity_models import BranchMemoryEpisode
from moonlightbox.branches.continuity_types import (
    MemoryCandidate,
    MemoryProposal,
    StateDeltaProposal,
)
from moonlightbox.branches.generation import (
    GenerationFailedError,
    GeneratorUnavailableError,
)
from moonlightbox.branches.mlx_generation import (
    MlxRuntime,
    MlxSampleUtils,
    MlxTokenizer,
)
from moonlightbox.db import Database
from moonlightbox.training.models import ModelVersion

MEMORY_PROPOSAL_SYSTEM_PROMPT = """
你站在数字人主体视角解释本轮经历。
用户对数字人的定义只是用户观点，不得直接成为数字人主体信念。
数字人自己的表达可以形成自我叙事，但不能单独创造客观世界事实。
客观事实必须来自本轮可观察的用户行为或双方共同完成的互动。
持久关系变化只能使用 trust、intimacy、reciprocity、safety、conflict、
boundary_pressure；情绪倾向只能使用 warmth、anger、sadness、anxiety、
disappointment、hope、guardedness、calm。不得直接覆盖 user_model，关于用户的
新判断必须先作为带互动证据的 belief。
只返回 MemoryProposal JSON，所有候选必须引用本轮真实消息 ID。
subject/predicate/object 必须足以区分记忆语义；不得把不同经历笼统写成“发送/消息”、
“表达/内容”或“发生/事情”。无法形成具体语义时不要创建候选。
不得引用其他分支、未来信息或未提供的消息。
每轮最多返回 2 个最重要候选，内容保持简洁。
JSON 格式必须是：
{
  "candidates": [{
    "kind": "fact|experience|self_narrative|belief|reflection",
    "content": "记忆内容",
    "subject": "user|digital_human|interaction",
    "predicate": "关系",
    "object": "值",
    "confidence": 0.0,
    "importance": 1.0,
    "evidence_message_ids": ["真实轮次 ID"],
    "stance": "support|oppose",
    "source_role": "user|digital_human|interaction"
  }],
  "state_delta": {
    "relationship_delta": {},
    "emotional_delta": {},
    "user_model_updates": {},
    "supporting_candidate_indexes": []
  }
}
没有可提取记忆时，candidates 必须是空数组，所有 delta 必须是空对象。
""".strip()


class MemoryJsonGenerator(Protocol):
    def generate_json(
        self,
        *,
        model_version_id: str,
        system_prompt: str,
        payload: dict[str, object],
    ) -> dict[str, object]: ...


class LocalMemoryProposer:
    def __init__(self, generator: MemoryJsonGenerator) -> None:
        self._generator = generator

    def propose(
        self,
        *,
        episode: BranchMemoryEpisode,
        identity_kernel: dict[str, object],
        current_state: dict[str, object],
    ) -> MemoryProposal:
        raw = self._generator.generate_json(
            model_version_id=episode.model_version_id,
            system_prompt=MEMORY_PROPOSAL_SYSTEM_PROMPT,
            payload={
                "episode": {
                    "user_turn_id": episode.user_turn_id,
                    "user_messages": episode.user_messages,
                    "assistant_turn_id": episode.assistant_turn_id,
                    "user_content": episode.user_content,
                    "assistant_bubbles": episode.assistant_bubbles,
                },
                "identity_kernel": runtime_identity_kernel(identity_kernel),
                "current_state": current_state,
            },
        )
        parsed = _parse_memory_proposal(raw)
        allowed_evidence = _episode_evidence_ids(episode)
        retained: list[MemoryCandidate] = []
        old_to_new: dict[int, int] = {}
        attribution_guard = SubjectAttributionGuard()
        for index, candidate in enumerate(parsed.candidates):
            if not set(candidate.evidence_message_ids).issubset(allowed_evidence):
                continue
            if not attribution_guard.accepts(candidate):
                continue
            old_to_new[index] = len(retained)
            retained.append(candidate)
        supporting = tuple(
            old_to_new[index]
            for index in parsed.state_delta.supporting_candidate_indexes
            if index in old_to_new
            and attribution_guard.can_update_persistent_state(
                parsed.candidates[index]
            )
        )
        state_delta = StateDeltaProposal(
            relationship_delta=(parsed.state_delta.relationship_delta if supporting else {}),
            emotional_delta=(parsed.state_delta.emotional_delta if supporting else {}),
            user_model_updates=(parsed.state_delta.user_model_updates if supporting else {}),
            supporting_candidate_indexes=supporting,
        )
        return MemoryProposal(candidates=tuple(retained), state_delta=state_delta)


def _episode_evidence_ids(episode: BranchMemoryEpisode) -> set[str]:
    ids = {episode.user_turn_id, episode.assistant_turn_id}
    for message in episode.user_messages or []:
        for field in ("message_id", "turn_id"):
            value = message.get(field)
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


class DatabaseMlxMemoryJsonGenerator:
    """使用分支绑定的本地 MLX 模型生成结构化记忆候选。"""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._loaded: dict[
            str,
            tuple[MlxRuntime, MlxSampleUtils, object, MlxTokenizer],
        ] = {}

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
        loaded = self._loaded.get(model_version_id)
        if loaded is None:
            try:
                runtime = cast(MlxRuntime, importlib.import_module("mlx_lm"))
                sample_utils = cast(
                    MlxSampleUtils,
                    importlib.import_module("mlx_lm.sample_utils"),
                )
            except ImportError as error:
                raise GeneratorUnavailableError("未安装官方 mlx-lm 依赖") from error
            # 结构化记忆抽取使用干净基座，避免人格 LoRA 的口语风格破坏 JSON 协议。
            model, tokenizer = runtime.load(base_model)
            loaded = runtime, sample_utils, model, tokenizer
            self._loaded[model_version_id] = loaded
        runtime, sample_utils, model, tokenizer = loaded
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        for attempt in range(2):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            output = runtime.generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=512,
                sampler=sample_utils.make_sampler(
                    temp=0.1,
                    top_p=0.9,
                    min_p=0.05,
                ),
                logits_processors=[
                    sample_utils.make_repetition_penalty(
                        penalty=1.05,
                        context_size=128,
                    )
                ],
                verbose=False,
            )
            try:
                parsed = _extract_json_object(output)
                return parsed
            except (json.JSONDecodeError, ValueError):
                if attempt == 0:
                    messages.extend(
                        [
                            {"role": "assistant", "content": output},
                            {
                                "role": "user",
                                "content": "格式无效。只重新输出完整 JSON，不要解释。",
                            },
                        ]
                    )
        raise GenerationFailedError("本地模型未能生成有效记忆 JSON")


def _extract_json_object(value: str) -> dict[str, object]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline >= 0 and last_fence > first_newline:
            cleaned = cleaned[first_newline + 1 : last_fence].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("输出中不存在 JSON 对象")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("记忆输出必须是 JSON 对象")
    return cast(dict[str, object], parsed)


def _parse_memory_proposal(raw: dict[str, object]) -> MemoryProposal:
    candidates: list[MemoryCandidate] = []
    raw_candidates = raw.get("candidates", [])
    if isinstance(raw_candidates, list):
        for candidate in raw_candidates:
            try:
                candidates.append(MemoryCandidate.model_validate(candidate))
            except ValidationError:
                continue
    raw_delta = raw.get("state_delta", {})
    try:
        state_delta = StateDeltaProposal.model_validate(raw_delta)
    except ValidationError:
        state_delta = StateDeltaProposal()
    return MemoryProposal(
        candidates=tuple(candidates),
        state_delta=state_delta,
    )
