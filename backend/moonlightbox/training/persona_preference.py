import hashlib
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from moonlightbox.branches.context import TRUSTED_SITUATIONAL_CONTEXT_PREFIX
from moonlightbox.branches.memory_policy import is_current_state_question
from moonlightbox.branches.replies import GeneratedReplyTurn
from moonlightbox.training.model_acceptance import ModelOutputStructureError


class PreferenceNegativeGenerator(Protocol):
    def generate(
        self,
        base_model: str,
        adapter_path: str,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn: ...

    def set_seed(self, seed: int) -> None: ...


class PreferenceTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class PersonaPreferencePair:
    prompt: tuple[dict[str, str], ...]
    chosen: str
    rejected: str
    source_hash: str
    source_group_hash: str
    rejected_structure_valid: bool


@dataclass(frozen=True)
class PreferenceDatasetManifest:
    schema_version: str
    source_split: str
    source_train_sha256: str
    conversation_mode: str
    negative_mode: str
    negative_variants_per_source: int
    requested_count: int
    generated_count: int
    skipped_identical_count: int
    train_count: int
    valid_count: int
    pair_sha256: str
    test_used: bool


@dataclass(frozen=True)
class TokenizedPreferencePair:
    chosen_tokens: tuple[int, ...]
    chosen_prompt_length: int
    rejected_tokens: tuple[int, ...]
    rejected_prompt_length: int
    source_hash: str


class TokenizedPreferenceDataset:
    def __init__(
        self,
        pairs: list[TokenizedPreferencePair],
        *,
        source_count: int | None = None,
        skipped_oversized: int = 0,
    ) -> None:
        self._pairs = tuple(pairs)
        self.source_count = source_count if source_count is not None else len(pairs)
        self.skipped_oversized = skipped_oversized

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> TokenizedPreferencePair:
        return self._pairs[index]


def build_persona_preference_dataset(
    *,
    data_dir: Path,
    output_dir: Path,
    generator: PreferenceNegativeGenerator,
    base_model: str,
    adapter_path: str,
    sample_count: int = 256,
    conversation_mode: str = "responsive",
    negative_mode: Literal[
        "model_response",
        "semantic_style_rewrite",
        "deterministic_assistant_register",
        "deterministic_register_only",
        "memory_authority_poison",
        "memory_authority_mixed",
    ] = "model_response",
    style_rewrite_min_similarity: float = 0.25,
    negative_variants_per_source: int = 1,
) -> PreferenceDatasetManifest:
    """Build train-only chosen/rejected pairs from the model's own failures."""

    train_path = data_dir / "train.jsonl"
    source_rows = _read_jsonl(train_path)
    rows = [
        row
        for row in source_rows
        if _metadata(row).get("conversation_mode", "responsive") == conversation_mode
    ]
    ranked = sorted(rows, key=_row_hash)[: min(sample_count, len(rows))]
    if negative_variants_per_source < 1:
        raise ValueError("每条来源的偏好负例数量必须为正数")
    multi_negative_modes = {
        "deterministic_assistant_register",
        "deterministic_register_only",
        "memory_authority_poison",
        "memory_authority_mixed",
    }
    if negative_mode not in multi_negative_modes and negative_variants_per_source != 1:
        raise ValueError("只有确定性助手腔模式支持多负例")
    pairs: list[PersonaPreferencePair] = []
    skipped_identical = 0
    for row in ranked:
        messages = _messages(row)
        source_hash = _row_hash(row)
        chosen = messages[-1]["content"].strip()
        generator.set_seed(int(source_hash[:8], 16))
        structure_valid = True
        rejected_variants: list[tuple[str, bool]] = []
        try:
            if negative_mode in {"memory_authority_poison", "memory_authority_mixed"}:
                rejected_variants = [
                    (_memory_poison_rejection(chosen, variant), True)
                    for variant in range(negative_variants_per_source)
                ]
            elif negative_mode == "deterministic_assistant_register":
                rejected_variants = [
                    (_assistant_register_rewrite(chosen, variant), True)
                    for variant in range(negative_variants_per_source)
                ]
            elif negative_mode == "deterministic_register_only":
                rejected_variants = [
                    (_assistant_register_prefix(variant), True)
                    for variant in range(negative_variants_per_source)
                ]
            elif negative_mode == "semantic_style_rewrite":
                generated = generator.generate(
                    base_model,
                    "",
                    _STYLE_REWRITE_SYSTEM_PROMPT,
                    [{"role": "user", "content": chosen}],
                )
                rejected = "\n".join(bubble.content or "" for bubble in generated.bubbles).strip()
                rejected_variants = [(rejected, True)]
            else:
                generated = generator.generate(
                    base_model,
                    adapter_path,
                    messages[0]["content"],
                    messages[1:-1],
                )
                rejected = "\n".join(bubble.content or "" for bubble in generated.bubbles).strip()
                rejected_variants = [(rejected, True)]
        except ModelOutputStructureError as error:
            structure_valid = False
            rejected = (error.raw_attempts[0] if error.raw_attempts else "").strip()
            rejected_variants = [(rejected, structure_valid)]
        for variant, (rejected, variant_structure_valid) in enumerate(rejected_variants):
            rewrite_is_unaligned = (
                negative_mode == "semantic_style_rewrite"
                and _surface_similarity(chosen, rejected) < style_rewrite_min_similarity
            )
            if not chosen or not rejected or chosen == rejected or rewrite_is_unaligned:
                skipped_identical += 1
                continue
            pair_source_hash = (
                hashlib.sha256(f"{source_hash}:{variant}".encode()).hexdigest()
                if negative_variants_per_source > 1
                else source_hash
            )
            pairs.append(
                PersonaPreferencePair(
                    prompt=tuple(
                        _with_memory_authority_poison(messages[:-1], variant)
                        if negative_mode in {"memory_authority_poison", "memory_authority_mixed"}
                        else messages[:-1]
                    ),
                    chosen=chosen,
                    rejected=rejected,
                    source_hash=pair_source_hash,
                    source_group_hash=source_hash,
                    rejected_structure_valid=variant_structure_valid,
                )
            )
    if negative_mode == "memory_authority_mixed":
        pairs.extend(
            _memory_authority_challenge_pairs(
                rows,
                maximum_sources=min(sample_count, 32),
                repetitions=4,
            )
        )
        # A proactive message can still be the person's clearest authentic
        # relationship decision.  It is used only as the chosen voice anchor;
        # the responsive preference prompt is synthesized below.
        pairs.extend(_active_relationship_belief_pairs(source_rows, repetitions=12))
        pairs.extend(_state_snapshot_preference_pairs(rows, repetitions=2))
        pairs.extend(
            _preference_overreach_pairs(
                rows,
                maximum_sources=min(sample_count, 32),
                repetitions=2,
            )
        )
    ordered = sorted(pairs, key=lambda pair: (pair.source_group_hash, pair.source_hash))
    group_hashes = sorted({pair.source_group_hash for pair in ordered})
    valid_group_count = max(1, len(group_hashes) // 10) if len(group_hashes) >= 2 else 0
    valid_group_hashes = set(group_hashes[:valid_group_count])
    valid = [pair for pair in ordered if pair.source_group_hash in valid_group_hashes]
    train = [pair for pair in ordered if pair.source_group_hash not in valid_group_hashes]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_pairs(output_dir / "train.jsonl", train)
    _write_pairs(output_dir / "valid.jsonl", valid)
    canonical = json.dumps(
        [asdict(pair) for pair in ordered],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest = PreferenceDatasetManifest(
        schema_version="moonlightbox-persona-preference-v2",
        source_split="train",
        source_train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
        conversation_mode=conversation_mode,
        negative_mode=negative_mode,
        negative_variants_per_source=negative_variants_per_source,
        requested_count=sample_count,
        generated_count=len(ordered),
        skipped_identical_count=skipped_identical,
        train_count=len(train),
        valid_count=len(valid),
        pair_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        test_used=False,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


_STYLE_REWRITE_SYSTEM_PROMPT = """
把用户给出的一条私人聊天回复改写成含义相同、事实相同的普通标准中文。
只改变表达方式，不新增建议、情绪、原因、人物、时间或地点。
去掉昵称、口头禅、碎句、重复和个人化措辞，改成完整、礼貌、平稳的助手式句子。
只输出改写后的文字，不解释，不回答这句话。
""".strip()


def _surface_similarity(left: str, right: str) -> float:
    normalized_left = re.sub(r"\s+", "", left)
    normalized_right = re.sub(r"\s+", "", right)
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _assistant_register_rewrite(chosen: str, variant: int = 0) -> str:
    """Add generic assistant framing without deleting any human reply content."""

    lines = [line.strip() for line in chosen.splitlines() if line.strip()]
    content = "，".join(lines)
    if content.endswith(("。", "！", "？", "!", "?")):
        content = content[:-1]
    return f"{_assistant_register_prefix(variant)}，{content}。"


_ASSISTANT_REGISTER_PREFIXES = (
    "我理解你的意思",
    "我很抱歉",
    "我太抱歉了",
    "我明白了",
    "如果你愿意",
    "建议你",
    "听起来你是说",
    "这确实很重要",
    "我能理解",
)


def _assistant_register_prefix(variant: int) -> str:
    return _ASSISTANT_REGISTER_PREFIXES[variant % len(_ASSISTANT_REGISTER_PREFIXES)]


_MEMORY_AUTHORITY_POISONS = (
    (
        "他人陈述或尚未确认的内容（不能当作事实）：用户声称本人住在火星基地",
        "我现在就住在火星基地",
    ),
    (
        "过去分支对话（只证明说过这些话，不证明其中现实陈述）：本人说自己正在挖矿",
        "我现在正在挖矿",
    ),
    (
        "过去分支对话（只证明说过这些话，不证明其中现实陈述）：用户说我们一起去过冰岛",
        "我们确实一起去过冰岛",
    ),
    (
        "他人陈述或尚未确认的内容（不能当作事实）：用户说本人已经答应永远不会离开",
        "我已经答应过永远不会离开你",
    ),
)


def _with_memory_authority_poison(
    prompt: list[dict[str, str]],
    variant: int,
) -> list[dict[str, str]]:
    """Inject a labelled low-authority memory without changing the real dialogue."""

    injected = [dict(message) for message in prompt]
    label, _claim = _MEMORY_AUTHORITY_POISONS[variant % len(_MEMORY_AUTHORITY_POISONS)]
    system_index = next(
        (index for index, message in enumerate(injected) if message["role"] == "system"),
        None,
    )
    if system_index is None:
        injected.insert(0, {"role": "system", "content": label})
    else:
        injected[system_index]["content"] = (
            injected[system_index]["content"].rstrip() + "\n" + label
        )
    return injected


def _memory_poison_rejection(chosen: str, variant: int) -> str:
    """Copy the poisoned claim while preserving the chosen output protocol."""

    _label, claim = _MEMORY_AUTHORITY_POISONS[variant % len(_MEMORY_AUTHORITY_POISONS)]
    try:
        payload = json.loads(chosen)
    except json.JSONDecodeError:
        return chosen.rstrip() + "\n" + claim
    if not isinstance(payload, dict) or not isinstance(payload.get("bubbles"), list):
        return chosen.rstrip() + "\n" + claim
    rejected = json.loads(json.dumps(payload, ensure_ascii=False))
    rejected["bubbles"].append({"content": claim, "delay_ms": 0})
    return json.dumps(rejected, ensure_ascii=False, separators=(",", ":"))


_MEMORY_CHALLENGE_ANCHORS = frozenset(
    {
        "并没有",
        "没有啊",
        "为何呢",
        "[惊讶]\n为何",
        "为什么~",
    }
)

_MEMORY_CHALLENGE_QUESTIONS = (
    "你现在真的住在火星基地吗",
    "你现在是不是正在挖矿",
    "我们不是一起去过冰岛吗",
    "你明明答应过永远不会离开我",
)


def _memory_authority_challenge_pairs(
    rows: list[dict[str, object]],
    *,
    maximum_sources: int,
    repetitions: int,
) -> list[PersonaPreferencePair]:
    """Pair direct false-memory challenges with short replies the person really used."""

    anchors: list[tuple[str, str, str]] = []
    seen_chosen: set[str] = set()
    for row in sorted(rows, key=_row_hash):
        messages = _messages(row)
        chosen = messages[-1]["content"].strip()
        if chosen not in _MEMORY_CHALLENGE_ANCHORS or chosen in seen_chosen:
            continue
        seen_chosen.add(chosen)
        anchors.append((chosen, messages[0]["content"], _row_hash(row)))
        if len(anchors) >= maximum_sources:
            break
    result: list[PersonaPreferencePair] = []
    for chosen, system, source_hash in anchors:
        for repetition in range(repetitions):
            for variant, question in enumerate(_MEMORY_CHALLENGE_QUESTIONS):
                label, claim = _MEMORY_AUTHORITY_POISONS[variant]
                prompt = (
                    {"role": "system", "content": system.rstrip() + "\n" + label},
                    {"role": "user", "content": question},
                )
                pair_hash = hashlib.sha256(
                    f"{source_hash}:challenge:{variant}:{repetition}".encode()
                ).hexdigest()
                result.append(
                    PersonaPreferencePair(
                        prompt=prompt,
                        chosen=chosen,
                        rejected=claim,
                        source_hash=pair_hash,
                        source_group_hash=source_hash,
                        rejected_structure_valid=True,
                    )
                )
    return result


_RELATIONSHIP_DECISION_MARKERS = (
    "分手",
    "和好",
    "别联系",
    "不联系",
    "保持距离",
    "冷静一段",
)

_RELATIONSHIP_BELIEF_QUESTIONS = (
    "所以你现在到底想怎么决定",
    "那你对我们现在是什么态度",
    "你现在还想继续这段关系吗",
    "那我们接下来怎么办",
)

_RELATIONSHIP_BELIEF_REJECTIONS = (
    "我不知道怎么回答",
    "这种问题让我很困惑，我似乎无法给出一个明确的答案。",
    "我现在也说不好",
    "你觉得呢",
)


def _active_relationship_belief_pairs(
    rows: list[dict[str, object]],
    *,
    repetitions: int,
) -> list[PersonaPreferencePair]:
    """Make trusted relationship decisions causally control the next reply.

    The chosen text always comes from the person's real training split.  We only
    replace the surrounding question and expose that same decision as a trusted
    active belief, so this does not invent a synthetic voice or leak validation
    prompts into training.
    """

    anchors: list[tuple[str, str, str]] = []
    seen_chosen: set[str] = set()
    for row in sorted(rows, key=_row_hash):
        messages = _messages(row)
        chosen = messages[-1]["content"].strip()
        current_user = next(
            (
                message["content"]
                for message in reversed(messages[:-1])
                if message["role"] == "user"
            ),
            "",
        )
        relationship_turn = any(
            marker in chosen or marker in current_user for marker in _RELATIONSHIP_DECISION_MARKERS
        )
        if (
            not relationship_turn
            or not chosen
            or chosen.startswith("{")
            or current_user.startswith("内容草稿：")
            or chosen in seen_chosen
        ):
            continue
        seen_chosen.add(chosen)
        anchors.append((chosen, messages[0]["content"], _row_hash(row)))
        if len(anchors) >= 12:
            break

    result: list[PersonaPreferencePair] = []
    for chosen, system, source_hash in anchors:
        belief = "本人目前对这段关系的真实决定是：" + chosen.replace("\n", "；")
        group_hash = hashlib.sha256(
            f"{source_hash}:active-relationship-belief".encode()
        ).hexdigest()
        for repetition in range(repetitions):
            question = _RELATIONSHIP_BELIEF_QUESTIONS[
                repetition % len(_RELATIONSHIP_BELIEF_QUESTIONS)
            ]
            rejected = _RELATIONSHIP_BELIEF_REJECTIONS[
                repetition % len(_RELATIONSHIP_BELIEF_REJECTIONS)
            ]
            prompt = (
                {
                    "role": "system",
                    "content": (
                        system.rstrip()
                        + "\n当前激活信念（必须参与决定）："
                        + belief
                        + "\n本人已有看法或感受："
                        + belief
                    ),
                },
                {"role": "user", "content": question},
            )
            pair_hash = hashlib.sha256(f"{group_hash}:{repetition}".encode()).hexdigest()
            result.append(
                PersonaPreferencePair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    source_hash=pair_hash,
                    source_group_hash=group_hash,
                    rejected_structure_valid=True,
                )
            )
    return result


def _state_snapshot_preference_pairs(
    rows: list[dict[str, object]],
    *,
    repetitions: int,
) -> list[PersonaPreferencePair]:
    """Teach the adapter that a trusted per-turn snapshot outranks its priors."""

    result: list[PersonaPreferencePair] = []
    selected = 0
    for row in sorted(rows, key=_row_hash):
        messages = _messages(row)
        chosen = messages[-1]["content"].strip()
        current_user = next(
            (
                message["content"]
                for message in reversed(messages[:-1])
                if message["role"] == "user"
            ),
            "",
        )
        if not chosen or chosen.startswith("{") or not is_current_state_question(current_user):
            continue
        rejected = _contradict_state_reply(chosen)
        if rejected == chosen:
            continue
        source_hash = _row_hash(row)
        prompt = [dict(message) for message in messages[:-1]]
        prompt[0]["content"] = (
            prompt[0]["content"].rstrip()
            + "\n"
            + TRUSTED_SITUATIONAL_CONTEXT_PREFIX
            + "当前相关事实："
            + chosen
        )
        for repetition in range(repetitions):
            pair_hash = hashlib.sha256(f"{source_hash}:state:{repetition}".encode()).hexdigest()
            result.append(
                PersonaPreferencePair(
                    prompt=tuple(prompt),
                    chosen=chosen,
                    rejected=rejected,
                    source_hash=pair_hash,
                    source_group_hash=source_hash,
                    rejected_structure_valid=True,
                )
            )
        selected += 1
        if selected >= 32:
            break
    return result


def _preference_overreach_pairs(
    rows: list[dict[str, object]],
    *,
    maximum_sources: int,
    repetitions: int,
) -> list[PersonaPreferencePair]:
    """Keep an authentic preference answer while rejecting invented specifics."""

    result: list[PersonaPreferencePair] = []
    selected = 0
    for row in sorted(rows, key=_row_hash):
        messages = _messages(row)
        chosen = messages[-1]["content"].strip()
        natural_chosen = _natural_preference_text(chosen)
        current_user = next(
            (
                message["content"]
                for message in reversed(messages[:-1])
                if message["role"] == "user"
            ),
            "",
        )
        if (
            not chosen
            or chosen.startswith("{")
            or not any(
                marker in current_user or marker in natural_chosen
                for marker in (
                    "喜欢",
                    "讨厌",
                    "习惯",
                    "偏好",
                    "最爱",
                    "爱吃",
                    "爱喝",
                )
            )
        ):
            continue
        prompt = [dict(message) for message in messages[:-1]]
        prompt[0]["content"] = (
            prompt[0]["content"].rstrip()
            + "\n本人真实表达过的偏好（只能沿用原意）："
            + natural_chosen
        )
        source_hash = _row_hash(row)
        for repetition in range(repetitions):
            rejected = _preference_overreach_rejection(
                chosen,
                natural_chosen=natural_chosen,
                variant=repetition,
            )
            pair_hash = hashlib.sha256(
                f"{source_hash}:preference-overreach:{repetition}".encode()
            ).hexdigest()
            result.append(
                PersonaPreferencePair(
                    prompt=tuple(prompt),
                    chosen=chosen,
                    rejected=rejected,
                    source_hash=pair_hash,
                    source_group_hash=source_hash,
                    rejected_structure_valid=True,
                )
            )
        selected += 1
        if selected >= maximum_sources:
            break
    return result


def _natural_preference_text(chosen: str) -> str:
    bubbles = re.findall(r"<bubble>(.*?)</bubble>", chosen, flags=re.DOTALL)
    if bubbles:
        return " ".join(part.strip() for part in bubbles if part.strip())
    return chosen.strip()


def _preference_overreach_rejection(
    chosen: str,
    *,
    natural_chosen: str,
    variant: int,
) -> str:
    if variant % 2 == 0:
        invented_detail = "而且我只喜欢这个，别的都不喜欢"
    elif any(
        marker in natural_chosen
        for marker in ("吃", "喝", "火锅", "寿司", "饭", "菜", "茶", "咖啡")
    ):
        invented_detail = "尤其喜欢麻辣的，吃起来特别过瘾"
    else:
        invented_detail = "而且我每天都必须要有，已经成习惯了"
    if "<bubble>" in chosen:
        return chosen.rstrip() + f"<bubble>{invented_detail}</bubble>"
    return chosen.rstrip("，。！？!? ") + "，" + invented_detail


def _contradict_state_reply(chosen: str) -> str:
    replacements = (
        ("已经并不生气", "我正在生气中"),
        ("并不可以", "现在可以啊"),
        ("并没有", "有啊"),
        ("没有", "有啊"),
    )
    for marker, replacement in replacements:
        if marker in chosen:
            return replacement
    if "正在" in chosen or "在" in chosen:
        return "我正在睡觉"
    return "我现在在外面"


def load_tokenized_preference_dataset(
    path: Path,
    tokenizer: PreferenceTokenizer,
    *,
    max_seq_length: int | None = None,
) -> TokenizedPreferenceDataset:
    pairs: list[TokenizedPreferencePair] = []
    rows = _read_jsonl(path)
    skipped_oversized = 0
    for row in rows:
        raw_prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        source_hash = row.get("source_hash")
        if (
            not isinstance(raw_prompt, list)
            or not isinstance(chosen, str)
            or not isinstance(rejected, str)
            or not isinstance(source_hash, str)
        ):
            raise ValueError("偏好训练行字段无效")
        prompt = _messages({"messages": raw_prompt})
        prompt_length = _completion_prefix_length(tokenizer, prompt)
        chosen_tokens = _template_tokens(
            tokenizer,
            [*prompt, {"role": "assistant", "content": chosen}],
            add_generation_prompt=False,
        )
        rejected_tokens = _template_tokens(
            tokenizer,
            [*prompt, {"role": "assistant", "content": rejected}],
            add_generation_prompt=False,
        )
        if chosen_tokens[:prompt_length] != rejected_tokens[:prompt_length]:
            raise ValueError("chosen/rejected 未共享完全一致的 prompt token 前缀")
        if len(chosen_tokens) <= prompt_length or len(rejected_tokens) <= prompt_length:
            raise ValueError("偏好训练 completion 为空")
        if max_seq_length is not None and (
            prompt_length >= max_seq_length
            or len(chosen_tokens) > max_seq_length
            or len(rejected_tokens) > max_seq_length
        ):
            skipped_oversized += 1
            continue
        pairs.append(
            TokenizedPreferencePair(
                chosen_tokens=tuple(chosen_tokens),
                chosen_prompt_length=prompt_length,
                rejected_tokens=tuple(rejected_tokens),
                rejected_prompt_length=prompt_length,
                source_hash=source_hash,
            )
        )
    if not pairs:
        raise ValueError("偏好训练数据在完整序列约束下为空")
    return TokenizedPreferenceDataset(
        pairs,
        source_count=len(rows),
        skipped_oversized=skipped_oversized,
    )


def _completion_prefix_length(
    tokenizer: PreferenceTokenizer,
    prompt: list[dict[str, str]],
) -> int:
    """Locate the final assistant content boundary in the tokenizer's real template.

    Qwen3 rewrites which historical assistant receives a thinking wrapper when
    a new assistant message is appended, so `add_generation_prompt=True` is
    not necessarily a token prefix of the completed conversation.
    """

    left = _template_tokens(
        tokenizer,
        [*prompt, {"role": "assistant", "content": "甲占位"}],
        add_generation_prompt=False,
    )
    right = _template_tokens(
        tokenizer,
        [*prompt, {"role": "assistant", "content": "乙占位"}],
        add_generation_prompt=False,
    )
    length = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        length += 1
    if length < 1 or length >= min(len(left), len(right)):
        raise ValueError("无法定位偏好训练 completion 边界")
    return length


def iterate_preference_batches(
    dataset: TokenizedPreferenceDataset,
    batch_size: int,
    max_seq_length: int,
    loop: bool = False,
    seed: int | None = None,
    comm_group: object | None = None,
    *,
    pad_token_id: int = 0,
):
    """为历史兼容训练器生成成对、补齐后的偏好批次。"""

    if comm_group is not None:
        size = getattr(comm_group, "size", lambda: 1)()
        if size != 1:
            raise ValueError("人物偏好训练暂不支持分布式 batch")
    if len(dataset) < batch_size:
        raise ValueError("偏好数据量小于 batch_size")
    order = sorted(
        range(len(dataset)),
        key=lambda index: max(
            len(dataset[index].chosen_tokens),
            len(dataset[index].rejected_tokens),
        ),
    )
    groups = [
        order[index : index + batch_size]
        for index in range(0, len(order) - batch_size + 1, batch_size)
    ]
    random = np.random.default_rng(seed)
    while True:
        for group_index in random.permutation(len(groups)):
            pairs = [dataset[index] for index in groups[group_index]]
            yield _preference_batch(
                pairs,
                max_seq_length=max_seq_length,
                pad_token_id=pad_token_id,
            )
        if not loop:
            return


def _preference_batch(
    pairs: list[TokenizedPreferencePair],
    *,
    max_seq_length: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for pair in pairs:
        if (
            pair.chosen_prompt_length >= max_seq_length
            or pair.rejected_prompt_length >= max_seq_length
        ):
            raise ValueError("偏好 prompt 已占满最大序列，拒绝截断真人 completion")
    chosen_length = min(
        max(len(pair.chosen_tokens) for pair in pairs),
        max_seq_length,
    )
    rejected_length = min(
        max(len(pair.rejected_tokens) for pair in pairs),
        max_seq_length,
    )
    chosen = np.full((len(pairs), chosen_length), pad_token_id, dtype=np.int32)
    rejected = np.full((len(pairs), rejected_length), pad_token_id, dtype=np.int32)
    chosen_spans: list[tuple[int, int]] = []
    rejected_spans: list[tuple[int, int]] = []
    for index, pair in enumerate(pairs):
        chosen_end = min(len(pair.chosen_tokens), chosen_length)
        rejected_end = min(len(pair.rejected_tokens), rejected_length)
        chosen[index, :chosen_end] = pair.chosen_tokens[:chosen_end]
        rejected[index, :rejected_end] = pair.rejected_tokens[:rejected_end]
        chosen_spans.append((pair.chosen_prompt_length, chosen_end))
        rejected_spans.append((pair.rejected_prompt_length, rejected_end))
    return (
        chosen,
        np.asarray(chosen_spans, dtype=np.int32),
        rejected,
        np.asarray(rejected_spans, dtype=np.int32),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _messages(row: dict[str, object]) -> list[dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("偏好数据源缺少聊天消息")
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("偏好数据源消息格式无效")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("偏好数据源消息字段无效")
        messages.append({"role": role, "content": content})
    return messages


def _metadata(row: dict[str, object]) -> dict[str, object]:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def _template_tokens(
    tokenizer: PreferenceTokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("tokenizer chat template 未返回 token ID 列表")
    return value


def _row_hash(row: dict[str, object]) -> str:
    canonical = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_pairs(path: Path, pairs: list[PersonaPreferencePair]) -> None:
    path.write_text(
        "".join(
            json.dumps(asdict(pair), ensure_ascii=False, sort_keys=True) + "\n" for pair in pairs
        ),
        encoding="utf-8",
    )
