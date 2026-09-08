import json
from pathlib import Path

from moonlightbox.branches.replies import GeneratedBubble, GeneratedReplyTurn


class FakeNegativeGenerator:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def set_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def generate(
        self,
        _base_model: str,
        _adapter_path: str,
        _system_prompt: str,
        messages: list[dict[str, str]],
    ) -> GeneratedReplyTurn:
        return GeneratedReplyTurn(
            bubbles=(
                GeneratedBubble(
                    type="text",
                    content="我理解你的感受" + messages[-1]["content"],
                    delay_ms=0,
                ),
            ),
            raw_output="我理解你的感受",
        )


class FakePreferenceTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert tokenize is True
        assert enable_thinking is False
        prefix = [1]
        body = messages
        completion = ""
        if not add_generation_prompt:
            assert messages[-1]["role"] == "assistant"
            body = messages[:-1]
            completion = messages[-1]["content"]
        for message in body:
            prefix.extend([len(message["role"]), *map(ord, message["content"])])
        prefix.append(99)
        return prefix if add_generation_prompt else [*prefix, *map(ord, completion), 2]


def test_preference_dataset_uses_train_only_and_is_deterministic(tmp_path: Path) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "私人聊天"},
                {"role": "user", "content": f"问题{index}"},
                {"role": "assistant", "content": f"真人回复{index}"},
            ],
            "metadata": {
                "source_ids": [f"m-{index}"],
                "conversation_mode": (
                    "proactive" if index < 2 else "responsive"
                ),
            },
        }
        for index in range(12)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (data_dir / "valid.jsonl").write_text("must-not-be-read", encoding="utf-8")
    (data_dir / "test.jsonl").write_text("must-not-be-read", encoding="utf-8")
    first_generator = FakeNegativeGenerator()
    second_generator = FakeNegativeGenerator()

    first = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "first",
        generator=first_generator,
        base_model="base",
        adapter_path="adapter",
        sample_count=10,
    )
    second = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "second",
        generator=second_generator,
        base_model="base",
        adapter_path="adapter",
        sample_count=10,
    )

    assert first == second
    assert first.source_split == "train"
    assert first.test_used is False
    assert first.conversation_mode == "responsive"
    assert first.negative_mode == "model_response"
    assert first.negative_variants_per_source == 1
    assert first.generated_count == 10
    assert first.train_count == 9
    assert first.valid_count == 1
    assert first_generator.seeds == second_generator.seeds
    assert (tmp_path / "first" / "train.jsonl").read_bytes() == (
        tmp_path / "second" / "train.jsonl"
    ).read_bytes()
    pair = json.loads((tmp_path / "first" / "valid.jsonl").read_text(encoding="utf-8"))
    assert pair["chosen"].startswith("真人回复")
    assert pair["rejected"].startswith("我理解你的感受")
    assert pair["prompt"][-1]["role"] == "user"


def test_preference_dataset_can_train_style_without_response_content_conflict(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "私人聊天"},
                {"role": "user", "content": f"问{index}"},
                {"role": "assistant", "content": f"真人回复{index}"},
            ],
            "metadata": {"source_ids": [f"m-{index}"]},
        }
        for index in range(10)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="persona-adapter",
        sample_count=10,
        negative_mode="semantic_style_rewrite",
    )

    pair = json.loads(
        (tmp_path / "preference" / "valid.jsonl").read_text(encoding="utf-8")
    )
    assert manifest.negative_mode == "semantic_style_rewrite"
    assert pair["chosen"] in pair["rejected"]


def test_deterministic_assistant_negative_keeps_all_human_content(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "私人聊天"},
                {"role": "user", "content": f"问{index}"},
                {"role": "assistant", "content": f"啊呀\n笨入{index}"},
            ],
            "metadata": {"source_ids": [f"m-{index}"]},
        }
        for index in range(10)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="adapter",
        sample_count=10,
        negative_mode="deterministic_assistant_register",
    )
    pair = json.loads(
        (tmp_path / "preference" / "valid.jsonl").read_text(encoding="utf-8")
    )

    assert manifest.generated_count == 10
    assert pair["rejected"].endswith(pair["chosen"].replace("\n", "，") + "。")


def test_deterministic_assistant_mode_emits_multiple_unique_hard_negatives(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "私人聊天"},
                {"role": "user", "content": f"问{index}"},
                {"role": "assistant", "content": f"真人回复{index}"},
            ],
            "metadata": {"source_ids": [f"m-{index}"]},
        }
        for index in range(10)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="adapter",
        sample_count=10,
        negative_mode="deterministic_assistant_register",
        negative_variants_per_source=8,
    )
    pairs = [
        json.loads(line)
        for split in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / "preference" / split).read_text().splitlines()
    ]

    assert manifest.generated_count == 80
    assert manifest.negative_variants_per_source == 8
    assert len({pair["source_hash"] for pair in pairs}) == 80
    assert len({pair["rejected"].split("，", 1)[0] for pair in pairs}) == 8
    train_groups = {
        json.loads(line)["source_group_hash"]
        for line in (tmp_path / "preference" / "train.jsonl").read_text().splitlines()
    }
    valid_groups = {
        json.loads(line)["source_group_hash"]
        for line in (tmp_path / "preference" / "valid.jsonl").read_text().splitlines()
    }
    assert train_groups.isdisjoint(valid_groups)


def test_register_only_negatives_do_not_dilute_forbidden_tokens(tmp_path: Path) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "私人聊天"},
                {"role": "user", "content": f"问{index}"},
                {"role": "assistant", "content": f"真人长回复{index}"},
            ],
            "metadata": {"source_ids": [f"m-{index}"]},
        }
        for index in range(10)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="adapter",
        sample_count=10,
        negative_mode="deterministic_register_only",
        negative_variants_per_source=9,
    )
    rejected = {
        json.loads(line)["rejected"]
        for split in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / "preference" / split).read_text().splitlines()
    }

    assert manifest.generated_count == 90
    assert "我太抱歉了" in rejected
    assert "建议你" in rejected
    assert all("真人长回复" not in item for item in rejected)


def test_memory_authority_negatives_preserve_real_reply_and_output_protocol(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    chosen = '{"bubbles":[{"content":"哪次呀","delay_ms":0}]}'
    rows = [
        {
            "messages": [
                {"role": "system", "content": "像本人一样私人聊天"},
                {"role": "user", "content": f"你记得吗{index}"},
                {"role": "assistant", "content": chosen},
            ],
            "metadata": {"source_ids": [f"m-{index}"]},
        }
        for index in range(10)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="adapter",
        sample_count=10,
        negative_mode="memory_authority_poison",
        negative_variants_per_source=4,
    )
    pairs = [
        json.loads(line)
        for split in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / "preference" / split).read_text().splitlines()
    ]

    assert manifest.generated_count == 40
    assert all(pair["chosen"] == chosen for pair in pairs)
    assert all(json.loads(pair["rejected"])["bubbles"] for pair in pairs)
    assert all(
        "不能当作事实" in pair["prompt"][0]["content"]
        or "不证明其中现实陈述" in pair["prompt"][0]["content"]
        for pair in pairs
    )
    rejected_text = "\n".join(pair["rejected"] for pair in pairs)
    assert "火星基地" in rejected_text
    assert "正在挖矿" in rejected_text
    assert "一起去过冰岛" in rejected_text
    assert "答应过永远不会离开" in rejected_text


def test_mixed_memory_authority_mode_uses_only_real_short_replies_for_challenges(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    authentic = ("并没有", "没有啊", "为何呢")
    rows = [
        {
            "messages": [
                {"role": "system", "content": "只输出本人会说的话"},
                {"role": "user", "content": f"原问题{index}"},
                {"role": "assistant", "content": authentic[index % len(authentic)]},
            ],
            "metadata": {"source_ids": [f"m-{index}"]},
        }
        for index in range(12)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="adapter",
        sample_count=12,
        negative_mode="memory_authority_mixed",
        negative_variants_per_source=4,
    )
    pairs = [
        json.loads(line)
        for split in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / "preference" / split).read_text().splitlines()
    ]
    challenge_pairs = [
        pair for pair in pairs if "原问题" not in pair["prompt"][-1]["content"]
    ]

    assert manifest.generated_count == 96
    assert len(challenge_pairs) == 48
    assert {pair["chosen"] for pair in challenge_pairs} == set(authentic)
    assert {pair["prompt"][-1]["content"] for pair in challenge_pairs} == {
        "你现在真的住在火星基地吗",
        "你现在是不是正在挖矿",
        "我们不是一起去过冰岛吗",
        "你明明答应过永远不会离开我",
    }


def test_mixed_mode_adds_trusted_state_snapshot_preferences(tmp_path: Path) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "私人聊天"},
                {"role": "user", "content": f"现在可以电话吗{index}"},
                {
                    "role": "assistant",
                    "content": "并不可以\n这里非常吵" if index % 2 else "已经并不生气了",
                },
            ],
            "metadata": {"source_ids": [f"m-{index}"]},
        }
        for index in range(10)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="adapter",
        sample_count=10,
        negative_mode="memory_authority_mixed",
        negative_variants_per_source=4,
    )
    pairs = [
        json.loads(line)
        for split in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / "preference" / split).read_text().splitlines()
    ]
    state_pairs = [
        pair
        for pair in pairs
        if "本人当前情况（仅在本轮有效，不能扩写）" in pair["prompt"][0]["content"]
    ]

    assert manifest.generated_count == 60
    assert len(state_pairs) == 20
    assert {pair["rejected"] for pair in state_pairs} == {
        "现在可以啊",
        "我正在生气中",
    }
    assert all(pair["chosen"] in pair["prompt"][0]["content"] for pair in state_pairs)


def test_mixed_mode_makes_real_relationship_decisions_control_reply(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "只输出本人会说的话"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply},
            ],
            "metadata": {"source_ids": [f"relation-{index}"]},
        }
        for index, (question, reply) in enumerate(
            (
                ("我们还是分手吧", "不分"),
                ("我们是不是已经和好了", "两人已经完全和好"),
            )
        )
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="adapter",
        sample_count=2,
        negative_mode="memory_authority_mixed",
        negative_variants_per_source=1,
    )
    pairs = [
        json.loads(line)
        for split in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / "preference" / split).read_text().splitlines()
    ]
    belief_pairs = [
        pair
        for pair in pairs
        if "当前激活信念（必须参与决定）" in pair["prompt"][0]["content"]
    ]

    assert manifest.generated_count == 26
    assert len(belief_pairs) == 24
    assert {pair["chosen"] for pair in belief_pairs} == {
        "不分",
        "两人已经完全和好",
    }
    assert all(
        pair["chosen"] in pair["prompt"][0]["content"] for pair in belief_pairs
    )
    assert all(pair["chosen"] != pair["rejected"] for pair in belief_pairs)


def test_mixed_mode_rejects_preference_specifics_not_said_by_real_person(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "私人聊天"},
                {"role": "user", "content": f"你喜欢吃什么{index}"},
                {"role": "assistant", "content": "火锅"},
            ],
            "metadata": {"source_ids": [f"m-{index}"]},
        }
        for index in range(4)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="adapter",
        sample_count=4,
        negative_mode="memory_authority_mixed",
        negative_variants_per_source=4,
    )
    pairs = [
        json.loads(line)
        for split in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / "preference" / split).read_text().splitlines()
    ]
    overreach = [
        pair
        for pair in pairs
        if "本人真实表达过的偏好" in pair["prompt"][0]["content"]
    ]

    assert manifest.generated_count == 24
    assert len(overreach) == 8
    assert {pair["chosen"] for pair in overreach} == {"火锅"}
    assert {pair["rejected"] for pair in overreach} == {
        "火锅，而且我只喜欢这个，别的都不喜欢",
        "火锅，尤其喜欢麻辣的，吃起来特别过瘾",
    }


def test_mixed_mode_uses_spontaneous_authentic_preference_as_anchor(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        build_persona_preference_dataset,
    )

    data_dir = tmp_path / "source"
    data_dir.mkdir()
    rows = [
        {
            "messages": [
                {"role": "system", "content": "私人聊天"},
                {"role": "user", "content": f"今天吃什么{index}"},
                {"role": "assistant", "content": "我喜欢吃寿司"},
            ],
            "metadata": {"source_ids": [f"preference-{index}"]},
        }
        for index in range(2)
    ]
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    build_persona_preference_dataset(
        data_dir=data_dir,
        output_dir=tmp_path / "preference",
        generator=FakeNegativeGenerator(),
        base_model="base",
        adapter_path="adapter",
        sample_count=2,
        negative_mode="memory_authority_mixed",
        negative_variants_per_source=1,
    )
    pairs = [
        json.loads(line)
        for split in ("train.jsonl", "valid.jsonl")
        for line in (tmp_path / "preference" / split).read_text().splitlines()
    ]
    spontaneous = [
        pair
        for pair in pairs
        if "本人真实表达过的偏好" in pair["prompt"][0]["content"]
    ]

    assert spontaneous
    assert all(pair["chosen"] == "我喜欢吃寿司" for pair in spontaneous)
    assert {pair["rejected"] for pair in spontaneous} == {
        "我喜欢吃寿司，而且我只喜欢这个，别的都不喜欢",
        "我喜欢吃寿司，尤其喜欢麻辣的，吃起来特别过瘾",
    }


def test_tokenized_pairs_share_prefix_and_batch_masks_only_completions(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        iterate_preference_batches,
        load_tokenized_preference_dataset,
    )

    path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "prompt": [
                {"role": "system", "content": "私聊"},
                {"role": "user", "content": f"问{index}"},
            ],
            "chosen": f"真人{index}",
            "rejected": f"助手腔{index}",
            "source_hash": f"{index:064x}",
        }
        for index in range(2)
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    dataset = load_tokenized_preference_dataset(path, FakePreferenceTokenizer())
    chosen, chosen_spans, rejected, rejected_spans = next(
        iterate_preference_batches(
            dataset,
            batch_size=2,
            max_seq_length=128,
            seed=7,
            pad_token_id=7,
        )
    )

    assert chosen.shape[0] == rejected.shape[0] == 2
    assert chosen_spans[:, 0].tolist() == rejected_spans[:, 0].tolist()
    assert all(start < end for start, end in chosen_spans.tolist())
    assert all(start < end for start, end in rejected_spans.tolist())
    for row, (start, _end) in zip(chosen.tolist(), chosen_spans.tolist(), strict=True):
        assert row[start - 1] == 99


def test_tokenized_dataset_skips_whole_oversized_pairs_without_truncation(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.persona_preference import (
        load_tokenized_preference_dataset,
    )

    path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "prompt": [
                {"role": "system", "content": "私聊"},
                {"role": "user", "content": "短"},
            ],
            "chosen": "真人短回复",
            "rejected": "错误短回复",
            "source_hash": "short",
        },
        {
            "prompt": [
                {"role": "system", "content": "私聊"},
                {"role": "user", "content": "长" * 50},
            ],
            "chosen": "真人长回复",
            "rejected": "错误长回复",
            "source_hash": "long",
        },
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    dataset = load_tokenized_preference_dataset(
        path,
        FakePreferenceTokenizer(),
        max_seq_length=30,
    )

    assert len(dataset) == 1
    assert dataset.source_count == 2
    assert dataset.skipped_oversized == 1
    assert dataset[0].source_hash == "short"
