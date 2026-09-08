from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from moonlightbox.branches.identity import EvidenceBackedIdentityKernelBuilder
from moonlightbox.branches.replies import (
    GeneratedBubble,
    GeneratedReplyTurn,
    drop_invalid_sticker_bubbles,
    validate_reply_sticker_ids,
)
from moonlightbox.embeddings import LocalChineseEmbedder
from moonlightbox.evaluation.blind_service import HumanBlindStudyService
from moonlightbox.imports.models import Participant
from moonlightbox.jobs.models import Job
from moonlightbox.training.bubble_protocol import (
    compact_protocol_instruction,
    parse_bubble_protocol,
)
from moonlightbox.training.dataset_builder import (
    ChatTurn,
    ConfirmedEventContext,
    DatasetManifest,
    TrainingExample,
)
from moonlightbox.training.model_acceptance import (
    AcceptanceMetric,
    AcceptanceReport,
    ModelAcceptanceRunner,
    ModelGateSnapshot,
    ModelOutputStructureError,
    PathReplyGenerator,
    build_task4_style_metrics,
    character_ngram_cosine,
    evaluate_activation_gates,
)
from moonlightbox.training.models import ModelVersion, TimelineConfirmation
from moonlightbox.training.registry import ModelRegistry
from moonlightbox.training.sticker_policy import (
    RecentStickerBubble,
    StickerContext,
    StickerPolicy,
    build_sticker_evaluation_cases_from_database,
    build_sticker_policy_from_database,
    evaluate_sticker_policy,
    rank_stickers,
    tune_sticker_policy_on_valid,
)
from moonlightbox.training.style_acceptance import (
    HeldOutCase,
    MemorizationIndex,
    MemorizationSource,
    SpeakerSample,
    SpeakerStyleIdentifier,
    compare_style_distributions,
)
from moonlightbox.training.style_features import StyleBubble, StyleTurn

ReplyProtocol = Literal["legacy_json", "compact", "persona_text"]


@dataclass(frozen=True)
class AcceptanceCorpus:
    held_out_cases: tuple[HeldOutCase, ...]
    human_style_turns: tuple[StyleTurn, ...]
    speaker_fit_samples: tuple[SpeakerSample, ...]
    memorization_sources: tuple[MemorizationSource, ...]
    known_sticker_ids: tuple[str, ...]
    actual_sticker_ids: tuple[str | None, ...]
    allowed_sticker_ids_by_case: dict[str, tuple[str, ...]]
    retrieval_rankings: tuple[tuple[str, ...], ...]
    retrieval_actual_ids: tuple[str, ...]
    sticker_coverage: dict[str, int | float]
    sample_hash: str
    split_hashes: dict[str, str]


@dataclass(frozen=True)
class HeldOutModelReport:
    model_id: str
    reply_protocol: ReplyProtocol
    sample_count: int
    sample_hash: str
    parse_failures: int
    raw_format_compliance: float
    normalization_rate: float
    style: dict[str, object]
    speaker: dict[str, object]
    sticker: dict[str, object]
    memorization: dict[str, object]
    sample_audit: tuple[dict[str, object], ...]
    passed: bool
    failure_reasons: tuple[str, ...]


def resolve_reply_protocol(training_config: dict[str, object]) -> ReplyProtocol:
    version = str(training_config.get("reply_protocol_version", ""))
    if version.startswith("persona-text"):
        return "persona_text"
    return "compact" if version.startswith("high-fidelity-compact-bubble") else "legacy_json"


def load_acceptance_corpus(
    data_dir: Path,
    *,
    sample_count: int = 20,
    minimum_sample_count: int | None = None,
    sticker_policy: StickerPolicy | None = None,
) -> AcceptanceCorpus:
    """加载严格隔离的训练/验证拟合集与确定性 test 留出样本。"""

    minimum = sample_count if minimum_sample_count is None else minimum_sample_count
    if sample_count < 1 or minimum < 1:
        raise ValueError("验收样本数必须为正数")
    rows = {split: _read_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "valid", "test")}
    if not rows["train"] or not rows["test"]:
        raise ValueError("验收数据的 train 或 test 样本不足")
    if len(rows["test"]) < minimum or len(rows["test"]) < sample_count:
        raise ValueError(f"test 验收样本不足，至少需要 {max(minimum, sample_count)} 条")
    fit_boundary = max(_row_target_at(row) for split in ("train", "valid") for row in rows[split])
    if sticker_policy is not None:
        policy_boundary_text = sticker_policy.boundary.get("effective_cutoff")
        if (
            policy_boundary_text
            and _aware_datetime(datetime.fromisoformat(policy_boundary_text)) > fit_boundary
        ):
            raise ValueError("sticker policy 越过 train/valid 历史边界")

    fit_samples: list[SpeakerSample] = []
    memorization_sources: list[MemorizationSource] = []
    known_stickers: set[str] = set()
    for split in ("train", "valid"):
        for row in rows[split]:
            messages = _messages(row)
            target = messages[-1]
            if target["role"] != "assistant":
                raise ValueError(f"{split} 样本缺少 assistant 目标")
            reply = _parse_dataset_target(target["content"])
            target_text = _reply_text(reply)
            fit_samples.append(SpeakerSample(split, "target", target_text))
            other_text = "\n".join(
                message["content"] for message in messages[:-1] if message["role"] == "user"
            )
            if other_text:
                fit_samples.append(SpeakerSample(split, "other", other_text))
            if split == "train":
                memorization_sources.append(
                    MemorizationSource("train", _canonical_hash(row), target_text)
                )
            known_stickers.update(_sticker_ids(reply))

    policy_asset_ids = (
        {asset.asset_id for asset in sticker_policy.assets}
        if sticker_policy is not None
        else known_stickers
    )
    historical_universe = known_stickers & policy_asset_ids
    positive_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    test_details: list[tuple[dict[str, object], HeldOutCase, tuple[str, ...]]] = []
    for row in rows["test"]:
        messages = _messages(row)
        target = messages[-1]
        if target["role"] != "assistant":
            raise ValueError("test 样本缺少 assistant 真人目标")
        source_hash = _canonical_hash(row)
        reply = _parse_dataset_target(target["content"])
        sticker_ids = _sticker_ids(reply)
        case = HeldOutCase(
            case_id=source_hash[:16],
            split="test",
            context=tuple(messages[:-1]),
            human_target=target["content"],
            source_hash=source_hash,
        )
        test_details.append((row, case, sticker_ids))
        (positive_rows if sticker_ids else negative_rows).append(row)

    positive_target = round(sample_count * len(positive_rows) / len(rows["test"]))
    if positive_rows and positive_target == 0 and sample_count > 1:
        positive_target = 1
    negative_target = sample_count - positive_target
    if positive_target > len(positive_rows) or negative_target > len(negative_rows):
        raise ValueError("test 分层样本不足，拒绝改变真实 sticker 比例")
    positive_rows.sort(
        key=lambda row: (
            not any(
                asset_id in historical_universe
                for asset_id in _sticker_ids(_parse_dataset_target(_messages(row)[-1]["content"]))
            ),
            _canonical_hash(row),
        )
    )
    negative_rows.sort(key=_canonical_hash)
    selected = positive_rows[:positive_target] + negative_rows[:negative_target]

    held_out: list[HeldOutCase] = []
    human_turns: list[StyleTurn] = []
    actual_stickers: list[str | None] = []
    for row in selected:
        messages = _messages(row)
        target = messages[-1]
        if target["role"] != "assistant":
            raise ValueError("test 样本缺少 assistant 真人目标")
        source_hash = _canonical_hash(row)
        reply = _parse_dataset_target(target["content"])
        held_out.append(
            HeldOutCase(
                case_id=source_hash[:16],
                split="test",
                context=tuple(messages[:-1]),
                human_target=target["content"],
                source_hash=source_hash,
            )
        )
        human_turns.append(_style_turn(reply, messages[:-1]))
        actual_stickers.append(
            next(
                (bubble.asset_id for bubble in reply.bubbles if bubble.type == "sticker"),
                None,
            )
        )

    allowed_by_case: dict[str, tuple[str, ...]] = {}
    retrieval_rankings: list[tuple[str, ...]] = []
    retrieval_actual_ids: list[str] = []
    all_sticker_ids = [
        asset_id for _row, _case, sticker_ids in test_details for asset_id in sticker_ids
    ]
    for _row, case, sticker_ids in test_details:
        ranking = _rank_case_stickers(sticker_policy, case)
        if case in held_out:
            allowed_by_case[case.case_id] = ranking
        for asset_id in sticker_ids:
            if asset_id in historical_universe:
                retrieval_rankings.append(ranking)
                retrieval_actual_ids.append(asset_id)
    seen_ids = [asset_id for asset_id in all_sticker_ids if asset_id in historical_universe]
    unseen_ids = [asset_id for asset_id in all_sticker_ids if asset_id not in historical_universe]
    sample_hash = _canonical_hash([case.source_hash for case in held_out])
    return AcceptanceCorpus(
        held_out_cases=tuple(held_out),
        human_style_turns=tuple(human_turns),
        speaker_fit_samples=tuple(fit_samples),
        memorization_sources=tuple(memorization_sources),
        known_sticker_ids=tuple(sorted(known_stickers)),
        actual_sticker_ids=tuple(actual_stickers),
        allowed_sticker_ids_by_case=allowed_by_case,
        retrieval_rankings=tuple(retrieval_rankings),
        retrieval_actual_ids=tuple(retrieval_actual_ids),
        sticker_coverage={
            "ground_truth_event_count": len(all_sticker_ids),
            "historical_seen_event_count": len(seen_ids),
            "unseen_ground_truth_event_count": len(unseen_ids),
            "historical_seen_unique_count": len(set(seen_ids)),
            "unseen_ground_truth_unique_count": len(set(unseen_ids)),
            "historical_seen_rate": (
                len(seen_ids) / len(all_sticker_ids) if all_sticker_ids else 0.0
            ),
        },
        sample_hash=sample_hash,
        split_hashes={
            split: _sha256(data_dir / f"{split}.jsonl") for split in ("train", "valid", "test")
        },
    )


def evaluate_held_out_model(
    corpus: AcceptanceCorpus,
    generator: PathReplyGenerator,
    *,
    model_id: str,
    base_model: str,
    adapter_path: str,
    reply_protocol: ReplyProtocol,
    sticker_policy: StickerPolicy | None = None,
    allowed_sticker_ids_by_case: Mapping[str, tuple[str, ...]] | None = None,
) -> HeldOutModelReport:
    generated: list[GeneratedReplyTurn | None] = []
    audits: list[dict[str, object]] = []
    reasons: list[str] = []
    invalid_attempt_count = 0
    resolved_allowed = (
        dict(allowed_sticker_ids_by_case)
        if allowed_sticker_ids_by_case is not None
        else {
            case.case_id: _rank_case_stickers(sticker_policy, case)
            for case in corpus.held_out_cases
        }
    )
    for case in corpus.held_out_cases:
        case_attempted_invalid: tuple[str, ...] = ()
        context = list(case.context)
        system = next(
            (item["content"] for item in context if item["role"] == "system"),
            "",
        )
        messages = [item for item in context if item["role"] != "system"]
        allowed_ids = resolved_allowed.get(
            case.case_id,
            _rank_case_stickers(sticker_policy, case),
        )
        try:
            reply = generator.generate(
                base_model,
                adapter_path,
                system + "\n" + _protocol_instruction(reply_protocol, allowed_ids),
                messages,
            )
            direct_invalid = tuple(
                bubble.asset_id or ""
                for bubble in reply.bubbles
                if bubble.type == "sticker" and bubble.asset_id not in set(allowed_ids)
            )
            case_attempted_invalid = (
                *reply.attempted_invalid_sticker_ids,
                *direct_invalid,
            )
            invalid_attempt_count += len(case_attempted_invalid)
            reply = drop_invalid_sticker_bubbles(reply, allowed_ids)
            validate_reply_sticker_ids(reply, allowed_ids)
            policy_appended = False
        except ModelOutputStructureError as error:
            case_attempted_invalid = error.attempted_invalid_sticker_ids
            invalid_attempt_count += len(case_attempted_invalid)
            generated.append(None)
            audits.append(
                {
                    "case_id": case.case_id,
                    "source_hash": case.source_hash,
                    "parsed": False,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "attempts": [asdict(item) for item in error.attempts],
                    "attempted_invalid_sticker_ids": list(case_attempted_invalid),
                }
            )
            continue
        except Exception as error:
            generated.append(None)
            audits.append(
                {
                    "case_id": case.case_id,
                    "source_hash": case.source_hash,
                    "parsed": False,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "attempted_invalid_sticker_ids": list(case_attempted_invalid),
                }
            )
            continue
        generated.append(reply)
        human_reply = _parse_dataset_target(case.human_target)
        latest_context = next(
            (item["content"] for item in reversed(case.context) if item["role"] == "user"),
            "",
        )
        audits.append(
            {
                "case_id": case.case_id,
                "source_hash": case.source_hash,
                "parsed": True,
                "raw_output_hash": hashlib.sha256(reply.raw_output.encode("utf-8")).hexdigest(),
                "allowed_sticker_ids": list(allowed_ids),
                "normalization": reply.normalization,
                "policy_appended": policy_appended,
                "attempted_invalid_sticker_ids": list(case_attempted_invalid),
                "prompt_excerpt": latest_context[:240],
                "human_excerpt": _reply_text(human_reply)[:240],
                "generated_excerpt": _reply_text(reply)[:240],
            }
        )
    parse_failures = sum(reply is None for reply in generated)
    normalized_count = sum(
        reply is not None and reply.normalization is not None for reply in generated
    )
    case_count = len(corpus.held_out_cases)
    raw_format_compliance = (
        (case_count - parse_failures - normalized_count) / case_count if case_count else 0.0
    )
    normalization_rate = normalized_count / case_count if case_count else 0.0
    if parse_failures:
        reasons.append(f"留出集有 {parse_failures} 条输出解析失败")
    if parse_failures == len(corpus.held_out_cases):
        return HeldOutModelReport(
            model_id=model_id,
            reply_protocol=reply_protocol,
            sample_count=len(corpus.held_out_cases),
            sample_hash=corpus.sample_hash,
            parse_failures=parse_failures,
            raw_format_compliance=raw_format_compliance,
            normalization_rate=normalization_rate,
            style={"passed": False, "reason": "样本解析失败"},
            speaker={"passed": False, "reason": "样本解析失败"},
            sticker={
                "status": "failed",
                "reason": "样本解析失败",
                "passed": False,
                "invalid_asset_rate": 0.0,
                "invalid_asset_count": 0,
                "attempted_invalid_rate": (invalid_attempt_count / max(1, invalid_attempt_count)),
                "attempted_invalid_asset_count": invalid_attempt_count,
            },
            memorization={"passed": False, "reason": "样本解析失败"},
            sample_audit=tuple(audits),
            passed=False,
            failure_reasons=tuple(reasons),
        )

    successful = tuple(
        (reply, case, index)
        for index, (reply, case) in enumerate(zip(generated, corpus.held_out_cases, strict=True))
        if reply is not None
    )
    style_turns = tuple(
        _style_turn(reply, list(case.context)) for reply, case, _index in successful
    )
    texts = tuple(_reply_text(reply) for reply, _case, _index in successful)
    memorization_texts = tuple(
        _reply_memorization_text(reply) for reply, _case, _index in successful
    )
    style_distance = compare_style_distributions(
        style_turns,
        tuple(corpus.human_style_turns[index] for _reply, _case, index in successful),
    )
    if not style_distance.passed:
        reasons.append("统一 style distance 超过硬门槛")

    identifier = SpeakerStyleIdentifier()
    identifier.fit(corpus.speaker_fit_samples)
    speaker = identifier.evaluate(
        tuple(SpeakerSample("test", "target", text) for text in texts),
        positive_speaker="target",
    )
    human_texts = tuple(
        _reply_text(_parse_dataset_target(case.human_target)) for _reply, case, _index in successful
    )
    human_speaker = identifier.evaluate(
        tuple(SpeakerSample("test", "target", text) for text in human_texts),
        positive_speaker="target",
    )
    speaker_probability_alignment = 1.0 - sum(
        abs(generated - human)
        for generated, human in zip(
            speaker.probabilities,
            human_speaker.probabilities,
            strict=True,
        )
    ) / max(1, len(texts))
    paired_response_similarity = sum(
        character_ngram_cosine(generated, human)
        for generated, human in zip(texts, human_texts, strict=True)
    ) / max(1, len(texts))
    memorization = MemorizationIndex(corpus.memorization_sources).check_many(memorization_texts)
    if not bool(memorization["passed"]):
        reasons.append("候选输出命中训练集查重硬门槛")

    predictions = tuple(
        (
            tuple(
                bubble.asset_id
                for bubble in reply.bubbles
                if bubble.type == "sticker" and bubble.asset_id is not None
            )
            if reply is not None
            else ()
        )
        for reply in generated
    )
    sample_positive_count = sum(item is not None for item in corpus.actual_sticker_ids)
    retrievable_positive_count = len(corpus.retrieval_actual_ids)
    if sample_positive_count or retrievable_positive_count:
        sticker_metrics = _sticker_metrics(
            predictions,
            corpus.actual_sticker_ids,
            corpus.retrieval_rankings,
            corpus.retrieval_actual_ids,
            previous_sticker_ids=tuple(
                _previous_sticker_id(case) for case in corpus.held_out_cases
            ),
        )
        predicted_with_allowed = [
            (asset_id, set(resolved_allowed.get(case.case_id, ())))
            for case, group in zip(corpus.held_out_cases, predictions, strict=True)
            for asset_id in group
        ]
        invalid_count = sum(
            asset_id not in allowed_ids for asset_id, allowed_ids in predicted_with_allowed
        )
        final_asset_count = len(predicted_with_allowed)
        invalid_rate = invalid_count / final_asset_count if final_asset_count else 0.0
        model_asset_attempt_count = final_asset_count + invalid_attempt_count
        attempted_invalid_rate = (
            invalid_attempt_count / model_asset_attempt_count if model_asset_attempt_count else 0.0
        )
        calibrated_minimums = _sticker_calibrated_minimums(sticker_policy)
        calibration_available = calibrated_minimums is not None
        sticker_gates = build_task4_style_metrics(
            sticker_metrics,
            emoji_distance=style_distance.distances["emoji"],
            invalid_asset_rate=invalid_rate,
            calibrated_minimums=(
                calibrated_minimums
                if calibrated_minimums is not None
                else {
                    "modality_f1": 1.0,
                    "sticker_recall_at_k": 1.0,
                    "sticker_mrr": 1.0,
                }
            ),
        )
        sticker_passed = (
            all(
                (metric.minimum is None or metric.value >= metric.minimum)
                and (metric.maximum is None or metric.value <= metric.maximum)
                for metric in sticker_gates.values()
            )
            and calibration_available
        )
        sticker_payload: dict[str, object] = {
            "status": "applicable",
            "passed": sticker_passed,
            "metrics": sticker_metrics,
            "modality_metrics": {
                name: sticker_metrics[name]
                for name in (
                    "modality_precision",
                    "modality_recall",
                    "modality_f1",
                    "generation_sample_positive_count",
                )
            },
            "retrieval_metrics": {
                name: sticker_metrics[name]
                for name in (
                    "recall_at_k",
                    "mrr",
                    "retrievable_positive_count",
                )
            },
            "invalid_asset_rate": invalid_rate,
            "invalid_asset_count": invalid_count,
            "predicted_asset_count": final_asset_count,
            "attempted_invalid_rate": attempted_invalid_rate,
            "attempted_invalid_asset_count": invalid_attempt_count,
            "coverage": corpus.sticker_coverage,
            "threshold_source": (
                sticker_policy.parameters.get("threshold_source")
                if sticker_policy is not None
                else None
            ),
            "calibration_available": calibration_available,
            "gates": {name: asdict(metric) for name, metric in sticker_gates.items()},
        }
        if not sticker_passed:
            reasons.append("sticker 模态或资产硬门槛失败")
    else:
        policy_requires_sticker_evaluation = bool(
            sticker_policy is not None and sticker_policy.enabled
        )
        sticker_payload = {
            "status": "not_applicable",
            "passed": not policy_requires_sticker_evaluation,
            "reason": (
                "活动 sticker policy 缺少可验收 test 正例"
                if policy_requires_sticker_evaluation
                else "历史中没有启用 sticker policy"
            ),
            "coverage": corpus.sticker_coverage,
            "invalid_asset_rate": 0.0,
            "invalid_asset_count": 0,
            "attempted_invalid_rate": (invalid_attempt_count / max(1, invalid_attempt_count)),
            "attempted_invalid_asset_count": invalid_attempt_count,
        }
        if policy_requires_sticker_evaluation:
            reasons.append("sticker policy 已启用但没有可验收 test 正例")

    return HeldOutModelReport(
        model_id=model_id,
        reply_protocol=reply_protocol,
        sample_count=len(corpus.held_out_cases),
        sample_hash=corpus.sample_hash,
        parse_failures=parse_failures,
        raw_format_compliance=raw_format_compliance,
        normalization_rate=normalization_rate,
        style={
            "version": style_distance.version,
            "overall_distance": style_distance.overall_distance,
            "threshold": style_distance.threshold,
            "distances": style_distance.distances,
            "passed": style_distance.passed,
        },
        speaker={
            "sample_count": speaker.sample_count,
            "accuracy": speaker.accuracy,
            "mean_positive_probability": speaker.mean_positive_probability,
            "human_oracle_speaker_probability": (human_speaker.mean_positive_probability),
            "speaker_probability_alignment": speaker_probability_alignment,
            "paired_response_similarity": paired_response_similarity,
            "brier_score": speaker.brier_score,
            "passed": True,
        },
        sticker=sticker_payload,
        memorization=memorization,
        sample_audit=tuple(audits),
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


def retry_acceptance_only(
    session: Session,
    *,
    job_id: str,
    data_root: Path,
    model_root: Path,
    acceptance_runner: ModelAcceptanceRunner,
    generator: PathReplyGenerator,
    sample_count: int = 20,
    candidate_adapter_path: Path | None = None,
) -> dict[str, object]:
    """校验原训练全部 CAS 后，仅复用 final adapter 重跑验收并原子发布。"""

    job = session.get(Job, job_id)
    if job is None or job.status != "failed":
        raise RuntimeError("acceptance-only 只接受原失败训练任务")
    confirmation_id = str(job.payload.get("confirmation_id", ""))
    confirmation = session.get(TimelineConfirmation, confirmation_id)
    if (
        confirmation is None
        or confirmation.project_id != job.payload.get("project_id")
        or confirmation.confirmation_fingerprint != job.payload.get("confirmation_fingerprint")
    ):
        raise RuntimeError("原任务、项目或确认记录 CAS 校验失败")
    project_id = confirmation.project_id
    output_dir = (model_root / project_id / job_id).resolve()
    expected_output_parent = (model_root / project_id).resolve()
    if output_dir.parent != expected_output_parent:
        raise RuntimeError("模型输出目录越过项目边界")
    state_path = output_dir / "search-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    full = state.get("full_training")
    if not isinstance(full, dict) or full.get("status") != "succeeded":
        raise RuntimeError("原任务没有已完成的 final adapter")
    adapter_path = (
        candidate_adapter_path.resolve()
        if candidate_adapter_path is not None
        else (output_dir / "final").resolve()
    )
    if not adapter_path.is_relative_to(output_dir):
        raise RuntimeError("候选 adapter 路径越过原训练任务目录")
    adapter_file = adapter_path / "adapters.safetensors"
    adapter_sha256 = _sha256(adapter_file)
    if candidate_adapter_path is None and adapter_sha256 != full.get("final_adapter_sha256"):
        raise RuntimeError("final adapter SHA 不匹配")

    data_dir = (data_root / "projects" / project_id / "training" / confirmation.id).resolve()
    manifest_path = data_dir / "manifest.json"
    fingerprint = state.get("fingerprint_payload")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("训练状态缺少数据指纹")
    expected_manifest_sha = fingerprint.get("declared_data_manifest_digest")
    if _sha256(manifest_path) != expected_manifest_sha:
        raise RuntimeError("dataset manifest SHA 不匹配")
    for item in fingerprint.get("data_files", []):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise RuntimeError("训练状态的数据文件指纹无效")
        if _sha256(data_dir / str(item["name"])) != item.get("sha256"):
            raise RuntimeError(f"数据文件 SHA 不匹配：{item['name']}")
    manifest = DatasetManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    if (
        manifest.confirmation_id != confirmation.id
        or manifest.base_model != confirmation.config_snapshot.get("base_model")
    ):
        raise RuntimeError("数据清单与原确认记录不一致")

    old_report_path = output_dir / "acceptance-report.json"
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    expected_active_id = (
        old_report.get("model_ids", {}).get("active")
        if isinstance(old_report.get("model_ids"), dict)
        else None
    )
    active = session.scalar(
        select(ModelVersion)
        .where(
            ModelVersion.project_id == project_id,
            ModelVersion.active.is_(True),
        )
        .order_by(ModelVersion.created_at.desc())
    )
    if active is None or active.id != expected_active_id:
        raise RuntimeError("当前活动模型已变化，拒绝 acceptance-only 覆盖")
    if (
        session.scalar(select(ModelVersion).where(ModelVersion.training_job_id == job_id))
        is not None
    ):
        raise RuntimeError("原任务已经发布过模型")

    target = session.scalar(
        select(Participant).where(
            Participant.project_id == project_id,
            Participant.role == "target",
        )
    )
    if target is None:
        raise RuntimeError("原项目缺少 target 参与者")
    candidate_protocol = resolve_reply_protocol(confirmation.config_snapshot)
    base_protocol = resolve_reply_protocol(manifest.training_config)
    active_protocol = resolve_reply_protocol(active.training_config)
    cutoff = manifest.cutoff
    split_boundaries = manifest.split_time_boundaries
    train_boundary_text = split_boundaries.get("train", {}).get("end")
    valid_start_text = split_boundaries.get("valid", {}).get("start")
    valid_end_text = split_boundaries.get("valid", {}).get("end")
    test_start_text = split_boundaries.get("test", {}).get("start")
    test_end_text = split_boundaries.get("test", {}).get("end")
    if not all(
        isinstance(value, str) and value
        for value in (
            train_boundary_text,
            valid_start_text,
            valid_end_text,
            test_start_text,
            test_end_text,
        )
    ):
        raise RuntimeError("数据清单缺少 train/valid/test 历史边界")
    train_boundary = datetime.fromisoformat(train_boundary_text)
    embedder_error: str | None = None
    try:
        sticker_embedder = LocalChineseEmbedder(
            model_root / "embeddings" / "fastembed-bge-small-zh-v1.5"
        )
    except Exception as error:
        sticker_embedder = None
        embedder_error = type(error).__name__
    train_sticker_policy = build_sticker_policy_from_database(
        session,
        project_id=project_id,
        target_id=target.id,
        cutoff=train_boundary,
        branch_time=train_boundary,
        import_id=confirmation.import_id,
        embedder=sticker_embedder,
    )
    tuned_policy, sticker_valid_metrics = tune_sticker_policy_on_valid(
        train_sticker_policy,
        build_sticker_evaluation_cases_from_database(
            session,
            project_id=project_id,
            target_id=target.id,
            start=datetime.fromisoformat(valid_start_text),
            end=datetime.fromisoformat(valid_end_text),
            split="valid",
            import_id=confirmation.import_id,
        ),
        parameter_grid=tuple(
            {
                "minimum_similarity": minimum_similarity,
                "modality_threshold": modality_threshold,
            }
            for minimum_similarity in (0.05, 0.12, 0.2)
            for modality_threshold in (0.38, 0.4, 0.42, 0.44, 0.46, 0.5)
        ),
    )
    sticker_policy = (
        replace(
            tuned_policy,
            parameters={
                **tuned_policy.parameters,
                "semantic_status": "unavailable",
                "semantic_failure": embedder_error,
            },
        )
        if embedder_error is not None
        else tuned_policy
    )
    sticker_test_metrics = evaluate_sticker_policy(
        sticker_policy,
        build_sticker_evaluation_cases_from_database(
            session,
            project_id=project_id,
            target_id=target.id,
            start=datetime.fromisoformat(test_start_text),
            end=datetime.fromisoformat(test_end_text),
            split="test",
            import_id=confirmation.import_id,
        ),
        expected_split="test",
    )
    corpus = load_acceptance_corpus(
        data_dir,
        sample_count=sample_count,
        sticker_policy=sticker_policy,
    )
    model_inputs = {
        "candidate": (
            f"training-job:{job_id}",
            str(manifest.base_model),
            str(adapter_path),
            candidate_protocol,
        ),
        "base": (
            str(manifest.base_model),
            str(manifest.base_model),
            "",
            base_protocol,
        ),
        "active": (
            active.id,
            active.base_model,
            active.adapter_path,
            active_protocol,
        ),
    }
    active_configured_policy = StickerPolicy.from_metadata(
        active.training_config.get("sticker_policy")
    )
    active_sticker_policy = (
        build_sticker_policy_from_database(
            session,
            project_id=project_id,
            target_id=target.id,
            cutoff=train_boundary,
            branch_time=train_boundary,
            import_id=confirmation.import_id,
            parameters=active_configured_policy.parameters,
            embedder=sticker_embedder,
        )
        if active_configured_policy.enabled
        else None
    )
    evaluation_policies = {
        "candidate": sticker_policy,
        "base": None,
        "active": active_sticker_policy,
    }
    fixed: dict[str, AcceptanceReport] = {}
    held_out: dict[str, HeldOutModelReport] = {}
    for name, (
        model_id,
        evaluated_base_model,
        evaluated_adapter_path,
        reply_protocol,
    ) in model_inputs.items():
        fixed[name] = acceptance_runner.run(
            base_model=evaluated_base_model,
            adapter_path=evaluated_adapter_path,
            persona=target.name,
            cutoff=cutoff,
            style_profile=manifest.style_profile,
            reply_protocol=reply_protocol,
        )
        held_out[name] = evaluate_held_out_model(
            corpus,
            generator,
            model_id=model_id,
            base_model=evaluated_base_model,
            adapter_path=evaluated_adapter_path,
            reply_protocol=reply_protocol,
            sticker_policy=evaluation_policies[name],
            allowed_sticker_ids_by_case=(
                corpus.allowed_sticker_ids_by_case
                if name == "candidate"
                else {
                    case.case_id: _rank_case_stickers(
                        evaluation_policies[name],
                        case,
                    )
                    for case in corpus.held_out_cases
                }
            ),
        )
    snapshots = {
        name: _gate_snapshot(
            held_out[name],
            fixed[name],
        )
        for name in ("candidate", "base", "active")
    }
    gate = evaluate_activation_gates(
        snapshots["candidate"],
        snapshots["base"],
        snapshots["active"],
    )
    reasons = [
        *gate.failure_reasons,
        *held_out["candidate"].failure_reasons,
    ]
    passed = gate.passed and fixed["candidate"].passed and held_out["candidate"].passed
    report: dict[str, object] = {
        "schema_version": "moonlightbox.high-fidelity-acceptance.v2",
        "mode": "acceptance-only",
        "passed": passed,
        "model_ids": {
            "candidate": f"training-job:{job_id}",
            "base": str(manifest.base_model),
            "active": active.id,
        },
        "protocols": {
            "candidate": candidate_protocol,
            "base": base_protocol,
            "active": active_protocol,
        },
        "dataset": {
            "dataset_hash": manifest.dataset_hash,
            "manifest_sha256": expected_manifest_sha,
            "split_hashes": corpus.split_hashes,
            "sample_count": len(corpus.held_out_cases),
            "sample_hash": corpus.sample_hash,
            "test_used_for_tuning": False,
        },
        "sticker_policy_valid_tuning": {
            "parameters": dict(sticker_policy.parameters),
            "metrics": sticker_valid_metrics,
            "split": "valid",
        },
        "sticker_policy_test_evaluation": {
            "parameters_frozen_from": "valid",
            "metrics": sticker_test_metrics,
            "split": "test",
            "test_used_for_tuning": False,
        },
        "sticker_embedding": {
            "status": sticker_policy.parameters.get("semantic_status"),
            "failure": sticker_policy.parameters.get("semantic_failure"),
            "model": "BAAI/bge-small-zh-v1.5",
        },
        "adapter": {
            "path": str(adapter_path),
            "sha256": adapter_sha256,
            "retrained": False,
            "checkpoint_override": candidate_adapter_path is not None,
        },
        "semantic": {name: asdict(value) for name, value in fixed.items()},
        "held_out": {name: asdict(value) for name, value in held_out.items()},
        "format": {
            name: {
                "raw_format_compliance": held_out[name].raw_format_compliance,
                "normalization_rate": held_out[name].normalization_rate,
            }
            for name in ("candidate", "base", "active")
        },
        "gates": asdict(gate),
        "failure_reasons": list(dict.fromkeys(reasons)),
        "report_path": str(old_report_path),
    }
    archived_report_path = _archive_report(old_report_path)
    report["archived_report_path"] = str(archived_report_path)
    _atomic_write_json(old_report_path, report)
    if not passed:
        session.rollback()
        return report

    examples = _load_training_examples(data_dir)
    kernel = EvidenceBackedIdentityKernelBuilder().build(
        persona=target.name,
        examples=examples,
        event_contexts=[_event_context(item) for item in confirmation.event_revision_snapshots],
    )
    validation_loss = full.get("best_validation_loss", 0.0)
    metrics = {
        "validation_loss": (
            float(validation_loss) if isinstance(validation_loss, int | float) else 0.0
        ),
        "acceptance_pass_rate": 1.0,
        "style_score": 1.0
        - _number_value(held_out["candidate"].style.get("overall_distance"), 1.0),
        "quality_gate_passed": 1.0,
    }
    blind_cases: list[dict[str, object]] = []
    raw_minimum_ratings = confirmation.config_snapshot.get(
        "human_blind_minimum_ratings",
        20,
    )
    minimum_ratings = (
        int(raw_minimum_ratings) if isinstance(raw_minimum_ratings, int | float) else 20
    )
    minimum_preference = _number_value(
        confirmation.config_snapshot.get("human_blind_minimum_preference"),
        0.45,
    )
    if bool(confirmation.config_snapshot.get("human_blind_required")):
        blind_cases = acceptance_runner.build_human_blind_cases(
            data_dir=data_dir,
            base_model=str(manifest.base_model),
            adapter_path=str(adapter_path),
            sample_count=minimum_ratings,
        )
    version = ModelRegistry(session).publish_and_activate(
        project_id=project_id,
        base_model=str(manifest.base_model),
        adapter_path=str(adapter_path),
        dataset_hash=manifest.dataset_hash,
        metrics=metrics,
        training_config={
            **confirmation.config_snapshot,
            "training_metadata": state,
            "data_manifest_digest": expected_manifest_sha,
            "data_manifest": asdict(manifest),
            "acceptance_only_retry": True,
        },
        kernel_proposal=kernel,
        evidence_message_ids=list(
            dict.fromkeys(source_id for example in examples for source_id in example.source_ids)
        ),
        acceptance_report=report,
        sticker_policy=sticker_policy.to_metadata(),
        expected_active_model_id=active.id,
        timeline_confirmation_id=confirmation.id,
        training_job_id=job_id,
        commit=False,
    )
    if blind_cases:
        HumanBlindStudyService(session).create(
            project_id=project_id,
            model_version_id=version.id,
            cases=blind_cases,
            minimum_ratings=minimum_ratings,
            minimum_preference=minimum_preference,
            commit=False,
        )
    confirmation.status = "trained"
    session.commit()
    session.refresh(version)
    report["published_model_id"] = version.id
    _atomic_write_json(old_report_path, report)
    return report


def _gate_snapshot(
    held_out: HeldOutModelReport,
    fixed: object,
) -> ModelGateSnapshot:
    case_count = int(getattr(fixed, "case_count", 0))
    structure_failures = int(getattr(fixed, "structure_failures", case_count))
    fact_failures = int(getattr(fixed, "forbidden_fact_failures", case_count))
    raw_failures = int(getattr(fixed, "raw_output_failures", case_count))
    passed_count = int(getattr(fixed, "passed_count", 0))
    fixed_regression_passed = (
        case_count > 0
        and passed_count == case_count
        and structure_failures == 0
        and fact_failures == 0
        and raw_failures == 0
    )
    semantic = {
        "fixed_regression": AcceptanceMetric(float(fixed_regression_passed)),
        "structure": AcceptanceMetric(1.0 - structure_failures / max(1, case_count)),
        "future_isolation": AcceptanceMetric(float(raw_failures == 0)),
        "unsupported_fact": AcceptanceMetric(float(fact_failures == 0)),
    }
    style_distance = _number_value(held_out.style.get("overall_distance"), 1.0)
    speaker_probability = _number_value(
        held_out.speaker.get("mean_positive_probability"),
        0.0,
    )
    human_oracle_speaker_probability = _number_value(
        held_out.speaker.get("human_oracle_speaker_probability"),
        0.0,
    )
    speaker_probability_alignment = _number_value(
        held_out.speaker.get("speaker_probability_alignment"),
        0.0,
    )
    paired_response_similarity = _number_value(
        held_out.speaker.get("paired_response_similarity"),
        0.0,
    )
    style = {
        "style_distance": AcceptanceMetric(
            style_distance,
            tolerance=0.03,
            higher_is_better=False,
        ),
        "speaker_probability": AcceptanceMetric(
            speaker_probability,
            tolerance=0.03,
        ),
        "human_oracle_speaker_probability": AcceptanceMetric(
            human_oracle_speaker_probability,
        ),
        "speaker_probability_alignment": AcceptanceMetric(
            speaker_probability_alignment,
            tolerance=0.03,
        ),
        "paired_response_similarity": AcceptanceMetric(
            paired_response_similarity,
            tolerance=0.01,
        ),
        "style_distance_threshold": AcceptanceMetric(
            style_distance,
            maximum=0.35,
            higher_is_better=False,
        ),
    }
    sticker_gates = held_out.sticker.get("gates")
    if isinstance(sticker_gates, dict):
        for name, payload in sticker_gates.items():
            if isinstance(payload, dict):
                style[str(name)] = AcceptanceMetric(
                    value=float(payload["value"]),
                    minimum=(
                        float(payload["minimum"]) if payload.get("minimum") is not None else None
                    ),
                    maximum=(
                        float(payload["maximum"]) if payload.get("maximum") is not None else None
                    ),
                    higher_is_better=bool(payload.get("higher_is_better", True)),
                )
    return ModelGateSnapshot(held_out.model_id, semantic, style)


def _load_training_examples(data_dir: Path) -> list[TrainingExample]:
    result: list[TrainingExample] = []
    for split in ("train", "valid", "test"):
        for row in _read_jsonl(data_dir / f"{split}.jsonl"):
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError("训练样本缺少 metadata")
            result.append(
                TrainingExample(
                    messages=[
                        ChatTurn(
                            role=message["role"],  # type: ignore[arg-type]
                            content=message["content"],
                        )
                        for message in _messages(row)
                    ],
                    source_ids=[str(item) for item in metadata.get("source_ids", [])],
                    target_at=datetime.fromisoformat(str(metadata["target_at"])),
                    kind="chat",
                )
            )
    return result


def _event_context(payload: dict[str, object]) -> ConfirmedEventContext:
    evidence = payload.get("evidence_ids")
    return ConfirmedEventContext(
        event_id=str(payload["event_id"]),
        title=str(payload["title"]),
        summary=str(payload["summary"]),
        lane=str(payload["lane"]),
        event_status=str(payload["event_status"]),
        evidence_ids=(tuple(str(item) for item in evidence) if isinstance(evidence, list) else ()),
        before_state=(
            str(payload["before_state"]) if payload.get("before_state") is not None else None
        ),
        after_state=(
            str(payload["after_state"]) if payload.get("after_state") is not None else None
        ),
    )


def _messages(row: dict[str, object]) -> list[dict[str, str]]:
    raw = row.get("messages")
    if not isinstance(raw, list) or not raw:
        raise ValueError("训练样本缺少 messages")
    result: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("训练消息结构无效")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("训练消息字段无效")
        result.append({"role": role, "content": content})
    return result


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name} 包含非对象样本")
        rows.append(payload)
    return rows


def _style_turn(
    reply: GeneratedReplyTurn,
    context: list[dict[str, str]],
) -> StyleTurn:
    previous = next(
        (message["content"] for message in reversed(context) if message["role"] == "user"),
        "",
    )
    return StyleTurn(
        previous_text=previous,
        bubbles=tuple(
            StyleBubble(
                text=bubble.content or "",
                delay_ms=bubble.delay_ms,
                kind=bubble.type,
                asset_id=bubble.asset_id,
            )
            for bubble in reply.bubbles
        ),
    )


def _reply_text(reply: GeneratedReplyTurn) -> str:
    return "\n".join(
        bubble.content or f"[{bubble.type}:{bubble.asset_id}]" for bubble in reply.bubbles
    )


def _reply_memorization_text(reply: GeneratedReplyTurn) -> str:
    """查重只评估模型文字，不把策略控制的历史资产 ID 当作语言复制。"""

    return "\n".join(
        bubble.content for bubble in reply.bubbles if bubble.type == "text" and bubble.content
    )


def _parse_dataset_target(content: str) -> GeneratedReplyTurn:
    """按训练协议读取真人 target，不套用运行时防复读策略。"""

    try:
        bubbles = parse_bubble_protocol(content)
    except ValueError:
        if "<" in content or ">" in content:
            raise
        lines = tuple(line.strip() for line in content.splitlines() if line.strip())
        if not lines:
            raise ValueError("纯文本训练目标不能为空") from None
        return GeneratedReplyTurn(
            bubbles=tuple(
                GeneratedBubble(
                    content=line,
                    delay_ms=0 if index == 0 else 800,
                )
                for index, line in enumerate(lines)
            ),
            raw_output=content,
        )
    return GeneratedReplyTurn(
        bubbles=tuple(
            GeneratedBubble(
                content=bubble.value if bubble.kind == "text" else None,
                delay_ms=bubble.delay_ms,
                type=bubble.kind,
                asset_id=bubble.value if bubble.kind != "text" else None,
            )
            for bubble in bubbles
        ),
        raw_output=content,
    )


def _protocol_instruction(
    reply_protocol: ReplyProtocol,
    allowed_sticker_ids: tuple[str, ...],
) -> str:
    if reply_protocol == "compact":
        return compact_protocol_instruction(allowed_sticker_ids)
    if allowed_sticker_ids:
        allowed = "、".join(allowed_sticker_ids)
        sticker_rule = f"本轮 sticker 仅允许以下资产 ID：{allowed}；禁止输出其他资产。"
    else:
        sticker_rule = "本轮禁止输出 sticker。"
    return "只输出合法统一气泡 JSON。" + sticker_rule


def _sticker_ids(reply: GeneratedReplyTurn) -> tuple[str, ...]:
    return tuple(
        bubble.asset_id
        for bubble in reply.bubbles
        if bubble.type == "sticker" and bubble.asset_id is not None
    )


def _rank_case_stickers(
    policy: StickerPolicy | None,
    case: HeldOutCase,
) -> tuple[str, ...]:
    if policy is None:
        return ()
    return tuple(item.asset_id for item in rank_stickers(policy, _sticker_context(case.context)))


def _sticker_context(context: tuple[dict[str, str], ...]) -> StickerContext:
    messages = [item for item in context if item["role"] != "system"]
    current_index = next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index]["role"] == "user"),
        -1,
    )
    current_text = messages[current_index]["content"] if current_index >= 0 else ""
    recent: list[RecentStickerBubble] = []
    for message in messages[:current_index]:
        if message["role"] == "assistant":
            try:
                reply = _parse_dataset_target(message["content"])
            except ValueError:
                recent.append(
                    RecentStickerBubble(
                        kind="text",
                        content=message["content"],
                    )
                )
            else:
                recent.extend(
                    RecentStickerBubble(
                        kind=bubble.type,
                        content=bubble.content or "",
                        asset_id=bubble.asset_id,
                    )
                    for bubble in reply.bubbles
                )
        else:
            recent.append(
                RecentStickerBubble(
                    kind="text",
                    content=message["content"],
                )
            )
    return StickerContext(current_text=current_text, recent=tuple(recent))


def _previous_sticker_id(case: HeldOutCase) -> str | None:
    return next(
        (
            item.asset_id
            for item in reversed(_sticker_context(case.context).recent)
            if item.kind == "sticker" and item.asset_id is not None
        ),
        None,
    )


def _sticker_metrics(
    predictions: tuple[tuple[str, ...], ...],
    actual: tuple[str | None, ...],
    retrieval_rankings: tuple[tuple[str, ...], ...],
    retrieval_actual_ids: tuple[str, ...],
    *,
    previous_sticker_ids: tuple[str | None, ...],
) -> dict[str, float]:
    predicted_positive = sum(bool(group) for group in predictions)
    actual_positive = sum(item is not None for item in actual)
    true_positive = sum(
        bool(group) and expected is not None
        for group, expected in zip(predictions, actual, strict=True)
    )
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    retrieval_hits = sum(
        expected in ranking
        for ranking, expected in zip(
            retrieval_rankings,
            retrieval_actual_ids,
            strict=True,
        )
    )
    reciprocal_ranks = [
        1.0 / (ranking.index(expected) + 1) if expected in ranking else 0.0
        for ranking, expected in zip(
            retrieval_rankings,
            retrieval_actual_ids,
            strict=True,
        )
    ]
    repeat_pairs = sum(
        bool(group) and previous is not None and group[0] == previous
        for group, previous in zip(
            predictions,
            previous_sticker_ids,
            strict=True,
        )
    ) + sum(
        current == previous
        for group in predictions
        for previous, current in zip(group, group[1:], strict=False)
    )
    predicted_asset_count = sum(len(group) for group in predictions)
    return {
        "modality_precision": precision,
        "modality_recall": recall,
        "modality_f1": f1,
        "recall_at_k": (
            retrieval_hits / len(retrieval_actual_ids) if retrieval_actual_ids else 0.0
        ),
        "mrr": (sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0),
        "consecutive_repeat_rate": repeat_pairs / max(1, predicted_asset_count),
        "generation_sample_positive_count": float(actual_positive),
        "retrievable_positive_count": float(len(retrieval_actual_ids)),
    }


def _canonical_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _row_target_at(row: dict[str, object]) -> datetime:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("target_at"), str):
        raise ValueError("训练样本缺少 target_at 时间边界")
    return _aware_datetime(datetime.fromisoformat(str(metadata["target_at"])))


def _aware_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _number_value(value: object, default: float) -> float:
    return float(value) if isinstance(value, int | float) else default


def _sticker_calibrated_minimums(
    policy: StickerPolicy | None,
) -> dict[str, float] | None:
    if (
        policy is None
        or policy.parameters.get("threshold_source") != "valid-only-confidence-lower-bound-v1"
    ):
        return None
    names = {
        "modality_f1": "minimum_modality_f1",
        "sticker_recall_at_k": "minimum_seen_positive_recall_at_5",
        "sticker_mrr": "minimum_seen_positive_mrr",
    }
    result: dict[str, float] = {}
    for output_name, parameter_name in names.items():
        value = policy.parameters.get(parameter_name)
        if not isinstance(value, int | float):
            return None
        result[output_name] = float(value)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _archive_report(path: Path) -> Path:
    """覆盖验收报告前保存带版本号的不可变副本。"""

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    digest = _sha256(path)[:12]
    archived = path.with_name(f"{path.stem}.{timestamp}-{digest}{path.suffix}")
    temporary = archived.with_suffix(archived.suffix + ".tmp")
    temporary.write_bytes(path.read_bytes())
    temporary.replace(archived)
    return archived
