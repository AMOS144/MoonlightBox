import json
from pathlib import Path

import pytest
from moonlightbox.training.peft_adapter import (
    BestCheckpointSelection,
    PeftCapabilities,
    PeftLoraConfig,
    PeftTrainingError,
    TrainingResult,
)


def _fake_capabilities() -> PeftCapabilities:
    return PeftCapabilities("0.20.0", "2.5.1", "5.14.1")


class SearchFakeAdapter:
    def __init__(
        self,
        losses: dict[str, float | tuple[float, ...]],
        *,
        fail: set[str] | None = None,
        interrupt_on: str | None = None,
        full_curve: tuple[float, ...] = (0.8, 0.7),
    ) -> None:
        self.losses = losses
        self.fail = fail or set()
        self.interrupt_on = interrupt_on
        self.full_curve = full_curve
        self.configs: list[PeftLoraConfig] = []
        self.evaluated_checkpoints: list[Path] = []
        self.capabilities = _fake_capabilities()

    def evaluate_loss(self, config: PeftLoraConfig, checkpoint: Path) -> float:
        self.evaluated_checkpoints.append(checkpoint)
        return 0.65

    def evaluate_split_loss(
        self,
        config: PeftLoraConfig,
        checkpoint: Path,
        *,
        split: str,
    ) -> float:
        assert split == "valid"
        self.evaluated_checkpoints.append(checkpoint)
        return 0.42

    def train(
        self,
        config: PeftLoraConfig,
        on_progress: object | None = None,
    ) -> TrainingResult:
        self.configs.append(config)
        run_name = config.adapter_dir.name
        candidate_id = run_name
        if candidate_id == self.interrupt_on:
            raise KeyboardInterrupt
        if candidate_id in self.fail:
            raise PeftTrainingError(f"{candidate_id} failed")
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        weights = config.adapter_dir / "adapter_model.safetensors"
        weights.write_bytes(candidate_id.encode("utf-8"))
        config_path = config.adapter_dir / "training.json"
        config_path.write_text(f"seed: {config.seed}\n", encoding="utf-8")
        (config.adapter_dir / "adapter_config.json").write_text(
            '{"fine_tune_type":"lora"}',
            encoding="utf-8",
        )
        curve = (
            self.full_curve
            if candidate_id == "full"
            else (
                self.losses[candidate_id]
                if isinstance(self.losses[candidate_id], tuple)
                else (self.losses[candidate_id] + 0.1, self.losses[candidate_id])
            )
        )
        all_points = tuple(
            {"iteration": index, "validation_loss": loss}
            for index, loss in enumerate(curve, start=1)
        )
        best_loss = float("inf")
        best_point: dict[str, float | int] | None = None
        stale = 0
        early_stopped = False
        stopped_iteration: int | None = None
        selected_points: list[dict[str, float | int]] = []
        for point in all_points:
            selected_points.append(point)
            iteration = int(point["iteration"])
            if iteration <= 1:
                continue
            loss = float(point["validation_loss"])
            if loss < best_loss:
                best_loss = loss
                best_point = point
                stale = 0
            else:
                stale += 1
            if config.early_stopping_patience > 0 and stale >= config.early_stopping_patience:
                early_stopped = True
                stopped_iteration = iteration
                break
        validation_curve = tuple(selected_points)
        if callable(on_progress):
            for point in validation_curve:
                on_progress(
                    {
                        "stage": "training",
                        **point,
                    }
                )
        for point in validation_curve:
            if int(point["iteration"]) <= 1:
                continue
            checkpoint = config.adapter_dir / (
                f"{int(point['iteration']) - 1:07d}_adapter_model.safetensors"
            )
            checkpoint.write_bytes(str(point["validation_loss"]).encode("utf-8"))
        best_checkpoint = weights
        selection: BestCheckpointSelection | None = None
        best_validation_iteration: int | None = None
        best_checkpoint_iteration: int | None = None
        if best_point is not None:
            best_validation_iteration = int(best_point["iteration"])
            best_checkpoint_iteration = best_validation_iteration - 1
            best_checkpoint = config.adapter_dir / (
                f"{best_checkpoint_iteration:07d}_adapter_model.safetensors"
            )
            selection = BestCheckpointSelection(
                validation_loss=float(best_point["validation_loss"]),
                validation_iteration=best_validation_iteration,
                checkpoint_iteration=best_checkpoint_iteration,
                checkpoint=best_checkpoint,
            )
        return TrainingResult(
            config.adapter_dir,
            config.resolved_iterations,
            config_path,
            "0.20.0",
            validation_curve,
            selection,
            (),
            early_stopped,
            stopped_iteration,
        )


def _training_paths(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "models" / "Qwen3-8B-4bit"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"weight-a")
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"weight-b")
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text('{"dataset_hash":"a"}', encoding="utf-8")
    (data_dir / "train.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    (data_dir / "valid.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    (data_dir / "test.jsonl").write_text('{"messages":[]}\n', encoding="utf-8")
    return model_dir, data_dir


def test_search_runs_candidates_serially_and_full_trains_best(tmp_path: Path) -> None:
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    adapter = SearchFakeAdapter(
        {
            "rank16-attn16": 1.4,
            "rank32-attn24": 0.8,
            "rank32-qv-all": 1.0,
        },
        full_curve=(0.75, 0.72, 0.74, 0.76),
    )
    style_scores = {
        "rank16-attn16": 0.9,
        "rank32-attn24": 0.95,
        "rank32-qv-all": 0.5,
    }

    result = run_lora_candidate_search(
        adapter,
        base_model=str(model_dir),
        data_dir=data_dir,
        output_dir=tmp_path / "search",
        data_manifest_digest="manifest-a",
        protocol_versions={"training": "v2", "style_features": "v4"},
        train_example_count=12,
        batch_size=2,
        style_evaluator=lambda path: {
            "style_score": style_scores[path.name],
        },
        max_epochs=6,
        early_stopping_patience=2,
    )

    short_configs = [
        config for config in adapter.configs if config.adapter_dir.parent.name == "candidates"
    ]
    full_configs = [config for config in adapter.configs if config.adapter_dir.name == "full"]
    assert [config.adapter_dir.name for config in short_configs] == [
        "rank16-attn16",
        "rank32-attn24",
        "rank32-qv-all",
    ]
    assert all(config.model == str(model_dir) for config in short_configs)
    assert all(config.resume_adapter_file is None for config in short_configs)
    assert all(config.save_every == config.steps_per_evaluation for config in short_configs)
    assert all(config.save_every > 1 for config in short_configs)
    assert all(config.early_stopping_patience == 1 for config in short_configs)
    assert result.best_candidate_id == "rank32-attn24"
    assert len(full_configs) == 1
    assert full_configs[0].resume_adapter_file is None
    assert full_configs[0].epochs == 6
    assert result.early_stopping_applied is True
    assert result.completed_epochs == 1
    assert result.best_checkpoint.is_file()
    assert result.validation_curve[-1]["validation_loss"] == 0.76
    assert (result.adapter_dir / "adapter_model.safetensors").read_bytes() == b"0.72"
    assert (result.adapter_dir / "adapter_config.json").is_file()
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert state["full_training"]["best_validation_iteration"] == 2
    assert state["full_training"]["best_checkpoint_iteration"] == 1
    assert state["full_training"]["approximation_steps"] == 0
    assert state["full_training"]["checkpoint_mapping_rule"] == ("peft-validation-at-save-step")
    assert state["full_training"]["selection_validation_loss"] == 0.72
    assert state["full_training"]["verified_test_loss"] == 0.65
    assert adapter.evaluated_checkpoints == [result.adapter_dir]
    assert state["full_training"]["final_adapter_sha256"]
    assert state["full_training"]["completed_steps"] == 3
    assert state["full_training"]["partial_epoch_steps"] == 1
    assert state["full_training"]["progress_epochs"] == 1.5


def test_selected_checkpoint_is_independently_revalidated_before_publish(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    adapter = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        }
    )
    evaluated: list[Path] = []

    result = run_lora_candidate_search(
        adapter,
        base_model=str(model_dir),
        data_dir=data_dir,
        output_dir=tmp_path / "search",
        data_manifest_digest="manifest-a",
        protocol_versions={"training": "v2"},
        train_example_count=8,
        batch_size=1,
        style_evaluator=lambda _path: {"style_score": 1.0},
        checkpoint_evaluator=lambda _config, path: evaluated.append(path) or 0.55,
        max_epochs=1,
    )

    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert evaluated == [result.adapter_dir]
    assert state["full_training"]["verified_test_loss"] == 0.55
    assert state["full_training"]["verification_kind"] == "independent_test_loss"


def test_full_run_selects_semantically_valid_checkpoint_by_held_out_style(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    adapter = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        },
        full_curve=(0.75, 0.72, 0.74, 0.76),
    )

    def evaluate_style(path: Path) -> dict[str, float]:
        if path.name.startswith(".checkpoint-style-eval-"):
            iteration = int(path.name.rsplit("-", 1)[1])
            fidelity = {1: 0.60, 2: 0.93, 3: 0.10}[iteration]
            return {
                "semantic_passed": 0.0 if iteration == 3 else 1.0,
                "composite_fidelity": fidelity,
                "speaker_probability": fidelity,
                "style_distance": 1.0 - fidelity,
                "style_score": fidelity,
            }
        return {
            "semantic_passed": 1.0,
            "composite_fidelity": 0.8,
            "speaker_probability": 0.8,
            "style_distance": 0.2,
            "style_score": 0.8,
        }

    result = run_lora_candidate_search(
        adapter,
        base_model=str(model_dir),
        data_dir=data_dir,
        output_dir=tmp_path / "style-search",
        data_manifest_digest="manifest-a",
        protocol_versions={"training": "v2"},
        train_example_count=12,
        batch_size=2,
        style_evaluator=evaluate_style,
        full_checkpoint_style_evaluator=evaluate_style,
        max_epochs=6,
        early_stopping_patience=2,
    )

    # Checkpoint 1 has the lowest validation loss, but checkpoint 2 is much
    # closer to the target speaker. Checkpoint 3 is excluded by the hard gate.
    assert (result.adapter_dir / "adapter_model.safetensors").read_bytes() == b"0.74"
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    full = state["full_training"]
    assert full["checkpoint_selection_strategy"] == "held_out_style_then_validation_v1"
    assert full["best_checkpoint_iteration"] == 2
    assert full["selection_validation_loss"] == 0.74
    assert [item["checkpoint_iteration"] for item in full["checkpoint_style_evaluations"]] == [
        1,
        2,
        3,
    ]


def test_search_resumes_completed_candidates_and_rejects_fingerprint_change(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.search.lock import TrainingFingerprintMismatch
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    output_dir = tmp_path / "search"
    interrupted = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        },
        interrupt_on="rank32-attn24",
    )
    common = {
        "base_model": str(model_dir),
        "data_dir": data_dir,
        "output_dir": output_dir,
        "data_manifest_digest": "manifest-a",
        "protocol_versions": {"training": "v2"},
        "train_example_count": 8,
        "batch_size": 1,
        "style_evaluator": lambda _path: {"style_score": 1.0},
        "max_epochs": 1,
    }
    with pytest.raises(KeyboardInterrupt):
        run_lora_candidate_search(interrupted, **common)

    # The manifest is enriched with artifact metadata during a real training
    # run. That mutable copy must not invalidate otherwise identical data.
    (data_dir / "manifest.json").write_text(
        '{"artifact":"resolved-after-start"}',
        encoding="utf-8",
    )

    resumed = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
            "full": 0.7,
        }
    )
    result = run_lora_candidate_search(resumed, **common)

    assert "rank16-attn16" not in {config.adapter_dir.name for config in resumed.configs}
    assert result.best_candidate_id == "rank32-qv-all"
    with pytest.raises(TrainingFingerprintMismatch):
        run_lora_candidate_search(
            resumed,
            **{
                **common,
                "data_manifest_digest": "manifest-b",
            },
        )


def test_search_reuses_valid_final_adapter_after_candidate_artifacts_are_pruned(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    output_dir = tmp_path / "search"
    common = {
        "base_model": str(model_dir),
        "data_dir": data_dir,
        "output_dir": output_dir,
        "data_manifest_digest": "manifest-a",
        "protocol_versions": {"training": "v2"},
        "train_example_count": 8,
        "batch_size": 1,
        "style_evaluator": lambda _path: {"style_score": 1.0},
        "max_epochs": 1,
    }
    first = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        }
    )
    original = run_lora_candidate_search(first, **common)
    resumed = SearchFakeAdapter({})

    reused = run_lora_candidate_search(resumed, **common)

    assert resumed.configs == []
    assert reused.best_checkpoint == original.best_checkpoint


def test_search_continues_after_one_failure_but_fails_when_all_candidates_fail(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.search.lock import AllCandidatesFailedError
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    partial = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.8,
            "rank32-qv-all": 0.7,
            "full": 0.6,
        },
        fail={"rank16-attn16"},
    )
    result = run_lora_candidate_search(
        partial,
        base_model=str(model_dir),
        data_dir=data_dir,
        output_dir=tmp_path / "partial",
        data_manifest_digest="manifest-a",
        protocol_versions={"training": "v2"},
        train_example_count=8,
        batch_size=1,
        style_evaluator=lambda _path: {"style_score": 1.0},
        max_epochs=1,
    )
    assert result.best_candidate_id == "rank32-qv-all"

    failed = SearchFakeAdapter(
        {},
        fail={"rank16-attn16", "rank32-attn24", "rank32-qv-all"},
    )
    with pytest.raises(AllCandidatesFailedError):
        run_lora_candidate_search(
            failed,
            base_model=str(model_dir),
            data_dir=data_dir,
            output_dir=tmp_path / "failed",
            data_manifest_digest="manifest-a",
            protocol_versions={"training": "v2"},
            train_example_count=8,
            batch_size=1,
            style_evaluator=lambda _path: {"style_score": 1.0},
            max_epochs=1,
        )


def test_resume_rejects_same_size_weight_and_train_data_changes(tmp_path: Path) -> None:
    from moonlightbox.training.search.lock import TrainingFingerprintMismatch
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    common = {
        "base_model": str(model_dir),
        "data_dir": data_dir,
        "output_dir": tmp_path / "search",
        "data_manifest_digest": "legacy-value-must-not-be-authoritative",
        "protocol_versions": {"training": "v2"},
        "train_example_count": 8,
        "batch_size": 1,
        "style_evaluator": lambda _path: {"style_score": 1.0},
        "max_epochs": 1,
    }
    interrupted = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        },
        interrupt_on="rank32-attn24",
    )
    with pytest.raises(KeyboardInterrupt):
        run_lora_candidate_search(interrupted, **common)

    weight = model_dir / "model-00001-of-00002.safetensors"
    weight.write_bytes(b"WEIGHT-A")
    with pytest.raises(TrainingFingerprintMismatch):
        run_lora_candidate_search(interrupted, **common)

    weight.write_bytes(b"weight-a")
    (data_dir / "train.jsonl").write_text('{"messages":[1]}\n', encoding="utf-8")
    with pytest.raises(TrainingFingerprintMismatch):
        run_lora_candidate_search(interrupted, **common)


def test_resume_rejects_actual_yaml_configuration_change(tmp_path: Path) -> None:
    from dataclasses import replace

    from moonlightbox.training.peft_adapter import default_lora_candidates
    from moonlightbox.training.search.lock import TrainingFingerprintMismatch
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    candidates = default_lora_candidates()
    common = {
        "base_model": str(model_dir),
        "data_dir": data_dir,
        "output_dir": tmp_path / "search",
        "data_manifest_digest": "manifest-a",
        "protocol_versions": {"training": "v2"},
        "train_example_count": 8,
        "batch_size": 1,
        "style_evaluator": lambda _path: {"style_score": 1.0},
        "max_epochs": 1,
    }
    interrupted = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        },
        interrupt_on="rank32-attn24",
    )
    with pytest.raises(KeyboardInterrupt):
        run_lora_candidate_search(interrupted, candidates=candidates, **common)

    changed = (replace(candidates[0], dropout=0.1), *candidates[1:])
    with pytest.raises(TrainingFingerprintMismatch):
        run_lora_candidate_search(interrupted, candidates=changed, **common)


def test_interrupted_full_run_restarts_clean_without_claiming_optimizer_resume(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    common = {
        "base_model": str(model_dir),
        "data_dir": data_dir,
        "output_dir": tmp_path / "search",
        "data_manifest_digest": "manifest-a",
        "protocol_versions": {"training": "v2"},
        "train_example_count": 8,
        "batch_size": 1,
        "style_evaluator": lambda _path: {"style_score": 1.0},
        "max_epochs": 4,
    }
    interrupted = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        },
        interrupt_on="full",
    )
    with pytest.raises(KeyboardInterrupt):
        run_lora_candidate_search(interrupted, **common)
    full_dir = tmp_path / "search" / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    (full_dir / "partial.bin").write_bytes(b"partial")

    resumed = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        },
        full_curve=(0.8, 0.7),
    )
    result = run_lora_candidate_search(resumed, **common)

    full_config = next(config for config in resumed.configs if config.adapter_dir.name == "full")
    assert full_config.resume_adapter_file is None
    assert not (tmp_path / "search" / "full" / "partial.bin").exists()
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert state["full_training"]["resume_strategy"] == "restart_from_base"
    assert state["full_training"]["optimizer_state_restored"] is False


def test_resume_reuses_terminal_saved_full_checkpoint_after_worker_exit(
    tmp_path: Path,
) -> None:
    """最后权重已保存时，恢复只补 valid 审计，不得重新训练完整 epoch。"""

    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    output_dir = tmp_path / "search"
    common = {
        "base_model": str(model_dir),
        "data_dir": data_dir,
        "output_dir": output_dir,
        "data_manifest_digest": "manifest-a",
        "protocol_versions": {"training": "v2"},
        "train_example_count": 8,
        "batch_size": 1,
        "style_evaluator": lambda _path: {"style_score": 1.0},
        "max_epochs": 1,
    }
    interrupted = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        },
        interrupt_on="full",
    )
    with pytest.raises(KeyboardInterrupt):
        run_lora_candidate_search(interrupted, **common)

    state_path = output_dir / "search-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    full = state["full_training"]
    full_dir = output_dir / "full"
    full_dir.mkdir(parents=True)
    (full_dir / "training.json").write_text(full["normalized_config"], encoding="utf-8")
    (full_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (full_dir / "adapter_model.safetensors").write_bytes(b"terminal")
    (full_dir / "0000008_adapter_model.safetensors").write_bytes(b"terminal")
    full["latest_progress"] = {
        "stage": "saved",
        "iteration": 8,
        "total_iterations": 8,
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    resumed = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        }
    )
    result = run_lora_candidate_search(resumed, **common)

    assert not any(config.adapter_dir.name == "full" for config in resumed.configs)
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert state["full_training"]["resume_strategy"] == "reuse_saved_completed_run"
    assert state["full_training"]["completed_steps"] == 8


def test_migrates_only_legacy_checkpoint_preservation_fingerprint(tmp_path: Path) -> None:
    from moonlightbox.training.model_identity import _canonical_digest
    from moonlightbox.training.search.state import (
        _migrate_legacy_full_checkpoint_preservation_fingerprint,
    )

    full_config = {
        "adapter_dir": "models/project/job/full",
        "preserve_checkpoints": True,
        "rank": 16,
    }
    current_config = json.dumps(full_config, ensure_ascii=False, indent=2, sort_keys=True)
    current_normalized = {
        "compact": {"best_candidate_full_run": current_config},
    }
    current_payload = {
        "normalized_config": current_normalized,
        "config_fingerprint": _canonical_digest(current_normalized),
    }
    legacy_config = dict(full_config)
    legacy_config["preserve_checkpoints"] = False
    legacy_normalized = {
        "compact": {
            "best_candidate_full_run": json.dumps(
                legacy_config,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        }
    }
    legacy_payload = {
        "normalized_config": legacy_normalized,
        "config_fingerprint": _canonical_digest(legacy_normalized),
    }
    state_path = tmp_path / "search-state.json"
    state_path.write_text(
        json.dumps(
            {
                "fingerprint": _canonical_digest(legacy_payload),
                "fingerprint_payload": legacy_payload,
                "best_candidate_id": "compact",
                "full_training": {
                    "normalized_config": legacy_normalized["compact"]["best_candidate_full_run"],
                },
            }
        ),
        encoding="utf-8",
    )

    _migrate_legacy_full_checkpoint_preservation_fingerprint(
        state_path,
        fingerprint=_canonical_digest(current_payload),
        fingerprint_payload=current_payload,
    )

    migrated = json.loads(state_path.read_text(encoding="utf-8"))
    assert migrated["fingerprint"] == _canonical_digest(current_payload)
    assert migrated["full_training"]["normalized_config"] == current_config


def test_checkpoint_style_selection_resumes_from_persisted_audits(tmp_path: Path) -> None:
    from moonlightbox.training.search.runner import _select_full_checkpoint_by_style

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    (full_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (full_dir / "README.md").write_text("adapter", encoding="utf-8")
    for iteration in (1, 2):
        (full_dir / f"{iteration:07d}_adapter_model.safetensors").write_bytes(
            str(iteration).encode("utf-8")
        )
    first_checkpoint = full_dir / "0000001_adapter_model.safetensors"
    fallback = BestCheckpointSelection(0.8, 2, 1, first_checkpoint)
    persisted: list[list[dict[str, object]]] = []

    def interrupted_evaluator(path: Path) -> dict[str, float]:
        iteration = int(path.name.rsplit("-", 1)[1])
        if iteration == 2:
            raise KeyboardInterrupt
        return {
            "semantic_passed": 1.0,
            "composite_fidelity": 0.9,
            "speaker_probability": 0.9,
            "style_distance": 0.1,
        }

    with pytest.raises(KeyboardInterrupt):
        _select_full_checkpoint_by_style(
            output_dir=tmp_path,
            full_dir=full_dir,
            validation_curve=[
                {"iteration": 2, "validation_loss": 0.8},
                {"iteration": 3, "validation_loss": 0.7},
            ],
            fallback=fallback,
            style_evaluator=interrupted_evaluator,
            on_audit=lambda audits: persisted.append(audits),
        )

    evaluated: list[int] = []

    def resumed_evaluator(path: Path) -> dict[str, float]:
        iteration = int(path.name.rsplit("-", 1)[1])
        evaluated.append(iteration)
        return {
            "semantic_passed": 1.0,
            "composite_fidelity": 0.8,
            "speaker_probability": 0.8,
            "style_distance": 0.2,
        }

    selected, audits = _select_full_checkpoint_by_style(
        output_dir=tmp_path,
        full_dir=full_dir,
        validation_curve=[
            {"iteration": 2, "validation_loss": 0.8},
            {"iteration": 3, "validation_loss": 0.7},
        ],
        fallback=fallback,
        style_evaluator=resumed_evaluator,
        existing_audits=persisted[-1],
    )

    assert evaluated == [2]
    assert selected.checkpoint_iteration == 1
    assert [audit["checkpoint_iteration"] for audit in audits] == [1, 2]


def test_training_metadata_records_environment_and_file_digests(tmp_path: Path) -> None:
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    adapter = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        }
    )
    result = run_lora_candidate_search(
        adapter,
        base_model=str(model_dir),
        data_dir=data_dir,
        output_dir=tmp_path / "search",
        data_manifest_digest="manifest-a",
        protocol_versions={"training": "v2"},
        train_example_count=8,
        batch_size=1,
        style_evaluator=lambda _path: {"style_score": 1.0},
        max_epochs=1,
    )

    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    environment = state["environment"]
    assert environment["python_version"]
    assert environment["platform"]
    assert environment["peft_version"] == "0.20.0"
    model_files = state["fingerprint_payload"]["base_model"]["files"]
    assert all(file["sha256"] for file in model_files)
    data_files = state["fingerprint_payload"]["data_files"]
    assert {file["name"] for file in data_files} == {
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
    }
    assert state["fingerprint_payload"]["normalized_config"]
    assert state["fingerprint_payload"]["config_fingerprint"]


def test_unpinned_remote_model_identifier_fails_closed(tmp_path: Path) -> None:
    from moonlightbox.training.search.lock import TrainingFingerprintMismatch
    from moonlightbox.training.search.runner import run_lora_candidate_search

    _model_dir, data_dir = _training_paths(tmp_path)
    adapter = SearchFakeAdapter({})

    with pytest.raises(TrainingFingerprintMismatch, match="commit"):
        run_lora_candidate_search(
            adapter,
            base_model="mlx-community/Qwen3-8B-4bit",
            data_dir=data_dir,
            output_dir=tmp_path / "search",
            data_manifest_digest="manifest-a",
            protocol_versions={"training": "v2"},
            train_example_count=8,
            batch_size=1,
            style_evaluator=lambda _path: {"style_score": 1.0},
            max_epochs=1,
        )


def test_candidate_ranking_ignores_unpublishable_iter_one_baseline(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    adapter = SearchFakeAdapter(
        {
            "rank16-attn16": (0.01, 0.9),
            "rank32-attn24": (0.5, 0.8),
            "rank32-qv-all": (0.4, 1.0),
        },
        full_curve=(0.9, 0.8),
    )

    result = run_lora_candidate_search(
        adapter,
        base_model=str(model_dir),
        data_dir=data_dir,
        output_dir=tmp_path / "search",
        data_manifest_digest="manifest-a",
        protocol_versions={"training": "v2"},
        train_example_count=8,
        batch_size=1,
        style_evaluator=lambda _path: {"style_score": 1.0},
        max_epochs=1,
        early_stopping_patience=0,
    )

    assert result.best_candidate_id == "rank32-attn24"
    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    ranked = state["search_results"]
    assert ranked[0]["metrics"]["validation_loss"] == 0.8


def test_candidate_ranking_requires_semantic_pass_before_style_and_loss() -> None:
    from moonlightbox.training.search.runner import _candidate_sort_key

    semantic_failure = {
        "candidate_id": "broken",
        "metrics": {
            "semantic_passed": 0.0,
            "composite_fidelity": 0.99,
            "style_distance": 0.01,
            "validation_loss": 0.1,
        },
    }
    valid_candidate = {
        "candidate_id": "valid",
        "metrics": {
            "semantic_passed": 1.0,
            "composite_fidelity": 0.8,
            "style_distance": 0.2,
            "validation_loss": 0.8,
        },
    }

    assert (
        sorted(
            [semantic_failure, valid_candidate],
            key=_candidate_sort_key,
        )[0]["candidate_id"]
        == "valid"
    )


def test_candidate_ranking_rejects_all_semantic_failures(tmp_path: Path) -> None:
    from moonlightbox.training.search.lock import AllCandidatesFailedError
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    adapter = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.8,
            "rank32-qv-all": 0.7,
        }
    )

    with pytest.raises(AllCandidatesFailedError, match="语义"):
        run_lora_candidate_search(
            adapter,
            base_model=str(model_dir),
            data_dir=data_dir,
            output_dir=tmp_path / "search",
            data_manifest_digest="manifest-a",
            protocol_versions={"training": "v2"},
            train_example_count=8,
            batch_size=1,
            style_evaluator=lambda _path: {
                "semantic_passed": 0.0,
                "composite_fidelity": 0.9,
                "style_distance": 0.1,
            },
            max_epochs=1,
        )


def test_explicit_single_candidate_fallback_only_selects_by_validation_loss(
    tmp_path: Path,
) -> None:
    """受限设备的唯一候选可进入全量训练，但不会把短跑当作发布验收。"""
    from moonlightbox.training.peft_adapter import default_lora_candidates
    from moonlightbox.training.search.runner import run_lora_candidate_search

    model_dir, data_dir = _training_paths(tmp_path)
    candidate = default_lora_candidates()[0]
    adapter = SearchFakeAdapter({candidate.candidate_id: 0.8})

    result = run_lora_candidate_search(
        adapter,
        base_model=str(model_dir),
        data_dir=data_dir,
        output_dir=tmp_path / "single-candidate-search",
        data_manifest_digest="manifest-a",
        protocol_versions={"training": "v2"},
        train_example_count=8,
        batch_size=1,
        style_evaluator=lambda _path: {
            "semantic_passed": 0.0,
            "composite_fidelity": 0.4,
            "style_distance": 0.6,
        },
        max_epochs=1,
        candidates=(candidate,),
        allow_single_candidate_semantic_fallback=True,
    )

    state = json.loads(result.state_path.read_text(encoding="utf-8"))
    assert result.best_candidate_id == candidate.candidate_id
    assert state["candidate_selection"] == {
        "mode": "single_candidate_validation_loss_fallback",
        "short_run_semantic_gate_passed": False,
        "final_model_acceptance_required": True,
    }
    assert state["full_training"]["status"] == "succeeded"
