import os
from dataclasses import replace
from pathlib import Path

import pytest
from moonlightbox.training.peft_adapter import default_lora_candidates


def test_search_output_lock_is_exclusive_and_records_owner(tmp_path: Path) -> None:
    from moonlightbox.training.jobs import SearchLockTimeout, SearchOutputLock

    output_dir = tmp_path / "search"
    with SearchOutputLock(output_dir, timeout_seconds=0.1) as owner:
        assert owner["pid"]
        assert owner["run_generation"]
        with pytest.raises(SearchLockTimeout):
            with SearchOutputLock(output_dir, timeout_seconds=0.02):
                pass


def test_state_store_rejects_stale_fencing_generation(tmp_path: Path) -> None:
    from moonlightbox.training.jobs import FencingTokenError, SearchStateStore

    path = tmp_path / "search-state.json"
    current = SearchStateStore(path, run_generation="generation-new")
    current.write({"stage": "running"})
    stale = SearchStateStore(path, run_generation="generation-old")

    with pytest.raises(FencingTokenError):
        stale.write({"stage": "stale"})


def test_candidate_id_rejects_path_traversal(tmp_path: Path) -> None:
    from moonlightbox.training.jobs import run_lora_candidate_search

    candidate = replace(default_lora_candidates()[0], candidate_id="../escape")

    with pytest.raises(ValueError, match="candidate_id"):
        run_lora_candidate_search(
            object(),  # type: ignore[arg-type]
            base_model=str(tmp_path / "model"),
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "search",
            data_manifest_digest="digest",
            protocol_versions={"training": "v2"},
            train_example_count=1,
            batch_size=1,
            style_evaluator=lambda _path: {"style_score": 1.0},
            max_epochs=1,
            candidates=(candidate,),
        )


def test_duplicate_candidate_ids_are_rejected_before_search(tmp_path: Path) -> None:
    from moonlightbox.training.jobs import run_lora_candidate_search

    first = default_lora_candidates()[0]
    duplicate = replace(
        default_lora_candidates()[1],
        candidate_id=first.candidate_id,
    )

    with pytest.raises(ValueError, match="重复 candidate_id"):
        run_lora_candidate_search(
            object(),  # type: ignore[arg-type]
            base_model=str(tmp_path / "model"),
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "search",
            data_manifest_digest="digest",
            protocol_versions={"training": "v2"},
            train_example_count=1,
            batch_size=1,
            style_evaluator=lambda _path: {"style_score": 1.0},
            max_epochs=1,
            candidates=(first, duplicate),
        )


def test_local_model_digest_cache_reuses_and_invalidates_hashes(tmp_path: Path) -> None:
    from moonlightbox.training.jobs import build_local_model_identity

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"first")
    cache_path = tmp_path / "digest-cache.json"
    calls: list[Path] = []

    def hash_file(path: Path) -> str:
        calls.append(path)
        return f"digest-{path.read_bytes().decode()}"

    first = build_local_model_identity(
        model_dir,
        cache_path=cache_path,
        hash_file=hash_file,
    )
    second = build_local_model_identity(
        model_dir,
        cache_path=cache_path,
        hash_file=hash_file,
    )
    assert first == second
    assert calls == [weight]

    weight.write_bytes(b"second")
    third = build_local_model_identity(
        model_dir,
        cache_path=cache_path,
        hash_file=hash_file,
    )
    assert calls == [weight, weight]
    assert third["digest"] != first["digest"]
    assert third["files"][0]["sha256"] == "digest-second"


def test_digest_cache_invalidates_same_size_rewrite_with_restored_mtime(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.jobs import build_local_model_identity

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"AAAA")
    original = weight.stat()
    calls = 0

    def hash_file(path: Path) -> str:
        nonlocal calls
        calls += 1
        return path.read_bytes().hex()

    cache_path = tmp_path / "digest-cache.json"
    first = build_local_model_identity(
        model_dir,
        cache_path=cache_path,
        hash_file=hash_file,
    )
    weight.write_bytes(b"BBBB")
    os.utime(weight, ns=(original.st_atime_ns, original.st_mtime_ns))
    second = build_local_model_identity(
        model_dir,
        cache_path=cache_path,
        hash_file=hash_file,
    )

    assert calls == 2
    assert second["digest"] != first["digest"]


def test_unexpected_candidate_error_aborts_instead_of_being_isolated(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.jobs import run_lora_candidate_search
    from test_lora_candidate_search import SearchFakeAdapter, _training_paths

    class BrokenAdapter(SearchFakeAdapter):
        def train(self, config: object, on_progress: object = None) -> object:
            raise RuntimeError("代码错误")

    model_dir, data_dir = _training_paths(tmp_path)
    adapter = BrokenAdapter({})
    with pytest.raises(RuntimeError, match="代码错误"):
        run_lora_candidate_search(
            adapter,  # type: ignore[arg-type]
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


def test_tampered_successful_candidate_is_quarantined_and_retrained(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.jobs import run_lora_candidate_search
    from test_lora_candidate_search import SearchFakeAdapter, _training_paths

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
        interrupt_on="rank32-attn24",
    )
    with pytest.raises(KeyboardInterrupt):
        run_lora_candidate_search(interrupted, **common)

    checkpoint = (
        output_dir
        / "candidates"
        / "rank16-attn16"
        / "adapter_model.safetensors"
    )
    checkpoint.write_bytes(b"half-written")
    resumed = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        }
    )
    run_lora_candidate_search(resumed, **common)

    assert "rank16-attn16" in {
        config.adapter_dir.name for config in resumed.configs
    }


def test_tampered_final_adapter_retrains_after_bounded_cleanup(tmp_path: Path) -> None:
    from moonlightbox.training.jobs import run_lora_candidate_search
    from test_lora_candidate_search import SearchFakeAdapter, _training_paths

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
    result = run_lora_candidate_search(first, **common)
    (result.adapter_dir / "adapter_model.safetensors").write_bytes(b"tampered")

    resumed = SearchFakeAdapter(
        {
            "rank16-attn16": 1.0,
            "rank32-attn24": 0.9,
            "rank32-qv-all": 0.8,
        }
    )
    run_lora_candidate_search(resumed, **common)

    assert [config.adapter_dir.name for config in resumed.configs] == [
        "rank16-attn16",
        "rank32-attn24",
        "rank32-qv-all",
        "full",
    ]


def test_full_training_progress_uses_streamed_iteration() -> None:
    from moonlightbox.training.jobs import _search_job_progress

    state = {
        "stage": "full_training",
        "full_training": {
            "latest_progress": {
                "iteration": 672,
                "total_iterations": 1344,
            }
        },
    }

    assert _search_job_progress(state, maximum_iterations=1344) == pytest.approx(0.575)
    state["full_training"]["latest_progress"]["iteration"] = 1344
    assert _search_job_progress(state, maximum_iterations=1344) == pytest.approx(0.75)


def test_checkpoint_cleanup_is_scoped_and_strict(tmp_path: Path) -> None:
    from moonlightbox.training.peft_adapter import cleanup_generated_checkpoints

    output_dir = tmp_path / "job-output"
    full_dir = output_dir / "full"
    full_dir.mkdir(parents=True)
    generated = full_dir / "0000042_adapter_model.safetensors"
    latest = full_dir / "adapter_model.safetensors"
    near_match = full_dir / "42_adapter_model.safetensors"
    outside = tmp_path / "0000042_adapter_model.safetensors"
    for path in (generated, latest, near_match, outside):
        path.write_bytes(path.name.encode())

    cleanup_generated_checkpoints(output_dir, include_latest=True)

    assert not generated.exists()
    assert not latest.exists()
    assert near_match.exists()
    assert outside.exists()
