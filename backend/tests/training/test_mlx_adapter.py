from pathlib import Path


class FakeRunner:
    def __init__(self) -> None:
        self.command: list[str] = []

    def run(self, command: list[str], on_line: object) -> int:
        self.command = command
        on_line("Iter 10: Train loss 1.23")
        on_line("Saved adapter weights")
        return 0


def test_mlx_adapter_builds_official_lora_command_and_reports_progress(
    tmp_path: Path,
) -> None:
    from moonlightbox.training.mlx_adapter import MlxLmAdapter, MlxLoraConfig

    runner = FakeRunner()
    progress: list[dict[str, object]] = []
    adapter = MlxLmAdapter(runner, check_environment=False)
    config = MlxLoraConfig(
        model="mlx-community/Qwen2.5-7B-Instruct-4bit",
        data_dir=tmp_path / "dataset",
        adapter_dir=tmp_path / "adapter",
        iterations=100,
    )

    result = adapter.train(config, progress.append)

    assert runner.command[:3] == ["python", "-m", "mlx_lm.lora"]
    assert "--model" in runner.command
    assert str(config.adapter_dir) in runner.command
    assert progress[0]["iteration"] == 10
    assert result.adapter_dir == config.adapter_dir
