from pathlib import Path

from moonlightbox.jobs.models import Job
from moonlightbox.jobs.registry import JobHandler
from moonlightbox.jobs.service import JobService
from moonlightbox.training.mlx_adapter import MlxLmAdapter, MlxLoraConfig


def create_mlx_training_handler(adapter: MlxLmAdapter) -> JobHandler:
    def handle(service: JobService, job: Job) -> None:
        payload = job.payload
        config = MlxLoraConfig(
            model=str(payload["model"]),
            data_dir=Path(str(payload["data_dir"])),
            adapter_dir=Path(str(payload["adapter_dir"])),
            iterations=int(payload.get("iterations", 600)),
            batch_size=int(payload.get("batch_size", 1)),
            learning_rate=float(payload.get("learning_rate", 1e-5)),
        )

        def save_progress(progress: dict[str, object]) -> None:
            service.checkpoint(job.id, progress)

        adapter.train(config, save_progress)

    return handle
