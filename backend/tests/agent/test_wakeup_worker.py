from pathlib import Path

from moonlightbox.db import Database


def test_only_cognition_and_all_roles_scan_due_wakeups() -> None:
    from moonlightbox.worker_main import role_scans_due_wakeups

    assert role_scans_due_wakeups("cognition") is True
    assert role_scans_due_wakeups("all") is True
    assert role_scans_due_wakeups("realtime") is False
    assert role_scans_due_wakeups("background") is False


def test_wakeup_scan_exception_is_isolated_from_worker_loop(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    import moonlightbox.worker_main as worker_main

    database = Database(f"sqlite:///{tmp_path / 'scan-error.db'}")

    def fail_scan(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("模拟扫描异常")

    monkeypatch.setattr(worker_main, "process_due_wakeups", fail_scan)

    assert worker_main.scan_due_wakeups(database) == 0
    database.close()


def test_interrupted_job_scan_exception_is_isolated_from_worker_loop(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    import moonlightbox.worker_main as worker_main

    database = Database(f"sqlite:///{tmp_path / 'interrupted-scan-error.db'}")

    def fail_recovery(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("模拟恢复异常")

    monkeypatch.setattr(worker_main, "recover_interrupted_jobs", fail_recovery)

    assert worker_main.scan_interrupted_jobs(database) == 0
    database.close()
