from pathlib import Path


def test_private_paths_are_ignored() -> None:
    rules = Path(".gitignore").read_text(encoding="utf-8")

    for path in (
        "wxecho/",
        ".superpowers/",
        ".worktrees/",
        "data/",
        "models/",
        ".env",
    ):
        assert path in rules
