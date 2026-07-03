from __future__ import annotations

from pathlib import Path

from app.bootstrap import build_container
from app.core.config import Settings


def _settings(root: Path) -> Settings:
    return Settings(
        data_dir=root,
        image_provider="mock",
        brain_provider="local",
        rate_limit_per_second=0,
    )


def test_build_container_creates_isolated_application_scopes(
    tmp_path: Path,
) -> None:
    first = build_container(_settings(tmp_path / "first"))
    second = build_container(_settings(tmp_path / "second"))

    assert first is not second
    assert first.storage is not second.storage
    assert first.jobs is not second.jobs
    assert first.task_queue is not second.task_queue
    assert first.job_dispatcher is not second.job_dispatcher
    assert first.storage.config.data_dir == tmp_path / "first"
    assert second.storage.config.data_dir == tmp_path / "second"
