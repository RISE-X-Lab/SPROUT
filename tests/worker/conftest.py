"""Fixtures of the worker tests: directory sandboxes for each run."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.worker.jobs.sandboxes import DirectorySandboxes


@pytest.fixture
def run_sandboxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DirectorySandboxes:
    """Directory sandboxes in the place of ``SandboxSession`` for each run, from the empty image ``base``."""
    (tmp_path / "images" / "base" / "tmp").mkdir(parents=True)
    sandboxes = DirectorySandboxes(tmp_path / "sandboxes", images={"base": tmp_path / "images" / "base"})
    # The runner makes its session where it needs one, from this module attribute.
    monkeypatch.setattr("sprout.worker.sandbox.session.SandboxSession", sandboxes)
    return sandboxes
