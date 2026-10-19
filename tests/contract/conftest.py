"""The fake sandbox backend of the tests that run an agent through the worker (``fake_sandbox.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.contract.fake_sandbox import FakeSandboxes


@pytest.fixture
def sandboxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeSandboxes:
    """The fake backend, in the place of ``SandboxSession`` where the runner makes each session."""
    fake = FakeSandboxes(tmp_path / "sandboxes")
    monkeypatch.setattr("sprout.worker.sandbox.session.SandboxSession", fake.session)
    return fake
