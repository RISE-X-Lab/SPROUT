"""The sandboxes of the tests of whole runs: directories in the place of ``SandboxSession``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.worker.run.directory_session import DirectoryBackend, DirectorySession

#: Where the runner finds the class of the session that it creates.
SESSION_CLASS = "sprout.worker.sandbox.session.SandboxSession"


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DirectorySession:
    """The session that each run of the test creates, on the directory ``tmp_path/sandbox``."""
    session = DirectorySession(tmp_path / "sandbox")
    monkeypatch.setattr(SESSION_CLASS, session.open)
    return session


@pytest.fixture
def sandboxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DirectoryBackend:
    """A session of its own for each run of the test, under ``tmp_path/sandboxes``, with shared snapshots."""
    backend = DirectoryBackend(tmp_path / "sandboxes")
    monkeypatch.setattr(SESSION_CLASS, backend.open)
    return backend
