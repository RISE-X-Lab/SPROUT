"""The directory sandbox of the run tests answers as a ``SandboxSession`` of the podman backend answers.

Each case runs on ``DirectorySession`` (small) and on a real ``SandboxSession`` (large, marked ``slow``).
The large side skips without podman, a btrfs store or the runtime binary, and while the store is busy.
It removes each sandbox and snapshot that it made. Run it alone:
``pytest tests/worker/run/test_directory_session.py -m slow``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.sandbox.session import SandboxSession
from sprout_sandbox.btrfs_store import BtrfsStore
from tests.support.paths import RUNTIME_BIN
from tests.worker.run.directory_session import DirectoryBackend

IMAGE = "docker.io/library/python:3.11-slim"
NO_RUNTIME = "needs runtime/sprout-runtime: cd runtime && go generate ./... && go build -o sprout-runtime ."


class DirectorySide:
    """Sessions of the directory backend."""

    def __init__(self, root: Path) -> None:
        self._backend = DirectoryBackend(root)

    def open(self, image: str) -> Any:
        session = self._backend.open(backend={}, quiet=True)
        assert session.create(image), session.create_error
        return session

    def close(self) -> None:
        """Nothing to remove: the directories are in ``tmp_path``."""


class PodmanSide:
    """Sessions of the podman backend on the store of the large tests."""

    def __init__(self, store: Path) -> None:
        self._store = store
        self._backend = {
            "backend": "podman",
            "podman": {"store": str(store), "runtime_bin": str(RUNTIME_BIN)},
        }
        self._sessions: list[SandboxSession] = []
        self._sandbox_ids: list[str] = []

    def open(self, image: str) -> Any:
        session = SandboxSession(backend=self._backend, quiet=True)
        self._sessions.append(session)
        assert session.create(image), session.create_error
        self._sandbox_ids.append(session.sandbox_id)
        return session

    def close(self) -> None:
        """Destroy each sandbox, and delete each snapshot of them: its id starts with the sandbox id."""
        for session in self._sessions:
            session.destroy()
        store = BtrfsStore(self._store)
        for path in sorted((self._store / "snapshots").glob("*")):
            if path.is_dir() and path.name.startswith(tuple(sandbox + "-" for sandbox in self._sandbox_ids)):
                asyncio.run(store.delete_snapshot(path.name))


@pytest.fixture(params=["directory", pytest.param("podman", marks=pytest.mark.slow)])
def side(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[DirectorySide | PodmanSide]:
    """The directory backend, or the podman backend of this host, with what it made removed afterwards."""
    if request.param == "directory":
        opened: DirectorySide | PodmanSide = DirectorySide(tmp_path / "sandboxes")
    else:
        store = request.getfixturevalue("live_podman_store")
        if not RUNTIME_BIN.exists():
            pytest.skip(NO_RUNTIME)
        opened = PodmanSide(store)
    try:
        yield opened
    finally:
        opened.close()


def test_a_command_that_exits_0_without_stderr_gives_its_bare_stdout(side):
    session = side.open(IMAGE)

    result = session.call("shell", {"command": "printf ok", "timeout": 30})

    assert (result.success, result.output, result.outcome) == (True, "ok", None)


def test_a_command_that_fails_gives_its_exit_code_and_both_streams(side):
    session = side.open(IMAGE)

    result = session.call("shell", {"command": "printf out; printf err >&2; exit 3", "timeout": 30})

    outcome = result.outcome
    assert (result.success, outcome.exit_code, outcome.stdout, outcome.stderr) == (False, 3, "out", "err")


def test_a_sandbox_created_from_a_snapshot_starts_with_its_files(side):
    session = side.open(IMAGE)
    session.call("shell", {"command": "printf kept > marker", "timeout": 30})
    snapshot = session.snapshot(disk_only=True)

    restored = side.open(snapshot.id)

    result = restored.call("shell", {"command": "cat marker", "timeout": 30})
    assert (snapshot.rootfs_layers, result.output) == (None, "kept")


def test_a_destroyed_session_names_no_sandbox(side):
    session = side.open(IMAGE)

    session.destroy()

    assert session.sandbox_id == "unknown"
