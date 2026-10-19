"""The directory sandboxes of ``sandboxes.py`` answer the job code as ``SandboxSession`` on podman answers.

Each test runs on the fake, as a medium test, and on ``SandboxSession`` with the podman backend, as a
large test marked ``slow``. The real side starts its sandboxes from ``IMAGE`` on the store of the large
tests, skips when ``live_podman_problem`` finds a reason or the runtime binary is missing, and removes
each sandbox and snapshot that it makes. Run it alone: ``pytest tests/worker/jobs/test_sandboxes.py -m slow``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.sandbox.commands import shell_outcome
from sprout.worker.sandbox.session import SandboxSession
from sprout_sandbox.btrfs_store import BtrfsStore
from tests.support.paths import RUNTIME_BIN
from tests.worker.jobs.sandboxes import DirectorySandboxes

IMAGE = "docker.io/library/python:3.11-slim"
NO_RUNTIME = "needs runtime/sprout-runtime (cd runtime && go generate ./... && go build -o sprout-runtime .)"
#: Bytes that are not UTF-8 text, with a NUL and a CR, which a text channel would change.
PAYLOAD = b"caf\xe9\x00\r\n" * 1000


class Side:
    """The sessions of one test on one side, and the snapshots that it takes. ``close`` removes them."""

    def __init__(self, new_session: Callable[[], Any], *, store: Path | None) -> None:
        self._new_session = new_session
        self._store = store
        self._sessions: list[Any] = []
        self.snapshots: list[str] = []

    def session(self) -> Any:
        """A new session without a sandbox."""
        session = self._new_session()
        self._sessions.append(session)
        return session

    def started(self) -> Any:
        """A new session with a sandbox from ``IMAGE``."""
        session = self.session()
        assert session.create(IMAGE), session.create_error
        return session

    def close(self) -> None:
        for session in self._sessions:
            session.destroy()
        for snapshot in self.snapshots if self._store is not None else []:
            asyncio.run(BtrfsStore(self._store).delete_snapshot(snapshot))


@pytest.fixture(params=["fake", pytest.param("real", marks=pytest.mark.slow)])
def side(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Side]:
    """The directory sandboxes, or ``SandboxSession`` on podman, for one test."""
    if request.param == "fake":
        (tmp_path / "image" / "tmp").mkdir(parents=True)
        sandboxes = DirectorySandboxes(tmp_path / "sandboxes", images={IMAGE: tmp_path / "image"})
        chosen = Side(lambda: sandboxes(quiet=True, backend={}), store=None)
    else:
        store = request.getfixturevalue("live_podman_store")
        if not RUNTIME_BIN.exists():
            pytest.skip(NO_RUNTIME)
        backend = {"backend": "podman", "podman": {"store": str(store), "runtime_bin": str(RUNTIME_BIN)}}
        chosen = Side(lambda: SandboxSession(quiet=True, backend=backend), store=store)
    try:
        yield chosen
    finally:
        chosen.close()


def test_a_command_reports_its_exit_code_stdout_and_stderr(side):
    session = side.started()

    outcome = shell_outcome(session, "echo out; echo err >&2; exit 3")

    assert outcome == ("out\n", "err\n", 3, False)


def test_a_plain_success_is_bare_stdout(side):
    session = side.started()

    result = session.execute("shell", {"command": "echo hello"})

    assert (result.success, result.output, result.outcome) == (True, "hello\n", None)


def test_a_command_past_its_timeout_is_killed_and_reported_as_timed_out(side):
    session = side.started()

    _, _, code, timed_out = shell_outcome(session, "sleep 30", timeout=1)

    assert (code, timed_out) == (137, True)


def test_a_command_longer_than_one_argument_may_be_does_not_run_and_exits_1(side):
    """Linux caps one argument at 128 KiB (MAX_ARG_STRLEN), and the runtime passes a command as one."""
    session = side.started()

    outcome = shell_outcome(session, "echo " + "x" * 200_000)

    assert outcome == ("", "", 1, False)


def test_a_command_runs_in_the_working_directory_that_it_names(side):
    session = side.started()

    result = session.execute("shell", {"command": "pwd -P", "working_dir": "/tmp"})

    assert result.output == "/tmp\n"


def test_a_file_moves_into_the_sandbox_and_out_byte_for_byte(side, tmp_path):
    (tmp_path / "upload.bin").write_bytes(PAYLOAD)
    session = side.started()

    uploaded = session.upload_file(tmp_path / "upload.bin", "/tmp/moved.bin")
    seen = shell_outcome(session, "wc -c < /tmp/moved.bin")[0]
    downloaded = session.download_file("/tmp/moved.bin", tmp_path / "download.bin")

    assert (uploaded, seen.strip(), downloaded) == (True, str(len(PAYLOAD)), True)
    assert (tmp_path / "download.bin").read_bytes() == PAYLOAD


def test_a_file_that_the_sandbox_lacks_does_not_download_and_the_session_says_why(side, tmp_path):
    session = side.started()

    downloaded = session.download_file("/tmp/absent.txt", tmp_path / "absent.txt")

    assert (downloaded, session.transfer_error.split(":")[0]) == (False, "PodmanError")


def test_a_file_does_not_upload_into_a_directory_that_the_sandbox_lacks(side, tmp_path):
    (tmp_path / "upload.txt").write_text("text\n")
    session = side.started()

    uploaded = session.upload_file(tmp_path / "upload.txt", "/tmp/absent/upload.txt")

    assert uploaded is False


def test_a_second_session_attaches_to_a_running_sandbox_and_sees_its_files(side):
    first = side.started()
    shell_outcome(first, "echo kept > /tmp/marker")
    second = side.session()

    attached = second.attach(first.sandbox_id)

    assert (attached, shell_outcome(second, "cat /tmp/marker")) == (True, ("kept\n", "", 0, False))


def test_a_destroyed_sandbox_does_not_answer_an_attach(side):
    first = side.started()
    sandbox_id = first.sandbox_id
    first.destroy()
    second = side.session()

    attached = second.attach(sandbox_id)

    assert (attached, second.create_error) == (False, "sandbox %s does not answer" % sandbox_id)


def test_a_sandbox_from_a_snapshot_has_the_files_of_its_source(side):
    source = side.started()
    shell_outcome(source, "echo saved > /tmp/marker")
    snapshot = source.snapshot(disk_only=True)
    side.snapshots.append(snapshot.id)
    restored = side.session()

    created = restored.create(snapshot.id)

    assert (created, shell_outcome(restored, "cat /tmp/marker")) == (True, ("saved\n", "", 0, False))
