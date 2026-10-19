"""A stand-in for ``SandboxSession`` whose sandbox is a directory of this host, for tests of whole runs.

The runner creates its session from ``sprout.worker.sandbox.session.SandboxSession``, so a test puts
``DirectorySession.open`` there. The directory is the sandbox's ``/testbed``, its default working
directory. Each command of the agent really runs in it, with bash, and its result has the shape that
the runtime gives: bare stdout for a command that exits 0 and writes no stderr, else an outcome. A
snapshot copies the files of the directory. Commands that name another absolute path would reach the
host, so a test sends only commands that stay in ``/testbed``. ``DirectoryBackend`` gives each run a
session of its own, and a sandbox that it creates from a snapshot id starts with that snapshot's files.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sprout.worker.tools.result import CommandOutcome, ToolResult
from sprout_sandbox.pool import Snapshot

#: The sandbox path of the directory.
WORKDIR = "/testbed"


class DirectorySession:
    """One sandbox: a directory, its snapshots and its lifecycle, recorded for the test to read.

    The knobs ``create_error``, ``destroy_error``, ``can_snapshot``, ``failing_snapshots``, ``compacting``
    and ``unsettled`` put the sandbox in the state that a test needs. A test sets them before the run."""

    def __init__(
        self,
        root: Path,
        *,
        sandbox_id: str = "sandbox-1",
        snapshots: dict[str, dict[str, bytes]] | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        #: Why ``create`` fails, or empty when it succeeds.
        self.create_error = ""
        #: Why ``destroy`` raises, or empty when it succeeds.
        self.destroy_error = ""
        #: Whether the backend of the sandbox can snapshot at all.
        self.can_snapshot = True
        #: Whether each capture fails, as a backend that can snapshot fails one.
        self.failing_snapshots = False
        #: Whether each capture reports the same layer count, as a chain that the server compacts.
        #: A re-board then moves the session to a sandbox of the next id.
        self.compacting = False
        #: A text whose agent command fails in transport, so that nobody knows whether it ran.
        self.unsettled: str | None = None
        #: The ``backend`` and the ``quiet`` that the runner made the session with.
        self.opened_with: dict[str, Any] | None = None
        #: The image that ``create`` was called with, or None before.
        self.image: str | None = None
        #: ``create``, ``snapshot``, ``reboard`` and ``destroy``, in call order, each with its subject.
        self.lifecycle: list[tuple[str, str]] = []
        #: The files of each snapshot by snapshot id: each path under the directory and its bytes.
        self.snapshots: dict[str, dict[str, bytes]] = {} if snapshots is None else snapshots
        self.on_swap: list[Any] = []
        self._sandbox_id = sandbox_id
        self._destroyed = False

    def open(self, *, backend: dict[str, Any], quiet: bool = False) -> DirectorySession:
        """This session, as the constructor of ``SandboxSession`` returns a new one."""
        self.opened_with = {"backend": backend, "quiet": quiet}
        return self

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    @property
    def sandbox_id(self) -> str:
        # A destroyed SandboxSession names no sandbox.
        return "unknown" if self._destroyed else self._sandbox_id

    def create(self, image: str, resources: dict[str, Any] | None = None) -> bool:
        self.image = image
        self.lifecycle.append(("create", image))
        for name, data in self.snapshots.get(image, {}).items():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_bytes(data)
        return not self.create_error

    def supports_snapshot(self) -> bool:
        return self.can_snapshot

    def snapshot(self, name: str | None = None, disk_only: bool = True) -> Snapshot | None:
        if not self.can_snapshot or self.failing_snapshots:
            return None
        snapshot_id = "snapshot-%d" % (len(self.snapshots) + 1)
        self.snapshots[snapshot_id] = {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.lifecycle.append(("snapshot", snapshot_id))
        return Snapshot(snapshot_id, rootfs_layers=5 if self.compacting else None, disk_only=disk_only)

    def swap_sandbox(self, snapshot: Snapshot) -> bool:
        """Continue on a sandbox of the next id. As in ``SandboxSession``, the listeners hear of it first."""
        if not self.compacting:
            return False
        previous = self._sandbox_id
        self._sandbox_id = "sandbox-%d" % (int(previous.rsplit("-", 1)[1]) + 1)
        self.lifecycle.append(("reboard", snapshot.id))
        for listener in list(self.on_swap):
            listener(self._sandbox_id)
        self.lifecycle.append(("destroy", previous))
        return True

    def execute(self, tool_name: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        return self._run(tool_name, args, timeout)

    def call(self, tool_name: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        if self.unsettled is not None and self.unsettled in args.get("command", ""):
            raise ConnectionError("connection reset by peer")
        return self._run(tool_name, args, timeout)

    def destroy(self) -> None:
        self.lifecycle.append(("destroy", self._sandbox_id))
        if self.destroy_error:
            raise RuntimeError(self.destroy_error)
        self._destroyed = True

    def _run(self, tool_name: str, args: dict[str, Any], timeout: float) -> ToolResult:
        if tool_name != "shell":
            return ToolResult(success=False, output="", error="unknown tool: %s" % tool_name)
        directory = args.get("working_dir", WORKDIR)
        if directory != WORKDIR:
            return ToolResult(
                False, "", "failed to start shell: chdir %s: no such file or directory" % directory
            )
        done = subprocess.run(
            ["bash", "-c", args["command"]],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=args.get("timeout") or timeout,
        )
        stdout = done.stdout.replace(str(self.root.resolve()), WORKDIR)
        if done.returncode == 0 and not done.stderr:
            return ToolResult(True, stdout)
        outcome = CommandOutcome(
            exit_code=done.returncode,
            stdout=stdout,
            stderr=done.stderr,
            stdout_bytes=len(stdout.encode()),
            stderr_bytes=len(done.stderr.encode()),
        )
        # The runtime's text of such a result is the JSON of its outcome.
        return ToolResult(done.returncode == 0, json.dumps(asdict(outcome)), outcome=outcome)


class DirectoryBackend:
    """Sandboxes as directories under ``root``, one session for each run, that share their snapshots."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.snapshots: dict[str, dict[str, bytes]] = {}
        #: The session of each run, in the order that the runs created them.
        self.sessions: list[DirectorySession] = []

    def open(self, *, backend: dict[str, Any], quiet: bool = False) -> DirectorySession:
        """A new session, as the constructor of ``SandboxSession`` returns one."""
        number = len(self.sessions) + 1
        session = DirectorySession(
            self.root / ("sandbox-%d" % number), sandbox_id="sandbox-%d" % number, snapshots=self.snapshots
        )
        self.sessions.append(session.open(backend=backend, quiet=quiet))
        return session
