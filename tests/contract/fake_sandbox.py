"""A fake sandbox backend for the tests that run an agent through the worker's ``Runner``.

``FakeSandboxes.session`` takes the place of ``SandboxSession``, and the ``sandboxes`` fixture of
``tests/contract/conftest.py`` puts it where the runner makes each session. Each sandbox is a
directory under the test's ``tmp_path``. Its commands run on the host, in that directory, and see it as
``/testbed``: the fake replaces that text in each command and in each output. A snapshot is a copy of
the directory's files. A sandbox from a snapshot id starts with that copy, from any other image empty.

Like the runtime, a ``shell`` call answers bare stdout for a command that exits 0 with no stderr, and
a ``CommandOutcome`` otherwise. A working directory outside ``/testbed`` does not start a shell.
"""

from __future__ import annotations

import itertools
import os
import subprocess
from pathlib import Path
from typing import Any

from sprout.worker.tools.result import CommandOutcome, ToolResult
from sprout_sandbox import Snapshot

#: The directory that a sandbox's commands see as theirs.
WORKDIR = "/testbed"


class FakeSandboxes:
    """The sandboxes of one test, their snapshots, and the answers that the test sets for agent calls."""

    def __init__(self, root: Path) -> None:
        self.root = root
        #: The files of each snapshot by its id, as relative path to bytes.
        self.snapshots: dict[str, dict[str, bytes]] = {}
        #: Each session that the runner made, in order.
        self.sessions: list[FakeSession] = []
        #: The OCI config of every image: ``User``, ``Env`` and ``WorkingDir``.
        self.image_config: dict[str, Any] = {}
        self._numbers = itertools.count(1)
        self._unsettled: CommandOutcome | Exception | None = None
        self._timed_out: str | None = None

    def session(
        self, *, backend: dict[str, Any] | None = None, quiet: bool = False, timeout: float = 300.0
    ) -> FakeSession:
        """A new session, as ``SandboxSession(backend=..., quiet=...)`` makes one."""
        session = FakeSession(
            self, backend=backend or {}, root=self.root / ("sandbox-%d" % next(self._numbers))
        )
        self.sessions.append(session)
        return session

    def unsettle(self, answer: CommandOutcome | Exception) -> None:
        """Answer each agent call with ``answer``, an outcome to return or an error to raise."""
        self._unsettled = answer

    def time_out(self, marker: str) -> None:
        """Run each agent command that contains ``marker``, then report it timed out with exit code 137."""
        self._timed_out = marker

    def snapshot_id(self) -> str:
        """A snapshot id that no other sandbox or snapshot of the test has."""
        return "snapshot-%d" % next(self._numbers)


class FakeSession:
    """One sandbox: a directory that its commands see as ``/testbed``."""

    def __init__(self, sandboxes: FakeSandboxes, *, backend: dict[str, Any], root: Path) -> None:
        self._sandboxes = sandboxes
        #: The backend config that the runner made the session with.
        self.backend = backend
        self.root = root
        #: The image or snapshot id, and the resources, that ``create`` started the sandbox from.
        self.image: str | None = None
        self.resources: dict[str, Any] | None = None
        #: The number of agent calls, which ``call`` counts.
        self.calls = 0
        self.destroyed = False
        self.create_error = ""
        self.on_swap: list[Any] = []

    @property
    def sandbox_id(self) -> str:
        return self.root.name

    def create(self, image: str, resources: dict[str, Any] | None = None) -> bool:
        self.image, self.resources = image, resources
        self.root.mkdir(parents=True)
        for name, data in self._sandboxes.snapshots.get(image, {}).items():
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_bytes(data)
        return True

    def supports_snapshot(self) -> bool:
        return True

    def snapshot(self, name: str | None = None, disk_only: bool = True) -> Snapshot:
        snapshot = Snapshot(id=self._sandboxes.snapshot_id(), disk_only=disk_only)
        self._sandboxes.snapshots[snapshot.id] = {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        return snapshot

    def swap_sandbox(self, snapshot: object) -> bool:
        return False

    def image_config(self, image: str | None) -> dict[str, Any]:
        return dict(self._sandboxes.image_config)

    def execute(self, tool_name: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        """Run a tool as the owner, such as the workspace probes."""
        return self._shell(args)

    def call(self, tool_name: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        """Run a tool for the agent, with the answer that the test set, if any."""
        self.calls += 1
        answer = self._sandboxes._unsettled
        if isinstance(answer, Exception):
            raise answer
        if answer is not None:
            return ToolResult(success=False, output="not settled", outcome=answer)
        result = self._shell(args)
        marker = self._sandboxes._timed_out
        if marker is not None and marker in args["command"]:
            stderr = result.outcome.stderr if result.outcome else ""
            outcome = CommandOutcome(exit_code=137, stdout=result.output, stderr=stderr, timed_out=True)
            return ToolResult(success=False, output=result.output, outcome=outcome)
        return result

    def destroy(self) -> None:
        self.destroyed = True

    def _shell(self, args: dict[str, Any]) -> ToolResult:
        directory = args.get("working_dir", WORKDIR)
        if directory != WORKDIR and not directory.startswith(WORKDIR + "/"):
            error = "failed to start shell: chdir %s: no such file or directory" % directory
            return ToolResult(success=False, output="", error=error)
        command = args["command"].replace(WORKDIR, str(self.root))
        # Without a ceiling, git would find the repository that holds the test's temporary directory.
        env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(self.root.parent)}
        done = subprocess.run(
            ["bash", "-c", command],
            cwd=self.root / directory[len(WORKDIR) + 1 :],
            env=env,
            capture_output=True,
            text=True,
            timeout=args.get("timeout", 300),
        )
        stdout = done.stdout.replace(str(self.root), WORKDIR)
        stderr = done.stderr.replace(str(self.root), WORKDIR)
        if done.returncode == 0 and not stderr:
            return ToolResult(success=True, output=stdout)
        outcome = CommandOutcome(exit_code=done.returncode, stdout=stdout, stderr=stderr)
        return ToolResult(success=done.returncode == 0, output=stdout, outcome=outcome)
