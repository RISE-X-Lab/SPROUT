"""A sandbox backend for the job tests: each sandbox is a directory, and each command runs in it.

``DirectorySandboxes`` makes the sessions of one test. A grade takes it as its ``session_factory``,
and a run gets it in the place of ``SandboxSession``. A session runs each ``shell`` command with
``sh -c`` in its sandbox directory, and answers as the runtime answers: bare stdout for a plain
success, else a ``CommandOutcome``. The fake maps the absolute paths under ``SANDBOX_DIRECTORIES``
into the sandbox directory, in a command and back in its output. It does not rewrite files, so a
script that runs in a sandbox finds the sandbox directory in ``$SANDBOX_ROOT``. ``test_sandboxes.py``
makes the same calls on ``SandboxSession`` and podman.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

from sprout.contract.backend import backend_name
from sprout.worker.tools.result import CommandOutcome, ToolResult

#: The top-level directories of a sandbox that a command can name by their absolute paths.
SANDBOX_DIRECTORIES = ("app", "logs", "project", "tests", "tmp")
_SANDBOX_PATH = re.compile(r"(?<![\w.~/-])/(?:%s)(?=[/\s'\"();|&>]|$)" % "|".join(SANDBOX_DIRECTORIES))
#: The exit code that the runtime reports for a command that it killed at its timeout: 128 + SIGKILL.
_KILLED = 128 + signal.SIGKILL


@dataclass(frozen=True)
class DirectorySnapshot:
    """A snapshot that a session took. Its ``id`` is an image that a new sandbox can start from."""

    id: str
    #: Always None: a directory has no layer chain, so a run never re-boards from one.
    rootfs_layers: int | None = None


class DirectorySandboxes:
    """The sandboxes of one test, kept under ``root``, and the images that they start from.

    ``images`` maps an image reference to a host directory that holds the files of that image.
    ``configs`` maps an image reference to its OCI config, such as ``{"User": "app"}``. A snapshot
    becomes an image too, with the config of the image that its sandbox started from."""

    def __init__(
        self,
        root: Path,
        images: dict[str, Path] | None = None,
        configs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.root = root
        self.images = dict(images or {})
        self.configs = dict(configs or {})
        #: Each session, in the order that the factory made them.
        self.sessions: list[DirectorySandbox] = []
        #: Each ``("create", image)``, ``("attach", sandbox id)`` and ``("destroy", sandbox id)``, in order.
        self.lifecycle: list[tuple[str, str]] = []
        self._refused: set[tuple[str, str]] = set()
        self._snapshots_fail = False
        self._destroy_gate: tuple[Path, Path] | None = None
        self._live: dict[str, Path] = {}
        self._origins: dict[str, str] = {}
        self._sandbox_numbers = itertools.count(1)
        self._snapshot_numbers = itertools.count(1)

    def __call__(
        self, *, quiet: bool = False, backend: dict[str, Any] | None = None, timeout: float = 300.0
    ) -> DirectorySandbox:
        """A new session without a sandbox, with the arguments of ``SandboxSession``."""
        session = DirectorySandbox(self, backend=backend or {})
        self.sessions.append(session)
        return session

    def refuse(self, direction: str, name: str) -> None:
        """Make each ``upload`` or ``download`` (``direction``) of a file whose name matches ``name`` fail.

        ``name`` is a file name or a shell pattern, such as ``sprout-harbor-*.tar.gz``."""
        self._refused.add((direction, name))

    def fail_snapshots(self) -> None:
        """Make each snapshot fail, as a capture fails on a backend that can snapshot."""
        self._snapshots_fail = True

    def hold_destroys(self, *, held: Path, released: Path) -> None:
        """Make each ``destroy`` create the file ``held``, and then wait until the file ``released`` exists.

        Until a test creates ``released``, a teardown that removes a sandbox hangs."""
        self._destroy_gate = (held, released)

    def kept(self, image: str) -> str:
        """The id of a running sandbox from ``image`` that no session holds, as a rollout keeps one.

        The lifecycle does not list its start."""
        return self._start(image)

    def live(self) -> list[str]:
        """The ids of the sandboxes that run now: started or kept, and not destroyed."""
        return sorted(self._live)

    def _start(self, image: str) -> str:
        sandbox_id = "sandbox-%d" % next(self._sandbox_numbers)
        shutil.copytree(self.images[image], self.root / "sandboxes" / sandbox_id, symlinks=True)
        self._live[sandbox_id] = self.root / "sandboxes" / sandbox_id
        self._origins[sandbox_id] = self._origins.get(image, image)
        return sandbox_id

    def _snapshot(self, sandbox_id: str) -> str | None:
        if self._snapshots_fail:
            return None
        snapshot_id = "snapshot-%d" % next(self._snapshot_numbers)
        shutil.copytree(self._live[sandbox_id], self.root / "images" / snapshot_id, symlinks=True)
        self.images[snapshot_id] = self.root / "images" / snapshot_id
        self._origins[snapshot_id] = self._origins[sandbox_id]
        return snapshot_id

    def _wait_to_destroy(self) -> None:
        if self._destroy_gate is None:
            return
        held, released = self._destroy_gate
        held.touch()
        while not released.exists():
            time.sleep(0.05)

    def _remove(self, sandbox_id: str) -> None:
        root = self._live.pop(sandbox_id, None)
        if root is not None:
            shutil.rmtree(root)


class DirectorySandbox:
    """One session of ``DirectorySandboxes``, with the part of ``SandboxSession`` that the jobs call."""

    def __init__(self, sandboxes: DirectorySandboxes, *, backend: dict[str, Any]) -> None:
        self.backend = backend
        self.create_error = ""
        self.transfer_error = ""
        self.on_swap: list[Any] = []
        #: The image and the shape of the last ``create``.
        self.image: str | None = None
        self.resources: dict[str, Any] | None = None
        #: The egress of each ``set_network`` call, in call order.
        self.networks: list[bool] = []
        self._sandboxes = sandboxes
        self._id: str | None = None

    @property
    def sandbox_id(self) -> str:
        return self._id or "unknown"

    @property
    def allow_internet(self) -> bool | None:
        """The egress that the backend config gives a new sandbox, None when it names none."""
        allow: bool | None = (self.backend.get(backend_name(self.backend)) or {}).get("allow_internet")
        return allow

    def create(self, image: str, resources: dict[str, Any] | None = None) -> bool:
        self.image, self.resources = image, resources
        if image not in self._sandboxes.images:
            self.create_error = "PodmanError: image %s not known" % image
            return False
        self._id = self._sandboxes._start(image)
        self._sandboxes.lifecycle.append(("create", image))
        return True

    def attach(self, sandbox_id: str) -> bool:
        self._sandboxes.lifecycle.append(("attach", sandbox_id))
        if sandbox_id not in self._sandboxes._live:
            self.create_error = "sandbox %s does not answer" % sandbox_id
            return False
        self._id = sandbox_id
        return True

    def destroy(self) -> None:
        if self._id is not None:
            self._sandboxes._wait_to_destroy()
            self._sandboxes._remove(self._id)
            self._sandboxes.lifecycle.append(("destroy", self._id))
            self._id = None

    def execute(self, tool_name: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        if tool_name != "shell" or self._id is None:
            return ToolResult(success=False, output="", error="No active sandbox, or no tool %s" % tool_name)
        return self._shell(args)

    def call(self, tool_name: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        return self.execute(tool_name, args, timeout)

    def upload_file(self, source: str | Path, destination: str) -> bool:
        target = self._host(destination)
        if not self._moves("upload", destination) or not target.parent.is_dir():
            self.transfer_error = self.transfer_error or "PodmanError: no directory for %s" % destination
            return False
        shutil.copyfile(source, target)
        return True

    def download_file(self, source: str, destination: str | Path, timeout: float = 120.0) -> bool:
        if not self._moves("download", source) or not self._host(source).is_file():
            self.transfer_error = self.transfer_error or "PodmanError: no such file %s" % source
            return False
        shutil.copyfile(self._host(source), destination)
        return True

    def image_config(self, image: str | None) -> dict[str, Any]:
        """As a podman sandbox, the config of the image of its line. As a microVM, the config of ``image``."""
        if backend_name(self.backend) != "podman":
            return self._sandboxes.configs.get(image or "", {})
        assert self._id is not None, "No active sandbox"
        return self._sandboxes.configs.get(self._sandboxes._origins[self._id], {})

    def set_network(self, allow: bool) -> None:
        if backend_name(self.backend) != "podman":
            raise NotImplementedError(
                "backend %r cannot change a running sandbox's network" % backend_name(self.backend)
            )
        self.networks.append(allow)

    def supports_snapshot(self) -> bool:
        return self._id is not None

    def snapshot(self, name: str | None = None, disk_only: bool = True) -> DirectorySnapshot | None:
        snapshot_id = self._sandboxes._snapshot(self._id) if self._id is not None else None
        return DirectorySnapshot(snapshot_id) if snapshot_id is not None else None

    def swap_sandbox(self, snapshot: object) -> bool:
        """False: a sandbox of this backend is never re-boarded."""
        return False

    def _moves(self, direction: str, path: str) -> bool:
        self.transfer_error = ""
        name = PurePosixPath(path).name
        if any(way == direction and fnmatchcase(name, refused) for way, refused in self._sandboxes._refused):
            self.transfer_error = "PodmanError: refused by the test"
            return False
        return True

    def _root(self) -> Path:
        assert self._id is not None, "No active sandbox"
        return self._sandboxes._live[self._id]

    def _host(self, path: str) -> Path:
        return self._root() / path.lstrip("/")

    def _shell(self, args: dict[str, Any]) -> ToolResult:
        root = str(self._root())
        command = _SANDBOX_PATH.sub(lambda match: root + match.group(0), args["command"])
        workdir = args.get("working_dir") or self.image_config(self.image).get("WorkingDir") or "/"
        try:
            process = subprocess.Popen(
                ["sh", "-c", command],
                cwd=self._host(workdir),
                env={**os.environ, "SANDBOX_ROOT": root},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError:
            # The runtime reports a command that does not start, such as one over MAX_ARG_STRLEN, as exit 1.
            return _answer(CommandOutcome(exit_code=1))
        try:
            stdout, stderr = process.communicate(timeout=args.get("timeout", 300))
            code, timed_out = process.returncode, False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            code, timed_out = _KILLED, True
        return _answer(
            CommandOutcome(
                exit_code=code if code >= 0 else 128 - code,
                stdout=_guest(stdout.decode(errors="replace"), root),
                stderr=_guest(stderr.decode(errors="replace"), root),
                timed_out=timed_out,
            )
        )


def _guest(text: str, root: str) -> str:
    """``text`` with each host path under the sandbox directory ``root`` as the sandbox names it."""
    return text.replace(root + "/", "/").replace(root, "/")


def _answer(outcome: CommandOutcome) -> ToolResult:
    """The result of a command, as the runtime reports it (``runtime/tools/boundedlog.go``)."""
    succeeded = outcome.exit_code == 0 and not outcome.timed_out
    if succeeded and not outcome.stderr:
        return ToolResult(success=True, output=outcome.stdout)
    return ToolResult(success=succeeded, output=json.dumps(asdict(outcome)), outcome=outcome)
