"""The container of a ``podman`` service, such as the Run Store's PostgreSQL, through the podman CLI.

:class:`Podman` has only the verbs of a container's lifecycle: inspect, run, start, stop
and logs. It runs on the podman that the caller names, which is the sandboxes' podman.

A value from the service's ``env``, such as the database password, never appears on a
command line, so the argv in the log holds names only. Nothing here removes a container.
Its data directory is a bind mount and survives a stop, and ``up`` of a stopped container
is :meth:`Podman.start`. The services and their images: ``docs/SERVICES.md``.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sprout.services.health import which
from sprout.services.spec import Context, Service, expand, expand_all
from sprout.services.supervisor import StartError, run_logged, tail

STOP_TIMEOUT_S = 20


@dataclass(frozen=True)
class ContainerState:
    """What podman reports about one container."""

    #: podman's State.Status (running, exited, created, paused, ...), or `absent`.
    status: str
    #: The pid of the container's main process. None if podman reports none.
    pid: int | None

    @property
    def running(self) -> bool:
        """True if the container is running."""
        return self.status == "running"

    @property
    def exists(self) -> bool:
        """True if podman knows the container, in any state."""
        return self.status != "absent"


class Podman:
    """The podman verbs for one service, logged into that service's log."""

    def __init__(self, log_path: Path, cwd: Path, environ: Mapping[str, str], *, podman_bin: str) -> None:
        self.log_path = log_path
        self.cwd = cwd
        self.environ = dict(environ)
        self.podman_bin = podman_bin
        self.executable = podman_bin if os.path.isabs(podman_bin) else which(podman_bin)

    def _require(self) -> str:
        if not self.executable:
            raise StartError(
                "no podman at %s (ops/setup_podman.sh installs one under ~/.local/opt/podman; "
                "$SPROUT_PODMAN_BIN names another)" % self.podman_bin
            )
        return self.executable

    def inspect(self, container: str) -> ContainerState:
        """The state of ``container``, ``absent`` if podman does not know it or does not answer.

        Writes nothing to the log, because ``status`` calls it on each run.
        """
        if not self.executable:
            return ContainerState("absent", None)
        try:
            done = subprocess.run(
                [
                    self.executable,
                    "inspect",
                    "--type",
                    "container",
                    "--format",
                    "{{.State.Status}} {{.State.Pid}}",
                    container,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ContainerState("absent", None)
        fields = done.stdout.split()
        if done.returncode != 0 or not fields:
            return ContainerState("absent", None)
        pid = int(fields[1]) if len(fields) > 1 and fields[1].isdigit() and fields[1] != "0" else None
        return ContainerState(fields[0], pid)

    def run(self, service: Service, ctx: Context) -> None:
        """Create and start the service's container, which does not exist yet, with ``podman run -d``.

        Expands the placeholders in ``publish``, ``volumes`` and ``env`` against ``ctx``, and
        makes the host side of each bind mount first. Raises :class:`StartError` on a failure.
        """
        podman = self._require()
        argv = [podman, "run", "-d", "--name", service.container]
        if service.publish:
            argv += ["-p", expand(service.publish, ctx)]
        for volume in service.volumes:
            mount = expand(volume, ctx)
            host_side = mount.split(":", 1)[0]
            # podman reads a source without a leading ``/`` as a named volume. podman would also make
            # a missing bind source, but then the container's first process decides its owner.
            if host_side.startswith("/"):
                Path(host_side).mkdir(parents=True, exist_ok=True)
            argv += ["-v", mount]
        env = expand_all(service.env, ctx)
        for name in env:
            argv += ["-e", name]  # without a value, podman copies it from the environment below
        argv.append(service.image)
        self._run(argv, {**self.environ, **env}, "run")

    def start(self, container: str) -> None:
        """Start the stopped ``container``. Raises :class:`StartError` on a failure."""
        self._run([self._require(), "start", container], self.environ, "start")

    def stop(self, container: str, timeout_s: int = STOP_TIMEOUT_S) -> None:
        """Stop ``container``, which podman kills after ``timeout_s``.

        Raises :class:`StartError` on a failure.
        """
        self._run([self._require(), "stop", "--time", str(timeout_s), container], self.environ, "stop")

    def logs(self, container: str, follow: bool) -> int:
        """Print the container's output with ``podman logs`` on the caller's terminal, and return its status.

        The container's own output is in podman. The service log holds only what the CLI did.
        """
        argv = [self._require(), "logs"] + (["-f"] if follow else []) + [container]
        try:
            return subprocess.run(argv).returncode
        except KeyboardInterrupt:
            return 0

    def _run(self, argv: list[str], env: Mapping[str, str], verb: str) -> None:
        status = run_logged(argv, env, self.log_path, self.cwd)
        if status != 0:
            raise StartError(
                "podman %s failed (exit %d); last lines of %s:\n%s"
                % (verb, status, self.log_path, tail(self.log_path))
            )
