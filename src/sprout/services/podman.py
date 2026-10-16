"""The database container through the podman CLI.

One container -- the Run Store's PostgreSQL -- and only the verbs its lifecycle
needs: inspect, run, start, stop, logs. It runs on the podman the sandboxes run
on -- ``$SPROUT_PODMAN_BIN``, else the podman 6 ``ops/setup_podman.sh``
installed, else ``podman`` on PATH (``spec.podman_bin_for``) -- and the image is
the official ``postgres:16-alpine``, so there is nothing to build and nothing to
install beyond what the sandbox backend already needs. The host's own PostgreSQL owns
5432; the container publishes its port on 127.0.0.1 only, at the port
``services.yaml`` names.

The password never appears on a command line: ``-e NAME`` without a value makes
podman copy the variable from its own environment, which is where the CLI puts
it, so the argv written to the log carries names only. ``podman stop`` is the
only way down (the data directory is a bind mount and survives; ``podman rm``
is left to a person), and ``up`` of an existing stopped container is ``podman
start``, so a reboot costs nothing but the start.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from sprout.services.health import which
from sprout.services.spec import Context, Service, expand, expand_all, podman_bin_for
from sprout.services.supervisor import StartError, run_logged, tail

STOP_TIMEOUT_S = 20


@dataclass(frozen=True)
class ContainerState:
    #: podman's State.Status (running, exited, created, paused, ...), or `absent`.
    status: str
    pid: int | None

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def exists(self) -> bool:
        return self.status != "absent"


class Podman:
    """The podman verbs for one service, logged into that service's log."""

    def __init__(self, log_path: Path, cwd: Path, environ: Mapping[str, str]) -> None:
        self.log_path = log_path
        self.cwd = cwd
        self.environ = dict(environ)
        chosen = podman_bin_for(self.environ)
        self.executable = chosen if os.path.isabs(chosen) else which(chosen)

    def _require(self) -> str:
        if not self.executable:
            raise StartError(
                "no podman at %s (ops/setup_podman.sh installs one under ~/.local/opt/podman; "
                "$SPROUT_PODMAN_BIN names another)" % podman_bin_for(self.environ)
            )
        return self.executable

    def inspect(self, container: str) -> ContainerState:
        """Quiet (status polls it): absent when podman does not know the name."""
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
        """``podman run -d`` for a container that does not exist yet.

        Every placeholder the service carries is resolved here, against one
        context: ``publish``, the ``volumes`` (the template mounts
        ``{state}/db/data``, and podman reads a source without a leading ``/``
        as a *named* volume, which ``{state}`` is not a legal name for) and
        the ``env`` values. The host side of every bind mount is then created
        first: podman would make it too, but as whatever the first process in
        the container decides, and the data directory should belong to this
        user from the start.
        """
        podman = self._require()
        argv = [podman, "run", "-d", "--name", service.container]
        if service.publish:
            argv += ["-p", expand(service.publish, ctx)]
        for volume in service.volumes:
            mount = expand(volume, ctx)
            host_side = mount.split(":", 1)[0]
            if host_side.startswith("/"):  # a bind mount, not a named volume
                Path(host_side).mkdir(parents=True, exist_ok=True)
            argv += ["-v", mount]
        env = expand_all(service.env, ctx)
        for name in env:
            argv += ["-e", name]  # the value comes from the environment below
        argv.append(service.image)
        self._run(argv, {**self.environ, **env}, "run")

    def start(self, container: str) -> None:
        self._run([self._require(), "start", container], self.environ, "start")

    def stop(self, container: str, timeout_s: int = STOP_TIMEOUT_S) -> None:
        self._run([self._require(), "stop", "--time", str(timeout_s), container], self.environ, "stop")

    def logs(self, container: str, follow: bool) -> int:
        """``podman logs`` on the caller's terminal: the container's own output
        lives in podman, not in the service log (which holds what the CLI did)."""
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
