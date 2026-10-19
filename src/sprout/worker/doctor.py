"""The checks of the machine a worker runs on, as ``python -m sprout.worker doctor`` reports them.

Each check asks for one thing that the podman backend needs before its first sandbox: a podman
that answers ``--version``, a store that :class:`sprout_sandbox.BtrfsStore` accepts, the
sprout-runtime binary, and a keyring quota for more than about 180 rootless containers. A check
reports a problem and never raises. No check changes the host. docs/SERVICES.md describes the
report and its JSON form, which ``python -m sprout.services`` reads.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sprout_sandbox.btrfs_store import REQUIRED_MOUNT_OPTION, store_problem

_OK = "ok"
#: An advisory result: the doctor prints it, and the report still passes.
_WARN = "warn"
_FAIL = "fail"
#: The sprout-runtime binary that ``go build -o sprout-runtime .`` in ``runtime/`` makes in this checkout.
CHECKOUT_RUNTIME = Path(__file__).resolve().parents[3] / "runtime" / "sprout-runtime"
#: Arbitrary: an upper bound for ``podman --version`` on a loaded host.
_VERSION_TIMEOUT_S = 15.0
#: Below this kernel.keys.maxkeys, rootless podman starts fail at about 180 concurrent containers.
#: Each container's user namespace takes keys from the per-user keyring, whose default quota is 200.
_KEYRING_MINIMUM = 20000
_KEYRING_FIX = "sudo sysctl kernel.keys.maxkeys=20000 kernel.keys.maxbytes=2000000"
_MAXKEYS = Path("/proc/sys/kernel/keys/maxkeys")


@dataclass(frozen=True)
class Check:
    """The result of one check: ``status`` is ``ok``, ``warn`` or ``fail``, and ``detail`` says why."""

    status: str
    detail: str


@dataclass(frozen=True)
class HostReport:
    """Each check of this host by name, in the order they ran, and the podman and store they checked."""

    podman_bin: str
    store: str
    checks: Mapping[str, Check]

    @property
    def ok(self) -> bool:
        """Whether no check failed. A ``warn`` does not fail the report."""
        return all(check.status != _FAIL for check in self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        """The report as the JSON object that ``doctor --json`` prints."""
        checks = {
            name: {"status": check.status, "detail": check.detail} for name, check in self.checks.items()
        }
        return {"podman_bin": self.podman_bin, "store": self.store, "checks": checks}


def host_report(*, podman_bin: str, store: str, runtime_bin: Path) -> HostReport:
    """Run every check of this host, and return the report.

    ``podman_bin`` is a path or a name on PATH. ``store`` is a directory.
    """
    checks = {
        "podman": _podman(podman_bin),
        "store": _store(Path(store)),
        "runtime": _runtime(runtime_bin),
        "keyring": _keyring(),
    }
    return HostReport(podman_bin=podman_bin, store=store, checks=checks)


def _podman(executable: str) -> Check:
    # A bare name is looked up on PATH only, because the pool executes it that way.
    found = executable if os.path.isabs(executable) else shutil.which(executable)
    if not found or not os.access(found, os.X_OK):
        return Check(
            _FAIL,
            "%s is not an executable podman (ops/setup_podman.sh installs one under "
            "~/.local/opt/podman; $SPROUT_PODMAN_BIN names another)" % executable,
        )
    try:
        done = subprocess.run(
            [found, "--version"], capture_output=True, text=True, timeout=_VERSION_TIMEOUT_S
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check(_FAIL, "%s --version: %s" % (found, error))
    if done.returncode != 0:
        return Check(
            _FAIL, "%s --version exited %d: %s" % (found, done.returncode, done.stderr.strip()[:120])
        )
    return Check(_OK, "%s (%s)" % (done.stdout.strip().replace("podman version ", "podman "), found))


def _store(path: Path) -> Check:
    problem = store_problem(path)
    if problem:
        return Check(_FAIL, problem)
    return Check(_OK, "%s: btrfs, %s" % (path, REQUIRED_MOUNT_OPTION))


def _runtime(path: Path) -> Check:
    if path.is_file():
        return Check(_OK, str(path))
    return Check(
        _FAIL, "%s is missing (cd runtime && go generate ./... && go build -o sprout-runtime .)" % path
    )


def _keyring() -> Check:
    try:
        value = int(_MAXKEYS.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return Check(_WARN, "kernel.keys.maxkeys is not readable here (%s)" % _MAXKEYS)
    if value < _KEYRING_MINIMUM:
        return Check(
            _WARN,
            "kernel.keys.maxkeys=%d < %d: rootless podman starts fail at ~180 concurrent "
            "containers; fix: %s" % (value, _KEYRING_MINIMUM, _KEYRING_FIX),
        )
    return Check(_OK, "kernel.keys.maxkeys=%d" % value)
