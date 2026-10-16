"""Probes: is a service answering, and is the host what the services need.

Every probe returns a :class:`Probe` rather than raising, because the callers
print a table or a doctor report and one unreachable service must not hide the
others. The probes are deliberately shallow -- a GET that must answer 200,
``pg_isready``, a pid that is alive -- since ``up`` polls them every half
second until a service comes up and ``status`` runs all of them on every call.
Nothing here starts anything.

The host checks at the bottom are the facts that have actually cost batches
(AGENTS.md): a store that is not btrfs with ``user_subvol_rm_allowed`` (every
spawn works, the first destroy fails after the container is gone), a per-user
keyring quota too small for ~180 concurrent rootless containers, a missing
runtime binary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from http.client import HTTPException  # by name: `http` below is the probe
from pathlib import Path

from sprout.services.spec import Context, ExpandError, Health, expand, podman_bin_for

HTTP_TIMEOUT_S = 3.0
TOOL_TIMEOUT_S = 15.0
#: kernel.keys.maxkeys below this makes rootless podman starts fail at about
#: 180 concurrent containers: each container's user namespace takes keys from
#: the per-user keyring, and the default quota is 200.
KEYRING_MINIMUM = 20000
KEYRING_FIX = "sudo sysctl kernel.keys.maxkeys=20000 kernel.keys.maxbytes=2000000"
REQUIRED_MOUNT_OPTION = "user_subvol_rm_allowed"


@dataclass(frozen=True)
class Probe:
    ok: bool
    detail: str
    #: Advisory: printed as `warn`, never a failure (doctor exits 0 on it).
    warning: bool = False

    @property
    def word(self) -> str:
        return "ok" if self.ok else ("warn" if self.warning else "fail")


# --- service probes -----------------------------------------------------------


def http(
    url: str, headers: Iterable[tuple[str, str]] = (), timeout: float = HTTP_TIMEOUT_S, expect: int = 200
) -> Probe:
    """GET ``url``; healthy is exactly ``expect`` (200). urllib, so no extra is needed."""
    try:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
    except urllib.error.HTTPError as error:  # a reply, just not 200: 401 says "wrong token"
        status = error.code
    except urllib.error.URLError as error:
        reason = error.reason
        text = reason.strerror if isinstance(reason, OSError) and reason.strerror else str(reason)
        return Probe(False, text)
    except TimeoutError:
        return Probe(False, "timed out after %gs" % timeout)
    except HTTPException as error:
        # urllib wraps only OSError into URLError; a port served by something
        # that is not HTTP (PostgreSQL, say) raises BadStatusLine unwrapped,
        # with the offending line, CRLF included.
        return Probe(False, "not an HTTP reply: %s" % " ".join(str(error).split()))
    except (OSError, ValueError) as error:  # ValueError: a URL urllib cannot parse
        return Probe(False, str(error))
    return Probe(status == expect, "HTTP %d" % status)


def pg_isready(host: str, port: int, user: str = "", container: str = "", podman: str = "podman") -> Probe:
    """``pg_isready`` from the host's client tools, else inside the container (through ``podman``).

    The host route is preferred because it proves the published port works,
    which is what the Run Store connects through; the in-container fallback
    only proves PostgreSQL is up.
    """
    user_flag = ["-U", user] if user else []
    tool = shutil.which("pg_isready")
    if tool:
        return _run_probe([tool, "-h", host, "-p", str(port), *user_flag])
    executable = podman if os.path.isabs(podman) else which(podman)
    if container and executable:
        return _run_probe(
            [executable, "exec", container, "pg_isready", *user_flag], prefix="in %s: " % container
        )
    return Probe(
        False, "pg_isready is not installed (postgresql-client) and there is no container to run it in"
    )


def _run_probe(argv: Sequence[str], prefix: str = "") -> Probe:
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True, timeout=TOOL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return Probe(False, prefix + "%s timed out" % argv[0])
    except OSError as error:
        return Probe(False, prefix + str(error))
    lines = (done.stdout or done.stderr).strip().splitlines()
    detail = lines[-1].strip() if lines else "exit %d" % done.returncode
    return Probe(done.returncode == 0, prefix + detail)


def probe(health: Health, ctx: Context, *, alive: bool | None, alive_detail: str) -> Probe:
    """The health of one service, given what is already known about its process.

    ``alive`` is the supervisor's or podman's answer (None for a service with
    no process of its own: an external or a check). For ``process`` and
    ``none`` it is the whole answer; ``http`` and ``pg_isready`` ask the
    service itself, which also catches a port served by something else.
    """
    if health.kind == "http":
        try:
            url = expand(health.url, ctx)
            headers = [(name, expand(value, ctx)) for name, value in health.headers]
        except ExpandError as error:  # an unset {env:...} or a missing secret
            return Probe(False, str(error))
        return http(url, headers)
    if health.kind == "pg_isready":
        return pg_isready(
            health.host, health.port or 0, health.user, health.container, podman_bin_for(ctx.environ)
        )
    if alive is None:
        return Probe(True, "not probed") if health.kind == "none" else Probe(False, "nothing to probe")
    return Probe(alive, alive_detail)


# --- host checks --------------------------------------------------------------


def which(name: str) -> str | None:
    """PATH, then ``~/.local/bin``: uv's installer puts it there, and a shell that
    did not source its profile (a cron job, a detached session) does not look."""
    found = shutil.which(name)
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / name
    return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None


def program(name: str) -> Probe:
    found = which(name)
    return Probe(bool(found), found or "%s is not on PATH or in ~/.local/bin" % name)


def file_present(path: Path, hint: str) -> Probe:
    return Probe(path.is_file(), str(path) if path.is_file() else "%s is missing (%s)" % (path, hint))


def podman_bin(executable: str) -> Probe:
    """The podman the sandboxes run on answers ``--version``: a path, or a name on PATH."""
    executable = os.path.expanduser(executable)
    found = executable if os.path.isabs(executable) else which(executable)
    if not found or not os.access(found, os.X_OK):
        return Probe(
            False,
            "%s is not an executable podman (ops/setup_podman.sh installs one under "
            "~/.local/opt/podman; $SPROUT_PODMAN_BIN names another)" % executable,
        )
    try:
        done = subprocess.run([found, "--version"], capture_output=True, text=True, timeout=TOOL_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        return Probe(False, "%s --version: %s" % (found, error))
    if done.returncode != 0:
        return Probe(
            False, "%s --version exited %d: %s" % (found, done.returncode, done.stderr.strip()[:120])
        )
    return Probe(True, "%s (%s)" % (done.stdout.strip().replace("podman version ", "podman "), found))


def btrfs_store(path: Path) -> Probe:
    """What ``PodmanPool`` refuses, asked the same way (``findmnt --target``): the
    filesystem holding the store must be btrfs mounted with ``user_subvol_rm_allowed``."""
    if not path.is_dir():
        return Probe(
            False,
            "%s does not exist; make it once as root: sudo ops/btrfs_store.sh %s "
            "/var/lib/sprout-btrfs-rootless.img 200G" % (path, path),
        )
    tool = shutil.which("findmnt")
    if not tool:
        return Probe(False, "findmnt (util-linux) is not installed; it is how the store's mount is checked")
    try:
        done = subprocess.run(
            [tool, "-no", "FSTYPE,OPTIONS", "--target", str(path)],
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Probe(False, "findmnt: %s" % error)
    fields = done.stdout.split()
    if done.returncode != 0 or len(fields) < 2:
        return Probe(
            False,
            "findmnt cannot tell what %s is mounted on: %s" % (path, done.stderr.strip() or "(no output)"),
        )
    fstype, options = fields[0], fields[1]
    if fstype != "btrfs":
        return Probe(
            False,
            "%s is on %s, not btrfs (sudo ops/btrfs_store.sh <mount point> <image file> <size>)"
            % (path, fstype),
        )
    if REQUIRED_MOUNT_OPTION not in options.split(","):
        return Probe(
            False, "%s is btrfs but mounted without %s (%s)" % (path, REQUIRED_MOUNT_OPTION, options)
        )
    return Probe(True, "%s: btrfs, %s" % (path, REQUIRED_MOUNT_OPTION))


def keyring_quota(minimum: int = KEYRING_MINIMUM) -> Probe:
    path = Path("/proc/sys/kernel/keys/maxkeys")
    try:
        value = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return Probe(False, "kernel.keys.maxkeys is not readable here (%s)" % path, warning=True)
    if value < minimum:
        return Probe(
            False,
            "kernel.keys.maxkeys=%d < %d: rootless podman starts fail at ~180 concurrent "
            "containers; fix: %s" % (value, minimum, KEYRING_FIX),
            warning=True,
        )
    return Probe(True, "kernel.keys.maxkeys=%d" % value)
