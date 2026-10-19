"""Probes: is a service answering, and is the host what the services need.

Each probe returns a :class:`Probe` and does not raise, so one unreachable service does
not hide the others in a table or a doctor report. Each probe is shallow, such as a GET
that must answer 200, ``pg_isready`` or a live pid, because ``up`` polls it every half
second and ``status`` runs every probe on each call. No probe starts anything.

The worker owns the checks of its host: the podman, the btrfs store, the runtime binary
and the keyring quota. :class:`WorkerDoctor` reads them from
``python -m sprout.worker doctor --json`` in a child process, so this package imports
nothing of the worker.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from http.client import HTTPException  # by name: `http` below is the probe
from pathlib import Path
from typing import Any

from sprout.services.spec import Context, ExpandError, Health, expand

HTTP_TIMEOUT_S = 3.0
TOOL_TIMEOUT_S = 15.0
#: Arbitrary: above the doctor's own limits, 15 s for `podman --version` and 30 s for findmnt.
_DOCTOR_TIMEOUT_S = 120.0
#: What a `check` service asks the worker's doctor, by its `check` value: the name
#: of the check in the doctor's report, and the doctor flag that the service's `path` sets.
_DOCTOR_CHECKS = {"podman": ("podman", "--podman-bin"), "btrfs_store": ("store", "--store")}
#: The doctor's status words, as (ok, warning) of a Probe.
_STATUSES = {"ok": (True, False), "warn": (False, True), "fail": (False, False)}


@dataclass(frozen=True)
class Probe:
    """The result of one probe or check: whether it passed, and a one-line detail."""

    ok: bool
    detail: str
    #: True for an advisory result. It prints as ``warn``, and ``doctor`` still exits 0.
    warning: bool = False

    @property
    def word(self) -> str:
        """The status word: ``ok``, ``warn`` or ``fail``."""
        return "ok" if self.ok else ("warn" if self.warning else "fail")


# --- service probes -----------------------------------------------------------


def http(
    url: str, headers: Iterable[tuple[str, str]] = (), timeout: float = HTTP_TIMEOUT_S, expect: int = 200
) -> Probe:
    """Probe ``url`` with a GET. The probe passes only if the reply status is ``expect``."""
    try:
        # urllib, not an HTTP client from an optional extra: this package needs none.
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
    except urllib.error.HTTPError as error:  # a reply that is not 200: 401 says "wrong token"
        status = error.code
    except urllib.error.URLError as error:
        reason = error.reason
        text = reason.strerror if isinstance(reason, OSError) and reason.strerror else str(reason)
        return Probe(False, text)
    except TimeoutError:
        return Probe(False, "timed out after %gs" % timeout)
    except HTTPException as error:
        # urllib wraps only OSError in URLError. A port that does not speak HTTP, such as
        # PostgreSQL's, raises BadStatusLine unwrapped, with the line and its CRLF.
        return Probe(False, "not an HTTP reply: %s" % " ".join(str(error).split()))
    except (OSError, ValueError) as error:  # ValueError: a URL urllib cannot parse
        return Probe(False, str(error))
    return Probe(status == expect, "HTTP %d" % status)


def pg_isready(host: str, port: int, *, user: str, container: str, podman: Callable[[], str]) -> Probe:
    """Probe PostgreSQL with the host's ``pg_isready``, else with the one in ``container``.

    The host's tool proves that the published port works, which the Run Store connects
    through. The probe calls ``podman()`` only for the fallback in the container.
    """
    user_flag = ["-U", user] if user else []
    tool = shutil.which("pg_isready")
    if tool:
        return _run_probe([tool, "-h", host, "-p", str(port), *user_flag])
    if container:
        try:
            chosen = podman()
        except ExpandError as error:  # the worker's doctor gave no report
            return Probe(False, str(error))
        executable = chosen if os.path.isabs(chosen) else which(chosen)
        if executable:
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
    """The health of one service, given ``alive``, what the caller knows about its process.

    ``alive`` is None for a service without a process of its own. It decides ``process`` and
    ``none``. ``http`` and ``pg_isready`` ask the service, so a port of another program fails.
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
            health.host,
            health.port or 0,
            user=health.user,
            container=health.container,
            podman=lambda: expand("{podman_bin}", ctx),
        )
    if alive is None:
        return Probe(True, "not probed") if health.kind == "none" else Probe(False, "nothing to probe")
    return Probe(alive, alive_detail)


# --- programs on this host ----------------------------------------------------


def which(name: str) -> str | None:
    """The path of the program ``name`` on PATH, else in ``~/.local/bin``, else None.

    uv's installer puts uv in ``~/.local/bin``. A shell that did not source its profile,
    such as a cron job, does not have that directory on PATH.
    """
    found = shutil.which(name)
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / name
    return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None


def program(name: str) -> Probe:
    """A probe that passes if :func:`which` finds the program ``name``."""
    found = which(name)
    return Probe(bool(found), found or "%s is not on PATH or in ~/.local/bin" % name)


# --- the worker's host checks -------------------------------------------------


class HostError(ValueError):
    """The worker's doctor gave no report.

    For example, the interpreter is missing, or it cannot import ``sprout.worker``.
    """


@dataclass(frozen=True)
class HostReport:
    """The worker's doctor report: the podman and store it resolved, and each check by name."""

    podman_bin: str
    store: str
    checks: Mapping[str, Probe]


class WorkerDoctor:
    """The worker's report on this host, from ``<python> -m sprout.worker doctor --json``.

    The doctor runs at most once for each set of flags, at the first question that needs it. So a
    command that reads several checks and placeholders starts one child process.
    """

    def __init__(self, *, python: str, environ: Mapping[str, str]) -> None:
        self._python = python
        self._environ = dict(environ)
        self._reports: dict[tuple[str, ...], HostReport | str] = {}

    def report(self, flags: tuple[str, ...] = ()) -> HostReport:
        """The report for these doctor flags. Raises :class:`HostError` when the doctor gives none."""
        if flags not in self._reports:
            argv = [self._python, "-m", "sprout.worker", "doctor", "--json", *flags]
            try:
                self._reports[flags] = _run_doctor(argv, self._environ)
            except HostError as error:
                self._reports[flags] = str(error)
        found = self._reports[flags]
        if isinstance(found, str):
            raise HostError(found)
        return found

    def check(self, *, kind: str, path: str) -> Probe:
        """The verdict on a ``check`` service of this kind, on ``path`` or, when it is empty, the default."""
        name, flag = _DOCTOR_CHECKS[kind]
        try:
            report = self.report((flag, path) if path else ())
        except HostError as error:
            return Probe(False, str(error))
        return report.checks.get(name) or Probe(False, "the worker's doctor reports no %s check" % name)

    def other_checks(self, kinds: Iterable[str]) -> list[tuple[str, Probe]]:
        """Each check of the default report that no ``check`` service of these kinds shows.

        When the doctor gives no report, this is one failed ``worker`` check that says why.
        """
        try:
            report = self.report()
        except HostError as error:
            return [("worker", Probe(False, str(error)))]
        shown = {_DOCTOR_CHECKS[kind][0] for kind in kinds}
        return [(name, probe) for name, probe in report.checks.items() if name not in shown]

    def fact(self, name: str) -> str:
        """``podman_bin`` or ``store`` as the doctor resolved them. Raises :class:`HostError`."""
        report = self.report()
        return report.podman_bin if name == "podman_bin" else report.store


def _run_doctor(argv: list[str], environ: Mapping[str, str]) -> HostReport:
    command = shlex.join(argv)
    try:
        done = subprocess.run(
            argv, env=dict(environ), capture_output=True, text=True, timeout=_DOCTOR_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        raise HostError("%s timed out after %g s" % (command, _DOCTOR_TIMEOUT_S)) from None
    except OSError as error:
        raise HostError("%s: %s" % (command, error)) from None
    try:
        return _parse_report(json.loads(done.stdout))
    except (ValueError, KeyError, TypeError, AttributeError):
        lines = done.stderr.strip().splitlines()
        raise HostError(
            "%s gave no report (exit %d): %s"
            % (command, done.returncode, lines[-1] if lines else "no output")
        ) from None


def _parse_report(data: Any) -> HostReport:
    """The report in the doctor's parsed JSON.

    Raises ``KeyError``, ``TypeError`` or ``AttributeError`` if ``data`` has another shape.
    """
    checks = {}
    for name, check in data["checks"].items():
        ok, warning = _STATUSES[check["status"]]
        checks[_text(name)] = Probe(ok, _text(check["detail"]), warning=warning)
    return HostReport(podman_bin=_text(data["podman_bin"]), store=_text(data["store"]), checks=checks)


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("not a string: %r" % (value,))
    return value
