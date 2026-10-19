"""Start, stop and inspect the processes that the services CLI owns, detached from the shell.

A started service runs in a session of its own, with stdin from ``/dev/null`` and its
output appended to ``<state>/<name>/log``. So a closed terminal, or a ``kill %1`` in the
shell that ran ``up``, does not stop it. :func:`start` returns only after the service's
health probe passes, not when the pid exists.

The pid file links a later CLI run to the process. A pid counts as the service's only
while ``/proc/<pid>/cmdline`` still holds the arguments that it was started with. So a
recycled pid is never signalled, and a stale pid file after a reboot reads as gone.
:func:`stop` sends SIGTERM to the process group, waits, and then sends SIGKILL.

Each command that the CLI runs for a service goes through :func:`run_logged`, so the
service's log shows what was done to the service and what the tool printed.
Details: ``docs/SERVICES.md``.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sprout.services.health import Probe, which
from sprout.services.state import Paths

POLL_S = 0.5
GRACE_S = 20.0
KILL_WAIT_S = 5.0
TAIL_LINES = 20


class StartError(RuntimeError):
    """A service could not be started or did not become healthy. The message says why."""


@dataclass(frozen=True)
class ProcessState:
    """What the pid file of a service says, and whether that process still runs."""

    #: The pid in the pid file. None if there is no pid file or it is unreadable.
    pid: int | None
    #: True if a process with ``pid`` exists and runs the service's command.
    alive: bool
    #: A short phrase that says why the process counts as alive or not, such as ``pid 12 is gone``.
    detail: str


# --- the log ------------------------------------------------------------------


def log_line(log_path: Path, text: str) -> None:
    """Append ``text`` to the log as one line of the CLI's own, with a timestamp."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("[services %s] %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), text))


def tail(log_path: Path, lines: int = TAIL_LINES) -> str:
    """The last ``lines`` lines of the log, or an empty string if it cannot be read."""
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            return "".join(deque(fh, maxlen=lines))
    except OSError:
        return ""


def run_logged(argv: Sequence[str], env: Mapping[str, str], log_path: Path, cwd: Path) -> int:
    """Run ``argv`` to completion with its output appended to the log, and return its exit status.

    The log records ``argv`` as it is, so ``argv`` must hold no secret. Pass a secret in
    ``env``. Returns 127 if the program cannot be started.
    """
    log_line(log_path, "run: %s" % shlex.join(argv))
    cwd.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        try:
            done = subprocess.run(
                list(argv),
                cwd=str(cwd),
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except OSError as error:
            log_line(log_path, "run: %s: %s" % (argv[0], error))
            return 127
    log_line(log_path, "run: exit %d" % done.returncode)
    return done.returncode


def resolve_program(name: str) -> str:
    """The executable for ``name``: a path as it is, a bare name from PATH, then ``~/.local/bin``.

    Raises :class:`StartError` if a bare name is in neither.
    """
    if "/" in name:
        return name
    found = which(name)
    if not found:
        raise StartError("%s is not on PATH or in ~/.local/bin" % name)
    return found


# --- what is running ----------------------------------------------------------


def read_pid(paths: Paths, name: str) -> int | None:
    """The pid in the pid file of the service ``name``, or None if it is missing or unreadable."""
    pid_file = paths.pid_file(name)
    try:
        return int(pid_file.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def cmdline(pid: int) -> list[str] | None:
    """The arguments of process ``pid``. None if they are unreadable or empty, as for a zombie."""
    try:
        raw = Path("/proc/%d/cmdline" % pid).read_bytes()
    except OSError:
        return None
    parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    return parts or None


def is_ours(argv: Sequence[str], command: Sequence[str]) -> bool:
    """True if the live arguments ``argv`` belong to a process started with ``command``.

    Each argument of ``command`` must be in ``argv``. For a command without arguments,
    the base names of the two programs must match.
    """
    # The program is not compared: a venv script's ``#!`` turns ``.../bin/litellm --config X``
    # into ``.../bin/python .../bin/litellm --config X``.
    arguments = list(command[1:])
    if not arguments:
        return bool(argv) and os.path.basename(argv[0]) == os.path.basename(command[0])
    return all(argument in argv for argument in arguments)


def _exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _zombie(pid: int) -> bool:
    try:
        status = Path("/proc/%d/status" % pid).read_text(encoding="ascii", errors="replace")
    except OSError:
        return False
    for line in status.splitlines():
        if line.startswith("State:"):
            return line.split()[1:2] == ["Z"]
    return False


def _reap(pid: int) -> None:
    """Collect ``pid`` if it is an exited child of this process.

    An exited child that nobody collects stays a zombie, and ``kill -0`` still finds it.
    """
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


def _gone(pid: int) -> bool:
    _reap(pid)
    return not _exists(pid) or _zombie(pid)


def process_state(paths: Paths, name: str, command: Sequence[str]) -> ProcessState:
    """The state of the process in the pid file of the service ``name``, started with ``command``."""
    pid_file = paths.pid_file(name)
    if not pid_file.is_file():
        return ProcessState(None, False, "no pid file")
    pid = read_pid(paths, name)
    if pid is None:
        return ProcessState(None, False, "unreadable pid file %s" % pid_file)
    if _gone(pid):
        return ProcessState(pid, False, "pid %d is gone" % pid)
    argv = cmdline(pid)
    if argv is None:
        return ProcessState(pid, False, "pid %d is not readable (another user's?)" % pid)
    if not is_ours(argv, command):
        return ProcessState(pid, False, "pid %d is another program (%s)" % (pid, argv[0]))
    return ProcessState(pid, True, "pid %d" % pid)


# --- start / stop -------------------------------------------------------------


def wait_for(
    probe: Callable[[], Probe],
    timeout_s: float,
    *,
    still_running: Callable[[], str | None],
    interval_s: float = POLL_S,
) -> tuple[Probe, float]:
    """Poll ``probe`` until it passes, the process dies or ``timeout_s`` runs out.

    Returns the last result and the seconds waited. ``still_running`` returns None while
    the process lives, else why it died. A dead process ends the wait at once.
    """
    started = time.monotonic()
    while True:
        result = probe()
        elapsed = time.monotonic() - started
        if result.ok:
            return result, elapsed
        gone = still_running()
        if gone is not None:
            return Probe(False, gone), elapsed
        if elapsed >= timeout_s:
            return result, elapsed
        time.sleep(min(interval_s, timeout_s - elapsed))


def start(
    name: str,
    command: Sequence[str],
    env: Mapping[str, str],
    paths: Paths,
    *,
    probe: Callable[[], Probe],
    timeout_s: float,
) -> int:
    """Start ``command`` detached in the service's state directory, wait for ``probe``, return the pid.

    Raises :class:`StartError` if the command cannot start, exits first, or is not healthy
    after ``timeout_s``. A process that still runs then is left running for inspection.
    """
    directory = paths.service(name)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = paths.log_file(name)
    argv = [resolve_program(command[0]), *command[1:]]
    log_line(log_path, "start: %s" % shlex.join(argv))
    with log_path.open("ab") as log:
        try:
            child = subprocess.Popen(
                argv,
                # Never the checkout: conmon can write an ``oom`` marker into its working directory.
                cwd=str(directory),
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            raise StartError("%s: cannot start %s: %s" % (name, argv[0], error)) from None
    paths.pid_file(name).write_text("%d\n" % child.pid, encoding="ascii")

    def exited() -> str | None:
        code = child.poll()
        return None if code is None else "exited with status %d" % code

    result, elapsed = wait_for(probe, timeout_s, still_running=exited)
    if result.ok:
        log_line(log_path, "start: pid %d healthy after %.1f s (%s)" % (child.pid, elapsed, result.detail))
        return child.pid
    if child.poll() is not None:
        paths.pid_file(name).unlink(missing_ok=True)
        log_line(log_path, "start: pid %d %s" % (child.pid, result.detail))
        raise StartError(
            "%s: %s before it answered its health probe; last lines of %s:\n%s"
            % (name, result.detail, log_path, tail(log_path))
        )
    log_line(
        log_path,
        "start: pid %d not healthy after %.0f s (%s); left running" % (child.pid, timeout_s, result.detail),
    )
    raise StartError(
        "%s: pid %d is running but not healthy after %.0f s (%s); it is left running for you "
        "to look at (`logs %s`, then `down %s`); last lines of %s:\n%s"
        % (name, child.pid, timeout_s, result.detail, name, name, log_path, tail(log_path))
    )


def _signal(pid: int, sig: int) -> None:
    """Send ``sig`` to the process group that ``pid`` leads, else to ``pid`` alone.

    A started service leads its own group, so its children get the signal too.
    """
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
        else:
            os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _wait_gone(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        if _gone(pid):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def stop(
    paths: Paths,
    name: str,
    command: Sequence[str],
    *,
    grace_s: float = GRACE_S,
    kill_wait_s: float = KILL_WAIT_S,
) -> str:
    """Stop the service's process, and return one line that says what happened.

    Sends SIGTERM, waits up to ``grace_s``, then sends SIGKILL. A stale pid file is removed,
    and its pid is not signalled. Raises ``RuntimeError`` if the process survives SIGKILL.
    """
    state = process_state(paths, name, command)
    pid_file = paths.pid_file(name)
    if not state.alive:
        pid_file.unlink(missing_ok=True)  # a stale pid is removed, never signalled
        return "not running (%s)" % state.detail
    pid = state.pid or 0
    log_path = paths.log_file(name)
    log_line(log_path, "stop: SIGTERM to pid %d" % pid)
    _signal(pid, signal.SIGTERM)
    if _wait_gone(pid, grace_s):
        pid_file.unlink(missing_ok=True)
        return "stopped (pid %d)" % pid
    log_line(log_path, "stop: pid %d still alive after %.0f s, SIGKILL" % (pid, grace_s))
    _signal(pid, signal.SIGKILL)
    if _wait_gone(pid, kill_wait_s):
        pid_file.unlink(missing_ok=True)
        return "killed (pid %d ignored SIGTERM for %.0f s)" % (pid, grace_s)
    raise RuntimeError("%s: pid %d survived SIGKILL" % (name, pid))
