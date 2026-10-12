"""Start, stop and inspect the processes this CLI owns, detached from the shell.

A service is started in a session of its own (``start_new_session``: fork, then
``setsid`` in the child), with stdin from ``/dev/null`` and both output streams
appended to ``<state>/<name>/log``, so closing the terminal that ran ``up`` or a
later ``kill %1`` in it cannot take the service down -- the hand-started proxy
this replaces died that way more than once. The CLI then waits for the
service's own health probe, not merely for the pid to exist: a proxy that
imports for twenty seconds and then dies on a bad config would otherwise be
reported up.

The pid file is the only link between a CLI run and a process started by an
earlier one, and pids are recycled, so a pid is believed to be ours only when
``/proc/<pid>/cmdline`` still carries the arguments we started it with; a stale
file after a reboot or a crash is reported as "gone" and removed, never
signalled. Stopping is SIGTERM to the process group, 20 s for a clean exit, then
SIGKILL -- uvicorn and the worker manager both handle SIGTERM, and the group
takes a worker's children along.

Every command the CLI runs on a service's behalf (a prepare step, a podman
verb) goes through :func:`run_logged`, so the service's log shows what was
done to it and what the tool said.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from sprout.services.health import Probe, which
from sprout.services.state import Paths

POLL_S = 0.5
GRACE_S = 20.0
KILL_WAIT_S = 5.0
TAIL_LINES = 20


class StartError(RuntimeError):
    """A service that could not be started or did not come up; the message says why."""


@dataclass(frozen=True)
class ProcessState:
    #: What the pid file says; None when there is none (or it is unreadable).
    pid: int | None
    #: A process with that pid exists and is the one we started.
    alive: bool
    detail: str


# --- the log ------------------------------------------------------------------

def log_line(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("[services %s] %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S%z"), text))


def tail(log_path: Path, lines: int = TAIL_LINES) -> str:
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            return "".join(deque(fh, maxlen=lines))
    except OSError:
        return ""


def run_logged(argv: Sequence[str], env: Mapping[str, str], log_path: Path, cwd: Path) -> int:
    """Run a command to completion with its output in the service's log; the exit status.

    The argv is written to the log as typed, so it must carry no secret: a
    secret travels in ``env`` (podman's ``-e NAME`` reads it from there).
    """
    log_line(log_path, "run: %s" % shlex.join(argv))
    cwd.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        try:
            done = subprocess.run(list(argv), cwd=str(cwd), env=dict(env), stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT)
        except OSError as error:
            log_line(log_path, "run: %s: %s" % (argv[0], error))
            return 127
    log_line(log_path, "run: exit %d" % done.returncode)
    return done.returncode


def resolve_program(name: str) -> str:
    """A path as given; a bare name through PATH, then ``~/.local/bin``."""
    if "/" in name:
        return name
    found = which(name)
    if not found:
        raise StartError("%s is not on PATH or in ~/.local/bin" % name)
    return found


# --- what is running ----------------------------------------------------------

def read_pid(paths: Paths, name: str) -> int | None:
    pid_file = paths.pid_file(name)
    try:
        return int(pid_file.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def cmdline(pid: int) -> list[str] | None:
    """``/proc/<pid>/cmdline`` split on NUL; None when unreadable or empty (a zombie)."""
    try:
        raw = Path("/proc/%d/cmdline" % pid).read_bytes()
    except OSError:
        return None
    parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    return parts or None


def is_ours(argv: Sequence[str], command: Sequence[str]) -> bool:
    """Does a live cmdline belong to ``command``?

    The program token is not compared: a venv script's ``#!`` turns
    ``.../bin/litellm --config X`` into ``.../bin/python .../bin/litellm
    --config X``. Every argument we passed must still be there instead; a
    recycled pid running something else fails that, and a command with no
    arguments falls back to the program's base name.
    """
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
        return True            # exists, owned by someone else
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
    """Collect an exited child of this very process, so it does not linger as a
    zombie that ``kill -0`` still finds (the CLI's own tests hit this)."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


def _gone(pid: int) -> bool:
    _reap(pid)
    return not _exists(pid) or _zombie(pid)


def process_state(paths: Paths, name: str, command: Sequence[str]) -> ProcessState:
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

def wait_for(probe: Callable[[], Probe], timeout_s: float, *, still_running: Callable[[], str | None],
             interval_s: float = POLL_S) -> tuple[Probe, float]:
    """Poll ``probe`` until it passes, the process dies or the timeout runs out.

    ``still_running`` returns None while the process lives, else why it does
    not; checked after each failed probe so a crash is reported at once instead
    of after the full timeout.
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


def start(name: str, command: Sequence[str], env: Mapping[str, str], paths: Paths, *,
          probe: Callable[[], Probe], timeout_s: float) -> int:
    """Start ``command`` detached and wait for ``probe``; the pid, or a StartError.

    The child runs from its own state directory -- never from the checkout,
    where conmon's ``oom`` marker once dirtied the tree (AGENTS.md).
    """
    directory = paths.service(name)
    directory.mkdir(parents=True, exist_ok=True)
    log_path = paths.log_file(name)
    argv = [resolve_program(command[0]), *command[1:]]
    log_line(log_path, "start: %s" % shlex.join(argv))
    with log_path.open("ab") as log:
        try:
            child = subprocess.Popen(argv, cwd=str(directory), env=dict(env), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
        raise StartError("%s: %s before it answered its health probe; last lines of %s:\n%s"
                         % (name, result.detail, log_path, tail(log_path)))
    log_line(log_path, "start: pid %d not healthy after %.0f s (%s); left running" % (child.pid, timeout_s, result.detail))
    raise StartError("%s: pid %d is running but not healthy after %.0f s (%s); it is left running for you "
                     "to look at (`logs %s`, then `down %s`); last lines of %s:\n%s"
                     % (name, child.pid, timeout_s, result.detail, name, name, log_path, tail(log_path)))


def _signal(pid: int, sig: int) -> None:
    """The whole process group when the pid leads one (it does: setsid made it the
    leader), so a worker's children get the signal too; else just the pid."""
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


def stop(paths: Paths, name: str, command: Sequence[str], *, grace_s: float = GRACE_S,
         kill_wait_s: float = KILL_WAIT_S) -> str:
    """SIGTERM, up to ``grace_s`` for a clean exit, then SIGKILL; one line saying what happened."""
    state = process_state(paths, name, command)
    pid_file = paths.pid_file(name)
    if not state.alive:
        pid_file.unlink(missing_ok=True)     # a stale pid is removed, never signalled
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
