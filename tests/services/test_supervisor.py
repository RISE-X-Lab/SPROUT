"""The supervisor with real child processes: a service starts detached and is up once its probe passes,
and a stop signals only the process that the pid file proves to be the service's.

The service is ``SERVICE_PROGRAM``. Its probe is a file that the program creates when it is ready.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from sprout.services.health import Probe
from sprout.services.supervisor import (
    ProcessState,
    StartError,
    is_ours,
    process_state,
    run_logged,
    start,
    stop,
)
from tests.services.helpers import is_running, log_lines, ready_probe, service_command
from tests.support.waiting import wait_until


def start_ready(paths, tmp_path: Path, *options: str) -> tuple[int, list[str]]:
    """Start SERVICE_PROGRAM with ``options`` as the service ``web``, and return its pid and command."""
    ready = tmp_path / "ready"
    command = service_command(tmp_path, *options, "--ready", str(ready))
    pid = start("web", command, dict(os.environ), paths, probe=ready_probe(ready), timeout_s=20)
    return pid, command


def write_pid_file(paths, pid: int) -> None:
    paths.service("web").mkdir(parents=True, exist_ok=True)
    paths.pid_file("web").write_text("%d\n" % pid)


def another_program(tmp_path: Path) -> subprocess.Popen:
    """A process that runs another program than the service. Its argv names tmp_path for the cleanup."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", str(tmp_path)])


# --- start --------------------------------------------------------------------


def test_start_returns_the_pid_once_the_probe_passes_and_the_pid_file_names_it(paths, tmp_path):
    pid, command = start_ready(paths, tmp_path)

    assert process_state(paths, "web", command) == ProcessState(pid, True, "pid %d" % pid)


def test_a_started_service_leads_a_session_of_its_own_and_reads_dev_null(paths, tmp_path):
    """So a closed terminal, or a ``kill %1`` in the shell that ran ``up``, does not stop it."""
    record = tmp_path / "record.json"

    pid, _ = start_ready(paths, tmp_path, "--record", str(record))

    assert (os.getsid(pid), json.loads(record.read_text())["stdin"]) == (pid, "/dev/null")


def test_a_service_runs_in_its_state_directory_and_never_in_the_checkout(paths, tmp_path):
    """conmon can write an ``oom`` marker into the working directory of the podman that it runs under."""
    record = tmp_path / "record.json"

    start_ready(paths, tmp_path, "--record", str(record))

    assert json.loads(record.read_text())["cwd"] == str(paths.service("web"))


def test_a_service_gets_the_environment_that_start_is_given_and_not_the_callers(paths, tmp_path, monkeypatch):
    record = tmp_path / "record.json"
    ready = tmp_path / "ready"
    command = service_command(tmp_path, "--record", str(record), "--ready", str(ready))
    monkeypatch.setenv("FROM_THE_CALLER", "caller")

    start("web", command, {"FOR_THE_SERVICE": "service"}, paths, probe=ready_probe(ready), timeout_s=20)

    env = json.loads(record.read_text())["env"]
    assert (env.get("FOR_THE_SERVICE"), env.get("FROM_THE_CALLER")) == ("service", None)


def test_the_output_of_a_service_is_appended_to_its_log_after_the_start_line(paths, tmp_path):
    paths.service("web").mkdir(parents=True)
    paths.log_file("web").write_text("an earlier line\n")

    _, command = start_ready(paths, tmp_path, "--say", "listening")

    assert log_lines(paths.log_file("web"))[:3] == [
        "an earlier line",
        "[services] start: %s" % shlex.join(command),
        "listening",
    ]


def test_a_service_that_exits_before_its_probe_passes_fails_the_start_with_the_last_lines_of_its_log(
    paths, tmp_path
):
    command = service_command(tmp_path, "--say", "no such config: /path/to/config.json", "--exit", "3")

    with pytest.raises(StartError) as caught:
        start("web", command, dict(os.environ), paths, probe=lambda: Probe(False, "not yet"), timeout_s=20)

    lines = str(caught.value).splitlines()
    assert (lines[0], "no such config: /path/to/config.json" in lines, paths.pid_file("web").exists()) == (
        "web: exited with status 3 before it answered its health probe; last lines of %s:"
        % paths.log_file("web"),
        True,
        False,
    )


def test_a_service_that_runs_but_is_not_healthy_in_time_fails_the_start_and_is_left_running(paths, tmp_path):
    command = service_command(tmp_path)

    with pytest.raises(StartError) as caught:
        start("web", command, dict(os.environ), paths, probe=lambda: Probe(False, "HTTP 503"), timeout_s=0.3)

    state = process_state(paths, "web", command)
    assert (str(caught.value).splitlines()[0], state.alive) == (
        "web: pid %d is running but not healthy after 0 s (HTTP 503); it is left running for you to look at "
        "(`logs web`, then `down web`); last lines of %s:" % (state.pid, paths.log_file("web")),
        True,
    )


def test_a_bare_program_name_that_is_nowhere_fails_the_start_and_starts_nothing(paths, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(StartError) as caught:
        start("web", ["litellm", "--port", "4000"], {}, paths, probe=lambda: Probe(True, ""), timeout_s=20)

    assert (str(caught.value), paths.pid_file("web").exists()) == (
        "litellm is not on PATH or in ~/.local/bin",
        False,
    )


# --- what the pid file says ---------------------------------------------------


def test_without_a_pid_file_the_service_is_not_running(paths, tmp_path):
    assert process_state(paths, "web", service_command(tmp_path)) == ProcessState(None, False, "no pid file")


def test_a_pid_file_that_holds_no_pid_reads_as_not_running(paths, tmp_path):
    paths.service("web").mkdir(parents=True)
    paths.pid_file("web").write_text("not a pid\n")

    state = process_state(paths, "web", service_command(tmp_path))

    assert state == ProcessState(None, False, "unreadable pid file %s" % paths.pid_file("web"))


def test_a_pid_file_of_a_process_that_ended_reads_as_gone(paths, tmp_path):
    """A pid file that survived a reboot is such a file."""
    ended = subprocess.Popen(service_command(tmp_path, "--exit", "0"))
    ended.wait()
    write_pid_file(paths, ended.pid)

    state = process_state(paths, "web", service_command(tmp_path))

    assert state == ProcessState(ended.pid, False, "pid %d is gone" % ended.pid)


def test_a_pid_file_of_an_exited_child_that_nobody_reaped_reads_as_gone(paths, tmp_path):
    """``kill -0`` still finds a zombie."""
    exited = subprocess.Popen(service_command(tmp_path, "--exit", "0"))
    wait_until(lambda: not is_running(exited.pid), what="the exit of the child")
    write_pid_file(paths, exited.pid)

    state = process_state(paths, "web", service_command(tmp_path))

    assert state == ProcessState(exited.pid, False, "pid %d is gone" % exited.pid)


def test_a_recycled_pid_that_runs_another_program_is_not_the_service(paths, tmp_path):
    other = another_program(tmp_path)
    write_pid_file(paths, other.pid)

    state = process_state(paths, "web", service_command(tmp_path, "--port", "4000"))

    assert state == ProcessState(
        other.pid, False, "pid %d is another program (%s)" % (other.pid, sys.executable)
    )


@pytest.mark.parametrize(
    "argv, command, expected",
    [
        # A venv script's #! turns `litellm --config X` into `python litellm --config X`.
        (
            ["/path/to/venv/bin/python", "/path/to/venv/bin/litellm", "--config", "/path/to/litellm.yaml"],
            ["/path/to/venv/bin/litellm", "--config", "/path/to/litellm.yaml"],
            True,
        ),
        (
            ["/path/to/venv/bin/python", "/path/to/venv/bin/litellm", "--config", "/path/to/other.yaml"],
            ["/path/to/venv/bin/litellm", "--config", "/path/to/litellm.yaml"],
            False,
        ),
        (["/usr/bin/sleep"], ["sleep"], True),
        (["/usr/bin/bash"], ["sleep"], False),
        ([], ["sleep"], False),
    ],
)
def test_a_process_is_the_service_when_it_has_each_argument_of_the_command(argv, command, expected):
    """A command without arguments compares the base names of the two programs."""
    assert is_ours(argv, command) is expected


# --- stop ---------------------------------------------------------------------


def test_stop_ends_the_service_with_sigterm_and_removes_its_pid_file(paths, tmp_path):
    pid, command = start_ready(paths, tmp_path)

    outcome = stop(paths, "web", command, grace_s=5)

    assert (outcome, is_running(pid), paths.pid_file("web").exists()) == (
        "stopped (pid %d)" % pid,
        False,
        False,
    )


def test_stop_also_ends_the_processes_that_the_service_started(paths, tmp_path):
    child = tmp_path / "child"
    _, command = start_ready(paths, tmp_path, "--child", str(child))

    stop(paths, "web", command, grace_s=5)

    wait_until(
        lambda: not is_running(int(child.read_text())), timeout_s=5, what="the end of the service's child"
    )


def test_a_service_that_ignores_sigterm_is_killed_after_the_grace_period(paths, tmp_path):
    pid, command = start_ready(paths, tmp_path, "--ignore-term")

    outcome = stop(paths, "web", command, grace_s=0.3)

    assert (outcome, is_running(pid)) == ("killed (pid %d ignored SIGTERM for 0 s)" % pid, False)


def test_stop_removes_a_pid_file_of_a_process_that_ended(paths, tmp_path):
    ended = subprocess.Popen(service_command(tmp_path, "--exit", "0"))
    ended.wait()
    write_pid_file(paths, ended.pid)

    outcome = stop(paths, "web", service_command(tmp_path))

    assert (outcome, paths.pid_file("web").exists()) == ("not running (pid %d is gone)" % ended.pid, False)


def test_stop_never_signals_a_recycled_pid_that_runs_another_program(paths, tmp_path):
    other = another_program(tmp_path)
    write_pid_file(paths, other.pid)

    outcome = stop(paths, "web", service_command(tmp_path, "--port", "4000"))

    assert (outcome, other.poll(), paths.pid_file("web").exists()) == (
        "not running (pid %d is another program (%s))" % (other.pid, sys.executable),
        None,
        False,
    )


# --- commands run for a service -----------------------------------------------


def test_a_command_run_for_a_service_logs_its_argv_its_output_and_its_status(paths):
    log = paths.log_file("db")

    status = run_logged(["/bin/sh", "-c", "echo created; exit 3"], {}, log, paths.service("db"))

    assert (status, log_lines(log)) == (
        3,
        ["[services] run: /bin/sh -c 'echo created; exit 3'", "created", "[services] run: exit 3"],
    )


def test_a_command_that_cannot_start_returns_127_and_logs_why(paths):
    log = paths.log_file("db")

    status = run_logged(["/path/to/absent/tool"], {}, log, paths.service("db"))

    assert (status, log_lines(log)) == (
        127,
        [
            "[services] run: /path/to/absent/tool",
            "[services] run: /path/to/absent/tool: "
            "[Errno 2] No such file or directory: '/path/to/absent/tool'",
        ],
    )
