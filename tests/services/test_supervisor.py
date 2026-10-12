"""The supervisor against a real throwaway child: a `python -c` HTTP server on a free port."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from sprout.services import health, supervisor

from tests.services.conftest import MARKER, free_port, server_command


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_start_waits_for_health_and_stop_terminates(paths):
    port = free_port()
    command = server_command(port)
    url = "http://127.0.0.1:%d/health" % port
    assert not health.http(url).ok
    pid = supervisor.start("web", command, dict(os.environ), paths, probe=lambda: health.http(url), timeout_s=20)
    try:
        assert health.http(url).ok
        assert int(paths.pid_file("web").read_text()) == pid
        state = supervisor.process_state(paths, "web", command)
        assert state.alive and state.pid == pid
        assert os.getpgid(pid) == pid, "the child leads a session of its own"
        log = paths.log_file("web").read_text()
        assert "start: " in log and MARKER in log and "healthy after" in log
    finally:
        outcome = supervisor.stop(paths, "web", command, grace_s=5)
    assert outcome == "stopped (pid %d)" % pid
    assert _gone(pid) and not paths.pid_file("web").exists()
    assert supervisor.process_state(paths, "web", command) == supervisor.ProcessState(None, False, "no pid file")


def test_a_stale_pid_file_for_a_dead_process_is_gone_not_alive(paths):
    dead = subprocess.Popen([sys.executable, "-c", "pass", MARKER])
    dead.wait()
    paths.service("web").mkdir(parents=True)
    paths.pid_file("web").write_text("%d\n" % dead.pid)
    state = supervisor.process_state(paths, "web", server_command(1))
    assert not state.alive and state.detail == "pid %d is gone" % dead.pid
    assert supervisor.stop(paths, "web", server_command(1)) == "not running (pid %d is gone)" % dead.pid
    assert not paths.pid_file("web").exists(), "a stale pid file is removed, never signalled"


def test_a_recycled_pid_running_another_program_is_not_ours(paths):
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", MARKER], start_new_session=True)
    try:
        paths.service("web").mkdir(parents=True)
        paths.pid_file("web").write_text("%d\n" % other.pid)
        state = supervisor.process_state(paths, "web", server_command(1))
        assert not state.alive and state.detail.startswith("pid %d is another program" % other.pid)
        assert supervisor.stop(paths, "web", server_command(1)).startswith("not running")
        assert other.poll() is None, "something that is not ours is never signalled"
    finally:
        other.kill()
        other.wait()


def test_a_child_that_dies_before_health_is_reported_with_its_log(paths):
    port = free_port()
    command = [sys.executable, "-c", "import sys; print('boom: bad config'); sys.exit(2)", MARKER]
    with pytest.raises(supervisor.StartError) as caught:
        supervisor.start("web", command, dict(os.environ), paths,
                         probe=lambda: health.http("http://127.0.0.1:%d/health" % port), timeout_s=20)
    assert "exited with status 2" in str(caught.value) and "boom: bad config" in str(caught.value)
    assert not paths.pid_file("web").exists()


def test_is_ours_compares_the_arguments_not_the_program():
    """A venv script's #! turns `litellm --config x` into `python litellm --config x`."""
    command = ["/v/bin/litellm", "--config", "/c/litellm.yaml", "--port", "4000"]
    assert supervisor.is_ours(["/v/bin/python", "/v/bin/litellm", "--config", "/c/litellm.yaml", "--port", "4000"], command)
    assert not supervisor.is_ours(["/v/bin/python", "/v/bin/litellm", "--config", "/other.yaml", "--port", "4000"], command)
    assert not supervisor.is_ours(["bash"], command)
    assert supervisor.is_ours(["/usr/bin/sleep"], ["sleep"]) and not supervisor.is_ours(["/usr/bin/bash"], ["sleep"])
