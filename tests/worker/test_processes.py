"""``process_identity`` and ``stop_process``: how a worker finds and stops the process group of an attempt.

Each test starts its own short-lived processes, and stops them when it ends.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.attempts import process_identity, stop_process
from tests.support.waiting import wait_until

SLEEP = [sys.executable, "-c", "import time; time.sleep(120)"]


@pytest.fixture
def group_leader() -> Iterator[subprocess.Popen[bytes]]:
    """A sleeping process that leads its own process group, as an attempt's child does."""
    process = subprocess.Popen(SLEEP, start_new_session=True)
    yield process
    process.kill()
    process.wait(timeout=5)


def test_a_live_process_with_another_start_time_than_the_identity_is_not_stopped(group_leader):
    """The kernel reused the pid of the attempt's child for this process."""
    identity = {**process_identity(group_leader.pid), "start_ticks": "0"}

    gone = stop_process(identity)

    assert (gone, group_leader.poll()) == (True, None)


@pytest.fixture
def orphaned_group(tmp_path: Path) -> Iterator[tuple[dict[str, Any], int]]:
    """The identity of a dead group leader, and the pid of its child that still runs in its group."""
    ready = tmp_path / "descendant.pid"
    program = (
        "import subprocess, sys, time\n"
        "descendant = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "open(%r, 'w').write(str(descendant.pid))\n"
        "time.sleep(120)\n" % str(ready)
    )
    leader = subprocess.Popen([sys.executable, "-c", program], start_new_session=True)
    identity = process_identity(leader.pid)
    wait_until(lambda: ready.exists() and ready.read_text(), what="the start of the descendant")
    descendant = int(ready.read_text())
    leader.kill()
    leader.wait(timeout=5)
    yield identity, descendant
    with contextlib.suppress(ProcessLookupError):
        os.kill(descendant, signal.SIGKILL)


def test_a_process_left_in_the_group_of_a_dead_leader_is_stopped(orphaned_group):
    """An attempt's child can die and leave its own children running in its group."""
    identity, descendant = orphaned_group

    gone = stop_process(identity)

    assert (gone, process_identity(descendant)) == (True, None)


def test_the_group_of_an_identity_from_another_boot_is_not_stopped(orphaned_group):
    """After a reboot, a group with the pid of the attempt's child belongs to another program."""
    identity, descendant = orphaned_group

    gone = stop_process({**identity, "boot_id": "another-boot"})

    assert gone is True
    assert process_identity(descendant) is not None, "the group of another program was stopped"


def test_a_process_that_exited_and_was_not_yet_reaped_has_no_identity(group_leader):
    """A zombie can never be stopped, so a stop that waited for it would wait for ever."""
    group_leader.kill()
    os.waitid(os.P_PID, group_leader.pid, os.WEXITED | os.WNOWAIT)

    assert process_identity(group_leader.pid) is None
    assert Path("/proc/%d" % group_leader.pid).exists(), "the process was reaped, so it is no zombie"
