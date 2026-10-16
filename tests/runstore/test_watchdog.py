import subprocess
import sys
from threading import Event
import time

import pytest

from tests.runstore.test_manager import eventually
from sprout.worker.attempts import AttemptStopped, Watchdog
from sprout.worker.attempts import process_identity, stop_process


@pytest.mark.parametrize("reason", ["timeout", "lease_expired", "worker_shutdown"])
def test_watchdog_stops_child_without_waiting_for_io_or_database(reason):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    identity = process_identity(process.pid)
    stop = Event()
    watchdog = Watchdog(stop_process, stop, 0.2 if reason == "lease_expired" else 10)
    try:
        watchdog.arm(identity, time.monotonic() + (0.2 if reason == "timeout" else 60))
        if reason == "worker_shutdown":
            stop.set()
        eventually(lambda: process.poll() is not None, timeout=3)
        assert watchdog.reason == reason
        with pytest.raises(AttemptStopped, match=reason):
            watchdog.check()
    finally:
        watchdog.close()
        stop_process(identity)
        process.wait(timeout=5)


def test_a_requested_stop_kills_the_child_and_is_not_overwritten_by_a_later_deadline():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    identity = process_identity(process.pid)
    watchdog = Watchdog(stop_process, Event(), 10)
    try:
        watchdog.arm(identity, time.monotonic() + 0.3)
        watchdog.request_stop("stopped")
        eventually(lambda: process.poll() is not None, timeout=3)
        time.sleep(0.5)   # past the deadline
        assert watchdog.reason == "stopped"
        with pytest.raises(AttemptStopped, match="stopped"):
            watchdog.check()
    finally:
        watchdog.close()
        stop_process(identity)
        process.wait(timeout=5)


def test_late_registration_cannot_start_after_watchdog_expired():
    killed = []
    watchdog = Watchdog(lambda identity: killed.append(identity), Event(), 0.1)
    try:
        eventually(lambda: watchdog.reason is not None)
        with pytest.raises(AttemptStopped):
            watchdog.arm({"pid": 123}, time.monotonic() + 60)
        assert {"pid": 123} in killed
    finally:
        watchdog.close()


def test_finished_child_is_not_misreported_as_timeout_during_collection():
    killed = []
    watchdog = Watchdog(lambda identity: killed.append(identity), Event(), 10)
    try:
        watchdog.arm({"pid": 123}, time.monotonic() + 0.05, running=lambda: False)
        time.sleep(0.2)
        watchdog.check()
        assert watchdog.reason is None and killed == []
    finally:
        watchdog.close()
