"""The ``Watchdog`` of an attempt: when it stops the attempt's process, and which reason it reports.

Its ``stop_process`` is a spy, so no process is started or stopped. The identity is never a real
process. The worker's own tests in ``test_supervision.py`` stop real children.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from threading import Event
from typing import Any

import pytest

from sprout.worker.attempts import AttemptStopped, Watchdog
from tests.support.waiting import wait_until

CHILD = {"pid": 4242, "start_ticks": "1000", "boot_id": "boot"}


class StopSpy:
    """A ``stop_process`` that records the identity of each process that it was asked to stop."""

    def __init__(self) -> None:
        self.identities: list[dict[str, Any] | None] = []

    def __call__(self, identity: dict[str, Any] | None) -> bool:
        self.identities.append(identity)
        return True


@pytest.fixture
def watchdog() -> Iterator[Callable[..., tuple[Watchdog, StopSpy]]]:
    """Make watchdogs with a fresh spy each, and close each one after the test."""
    made: list[Watchdog] = []

    def make(
        *, stop: Event | None = None, cut_off: Callable[[], bool] = lambda: False
    ) -> tuple[Watchdog, StopSpy]:
        spy = StopSpy()
        made.append(Watchdog(spy, stop if stop is not None else Event(), cut_off=cut_off))
        return made[-1], spy

    yield make
    for each in made:
        each.close()


def test_the_watchdog_stops_the_process_at_its_deadline(watchdog):
    dog, spy = watchdog()

    dog.arm(CHILD, time.monotonic() + 0.1)
    wait_until(lambda: spy.identities, what="the stop of the process")

    assert (dog.reason, spy.identities) == ("timeout", [CHILD])
    with pytest.raises(AttemptStopped, match="^timeout$"):
        dog.check()


def test_the_watchdog_stops_the_process_when_the_worker_shuts_down(watchdog):
    stop = Event()
    dog, spy = watchdog(stop=stop)
    dog.arm(CHILD, time.monotonic() + 60)

    stop.set()
    wait_until(lambda: spy.identities, what="the stop of the process")

    assert (dog.reason, spy.identities) == ("worker_shutdown", [CHILD])
    with pytest.raises(AttemptStopped, match="^worker_shutdown$"):
        dog.check()


def test_the_watchdog_stops_the_process_when_the_worker_is_cut_off_from_the_run_store(watchdog):
    """Another worker can take the job over once the worker is dead, so the child must stop first."""
    cut_off = Event()
    dog, spy = watchdog(cut_off=cut_off.is_set)
    dog.arm(CHILD, time.monotonic() + 60)

    cut_off.set()
    wait_until(lambda: spy.identities, what="the stop of the process")

    assert (dog.reason, spy.identities) == ("worker_cut_off", [CHILD])


def test_a_requested_stop_stops_the_process_at_once_with_its_own_reason(watchdog):
    dog, spy = watchdog()
    dog.arm(CHILD, time.monotonic() + 60)

    dog.request_stop("stopped")

    assert (dog.reason, spy.identities) == ("stopped", [CHILD])


def test_the_first_reason_is_kept_and_the_process_is_stopped_once(watchdog):
    dog, spy = watchdog()
    dog.arm(CHILD, time.monotonic())
    wait_until(lambda: spy.identities, what="the stop at the deadline")

    dog.request_stop("stopped")

    assert (dog.reason, spy.identities) == ("timeout", [CHILD])


def test_a_deadline_that_passes_after_a_requested_stop_does_not_replace_its_reason(watchdog):
    """A stopped job is never retried, and a timed-out one can be. The test waits 0.2 s past the deadline,
    four times the 50 ms in which the watchdog notices a deadline."""
    dog, spy = watchdog()
    deadline = time.monotonic() + 0.1
    dog.arm(CHILD, deadline)

    dog.request_stop("stopped")
    wait_until(lambda: time.monotonic() > deadline + 0.2, what="0.2 s past the deadline")

    assert (dog.reason, spy.identities) == ("stopped", [CHILD])


def test_a_process_registered_after_the_attempt_stopped_is_stopped_at_once(watchdog):
    """The child can start after a shutdown, for example while the worker waited for the database. The
    attempt then ends as the shutdown ends it, so ``arm`` raises nothing."""
    stop = Event()
    dog, spy = watchdog(stop=stop)
    stop.set()
    wait_until(lambda: dog.reason is not None, what="the shutdown")

    dog.arm(CHILD, time.monotonic() + 60)

    assert (dog.reason, spy.identities) == ("worker_shutdown", [None, CHILD])


class CallCounter:
    """A ``cut_off`` that is never true, and counts its calls. The watchdog makes one call each pass."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return False


def test_a_deadline_does_not_stop_a_process_that_has_already_exited(watchdog):
    """The worker reads the outcome of an exited child after its deadline. That attempt did not time out.

    The second call of ``cut_off`` comes after the first pass checked the deadline."""
    calls = CallCounter()
    dog, spy = watchdog(cut_off=calls)

    dog.arm(CHILD, time.monotonic(), running=lambda: False)
    wait_until(lambda: calls.calls >= 2, what="a pass of the watchdog after the deadline")

    assert (dog.reason, spy.identities) == (None, [])
