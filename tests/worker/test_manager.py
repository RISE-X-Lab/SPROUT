"""``WorkerManager``: the claim loop that registers its worker, bounds its attempts and sweeps dead ones.

The manager runs against ``FakeWorker``, whose queue is in memory and whose attempts block until the
test releases them. Each manager runs in a thread of the test, which stops it at the end.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterator
from threading import Event, Lock, Thread
from types import SimpleNamespace
from typing import Any

import pytest

from sprout.runstore.store import Conflict
from sprout.worker.manager import CUT_OFF_S, Heartbeat, WorkerManager
from tests.support.waiting import wait_until


class FakeQueue:
    """The part of the Run Store that the manager uses, in memory: workers and queued jobs."""

    def __init__(self, *, queued: int = 0, left: tuple[str, ...] = ()) -> None:
        self.queued = deque({"id": "job-%d" % number} for number in range(queued))
        #: The running jobs of the worker in the Run Store: those that ``left`` names.
        self.running = [{"id": job_id} for job_id in left]
        #: "sweep" or "claim" for each call of ``settle_dead_workers`` or ``claim``, in order.
        self.calls: list[str] = []
        #: The monotonic time of each ``claim`` call.
        self.claims: list[float] = []
        #: Raised by the next ``claim`` calls, one each, before the queue answers again.
        self.claim_errors: list[Exception] = []
        #: ("register", worker id, host, session, grades) or ("stopped", worker id, session), in order.
        self.workers: list[tuple[Any, ...]] = []
        #: Whether a registration finds the worker live in another session.
        self.held_elsewhere = False
        #: The answer of each heartbeat: True for an acknowledgment.
        self.acknowledge = True
        self.heartbeats = 0

    def worker(self, worker_id: str) -> dict[str, Any] | None:
        """None: no earlier session left a row of the worker."""
        return None

    def register_worker(self, worker_id: str, *, host: str, session: str, **options: Any) -> None:
        if self.held_elsewhere:
            raise Conflict(f"Worker {worker_id} is live in another session")
        self.workers.append(("register", worker_id, host, session, options["grades"]))

    def worker_heartbeat(self, worker_id: str, session: str) -> bool:
        self.heartbeats += 1
        return self.acknowledge

    def worker_stopped(self, worker_id: str, session: str) -> None:
        self.workers.append(("stopped", worker_id, session))

    def settle_dead_workers(self) -> list[str]:
        self.calls.append("sweep")
        return []

    def running_jobs(self, worker_id: str) -> list[dict[str, Any]]:
        return list(self.running)

    def claim(self, worker_id: str) -> dict[str, Any] | None:
        self.calls.append("claim")
        self.claims.append(time.monotonic())
        if self.claim_errors:
            raise self.claim_errors.pop(0)
        return self.queued.popleft() if self.queued else None


class FakeWorker:
    """A worker whose attempts block until the test releases them, and fail on request.

    ``blocked`` names the jobs whose attempts block. None, the default, names every job."""

    worker_id = "host-1/fake"
    host = "host-1"
    grades = (("dataset-sha256", "sha256:parser"),)

    def __init__(self, store: FakeQueue) -> None:
        self.store = store
        self.release = Event()
        self.blocked: set[str] | None = None
        #: The ids of the jobs whose attempts raise once released.
        self.failing: set[str] = set()
        #: ("attempt", job id) or ("settle", job id) for each call, in the order of the calls.
        self.events: list[tuple[str, str]] = []
        #: The ``stop`` event that each call got, by job id.
        self.stops: dict[str, Event | None] = {}
        #: The ``cut_off`` of each attempt, by job id.
        self.cut_offs: dict[str, Callable[[], bool] | None] = {}
        #: The most attempts that ran at the same time.
        self.peak = 0
        #: The calls of ``clean_up``, which raise with ``clean_up_fails``.
        self.clean_ups = 0
        self.clean_up_fails = False
        self._running: set[str] = set()
        self._lock = Lock()

    def run_claimed(
        self, job: dict[str, Any], *, stop: Event | None = None, cut_off: Callable[[], bool] | None = None
    ) -> str:
        with self._lock:
            self.events.append(("attempt", job["id"]))
            self.stops[job["id"]] = stop
            self.cut_offs[job["id"]] = cut_off
            self._running.add(job["id"])
            self.peak = max(self.peak, len(self._running))
        if self.blocked is None or job["id"] in self.blocked:
            self.release.wait(15)
        with self._lock:
            self._running.discard(job["id"])
        if job["id"] in self.failing:
            raise RuntimeError("the attempt of %s failed" % job["id"])
        return str(job["id"])

    def settle(self, job: dict[str, Any]) -> str:
        with self._lock:
            self.events.append(("settle", job["id"]))
        return str(job["id"])

    def clean_up(self) -> None:
        self.clean_ups += 1
        if self.clean_up_fails:
            raise OSError("podman did not answer")

    def idle(self) -> bool:
        """Whether no attempt runs now."""
        with self._lock:
            return not self._running


def attempts(*job_ids: str) -> list[tuple[str, str]]:
    return [("attempt", job_id) for job_id in job_ids]


@pytest.fixture
def running_manager() -> Iterator[Callable[..., tuple[Event, Thread]]]:
    """Start managers in threads, and give each one's stop event and thread.

    Each manager stops, and its worker's attempts are released, after the test."""
    started: list[tuple[Thread, Event, FakeWorker]] = []

    def start(worker: FakeWorker, **options: Any) -> tuple[Event, Thread]:
        manager = WorkerManager(worker, **{"poll_s": 0.01, **options})
        stop = Event()
        thread = Thread(target=manager.run, kwargs={"stop": stop}, daemon=True)
        thread.start()
        started.append((thread, stop, worker))
        return stop, thread

    yield start
    for thread, stop, worker in started:
        stop.set()
        worker.release.set()
        thread.join(timeout=10)


def test_the_manager_runs_at_most_concurrency_attempts_and_refills_a_slot_when_one_ends(running_manager):
    worker = FakeWorker(FakeQueue(queued=6))
    running_manager(worker, concurrency=4)
    wait_until(lambda: len(worker.events) == 4, what="four running attempts")

    worker.release.set()
    wait_until(lambda: len(worker.events) == 6 and worker.idle(), what="all six attempts")

    assert (worker.events, worker.peak) == (attempts("job-0", "job-1", "job-2", "job-3", "job-4", "job-5"), 4)


def test_a_blocked_attempt_holds_only_its_own_slot_and_the_others_go_on_with_the_queue(running_manager):
    worker = FakeWorker(FakeQueue(queued=4))
    worker.blocked = {"job-0"}

    running_manager(worker, concurrency=2)
    wait_until(lambda: len(worker.events) == 4, what="the attempts of all four jobs")

    assert (worker.events, worker.idle()) == (attempts("job-0", "job-1", "job-2", "job-3"), False)


def test_an_attempt_that_fails_does_not_stop_the_manager(running_manager):
    worker = FakeWorker(FakeQueue(queued=3))
    worker.failing = {"job-0"}
    worker.release.set()

    running_manager(worker, concurrency=1)
    wait_until(lambda: len(worker.events) == 3 and worker.idle(), what="all three attempts")

    assert worker.events == attempts("job-0", "job-1", "job-2")


def test_once_claims_one_job_waits_for_its_attempt_and_returns():
    queue = FakeQueue(queued=5)
    worker = FakeWorker(queue)
    worker.release.set()

    WorkerManager(worker, concurrency=4, poll_s=0.01).run(once=True)

    assert (worker.events, len(queue.queued), worker.idle()) == (attempts("job-0"), 4, True)


def test_the_manager_sets_its_stop_event_when_it_returns_so_that_running_attempts_stop():
    stop = Event()

    WorkerManager(FakeWorker(FakeQueue()), poll_s=0.01).run(stop=stop, once=True)

    assert stop.is_set(), "the manager returned and left its stop event unset"


def test_attempts_get_the_stop_event_of_the_manager(running_manager):
    worker = FakeWorker(FakeQueue(queued=2))

    stop, _ = running_manager(worker, concurrency=2)
    wait_until(lambda: len(worker.events) == 2, what="two attempts")

    assert worker.stops == {"job-0": stop, "job-1": stop}


def test_the_manager_first_settles_the_jobs_that_an_earlier_session_of_its_worker_left():
    """The Run Store renewed their leases at the registration, so no other worker takes them."""
    queue = FakeQueue(queued=1, left=("left-job",))
    worker = FakeWorker(queue)
    worker.release.set()

    WorkerManager(worker, poll_s=0.01).run(once=True)

    assert worker.events == [("settle", "left-job"), ("attempt", "job-0")]


def test_the_manager_settles_the_jobs_of_dead_workers_before_each_claim():
    """Another worker's sweep is what runs the jobs of a dead worker again."""
    queue = FakeQueue(queued=1)
    worker = FakeWorker(queue)
    worker.release.set()

    WorkerManager(worker, poll_s=0.01).run(once=True)

    assert queue.calls == ["sweep", "claim"]


def test_the_manager_registers_its_worker_and_its_grades_before_it_claims_and_stops_it_when_it_returns():
    """A claim gives the worker only the grades that its registration names."""
    queue = FakeQueue(queued=1)
    worker = FakeWorker(queue)
    worker.release.set()

    WorkerManager(worker, poll_s=0.01).run(once=True)

    [registered, stopped] = queue.workers
    session = registered[3]
    assert (registered, stopped) == (
        ("register", "host-1/fake", "host-1", session, (("dataset-sha256", "sha256:parser"),)),
        ("stopped", "host-1/fake", session),
    )
    assert worker.events == attempts("job-0")


def test_a_worker_that_is_live_in_another_session_claims_nothing():
    """Two sessions under one worker id would each settle the jobs of the other."""
    queue = FakeQueue(queued=1)
    queue.held_elsewhere = True

    with pytest.raises(Conflict, match="^Worker host-1/fake is live in another session$"):
        WorkerManager(FakeWorker(queue), poll_s=0.01).run(once=True)

    assert (queue.claims, queue.workers) == ([], [])


def test_each_attempt_is_cut_off_once_the_run_store_refuses_a_heartbeat_of_its_manager(running_manager):
    worker = FakeWorker(FakeQueue(queued=1))

    running_manager(worker, heartbeat_s=0.01)
    wait_until(lambda: worker.events, what="the attempt of the queued job")
    worker.store.acknowledge = False

    wait_until(lambda: worker.cut_offs["job-0"](), what="the cut-off of the attempt")


def test_a_manager_whose_session_lost_the_row_of_its_worker_stops(running_manager):
    """Another session runs under the worker id now, so this one must leave the jobs to it."""
    queue = FakeQueue()
    queue.acknowledge = False

    stop, thread = running_manager(FakeWorker(queue), heartbeat_s=0.01)
    thread.join(timeout=5)

    assert (thread.is_alive(), stop.is_set(), queue.workers[-1][0]) == (False, True, "stopped")


class Clock:
    """A monotonic clock that a test moves."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now


class SlowRegistration(FakeQueue):
    """A queue of one job whose registration answers ``CUT_OFF_S`` after it was sent, by ``clock``.

    Each heartbeat records how many claims came before it."""

    def __init__(self, clock: Clock) -> None:
        super().__init__(queued=1)
        self._clock = clock
        self.claims_at_heartbeat: list[int] = []

    def register_worker(self, worker_id: str, *, host: str, session: str, **options: Any) -> None:
        super().register_worker(worker_id, host=host, session=session, **options)
        self._clock.now += CUT_OFF_S

    def worker_heartbeat(self, worker_id: str, session: str) -> bool:
        self.claims_at_heartbeat.append(len(self.claims))
        return super().worker_heartbeat(worker_id, session)


def test_a_cut_off_worker_claims_nothing_until_a_heartbeat_is_acknowledged(running_manager, monkeypatch):
    clock = Clock()
    monkeypatch.setattr("sprout.worker.manager.time", clock)
    queue = SlowRegistration(clock)
    worker = FakeWorker(queue)
    worker.release.set()

    running_manager(worker, heartbeat_s=0.05)
    wait_until(lambda: worker.events, what="the attempt of the queued job")

    assert queue.claims_at_heartbeat[0] == 0, "the worker claimed while it was cut off"


class HeartbeatStore:
    """A Run Store whose worker heartbeats answer ``answers`` in order. An exception is raised.

    Each heartbeat moves ``clock`` by ``takes_s`` before it answers, as a slow database does."""

    def __init__(self, clock: Clock, answers: list[bool | Exception], *, takes_s: float = 0.0) -> None:
        self._clock, self._answers, self._takes_s = clock, answers, takes_s

    def worker_heartbeat(self, worker_id: str, session: str) -> bool:
        self._clock.now += self._takes_s
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def heartbeat_of(store: HeartbeatStore, clock: Clock) -> Heartbeat:
    """The heartbeat of a session whose registration was sent at time 0. Its thread is never started."""
    return Heartbeat(store, worker_id="host-1/w", session="s", acknowledged_at=0.0, clock=clock.monotonic)


def test_without_an_acknowledged_heartbeat_the_worker_is_cut_off_540_s_after_its_registration_was_sent():
    clock = Clock()
    heartbeat = heartbeat_of(HeartbeatStore(clock, [OSError("the database is unavailable")]), clock)
    clock.now = 100.0
    heartbeat.beat()

    clock.now = CUT_OFF_S - 1
    before = heartbeat.cut_off()
    clock.now = CUT_OFF_S
    at = heartbeat.cut_off()

    assert (CUT_OFF_S, before, at) == (540.0, False, True)


def test_an_acknowledged_heartbeat_moves_the_cut_off_to_540_s_after_its_send_time_not_its_answer():
    """The Run Store renews the leases at the heartbeat, so a slow answer must not extend the cut-off."""
    clock = Clock()
    heartbeat = heartbeat_of(HeartbeatStore(clock, [True], takes_s=30.0), clock)
    clock.now = 100.0
    heartbeat.beat()

    clock.now = 100.0 + CUT_OFF_S - 1
    before = heartbeat.cut_off()
    clock.now = 100.0 + CUT_OFF_S
    at = heartbeat.cut_off()

    assert (before, at) == (False, True)


def test_an_answer_that_the_session_no_longer_holds_the_row_cuts_the_worker_off_at_once():
    clock = Clock()
    heartbeat = heartbeat_of(HeartbeatStore(clock, [False]), clock)

    heartbeat.beat()

    assert (heartbeat.cut_off(), heartbeat.replaced) == (True, True)


def test_a_queue_error_doubles_the_wait_before_the_next_claim_and_the_manager_claims_on(running_manager):
    queue = FakeQueue(queued=1)
    queue.claim_errors = [OSError("the database is unavailable"), OSError("the database is unavailable")]
    worker = FakeWorker(queue)
    worker.release.set()

    running_manager(worker, poll_s=0.05)
    wait_until(lambda: worker.events, what="the attempt of the queued job")

    assert worker.events == attempts("job-0")
    assert queue.claims[2] - queue.claims[1] >= 0.1, "the second error did not double the 0.05 s backoff"


def test_shutdown_waits_at_most_shutdown_s_for_a_blocked_attempt_and_then_returns(running_manager):
    """The blocked attempt, which would block for 15 s, is left to lease-expiry reconciliation."""
    worker = FakeWorker(FakeQueue(queued=1))
    stop, thread = running_manager(worker, shutdown_s=0.1)
    wait_until(lambda: worker.events, what="the first attempt")

    stop.set()
    thread.join(timeout=5)

    assert thread.is_alive() is False, "the manager still waits for the blocked attempt"


class FakeKept:
    """The kept sandboxes of a worker: ``reap`` counts its calls, and raises on the first if asked."""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.reaps = 0
        self.fail_first = fail_first

    def reap(self) -> list[str]:
        self.reaps += 1
        if self.fail_first and self.reaps == 1:
            raise OSError("podman did not answer")
        return []


def test_the_manager_removes_the_kept_sandboxes_at_its_start_and_then_once_a_minute(running_manager):
    queue = FakeQueue()
    kept = FakeKept()

    running_manager(FakeWorker(queue), kept=kept)
    wait_until(lambda: len(queue.claims) > 20, what="twenty claims")

    assert kept.reaps == 1


def test_a_failed_removal_of_kept_sandboxes_does_not_stop_the_manager(running_manager):
    worker = FakeWorker(FakeQueue(queued=1))
    worker.release.set()

    running_manager(worker, kept=FakeKept(fail_first=True))
    wait_until(lambda: worker.events, what="the attempt of the queued job")

    assert worker.events == attempts("job-0")


class IdleQueue(FakeQueue):
    """A Run Store, by default without jobs. Each claim moves ``clock`` by ``step_s``. The tenth sets stop.

    A claimed job is running in it from then on."""

    def __init__(self, clock: Clock, *, step_s: float, stop: Event) -> None:
        super().__init__()
        self._clock, self._step_s, self._stop = clock, step_s, stop

    def claim(self, worker_id: str) -> dict[str, Any] | None:
        self.claims.append(self._clock.now)
        self._clock.now += self._step_s
        if len(self.claims) == 10:
            self._stop.set()
        if not self.queued:
            return None
        self.running.append(self.queued[0])
        return self.queued.popleft()


def test_the_manager_does_not_settle_a_running_job_that_its_own_attempt_runs(monkeypatch):
    """Its attempt settles it. Only a job that no attempt runs, such as one whose end failed, is settled."""
    clock, stop = Clock(), Event()
    monkeypatch.setattr("sprout.worker.manager.time", clock)
    queue = IdleQueue(clock, step_s=20, stop=stop)
    queue.queued.append({"id": "job-0"})
    worker = FakeWorker(queue)

    WorkerManager(worker, concurrency=2, poll_s=0, shutdown_s=0).run(stop=stop)

    assert (clock.now >= 60, worker.events) == (True, [("attempt", "job-0")])
    worker.release.set()


def test_the_manager_cleans_up_after_the_attempts_of_its_worker_at_most_once_a_minute(monkeypatch):
    clock, stop = Clock(), Event()
    monkeypatch.setattr("sprout.worker.manager.time", clock)
    worker = FakeWorker(IdleQueue(clock, step_s=20, stop=stop))

    WorkerManager(worker, poll_s=0).run(stop=stop)

    assert worker.clean_ups == 4


def test_a_failed_clean_up_does_not_stop_the_manager(running_manager):
    worker = FakeWorker(FakeQueue(queued=1))
    worker.clean_up_fails = True
    worker.release.set()

    running_manager(worker)
    wait_until(lambda: worker.events, what="the attempt of the queued job")

    assert (worker.clean_ups, worker.events) == (1, attempts("job-0"))


def test_the_manager_removes_the_kept_sandboxes_whose_time_passed_at_most_once_a_minute(monkeypatch):
    clock, stop, reaps = Clock(), Event(), []
    monkeypatch.setattr("sprout.worker.manager.time", clock)
    queue = IdleQueue(clock, step_s=20, stop=stop)
    kept = SimpleNamespace(reap=lambda: reaps.append(clock.now))
    manager = WorkerManager(FakeWorker(queue), poll_s=0, kept=kept)

    manager.run(stop=stop)

    assert (len(queue.claims), reaps) == (10, [0.0, 60.0, 120.0, 180.0])


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_a_concurrency_that_is_not_a_positive_integer_is_refused(concurrency):
    with pytest.raises(ValueError, match="^concurrency must be a positive integer$"):
        WorkerManager(FakeWorker(FakeQueue()), concurrency=concurrency)
