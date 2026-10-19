"""The worker's claim loop: register the worker, report its heartbeats, and supervise each attempt.

The manager runs no attempt itself. :class:`~sprout.worker.attempts.Worker` runs each one in a
thread, and the manager bounds how many run at once. :class:`Heartbeat` reports that the worker
lives, and says when the worker is cut off from the Run Store. docs/RUNSTORE.md ("Supervision")
gives the rules.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from threading import Event, Thread
from typing import Any
from uuid import uuid4

from sprout.runstore.store import WORKER_LIVE_S, Store
from sprout.worker.attempts import Worker, process_identity
from sprout.worker.kept import KeptSandboxes

#: How often, in seconds, the manager settles the jobs that no attempt of it runs, and frees what its
#: worker's attempts left (Worker.clean_up and the kept sandboxes). Arbitrary.
TIDY_S = 60.0
#: How often, in seconds, the manager reports a heartbeat of its worker. ADR-0006 sets the value.
HEARTBEAT_S = 30.0
#: How long attempts run without an acknowledged heartbeat, in seconds from its send time. It ends
#: 60 s before the worker is dead, so that each child stops before another worker can run its job.
CUT_OFF_S = WORKER_LIVE_S - 60.0

_log = logging.getLogger(__name__)


def _gone_session(row: dict[str, Any] | None, *, host: str) -> str | None:
    """The session of the worker row ``row`` whose process is gone from ``host``, else None.

    The process is gone if it exited, its pid names another process now, or ``host`` booted again. A row
    of another host or without a process gives None: this host cannot tell whether that process runs.
    """
    if row is None or row["host"] != host or row["process"] is None:
        return None
    alive = process_identity(row["process"]["pid"]) == row["process"]
    return None if alive else str(row["session"])


class Heartbeat:
    """The heartbeats of one session of a worker, sent from a daemon thread, and whether it is cut off.

    It is cut off ``cut_off_s`` after the send time of the last acknowledged heartbeat (at first, the
    registration's ``acknowledged_at``), and for good once an answer says another session holds the row.
    """

    def __init__(
        self,
        store: Store,
        *,
        worker_id: str,
        session: str,
        acknowledged_at: float,
        interval_s: float = HEARTBEAT_S,
        cut_off_s: float = CUT_OFF_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._worker_id = worker_id
        self.session = session
        self._acknowledged_at = acknowledged_at
        self._interval_s = interval_s
        self._cut_off_s = cut_off_s
        self._clock = clock
        #: Whether an answer said that another session holds the worker's row.
        self.replaced = False
        self._finished = Event()
        self._thread = Thread(target=self._run, name="worker-heartbeat", daemon=True)

    def start(self) -> None:
        """Send a heartbeat every ``interval_s`` from the thread, until :meth:`close`."""
        self._thread.start()

    def close(self) -> None:
        """Stop the thread, and wait up to 15 s for a heartbeat that it is sending."""
        self._finished.set()
        # Longer than a call to an unreachable database takes to fail (Store.transaction).
        self._thread.join(timeout=15)

    def beat(self) -> None:
        """Send one heartbeat. A heartbeat that raises changes nothing, and is logged."""
        sent = self._clock()
        try:
            acknowledged = self._store.worker_heartbeat(self._worker_id, self.session)
        except Exception:
            _log.exception("The Run Store did not record the heartbeat of %s", self._worker_id)
            return
        if acknowledged:
            self._acknowledged_at = sent
        else:
            self.replaced = True

    def cut_off(self) -> bool:
        """Whether the worker's attempts must stop and it must claim nothing. Any thread can call it."""
        return self.replaced or self._clock() - self._acknowledged_at >= self._cut_off_s

    def _run(self) -> None:
        while not self._finished.wait(self._interval_s):
            self.beat()


class WorkerManager:
    """Claims jobs for ``worker`` and runs up to ``concurrency`` attempts at once, each in a thread.

    The manager registers the worker, settles the jobs that an earlier session left, and claims nothing
    while the worker is cut off. Before it claims, it settles the jobs of dead workers and lost hosts."""

    def __init__(
        self,
        worker: Worker,
        *,
        concurrency: int = 1,
        poll_s: float = 1,
        shutdown_s: float = 20,
        heartbeat_s: float = HEARTBEAT_S,
        kept: KeptSandboxes | None = None,
    ) -> None:
        """``kept`` holds the worker's kept sandboxes. The manager removes each one when its time passes.

        With ``kept=None``, it removes none. Raises ``ValueError`` if ``concurrency`` is not positive.
        """
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        self.worker = worker
        self.concurrency = concurrency
        self.poll_s = poll_s
        self.shutdown_s = shutdown_s
        self.heartbeat_s = heartbeat_s
        self.kept = kept
        self.tidied_at = -TIDY_S
        self.active: dict[str, Future[object]] = {}
        self.wake = Event()
        self.log = logging.getLogger(__name__)

    def _start(self, job_id: str, action: Callable[[], object]) -> None:
        future: Future[object] = Future()

        def supervise() -> None:
            try:
                future.set_result(action())
            except BaseException as error:
                future.set_exception(error)
            finally:
                self.wake.set()

        thread = Thread(target=supervise, name=f"attempt-{job_id[:8]}", daemon=True)
        thread.start()
        self.active[job_id] = future

    def _reap(self) -> None:
        for job_id, future in list(self.active.items()):
            if not future.done():
                continue
            del self.active[job_id]
            try:
                future.result()
            except BaseException:
                self.log.exception("Attempt supervision failed for job %s", job_id)

    def _tidy(self) -> None:
        """Every ``TIDY_S``, settle the unsupervised jobs, clean up, and remove the kept sandboxes past time.

        A failure is logged, and the next call tries again.
        """
        if time.monotonic() < self.tidied_at + TIDY_S:
            return
        self.tidied_at = time.monotonic()
        self._settle_unsupervised()
        try:
            self.worker.clean_up()
        except Exception:
            self.log.exception("Could not free what the attempts of worker %s left", self.worker.worker_id)
        if self.kept is None:
            return
        try:
            self.kept.reap()
        except Exception:
            self.log.exception("Could not remove the kept sandboxes whose time has passed")

    def _settle_unsupervised(self) -> None:
        """Start to settle each running job of the worker that no attempt of the manager runs.

        These are the jobs that an earlier session left, and the jobs whose attempt ended unsettled.
        """
        try:
            jobs = self.worker.store.running_jobs(self.worker.worker_id)
        except Exception:
            self.log.exception("Could not list the running jobs of worker %s", self.worker.worker_id)
            return
        for job in jobs:
            if job["id"] not in self.active:
                self._start(job["id"], partial(self.worker.settle, job))

    def _register(self) -> Heartbeat:
        """Register a new session of the worker with its grades, and return its heartbeat, not started yet.

        The session replaces at once a session whose process is gone from this host. Raises Conflict if
        the worker is live in another session.
        """
        session = uuid4().hex
        holder = self.worker.store.worker(self.worker.worker_id)
        sent = time.monotonic()
        self.worker.store.register_worker(
            self.worker.worker_id,
            host=self.worker.host,
            session=session,
            process=process_identity(os.getpid()),
            grades=self.worker.grades,
            replaces=_gone_session(holder, host=self.worker.host),
        )
        return Heartbeat(
            self.worker.store,
            worker_id=self.worker.worker_id,
            session=session,
            acknowledged_at=sent,
            interval_s=self.heartbeat_s,
            clock=time.monotonic,
        )

    def run(self, *, stop: Event | None = None, once: bool = False) -> None:
        """Register the worker, then claim and supervise jobs until ``stop`` is set, which it sets on return.

        With ``once``, it makes one claim and returns after the attempts that it started. Raises Conflict,
        and claims nothing, if the worker is live in another session.
        """
        stop = stop if stop is not None else Event()
        heartbeat = self._register()
        heartbeat.start()
        try:
            self._supervise(stop, once=once, heartbeat=heartbeat)
        finally:
            stop.set()
            self._drain()
            heartbeat.close()
            self._mark_stopped(heartbeat.session)

    def _supervise(self, stop: Event, *, once: bool, heartbeat: Heartbeat) -> None:
        backoff = self.poll_s
        retry_at = 0.0
        draining = False
        while not stop.is_set():
            self.wake.clear()
            self._reap()
            self._tidy()
            if draining and not self.active:
                return
            if heartbeat.replaced:
                self.log.error(
                    "Another session holds the row of worker %s, so it stops", self.worker.worker_id
                )
                return
            if (
                not draining
                and not heartbeat.cut_off()
                and len(self.active) < self.concurrency
                and time.monotonic() >= retry_at
            ):
                try:
                    self._settle_dead_workers()
                    self._claim(stop, once=once, heartbeat=heartbeat)
                    # With once, the manager makes one claim and then waits for its attempts.
                    draining = once
                    backoff = self.poll_s
                except Exception:
                    self.log.exception("Worker queue operation failed; backing off")
                    retry_at = time.monotonic() + backoff
                    backoff = min(30, backoff * 2)
            self.wake.wait(self.poll_s)

    def _settle_dead_workers(self) -> None:
        settled = self.worker.store.settle_dead_workers()
        if settled:
            self.log.info("Settled the jobs of dead workers and lost hosts: %s", ", ".join(settled))

    def _claim(self, stop: Event, *, once: bool, heartbeat: Heartbeat) -> None:
        """Start an attempt for each job that a claim takes, up to ``concurrency``. ``once`` claims once."""
        while not stop.is_set() and len(self.active) < self.concurrency:
            claimed = self.worker.store.claim(self.worker.worker_id)
            if claimed is not None:
                attempt = partial(self.worker.run_claimed, claimed, stop=stop, cut_off=heartbeat.cut_off)
                self._start(claimed["id"], attempt)
            if claimed is None or once:
                break

    def _drain(self) -> None:
        """Wait up to ``shutdown_s`` for the running attempts, which the set ``stop`` stops."""
        deadline = time.monotonic() + self.shutdown_s
        while self.active and time.monotonic() < deadline:
            self._reap()
            self.wake.wait(0.05)
            self.wake.clear()
        if self.active:
            self.log.warning(
                "Leaving %s blocked attempts: their jobs run again once their leases expire", len(self.active)
            )

    def _mark_stopped(self, session: str) -> None:
        try:
            self.worker.store.worker_stopped(self.worker.worker_id, session)
        except Exception:
            self.log.exception("Could not mark the row of worker %s stopped", self.worker.worker_id)
