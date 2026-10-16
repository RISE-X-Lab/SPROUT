"""One claim loop, bounded attempt supervision and no execution in the manager."""

from __future__ import annotations

from concurrent.futures import Future
import logging
from threading import Event, Thread
import time
from typing import Callable

from sprout.runstore.worker import Worker


class WorkerManager:
    def __init__(
        self, worker: Worker, *, concurrency: int = 1, poll_s: float = 1, shutdown_s: float = 20
    ) -> None:
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        self.worker = worker
        self.concurrency = concurrency
        self.poll_s = poll_s
        self.shutdown_s = shutdown_s
        self.active: dict[str, Future] = {}
        self.recovery_after: dict[str, float] = {}
        self.wake = Event()
        self.log = logging.getLogger(__name__)

    def _start(self, job_id: str, action: Callable) -> None:
        future = Future()

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

    def run(self, *, stop: Event | None = None, once: bool = False) -> None:
        stop = stop if stop is not None else Event()
        backoff = self.poll_s
        retry_at = 0.0
        draining = False
        try:
            while not stop.is_set():
                self.wake.clear()
                self._reap()
                if draining and not self.active:
                    return
                if not draining and len(self.active) < self.concurrency and time.monotonic() >= retry_at:
                    try:
                        self.worker.store.expire()
                        now = time.monotonic()
                        self.recovery_after = {
                            job_id: deadline
                            for job_id, deadline in self.recovery_after.items()
                            if deadline > now
                        }
                        excluded = tuple(set(self.active) | set(self.recovery_after))
                        for job in self.worker.store.recoverable_jobs(
                            exclude=excluded, limit=self.concurrency - len(self.active)
                        ):
                            if stop.is_set():
                                break
                            job_id = job["id"]
                            self.recovery_after[job_id] = now + 30
                            self._start(
                                job_id, lambda job_id=job_id: self.worker.reconcile(job_id, stop=stop)
                            )
                        while not stop.is_set() and len(self.active) < self.concurrency:
                            job = self.worker.store.claim(self.worker.worker_id, lease_s=self.worker.lease_s)
                            if job is not None:
                                self._start(
                                    job["id"], lambda job=job: self.worker.run_claimed(job, stop=stop)
                                )
                            if once:
                                draining = True
                            if job is None or once:
                                break
                        backoff = self.poll_s
                    except Exception:
                        self.log.exception("Worker queue operation failed; backing off")
                        retry_at = time.monotonic() + backoff
                        backoff = min(30, backoff * 2)
                self.wake.wait(self.poll_s)
        finally:
            stop.set()
            deadline = time.monotonic() + self.shutdown_s
            while self.active and time.monotonic() < deadline:
                self._reap()
                self.wake.wait(0.05)
                self.wake.clear()
            if self.active:
                self.log.warning(
                    "Leaving %s blocked supervisors for lease-expiry reconciliation", len(self.active)
                )
