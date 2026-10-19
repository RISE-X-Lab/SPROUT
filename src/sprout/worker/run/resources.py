"""The resource ledger: which sandboxes each run claimed, so that they can be reclaimed after a crash.

A run destroys its sandbox in a ``finally`` on a clean exit, but SIGKILL, OOM and a hard Ctrl-C
skip it, and a backend's listing names no owner. So a run records each claim before it allocates
the sandbox, and records the release after it frees it. A crash leaves a stale claim, never an
unrecorded sandbox, and ``sprout.worker reap`` checks each claim against the backend::

    ledger = ResourceLedger()                      # runs/resources.jsonl
    with ledger.run("run-42") as claim:            # marks the run done on exit
        claim.sandbox(sandbox_id)
        claim.sandbox(kept_id, keep=True)          # not reaped until released, or until keep_until

Snapshots are not claimed. They are a run's durable result, and only
``sprout.worker sweep <journal>`` deletes them.

The ledger is append-only JSONL, like the journal. Two processes that append at the same time
do not corrupt each other's records, and a reader skips a torn line.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sprout.worker.run.journal import utc_now

DEFAULT_LEDGER = "runs/resources.jsonl"

CLAIM = "claim"  # {run_id, kind, id, pid, keep, backend, store?, keep_until?}
RELEASE = "release"  # {run_id, kind, id}
RUN_DONE = "run_done"  # {run_id}


@dataclass
class Resource:
    """One claimed resource as the ledger replays it: who claimed it, and whether it is still held."""

    kind: str  # "sandbox"
    id: str
    run_id: str
    pid: int | None = None
    keep: bool = False
    released: bool = False
    run_done: bool = False
    created: str | None = None
    #: The backend that made it, so that the reaper frees it through that backend's client. A
    #: claim without the field is ``microvm``.
    backend: str = "microvm"
    #: For podman, the store that holds the sandbox's rootfs subvolume. The reaper looks there
    #: for what a container leaves behind after the container is gone.
    store: str | None = None
    #: The ISO-8601 time until which a kept resource is kept. None: kept until released.
    keep_until: str | None = None

    def expired(self, now: datetime) -> bool:
        """Whether it is kept, never released, and its ``keep_until`` is not after ``now``.

        ``now`` must be timezone-aware."""
        return (
            self.keep
            and not self.released
            and self.keep_until is not None
            and datetime.fromisoformat(self.keep_until) <= now
        )

    def orphaned(self) -> bool:
        """Whether it is not kept, never released, and its run is done or its process is gone."""
        if self.released or self.keep:
            return False
        if self.run_done:
            # The run finished but this resource was never released: a bug or a kill between the
            # two writes. It is still an orphan.
            return True
        return not _pid_alive(self.pid)


class RunClaim:
    """Records what one run allocates. Obtained from ``ResourceLedger.run``."""

    def __init__(self, ledger: ResourceLedger, run_id: str) -> None:
        self._ledger = ledger
        self.run_id = run_id
        self.sandboxes: list[str] = []

    def sandbox(
        self,
        sandbox_id: str,
        *,
        keep: bool = False,
        backend: str = "microvm",
        store: str | None = None,
        keep_until: str | None = None,
    ) -> str:
        """Record the claim of ``sandbox_id``, and return it.

        The reaper leaves a ``keep`` sandbox alone until its ``keep_until`` passes. Without a
        ``keep_until``, it leaves the sandbox alone until the sandbox is released.
        """
        self.sandboxes.append(sandbox_id)
        self._ledger._append(
            CLAIM,
            run_id=self.run_id,
            kind="sandbox",
            id=sandbox_id,
            pid=os.getpid(),
            keep=keep,
            backend=backend,
            **({"store": store} if store else {}),
            **({"keep_until": keep_until} if keep_until else {}),
        )
        return sandbox_id

    def released(self, kind: str, resource_id: str) -> None:
        """Record that this run freed the resource ``resource_id`` of ``kind``."""
        self._ledger._append(RELEASE, run_id=self.run_id, kind=kind, id=resource_id)


class ResourceLedger:
    """The append-only ledger at ``path``, and its replay into the current state of each resource.

    A relative ``path`` is relative to the working directory. Appends are thread-safe.
    """

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_LEDGER) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # --- writing -----------------------------------------------------------
    def _append(self, event: str, **payload: Any) -> None:
        record = {"event": event, "ts": utc_now("seconds")}
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()

    def run(self, run_id: str) -> _RunContext:
        """A context manager that yields the :class:`RunClaim` of ``run_id``.

        On exit, also after an exception, it marks the run done."""
        return _RunContext(self, run_id)

    def mark_run_done(self, run_id: str) -> None:
        """Record that ``run_id`` ended. Each of its claims that is not released is then an orphan."""
        self._append(RUN_DONE, run_id=run_id)

    # --- reading -----------------------------------------------------------
    def entries(self) -> Iterator[dict[str, Any]]:
        """Every record of the ledger, in file order. A missing file has none, and a torn line is skipped."""
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue  # torn write: skip, never abort the reap

    def resources(self) -> list[Resource]:
        """The current state of each claimed resource, replayed from the ledger."""
        # A ledger line is JSON from another process: its values are untyped.
        table: dict[tuple[Any, Any], Resource] = {}
        done_runs: set[Any] = set()

        for entry in self.entries():
            event = entry.get("event")
            if event == RUN_DONE:
                done_runs.add(entry.get("run_id"))
                continue
            key: tuple[Any, Any] = (entry.get("kind"), entry.get("id"))
            if not key[1]:
                continue
            if event == CLAIM:
                table[key] = Resource(
                    kind=key[0],
                    id=key[1],
                    run_id=entry.get("run_id") or "",
                    pid=entry.get("pid"),
                    keep=bool(entry.get("keep")),
                    created=entry.get("ts"),
                    backend=str(entry.get("backend") or "microvm"),
                    store=entry.get("store") or None,
                    keep_until=entry.get("keep_until") or None,
                )
            elif event == RELEASE and key in table:
                table[key].released = True

        for resource in table.values():
            resource.run_done = resource.run_id in done_runs
        return list(table.values())


class _RunContext:
    """Context manager marking the run done on exit (even on exception)."""

    def __init__(self, ledger: ResourceLedger, run_id: str) -> None:
        self._ledger = ledger
        self.claim = RunClaim(ledger, run_id)

    def __enter__(self) -> RunClaim:
        return self.claim

    def __exit__(self, *exc: object) -> None:
        self._ledger.mark_run_done(self.claim.run_id)


def _pid_alive(pid: int | None) -> bool:
    """True if the process still exists. An unknown pid counts as alive.

    Reaping a sandbox that a live run still needs is worse than leaving a stale entry for the
    next pass."""
    if not pid:
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except (TypeError, ValueError):
        return True  # a ledger pid that is not a number
