"""What a worker deletes and frees after an attempt that it cannot settle cleanly.

``forget`` deletes the local data of an attempt whose job runs again, or that a shutdown released:
the snapshots that its journal names, then the attempt directory. It keeps the last
``LOG_TAIL_CHARS`` characters of the attempt's ``worker.log``, for the attempt's ``error`` in the
Run Store. ``Leftovers`` records the sandboxes whose removal after an attempt the worker could not
confirm, and frees them later. ADR-0006 records the decisions, and docs/RUNSTORE.md ("Failure
handling") the behavior.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from threading import Lock
from typing import Any

from sprout.contract.backend import backend_name, snapshot_store
from sprout.worker.run.reap import AgentEnvClient, PodmanClient, Reaper
from sprout.worker.run.resources import ResourceLedger
from sprout.worker.run.sweep import journal_snapshots, snapshot_deleter, sweep

#: How much of a deleted attempt's ``worker.log`` its attempt row keeps, in characters. ADR-0006 sets it.
LOG_TAIL_CHARS = 4000
#: The name of the ledger of ``Leftovers`` under a worker's artifact root.
LEFTOVERS_NAME = "leftover-sandboxes.jsonl"

_log = logging.getLogger(__name__)


def forget(directory: Path, backend: dict[str, Any]) -> str:
    """Delete the attempt ``directory`` and the snapshots that its journal names, and return its log's tail.

    ``backend`` is the attempt's resolved backend config. A snapshot that is not deleted is logged, and
    stays. A missing directory, journal or log is no error.
    """
    tail = _log_tail(directory / "worker.log")
    journal = directory / "trajectory.jsonl"
    if journal.exists():
        _delete_snapshots(journal, backend)
    shutil.rmtree(directory, ignore_errors=True)
    return tail


def log_section(log: str) -> str:
    """The text that gives ``log``, the tail of an attempt's ``worker.log``, after an error. Empty: no log."""
    return f"\n--- the last {len(log)} characters of worker.log ---\n{log}" if log else ""


def with_log(error: str | None, log: str) -> str | None:
    """``error`` followed by ``log``, the tail of the attempt's ``worker.log``. ``error`` alone for no log."""
    return f"{error}{log_section(log)}" if log else error


class Leftovers:
    """The sandboxes that attempts of a worker left, recorded in the ledger at ``path``, until it frees them.

    Calls are serialized within one process.
    """

    def __init__(self, path: Path) -> None:
        self._ledger = ResourceLedger(path)
        self._lock = Lock()

    def leave(self, directory: Path, *, backend: dict[str, Any], held: set[str]) -> None:
        """Record each sandbox that the ledger of the attempt in ``directory`` claims and did not release.

        ``backend`` is the attempt's resolved backend config. A sandbox in ``held`` is left out. A start of
        a sandbox that no claim names cannot be recorded, so it is logged.
        """
        path = directory / "resources.jsonl"
        if not path.exists():
            return
        attempt = ResourceLedger(path)
        claims = [
            resource
            for resource in attempt.resources()
            if resource.kind == "sandbox" and not resource.released and resource.id not in held
        ]
        events = [entry.get("event") for entry in attempt.entries()]
        if "allocation_unknown" in events or events.count("allocation_started") > events.count("claim"):
            _log.warning("A sandbox of %s has no id. Free it with python -m sprout.worker reap", directory)
        if not claims:
            return
        with self._lock, self._ledger.run(directory.name) as left:
            for resource in claims:
                store = resource.store or snapshot_store(backend)
                left.sandbox(resource.id, backend=backend_name(backend), store=store)

    def free(self) -> list[str]:
        """Remove each recorded sandbox that is still there, and return the ids of those that it removed.

        A sandbox that its backend no longer lists is gone already. One that it fails to remove stays
        recorded, so that a later call tries again.
        """
        with self._lock:
            pending = [resource for resource in self._ledger.resources() if not resource.released]
            if not pending:
                return []
            stores = [resource.store for resource in pending if resource.store]
            clients = {
                resource.backend: PodmanClient(stores=stores)
                if resource.backend == "podman"
                else AgentEnvClient()
                for resource in pending
            }
            reaper = Reaper(clients, self._ledger)
            plan = reaper.plan()
            freed = reaper.apply(plan)["sandboxes"]
            gone = {resource.id for resource in pending} - set(plan.sandboxes) - set(plan.foreign)
            with self._ledger.run("free") as claim:
                for sandbox_id in sorted(gone):
                    claim.released("sandbox", sandbox_id)
            return freed


def _log_tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            # A UTF-8 character takes at most 4 bytes.
            stream.seek(max(0, size - 4 * LOG_TAIL_CHARS))
            data = stream.read()
    except FileNotFoundError:
        return ""
    return data.decode("utf-8", errors="replace")[-LOG_TAIL_CHARS:]


def _delete_snapshots(journal: Path, backend: dict[str, Any]) -> None:
    snapshots = journal_snapshots(journal)
    if snapshots.swept or not snapshots.ids:
        return
    microvm = backend.get("microvm") or {}
    delete = snapshot_deleter(
        snapshots.backend,
        snapshots.store or snapshot_store(backend),
        agentenv=AgentEnvClient(microvm.get("server_url"), microvm.get("api_key")),
    )
    if isinstance(delete, str):
        _log.warning("The snapshots of %s stay: %s", journal, delete)
        return
    _, failed = sweep(snapshots, delete)
    if failed:
        _log.warning("%d snapshots of %s stay: they could not be deleted", failed, journal)
