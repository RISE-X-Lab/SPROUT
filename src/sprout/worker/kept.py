"""The sandboxes that queued rollouts keep on this host for a grade in place.

``KeptSandboxes`` records each kept sandbox with the time until which its rollout keeps it. A
grade in place takes its sandbox over with ``take``. A grade whose attempt runs nothing gives it
back with ``give_back``. ``reap`` removes each kept sandbox that no grade took, once its time has
passed. The record is a resource ledger (``worker.run.resources``), so ``python -m sprout.worker
reap --ledger <ledger>`` plans the same removals by hand. Calls are serialized within one process.
ADR-0004 (docs/decisions/0004-grade-in-kept-sandbox.md) records the design.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from sprout.contract.backend import backend_name, snapshot_store
from sprout.worker.run.reap import AgentEnvClient, PodmanClient, Reaper
from sprout.worker.run.resources import ResourceLedger, RunClaim

#: The ledger's name under a worker's artifact root.
LEDGER_NAME = "kept-sandboxes.jsonl"


class KeptSandboxes:
    """The kept sandboxes recorded in the ledger at ``path``."""

    def __init__(self, path: Path) -> None:
        self._ledger = ResourceLedger(path)
        self._lock = Lock()

    def keep(self, *, sandbox_id: str, keep_until: str, attempt_id: str, backend: dict[str, Any]) -> None:
        """Record that the attempt ``attempt_id`` keeps ``sandbox_id`` on ``backend`` until ``keep_until``.

        ``backend`` is the backend config of the attempt. ``keep_until`` is an ISO-8601 time with a
        timezone. The record replaces an earlier record of the same sandbox.
        """
        with self._lock, self._ledger.run(attempt_id) as claim:
            claim.sandbox(
                sandbox_id,
                keep=True,
                keep_until=keep_until,
                backend=backend_name(backend),
                store=snapshot_store(backend),
            )

    def take(self, sandbox_id: str, *, claim: RunClaim) -> bool:
        """Hand the kept sandbox ``sandbox_id`` over to the attempt whose ledger claim is ``claim``.

        Returns True, after ``claim`` records the sandbox, when a rollout keeps it here and its
        ``keep_until`` has not passed. ``held`` and ``reap`` then leave it to that attempt.
        """
        with self._lock:
            now = datetime.now(UTC)
            kept = [
                resource
                for resource in self._ledger.resources()
                if resource.id == sandbox_id and resource.keep and not resource.released
            ]
            if not kept or kept[0].expired(now):
                return False
            claim.sandbox(sandbox_id, backend=kept[0].backend, store=kept[0].store)
            with self._ledger.run(claim.run_id) as handed:
                handed.released("sandbox", sandbox_id)
            return True

    def give_back(self, sandbox_id: str, *, claim: RunClaim) -> None:
        """Keep the sandbox ``sandbox_id`` again, which ``take`` handed over to the attempt of ``claim``.

        ``claim`` records that its attempt freed the sandbox. The sandbox is kept until its earlier
        ``keep_until``, as before ``take``. Does nothing if no ``take`` handed the sandbox over.
        """
        with self._lock:
            handed = [
                resource
                for resource in self._ledger.resources()
                if resource.id == sandbox_id and resource.keep and resource.released
            ]
            if not handed:
                return
            with self._ledger.run(handed[0].run_id) as kept:
                kept.sandbox(
                    sandbox_id,
                    keep=True,
                    keep_until=handed[0].keep_until,
                    backend=handed[0].backend,
                    store=handed[0].store,
                )
            claim.released("sandbox", sandbox_id)

    def held(self) -> set[str]:
        """The ids of the sandboxes kept here that no grade has taken and no ``reap`` has removed."""
        with self._lock:
            return {
                resource.id
                for resource in self._ledger.resources()
                if resource.keep and not resource.released
            }

    def reap(self) -> list[str]:
        """Remove each kept sandbox that no grade took and whose ``keep_until`` has passed.

        Returns the ids of the sandboxes that it removed. A sandbox that it fails to remove stays
        recorded, so that a later call tries again.
        """
        with self._lock:
            now = datetime.now(UTC)
            expired = [resource for resource in self._ledger.resources() if resource.expired(now)]
            if not expired:
                return []
            # Each client lists every sandbox of its backend, so only the backends in question get one.
            backends = {resource.backend for resource in expired}
            stores = [resource.store for resource in expired if resource.store]
            reaper = Reaper(
                {
                    name: PodmanClient(stores=stores) if name == "podman" else AgentEnvClient()
                    for name in backends
                },
                self._ledger,
            )
            return reaper.apply(reaper.plan())["sandboxes"]
