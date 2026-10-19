"""Reclaim sandboxes that outlived their run. Snapshots are never reclaimed here.

A reap plan uses two sources, because each one alone misses cases:

- the ledger (``sprout.worker.run.resources``) records what each run allocated and on which
  backend, so it attributes an orphan to a dead pid. It does not see a sandbox from before the
  ledger or from another tool.
- each backend lists everything that exists, with no owner. Age is the only signal, so a sandbox
  that the ledger never saw is planned only with ``include_unknown``, and ``older_than`` bounds it.

By default the plan holds only what the ledger attributes, because a sweep by age alone can delete
a colleague's experiment in progress. A claim that names a backend with no client is reported as
``foreign`` and never touched. A snapshot stays, because each step of a finished run is a branch
point: ``python -m sprout.worker sweep <journal> --yes`` deletes the snapshots that a journal names.

    python -m sprout.worker reap --dry-run
    python -m sprout.worker reap
    python -m sprout.worker reap --include-unknown --older-than 24h   # backend-wide sweep
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from sprout.worker.run.resources import ResourceLedger
from sprout.worker.sandbox.backends import default_aenv_server_url
from sprout_sandbox.btrfs_store import BtrfsStore
from sprout_sandbox.podman_cli import default_podman_bin
from sprout_sandbox.podman_containers import MANAGED_LABEL, SOCKET_MOUNT, STORE_LABEL

_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd])$")


def parse_duration(text: str) -> timedelta:
    """A number and a unit (``s``, ``m``, ``h`` or ``d``), such as ``30m``, as a timedelta.

    Raises ``ValueError`` for any other text."""
    match = _DURATION.match(text.strip().lower())
    if not match:
        raise ValueError("bad duration %r (use 30m, 24h, 7d)" % text)
    value, unit = float(match.group(1)), match.group(2)
    return timedelta(**{{"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]: value})


class SandboxClient(Protocol):
    """One backend's sandboxes, listed and deleted without a pool, a session or an event loop."""

    def list_sandboxes(self) -> list[dict[str, Any]]:
        """Each sandbox as ``{"id", "created_at", "names"}``. Empty when the backend cannot answer.

        ``created_at`` is an ISO 8601 time or None. A sandbox with ``names`` is a deliberate artifact."""

    def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete the sandbox. True if nothing of it is left afterwards, also when it was gone before."""


@dataclass
class ReapPlan:
    """The sandboxes that ``Reaper.apply`` frees, the ones it leaves, and the reason for each."""

    sandboxes: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    #: Orphans of a backend this reaper has no client for: reported, never applied.
    foreign: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    #: Which backend each planned sandbox belongs to, so ``apply`` asks the
    #: right client to delete it.
    backends: dict[str, str] = field(default_factory=dict)

    def total(self) -> int:
        return len(self.sandboxes)


class Reaper:
    """Builds and applies a reap plan against the backends it has clients for.

    ``clients`` maps a backend name, as ledger claims spell it (``microvm``, ``podman``), to its client."""

    def __init__(self, clients: Mapping[str, SandboxClient], ledger: ResourceLedger | None = None) -> None:
        self.clients = dict(clients)
        self.ledger = ledger or ResourceLedger()

    # --- planning ----------------------------------------------------------
    def plan(
        self,
        *,
        include_unknown: bool = False,
        older_than: timedelta | None = None,
    ) -> ReapPlan:
        """What ``apply`` would free: the ledger's orphans and its kept sandboxes past their ``keep_until``.

        With ``include_unknown``, also the backends' sandboxes that the ledger never saw, past ``older_than``.
        """
        plan = ReapPlan()

        live = {
            backend: {s["id"]: s for s in client.list_sandboxes()} for backend, client in self.clients.items()
        }

        ledger_ids: set[str] = set()
        now = _utcnow()
        for resource in self.ledger.resources():
            if resource.kind != "sandbox":
                # An older ledger can still hold ``kind: snapshot`` rows. They are never planned
                # or counted: only ``python -m sprout.worker sweep`` deletes a snapshot.
                continue
            ledger_ids.add(resource.id)
            expired = resource.expired(now)
            if resource.keep and not expired:
                plan.kept.append(resource.id)
                continue
            if not (expired or resource.orphaned()):
                continue
            if resource.backend not in live:
                # Not ours to free, and not silently "already gone" either.
                plan.foreign.append(resource.id)
                plan.reasons[resource.id] = (
                    "%s sandbox of run %s (pid %s gone): no client for that backend"
                    % (resource.backend, resource.run_id or "?", resource.pid)
                )
                continue
            if resource.id not in live[resource.backend]:
                continue  # already gone: the ledger entry is stale
            plan.sandboxes.append(resource.id)
            plan.backends[resource.id] = resource.backend
            plan.reasons[resource.id] = (
                "kept by run %s until %s" % (resource.run_id or "?", resource.keep_until)
                if expired
                else "orphan of run %s (pid %s gone)" % (resource.run_id or "?", resource.pid)
            )

        if include_unknown:
            # An unknown RUNNING sandbox burns CPU and memory right now, which
            # is what makes it worth reclaiming without an owner to ask.
            cutoff = _utcnow() - older_than if older_than else None
            for backend, sandboxes in live.items():
                for sandbox_id, meta in sandboxes.items():
                    if sandbox_id in ledger_ids or sandbox_id in plan.sandboxes:
                        continue
                    if meta.get("names"):
                        # A named resource is someone's deliberate artifact.
                        plan.kept.append(sandbox_id)
                        continue
                    created = _parse_ts(meta.get("created_at"))
                    if cutoff is not None and (created is None or created > cutoff):
                        continue
                    plan.sandboxes.append(sandbox_id)
                    plan.backends[sandbox_id] = backend
                    plan.reasons[sandbox_id] = "not in ledger, created %s" % (
                        meta.get("created_at") or "unknown",
                    )
        return plan

    # --- applying ----------------------------------------------------------
    def apply(self, plan: ReapPlan) -> dict[str, list[str]]:
        """Delete each planned sandbox through its backend's client, and return the ids by result.

        ``sandboxes`` lists the deleted ones, which the ledger then records as released. ``failed``
        lists the others. A failure is reported, never raised."""
        done: dict[str, list[str]] = {"sandboxes": [], "failed": []}
        for sandbox_id in plan.sandboxes:
            client = self.clients[plan.backends[sandbox_id]]
            if client.delete_sandbox(sandbox_id):
                done["sandboxes"].append(sandbox_id)
                self.ledger._append("release", kind="sandbox", id=sandbox_id, run_id="reap")
            else:
                done["failed"].append(sandbox_id)
        return done


class AgentEnvClient:
    """A minimal AgentENV REST client that lists and deletes microVM sandboxes.

    It needs no pool, session or event loop, so a reap works from a cron job long after the
    process that leaked the sandboxes is gone. ``delete_snapshot`` deletes history. The reaper
    never calls it."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        import os

        self.base_url = (base_url or default_aenv_server_url()).rstrip("/")
        self.api_key = api_key or os.environ.get("AENV_API_KEY") or ""

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    # Any: the items of AgentENV's JSON list, whatever their shape.
    def _get(self, path: str) -> list[Any]:
        import httpx

        try:
            response = httpx.get(self.base_url + path, headers=self._headers(), timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception:  # noqa: BLE001 - an unreachable backend reaps nothing
            return []
        return payload if isinstance(payload, list) else []

    def _delete(self, path: str) -> bool:
        import httpx

        try:
            response = httpx.delete(self.base_url + path, headers=self._headers(), timeout=60)
        except Exception:  # noqa: BLE001
            return False
        return response.status_code < 300 or response.status_code == 404

    def list_sandboxes(self) -> list[dict[str, Any]]:
        return [
            {
                "id": item.get("sandboxID") or "",
                "created_at": item.get("startedAt") or item.get("createdAt"),
                "names": item.get("names") or [],
            }
            for item in self._get("/sandboxes")
        ]

    def delete_sandbox(self, sandbox_id: str) -> bool:
        return self._delete("/sandboxes/%s" % sandbox_id)

    def delete_snapshot(self, snapshot_id: str) -> bool:
        return self._delete("/snapshots/%s" % snapshot_id)


class PodmanClient:
    """The podman backend's sandboxes, listed and removed without a pool or a runtime binary.

    A sandbox is a container, a network, a rootfs subvolume and a socket directory. The client
    finds them when the container is gone too: a rootfs in one of ``stores``, a network by the
    pool's labels. A host without podman has nothing to list."""

    def __init__(self, podman_bin: str | None = None, stores: Iterable[str] = ()) -> None:
        self.podman_bin = podman_bin or default_podman_bin()
        self.stores = sorted({str(Path(store).expanduser().resolve()) for store in stores})

    def _podman(self, *args: str) -> str | None:
        """stdout of a podman command, or ``None`` (with stderr reported) when it failed."""
        try:
            completed = subprocess.run([self.podman_bin, *args], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            if completed.stderr.strip():
                print("podman %s: %s" % (args[0], completed.stderr.strip().splitlines()[-1]), file=sys.stderr)
            return None
        return completed.stdout

    def list_sandboxes(self) -> list[dict[str, Any]]:
        """Every managed container, every rootfs in a known store, and every sandbox network.

        A rootfs or a network appears only when no container of the same name is listed."""
        result = self._containers()
        seen = {item["id"] for item in result}
        # A rootfs whose container is gone (a destroy that failed after ``podman rm -f``, or a
        # removal by hand) has no label to find it by, so each known store is searched.
        for store in self.stores:
            directory = Path(store) / "sandboxes"
            if not directory.is_dir():
                continue
            for rootfs in sorted(directory.iterdir()):
                if not rootfs.is_dir() or rootfs.name in seen:
                    continue
                seen.add(rootfs.name)
                result.append(
                    {
                        "id": rootfs.name,
                        "created_at": datetime.fromtimestamp(rootfs.stat().st_mtime, tz=UTC).isoformat(),
                        "names": [],
                    }
                )
        # A network left without its container keeps the pool's labels. Its store label lets
        # ``delete_sandbox`` find the rootfs with no store given.
        for network in self._networks():
            if network["name"] not in seen:
                seen.add(network["name"])
                created = _parse_ts(network["created"])
                result.append(
                    {
                        "id": network["name"],
                        "created_at": created.isoformat() if created else None,
                        "names": [],
                    }
                )
        return result

    def _networks(self) -> list[dict[str, Any]]:
        """The sandbox networks podman holds -- name, the store its label names, creation time."""
        raw = self._podman("network", "ls", "--filter", "label=" + MANAGED_LABEL, "--format", "json")
        try:
            items = json.loads(raw or "[]")
        except ValueError:
            return []
        return [
            {
                "name": item["name"],
                "store": (item.get("labels") or {}).get(STORE_LABEL),
                "created": item.get("created"),
            }
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, dict) and item.get("name")
        ]

    def _containers(self) -> list[dict[str, Any]]:
        raw = self._podman("ps", "-a", "--filter", "label=" + MANAGED_LABEL, "--format", "json")
        try:
            items = json.loads(raw or "[]")
        except ValueError:
            return []
        result: list[dict[str, Any]] = []
        for item in items if isinstance(items, list) else []:
            names = item.get("Names") or []
            if not names:
                continue
            created = item.get("Created")
            result.append(
                {
                    "id": names[0],
                    "created_at": (
                        datetime.fromtimestamp(created, tz=UTC).isoformat()
                        if isinstance(created, (int, float))
                        else None
                    ),
                    # The pool names every container here, so none is a
                    # deliberately named artifact.
                    "names": [],
                }
            )
        return result

    def leftovers(self, sandbox_id: str) -> list[str]:
        """What of a sandbox still exists: its container, its network, a rootfs in a known store.

        It writes nothing to stderr, because most sandboxes asked about are already gone. A podman
        that cannot run gives an ``unknown`` entry, never an empty answer: "nothing left" must be shown."""
        found = []
        for kind in ("container", "network"):
            try:
                code = subprocess.run(
                    [self.podman_bin, kind, "exists", sandbox_id], capture_output=True, timeout=120
                ).returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                found.append("unknown (%s exists: %s)" % (kind, error))
                continue
            if code == 0:
                found.append(kind)
            elif code != 1:
                found.append("unknown (%s exists exited %d)" % (kind, code))
        found += [
            "rootfs in %s" % store
            for store in self.stores
            if (Path(store) / "sandboxes" / sandbox_id).exists()
        ]
        return found

    def delete_sandbox(self, sandbox_id: str) -> bool:
        """Remove the sandbox's container, network, rootfs and socket directory.

        True if the container, the network and the rootfs are gone afterwards."""
        raw = self._podman("inspect", "--type", "container", "--format", "json", sandbox_id)
        if raw is None:
            if any(item["id"] == sandbox_id for item in self._containers()):
                return False  # still listed but not inspectable: not reclaimed, and not claimed to be
            # No container is left. The rootfs can be in a known store or in the store that the
            # network's label names, and the socket directory is where the pool puts it by default.
            known = sorted(
                set(self.stores)
                | {
                    network["store"]
                    for network in self._networks()
                    if network["name"] == sandbox_id and network["store"]
                }
            )
            stores = [store for store in known if (Path(store) / "sandboxes" / sandbox_id).is_dir()]
            return self._discard_traces(
                sandbox_id, stores, [str(Path(store) / "sockets" / sandbox_id) for store in known]
            )
        # The container's store label and socket mount name the rest of the sandbox. Removal then
        # follows the pool's own teardown order: container, network, rootfs, socket directory.
        [info] = json.loads(raw)
        store = ((info.get("Config") or {}).get("Labels") or {}).get(STORE_LABEL)
        sockets = [
            m.get("Source")
            for m in info.get("Mounts") or []
            if m.get("Destination") == SOCKET_MOUNT and m.get("Source")
        ]
        if self._podman("rm", "-f", sandbox_id) is None:
            return False
        return self._discard_traces(sandbox_id, [store] if store else [], sockets)

    def _discard_traces(self, sandbox_id: str, stores: list[str], sockets: list[str]) -> bool:
        """Remove the sandbox's network, its rootfs in each of ``stores``, and the ``sockets`` directories.

        True if the network and each rootfs are gone afterwards."""
        # A sandbox of an older pool has no network of its own. ``network exists`` then fails
        # without output, so it counts as gone.
        reclaimed = (
            self._podman("network", "exists", sandbox_id) is None
            or self._podman("network", "rm", sandbox_id) is not None
        )
        for store in stores:
            try:
                btrfs = BtrfsStore(store)
                asyncio.run(btrfs.discard_rootfs(btrfs.sandboxes / sandbox_id, podman_bin=self.podman_bin))
            except Exception as error:  # noqa: BLE001 - the container is gone, so say what is left
                print(
                    "%s: container removed but its rootfs in %s was not: %s" % (sandbox_id, store, error),
                    file=sys.stderr,
                )
                reclaimed = False
        for directory in sockets:
            shutil.rmtree(directory, ignore_errors=True)
        return reclaimed


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_ts(text: object) -> datetime | None:
    if not text or not isinstance(text, str):
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
