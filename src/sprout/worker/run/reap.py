"""Reclaim sandboxes that outlived their run.

Two independent sources, because either alone misses cases:

- the **ledger** (worker/run/resources.py) knows what each run
  allocated and on which backend, so it can attribute an orphan to a dead pid.
  It cannot see sandboxes created before the ledger existed, or by other tools.
- the **backends** know everything that exists. They have no owner field, so age
  is the only signal -- hence ``--older-than`` and an explicit opt-in
  (``--include-unknown``) before touching anything the ledger never saw.

Default behaviour is deliberately narrow: reap only ledger-attributed orphans.
Wider sweeps must be asked for, because "delete everything older than N hours"
is exactly the command that eats a colleague's in-progress experiment.

The reaper holds one client per backend, keyed by the name a ledger claim
carries: :class:`AgentEnvClient` for microVMs and :class:`PodmanClient` for the
containers of the podman backend. A claim naming a backend with no client is
listed as ``foreign`` and never touched.

Snapshots are never reaped. A snapshot's whole point is that any step of any
FINISHED run can be branched later, so "owning run died" -- the condition this
planner keys on -- describes every snapshot worth keeping. Snapshot deletion is
``harness sweep <journal> --yes``: the journal names exactly what dies, and
typing it is the consent.

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
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sprout.worker.run.resources import ResourceLedger
from sprout.worker.sandbox.backends import default_aenv_server_url
from sprout_sandbox.btrfs_store import BtrfsStore
from sprout_sandbox.podman_cli import default_podman_bin
from sprout_sandbox.podman_containers import MANAGED_LABEL, SOCKET_MOUNT, STORE_LABEL

_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd])$")


def parse_duration(text: str) -> timedelta:
    """``30m`` / ``24h`` / ``7d`` -> timedelta."""
    match = _DURATION.match(text.strip().lower())
    if not match:
        raise ValueError("bad duration %r (use 30m, 24h, 7d)" % text)
    value, unit = float(match.group(1)), match.group(2)
    return timedelta(**{{"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]: value})


@dataclass
class ReapPlan:
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

    ``clients`` maps a backend name (as ledger claims spell it: ``microvm``,
    ``podman``) to an object with ``list_sandboxes()`` and
    ``delete_sandbox(id) -> bool``.
    """

    def __init__(self, clients: dict[str, object], ledger: ResourceLedger | None = None) -> None:
        self.clients = dict(clients)
        self.ledger = ledger or ResourceLedger()

    # --- planning ----------------------------------------------------------
    def plan(
        self,
        *,
        include_unknown: bool = False,
        older_than: timedelta | None = None,
    ) -> ReapPlan:
        plan = ReapPlan()

        live = {
            backend: {s["id"]: s for s in client.list_sandboxes()} for backend, client in self.clients.items()
        }

        ledger_ids = set()
        for resource in self.ledger.resources():
            if resource.kind != "sandbox":
                # A ledger written before snapshot claims were dropped still
                # carries `kind: snapshot` rows. They are neither planned nor
                # counted: snapshots die only by `harness sweep`.
                continue
            ledger_ids.add(resource.id)
            if resource.keep:
                plan.kept.append(resource.id)
                continue
            if not resource.orphaned():
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
                continue  # already gone; the ledger entry is just stale
            plan.sandboxes.append(resource.id)
            plan.backends[resource.id] = resource.backend
            plan.reasons[resource.id] = "orphan of run %s (pid %s gone)" % (
                resource.run_id or "?",
                resource.pid,
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
        """Delete planned sandboxes, each through its backend's client. Failures
        are reported, never raised."""
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
    """Minimal AgentENV REST client for listing and deleting.

    Kept separate from ``sprout_sandbox`` on purpose: reaping must work when no
    pool, session or event loop exists -- including from a cron job after the
    process that leaked the resources is long gone. ``delete_snapshot`` is here
    for the two explicit callers that delete history (``harness sweep`` and
    ``methods.evaluate`` releasing a failed attempt's memory snapshot); the
    reaper itself never calls it.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        import os

        self.base_url = (base_url or default_aenv_server_url()).rstrip("/")
        self.api_key = api_key or os.environ.get("AENV_API_KEY") or ""

    def _headers(self) -> dict:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _get(self, path: str) -> list:
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

    def list_sandboxes(self) -> list[dict]:
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
    """The podman backend's containers, listed and removed without a pool.

    Self-contained for the same reason as :class:`AgentEnvClient`: a pool
    needs the runtime binary and a running loop, and a cleanup command should
    need neither. ``podman ps`` finds every container the pool started (label
    ``sprout.managed=1``); ``podman inspect`` says which store holds its rootfs
    subvolume (label ``sprout.store``) and where its socket directory is (the
    ``/run/sprout`` mount), so a container carries all that removing it takes.
    Removal is the pool's own teardown order: ``podman rm -f``, then the
    network named after the sandbox, the rootfs subvolume (subvolumes made
    inside it first) and the socket directory.

    A rootfs whose container is already gone -- a destroy that failed after
    ``podman rm -f`` (a nested subvolume, a store mounted without
    ``user_subvol_rm_allowed``), or a container removed by hand -- carries no
    label to find it by, so the client also lists the ``sandboxes/`` of every
    store it is given: the ones the ledger's claims name, and ``reap --store``.
    A network left without its container carries the pool's labels: it is
    listed from ``podman network ls``, and its ``sprout.store`` label says
    where the sandbox's rootfs is, so a container removed by hand is reclaimed
    whole with no store given. A host without podman has nothing to list.
    """

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

    def list_sandboxes(self) -> list[dict]:
        """Every managed container, plus every rootfs in a known store and every sandbox network without one."""
        result = self._containers()
        seen = {item["id"] for item in result}
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

    def _networks(self) -> list[dict]:
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

    def _containers(self) -> list[dict]:
        raw = self._podman("ps", "-a", "--filter", "label=" + MANAGED_LABEL, "--format", "json")
        try:
            items = json.loads(raw or "[]")
        except ValueError:
            return []
        result = []
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
                    # Every container here is named by the pool; none is a
                    # deliberately named artifact.
                    "names": [],
                }
            )
        return result

    def leftovers(self, sandbox_id: str) -> list[str]:
        """What of a sandbox still exists: its container, its network, a rootfs in a known store.

        The podman counterpart of AgentENV answering 404 for a deleted
        sandbox, asked without a word on stderr (most sandboxes asked about
        are already gone). A podman that cannot be run is reported as such,
        never as an empty answer: "nothing left" has to be established.
        """
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
        raw = self._podman("inspect", "--type", "container", "--format", "json", sandbox_id)
        if raw is None:
            if any(item["id"] == sandbox_id for item in self._containers()):
                return False  # still listed but not inspectable: not reclaimed, and not claimed to be
            # No container: what a destroy that failed after `podman rm -f` left,
            # or one removed by hand. Its rootfs, wherever a known store -- or
            # the store its network's label names -- holds it, and the socket
            # directory the pool gives such a sandbox by default.
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
        """The sandbox's network, the rootfs in each store (subvolumes made inside it first) and the socket directories."""
        # A sandbox of an older pool joined one network per store and has none
        # of its own; `network exists` answers that with no output.
        reclaimed = (
            self._podman("network", "exists", sandbox_id) is None
            or self._podman("network", "rm", sandbox_id) is not None
        )
        for store in stores:
            try:
                btrfs = BtrfsStore(store)
                asyncio.run(btrfs.discard_rootfs(btrfs.sandboxes / sandbox_id, podman_bin=self.podman_bin))
            except Exception as error:  # noqa: BLE001 - the container is gone; say what is not
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


def _parse_ts(text) -> datetime | None:
    if not text or not isinstance(text, str):
        return None
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
