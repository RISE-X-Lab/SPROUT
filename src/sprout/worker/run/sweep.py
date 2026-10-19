"""The deletion of the snapshots that a run's journal names, and the mark of a swept journal.

A run names each snapshot that it captured in a ``checkpoint.captured`` record of its journal, and
its backend and podman store in ``sandbox.created``. A sweep appends a ``snapshots.swept`` record,
and a later sweep of the same journal skips it. ``python -m sprout.worker sweep`` sweeps here.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sprout.worker.run.events import CHECKPOINT_CAPTURED, SANDBOX_CREATED
from sprout.worker.run.journal import journal_events
from sprout.worker.run.reap import AgentEnvClient
from sprout_sandbox import BtrfsStore


@dataclass(frozen=True)
class JournalSnapshots:
    """The snapshots that the journal at ``path`` names, in the order of their first record."""

    path: Path
    ids: tuple[str, ...]
    #: The backend of the run's sandbox: ``microvm`` when the journal names none.
    backend: str
    #: For podman, the store that the journal names, if any.
    store: str | None
    #: Whether a sweep marked the journal already.
    swept: bool
    #: The highest ``seq`` of the journal's records.
    last_seq: int


def journal_snapshots(path: Path) -> JournalSnapshots:
    """What the journal at ``path`` says about its snapshots. A torn last line counts as no record."""
    ids: list[str] = []
    last_seq, swept, backend, store = 0, False, "microvm", None
    for record in journal_events(path):
        last_seq = max(last_seq, record.get("seq") or 0)
        if record.get("type") == SANDBOX_CREATED and record.get("backend"):
            backend, store = record["backend"], record.get("store")
        if record.get("type") == CHECKPOINT_CAPTURED and record.get("snapshot_id"):
            if record["snapshot_id"] not in ids:
                ids.append(record["snapshot_id"])
        if record.get("type") == "snapshots.swept":
            swept = True
    return JournalSnapshots(path, tuple(ids), backend, store, swept, last_seq)


def snapshot_deleter(
    backend: str, store: str | None, *, agentenv: AgentEnvClient | None = None
) -> Callable[[str], bool] | str:
    """A function that deletes one snapshot of ``backend``, or the reason that there is none.

    It returns True if the snapshot is gone afterwards. For podman, it deletes the subvolume in ``store``,
    and reports a failure on stderr. A microVM's goes to ``agentenv``, by default at ``AENV_SERVER_URL``.
    """
    if backend == "microvm":
        return (agentenv or AgentEnvClient()).delete_snapshot
    if backend != "podman":
        return "a %s run; no way to delete its snapshots is known here" % backend
    if not store:
        return (
            "a podman run whose journal does not name its store; pass --store <dir> "
            "(the store its subvolumes live in)"
        )
    try:
        btrfs = BtrfsStore(store)
    except ValueError as error:
        return "a podman run: %s" % error

    def delete(snapshot_id: str) -> bool:
        if not (btrfs.snapshots / snapshot_id).is_dir():
            return True
        try:
            asyncio.run(btrfs.delete_snapshot(snapshot_id))
        except Exception as error:  # noqa: BLE001 - reported per snapshot, never raised
            print("%s: %s" % (snapshot_id, error), file=sys.stderr)
            return False
        return True

    return delete


def sweep(snapshots: JournalSnapshots, delete: Callable[[str], bool]) -> tuple[int, int]:
    """Delete each of ``snapshots`` with ``delete``, mark the journal swept, and return the counts.

    The counts are the deleted and the failed snapshots. A torn last line of the journal moves into
    the mark, as ``torn_tail``, so that the journal stays a file of whole lines.
    """
    deleted = failed = 0
    for snapshot_id in snapshots.ids:
        if delete(snapshot_id):
            deleted += 1
        else:
            failed += 1
    marker = {
        "v": 2,
        "type": "snapshots.swept",
        "seq": snapshots.last_seq + 1,
        "run_id": "sweep",
        "agent_id": "sweep",
        "backend": snapshots.backend,
        "deleted": deleted,
        "failed": failed,
    }
    data = snapshots.path.read_bytes()
    whole_lines_end = data.rfind(b"\n") + 1
    torn = data[whole_lines_end:].decode("utf-8", errors="replace")
    if torn:
        marker["torn_tail"] = torn
        with snapshots.path.open("r+b") as stream:
            stream.truncate(whole_lines_end)
    with snapshots.path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(marker) + "\n")
    return deleted, failed
