"""The resource ledger: the claims that a run records, and their replay into orphans and kept sandboxes.

A ledger is the record that ``sprout.worker reap`` reads in another process, often after the run's own
process died. So the tests write the lines of such a process by hand where the API of a live run
cannot, for example a claim of a dead pid.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.run.resources import ResourceLedger

#: RFC 3339 UTC to the second. Ledgers on disk have this precision, so it cannot change.
SECOND_UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
#: Above the largest pid_max of Linux (2**22), so no process has it.
DEAD_PID = 999_999_999
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)


def write_ledger(path: Path, *entries: dict[str, Any]) -> ResourceLedger:
    """The ledger at ``path``, after another process appended ``entries`` to it."""
    with path.open("a") as stream:
        stream.writelines(json.dumps(entry) + "\n" for entry in entries)
    return ResourceLedger(path)


def claim(sandbox_id: str = "sandbox-1", **fields: Any) -> dict[str, Any]:
    """A ``claim`` line of run ``run-1`` for ``sandbox_id``, by a process that is gone."""
    return {
        "event": "claim",
        "run_id": "run-1",
        "kind": "sandbox",
        "id": sandbox_id,
        "pid": DEAD_PID,
        **fields,
    }


def claim_history(
    *,
    pid: object = DEAD_PID,
    keep: bool = False,
    keep_until: str | None = None,
    released: bool = False,
    run_done: bool = False,
) -> list[dict[str, Any]]:
    """The lines of run ``run-1`` that claimed ``sandbox-1``, then released it and ended, as the flags say."""
    lines = [claim(pid=pid, keep=keep, **({"keep_until": keep_until} if keep_until else {}))]
    if released:
        lines.append({"event": "release", "run_id": "run-1", "kind": "sandbox", "id": "sandbox-1"})
    if run_done:
        lines.append({"event": "run_done", "run_id": "run-1"})
    return lines


def without_timestamps(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in entry.items() if key != "ts"} for entry in entries]


def test_a_claim_and_the_end_of_its_run_are_written_in_the_format_that_reap_reads(tmp_path):
    ledger = ResourceLedger(tmp_path / "resources.jsonl")

    with ledger.run("run-1") as run:
        run.sandbox(
            "sandbox-1",
            keep=True,
            backend="podman",
            store="/path/to/store",
            keep_until="2030-01-01T00:20:00+00:00",
        )
        run.sandbox("sandbox-2")

    entries = list(ledger.entries())
    assert without_timestamps(entries) == [
        {"event": "claim", "run_id": "run-1", "kind": "sandbox", "id": "sandbox-1", "pid": os.getpid(),
         "keep": True, "backend": "podman", "store": "/path/to/store",
         "keep_until": "2030-01-01T00:20:00+00:00"},
        {"event": "claim", "run_id": "run-1", "kind": "sandbox", "id": "sandbox-2", "pid": os.getpid(),
         "keep": False, "backend": "microvm"},
        {"event": "run_done", "run_id": "run-1"},
    ]  # fmt: skip
    assert [bool(SECOND_UTC.fullmatch(entry["ts"])) for entry in entries] == [True, True, True]


def test_a_sandbox_that_its_run_released_replays_as_released(tmp_path):
    ledger = ResourceLedger(tmp_path / "resources.jsonl")

    with ledger.run("run-1") as run:
        run.sandbox("sandbox-1")
        run.released("sandbox", "sandbox-1")

    [resource] = ledger.resources()
    assert (resource.id, resource.released, resource.run_done, resource.orphaned()) == (
        "sandbox-1",
        True,
        True,
        False,
    )


def test_the_run_is_marked_done_also_when_it_raises(tmp_path):
    ledger = ResourceLedger(tmp_path / "resources.jsonl")

    with pytest.raises(RuntimeError, match="agent exploded"):
        with ledger.run("run-1") as run:
            run.sandbox("sandbox-1")
            raise RuntimeError("agent exploded")

    [resource] = ledger.resources()
    assert (resource.run_done, resource.orphaned()) == (True, True)


PIDS = {"live": os.getpid(), "dead": DEAD_PID, "unknown": None}


# 4-way covering array (covering_array.py), the full product: the orphan rule joins all four parameters.
@pytest.mark.parametrize("released,keep,run_done,pid", [
    (False, False, False, "live"),
    (False, False, False, "dead"),
    (False, False, False, "unknown"),
    (False, False, True, "live"),
    (False, False, True, "dead"),
    (False, False, True, "unknown"),
    (False, True, False, "live"),
    (False, True, False, "dead"),
    (False, True, False, "unknown"),
    (False, True, True, "live"),
    (False, True, True, "dead"),
    (False, True, True, "unknown"),
    (True, False, False, "live"),
    (True, False, False, "dead"),
    (True, False, False, "unknown"),
    (True, False, True, "live"),
    (True, False, True, "dead"),
    (True, False, True, "unknown"),
    (True, True, False, "live"),
    (True, True, False, "dead"),
    (True, True, False, "unknown"),
    (True, True, True, "live"),
    (True, True, True, "dead"),
    (True, True, True, "unknown"),
])  # fmt: skip
def test_a_claim_is_an_orphan_when_it_is_neither_released_nor_kept_and_its_run_or_process_is_gone(
    tmp_path, released, keep, run_done, pid
):
    """A finished run's unreleased claim is an orphan too: a kill between its two last writes leaves one."""
    history = claim_history(pid=PIDS[pid], keep=keep, released=released, run_done=run_done)
    ledger = write_ledger(tmp_path / "resources.jsonl", *history)

    [resource] = ledger.resources()

    assert resource.orphaned() is (not released and not keep and (run_done or pid == "dead"))


@pytest.mark.parametrize(
    "pid", [pytest.param(1, id="another-users-process"), pytest.param("not-a-pid", id="not-a-number")]
)
def test_a_process_that_the_reaper_cannot_check_counts_as_alive(tmp_path, pid):
    """Reaping a sandbox that a live run still needs is worse than a stale claim for the next pass."""
    ledger = write_ledger(tmp_path / "resources.jsonl", *claim_history(pid=pid))

    [resource] = ledger.resources()

    assert resource.orphaned() is False


# The full product: the expiry rule joins all three parameters.
@pytest.mark.parametrize("keep,released,keep_until,expired", [
    (False, False, None, False),
    (False, False, "2030-01-01T11:59:59+00:00", False),
    (False, False, "2030-01-01T12:00:00+00:00", False),
    (False, False, "2030-01-01T12:00:01+00:00", False),
    (False, True, None, False),
    (False, True, "2030-01-01T11:59:59+00:00", False),
    (False, True, "2030-01-01T12:00:00+00:00", False),
    (False, True, "2030-01-01T12:00:01+00:00", False),
    (True, False, None, False),
    (True, False, "2030-01-01T11:59:59+00:00", True),
    (True, False, "2030-01-01T12:00:00+00:00", True),
    (True, False, "2030-01-01T12:00:01+00:00", False),
    (True, True, None, False),
    (True, True, "2030-01-01T11:59:59+00:00", False),
    (True, True, "2030-01-01T12:00:00+00:00", False),
    (True, True, "2030-01-01T12:00:01+00:00", False),
])  # fmt: skip
def test_a_kept_claim_expires_when_its_keep_until_is_not_after_now(
    tmp_path, keep, released, keep_until, expired
):
    history = claim_history(keep=keep, keep_until=keep_until, released=released)
    ledger = write_ledger(tmp_path / "resources.jsonl", *history)

    [resource] = ledger.resources()

    assert resource.expired(NOW) is expired


def test_a_claim_without_a_backend_replays_as_a_microvm_claim_without_a_store(tmp_path):
    """Ledgers from before the podman backend name no backend."""
    ledger = write_ledger(tmp_path / "resources.jsonl", claim("sandbox-old"))

    [resource] = ledger.resources()

    assert (resource.backend, resource.store) == ("microvm", None)


def test_a_torn_line_is_skipped_and_the_other_claims_replay(tmp_path):
    """A process killed in the middle of a write leaves half a line."""
    path = tmp_path / "resources.jsonl"
    write_ledger(path, claim("sandbox-1"))
    with path.open("a") as stream:
        stream.write('{"event": "claim", "kind": "sandb\n')
    write_ledger(path, claim("sandbox-2"))

    resources = ResourceLedger(path).resources()

    assert [resource.id for resource in resources] == ["sandbox-1", "sandbox-2"]


def test_a_missing_ledger_has_no_entries(tmp_path):
    ledger = ResourceLedger(tmp_path / "runs" / "resources.jsonl")

    assert (list(ledger.entries()), ledger.resources()) == ([], [])
