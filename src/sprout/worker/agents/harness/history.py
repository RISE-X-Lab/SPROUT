"""The harness's native history on disk (``History``) and the branch points in it (``index_harness``).

The record format is in ``sprout.contract.harness_history``. A branch point is a turn that ran its
one call, closed in the journal, and paired with the snapshot taken at its end. Rejected and
submitted turns are message steps, but not branch points.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from sprout.contract.harness_history import FORMAT, MESSAGE, SESSION, SLOT, TURN
from sprout.contract.mini_history import NativePoint, read_entries, read_prefix
from sprout.worker.checkpoint.rollback import (
    branch_checkpoints,
    closed_in_order,
    closed_turns,
    exact_snapshot,
)
from sprout.worker.run.events import TOOL_SKIPPED


class History:
    """The native history of one harness session: a file of JSON lines that only grows.

    Each ``append`` returns after its line is on disk. Raises ``FileExistsError`` if ``path`` exists."""

    def __init__(
        self, path: Path | str, session_id: str, *, workspace: dict[str, Any], tools: list[dict[str, Any]]
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        header = {
            "type": SESSION,
            "format": FORMAT,
            "session_id": session_id,
            "workspace": deepcopy(workspace),
        }
        self.append({**header, "tools": deepcopy(tools)})

    def append(self, entry: dict[str, Any]) -> None:
        """Write ``entry`` as one JSON line, and return after it is on disk."""
        self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def message(self, message: dict[str, Any]) -> None:
        """Record one chat message."""
        self.append({"type": MESSAGE, "message": message})

    def turn(self, turn_id: str, call_ids: list[str], *, source: str = "actor", **marks: Any) -> None:
        """Record the end of the model response ``turn_id``, with ``rejected`` or ``submitted`` marks."""
        self.append({"type": TURN, "turn_id": turn_id, "call_ids": list(call_ids), "source": source, **marks})

    def close(self) -> None:
        self.stream.close()


def index_harness(
    journal: str | Path,
    transcript: str | Path,
    session_id: str,
    *,
    events: list[dict[str, Any]],
    inherited_native: dict[str, Any] | None = None,
) -> list[NativePoint]:
    """The branch points of the transcript of ``session_id`` that the journal's ``events`` confirm, in order.

    Indexing stops at the first record that the journal does not confirm. ``inherited_native`` is the
    reference of a restored prefix, or None. Raises ``ValueError`` for another format or an unknown record."""
    local = [e["call_id"] for e in events if e.get("type") == "tool.started"]
    positions = {call: index for index, call in enumerate(local, 1)}
    admitted = set(local) | {
        e["call_id"] for e in events if e.get("type") == TOOL_SKIPPED and e.get("call_id")
    }
    if inherited_native:
        if inherited_native.get("slot") != SLOT:
            raise ValueError("Inherited native prefix uses a different slot")
        for entry in read_entries(read_prefix(inherited_native)):
            admitted.update(call["id"] for call in (entry.get("message") or {}).get("tool_calls") or [])
    checkpoints = branch_checkpoints(journal, events=events)
    closed = closed_turns(events)
    points: list[NativePoint] = []
    pending: set[str] = set()
    seen: set[str] = set()
    group: list[str] = []
    checksum, offset, message_step = hashlib.sha256(), 0, 0
    for line in Path(transcript).read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        row = json.loads(line)
        checksum.update(line)
        offset += len(line)
        kind = row.get("type")
        if kind == SESSION:
            if row.get("format") != FORMAT:
                raise ValueError("Unsupported harness native history format")
        elif kind == MESSAGE:
            message = row["message"]
            calls = [call["id"] for call in message.get("tool_calls") or []]
            if calls:
                if (
                    pending
                    or group
                    or len(set(calls)) != len(calls)
                    or not set(calls) <= admitted
                    or set(calls) & seen
                ):
                    break
                group = calls
                pending.update(calls)
            if message.get("role") == "tool":
                if message.get("tool_call_id") not in pending:
                    break
                pending.remove(message["tool_call_id"])
                seen.add(message["tool_call_id"])
        elif kind == TURN:
            if pending or row.get("call_ids") != group:
                break
            message_step += 1
            local_calls = [call for call in group if call in positions]
            if not row.get("rejected") and not row.get("submitted") and local_calls:
                depth = max(positions[call] for call in local_calls)
                turn = closed.get(row.get("turn_id"), {})
                if not closed_in_order(turn, group=group, depth=depth, local_calls=local, seen=seen):
                    break
                snapshot_id = exact_snapshot(
                    checkpoints.get(depth), turn_id=row["turn_id"], session_id=session_id
                )
                if snapshot_id is not None:
                    reference = {
                        "slot": SLOT,
                        "session_id": session_id,
                        "path": str(Path(transcript).resolve()),
                        "byte_length": offset,
                        "sha256": checksum.hexdigest(),
                        "cut": row["turn_id"],
                    }
                    points.append(NativePoint(message_step, depth, snapshot_id, reference))
            group = []
        else:
            raise ValueError("Unknown harness native record %r" % kind)
    return points
