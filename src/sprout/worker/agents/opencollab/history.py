"""Exact Single2 session states paired with closed Sprout runtime turns."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from sprout.contract.agents.opencollab import (
    FORMAT,
    SLOT,
    training_messages,
)
from sprout.contract.mini_history import NativePoint
from sprout.worker.checkpoint.rollback import branch_checkpoints, closed_turns
from sprout.worker.run.journal import read_journal


class History:
    def __init__(
        self,
        path: Any,
        session_id: Any,
        workspace: Any,
        prefix: Any = (),
        *,
        tool_protocol: Any = "single2",
        tools: Any = (),
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        self.append(
            {
                "type": "oc.session",
                "format": FORMAT,
                "profile": "single2",
                "session_id": session_id,
                "workspace": workspace,
                "tool_protocol": tool_protocol,
                "tools": list(tools),
            }
        )
        for row in prefix:
            if row["type"] != "oc.session":
                self.append(row)
        self.messages = training_messages(prefix)

    def append(self, row: Any) -> None:
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def persist_messages(self, messages: Any) -> None:
        if messages[: len(self.messages)] == self.messages:
            for message in messages[len(self.messages) :]:
                self.append({"type": "oc.message", "message": deepcopy(message)})
        else:
            # Native context shaping may rewrite observations or remove old
            # turns. Persist that state, not a stale append-only conversation.
            self.append({"type": "oc.context", "messages": deepcopy(messages)})
        self.messages = deepcopy(messages)

    def close(self) -> None:
        self.stream.close()


def index_native(
    journal: Any, transcript: Any, session_id: Any, *, events: Any = None, inherited_native: Any = None
) -> list[NativePoint]:
    events = list(read_journal(journal)) if events is None else events
    calls = [row["call_id"] for row in events if row.get("type") == "tool.started"]
    positions = {identifier: index for index, identifier in enumerate(calls, 1)}
    checkpoints = branch_checkpoints(journal, events=events)
    closed = closed_turns(events)
    points: list[NativePoint] = []
    seen: set[str] = set()
    checksum, offset, message_step = hashlib.sha256(), 0, 0
    pending: set[str] = set()
    used: set[str] = set()
    for line in Path(transcript).read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        checksum.update(line)
        offset += len(line)
        row = json.loads(line)
        if row.get("type") == "oc.session":
            if row.get("format") != FORMAT:
                raise ValueError("Unsupported OpenCollab native format")
        elif row.get("type") == "oc.context":
            pending, used = set(), set()
            for message in row["messages"]:
                identifiers = [call["id"] for call in message.get("tool_calls") or []]
                if identifiers:
                    if pending or len(set(identifiers)) != len(identifiers) or set(identifiers) & used:
                        raise ValueError("Invalid shaped Single2 tool-call history")
                    pending.update(identifiers)
                    used.update(identifiers)
                if message.get("role") == "tool":
                    identifier = message.get("tool_call_id")
                    if identifier not in pending:
                        raise ValueError("Unpaired shaped Single2 tool result")
                    pending.remove(identifier)
        elif row.get("type") == "oc.message":
            message = row["message"]
            identifiers = [call["id"] for call in message.get("tool_calls") or []]
            if identifiers:
                if pending or len(set(identifiers)) != len(identifiers) or set(identifiers) & used:
                    break
                pending.update(identifiers)
                used.update(identifiers)
            if message.get("role") == "tool":
                identifier = message.get("tool_call_id")
                if identifier not in pending:
                    break
                pending.remove(identifier)
        elif row.get("type") == "oc.turn":
            if pending or not isinstance(row.get("session_snapshot"), dict):
                break
            message_step += 1
            runtime_ids = row.get("runtime_call_ids", [])
            local = [identifier for identifier in runtime_ids if identifier in positions]
            if not local:
                continue  # inherited turn or schema rejection: no local disk change
            if len(local) != len(runtime_ids) or set(local) & seen:
                break
            seen.update(local)
            depth = max(positions[identifier] for identifier in local)
            turn = closed.get(row["turn_id"], {})
            point = checkpoints.get(depth)
            if turn.get("call_ids") != runtime_ids or turn.get("step") != depth or set(calls[:depth]) != seen:
                break
            if (
                point
                and point.snapshot_id is not None
                and point.turn_id == row["turn_id"]
                and point.session_ckpt == session_id
            ):
                points.append(
                    NativePoint(
                        message_step,
                        depth,
                        point.snapshot_id,
                        {
                            "slot": SLOT,
                            "session_id": session_id,
                            "path": str(Path(transcript).resolve()),
                            "byte_length": offset,
                            "sha256": checksum.hexdigest(),
                            "cut": row["turn_id"],
                        },
                    )
                )
        else:
            raise ValueError("Unknown OpenCollab native record")
    return points


def reference_at(journal: Any, step: Any, session_id: Any) -> dict[str, Any] | None:
    events = list(read_journal(journal))
    reference = next(
        (
            row
            for row in reversed(events)
            if row.get("type") == "session.ref" and row.get("native_session_id") == session_id
        ),
        None,
    )
    if reference is None:
        return None
    points = index_native(journal, reference["transcript_path"], session_id, events=events)
    return next((point.native for point in points if point.tool_depth == step), None)
