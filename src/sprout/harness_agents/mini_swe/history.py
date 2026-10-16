"""mini's native history: durable messages, closed turns and their branch points.

A branch point is a closed model turn whose last tool call has an exact
snapshot; ``index_mini`` finds them, ``reference_at`` names the prefix a branch
resumes from, and ``materialize`` writes that prefix for a new attempt.
"""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

from sprout.contract.mini_history import (
    FORMAT,
    FORMATS,
    NativePoint,
    read_entries,
    read_prefix,
    validate_mini_tools,
)
from sprout.harness_checkpoint.rollback import branch_checkpoints, closed_turns
from sprout.harness_core.journal import journal_events, read_journal


class History:
    def __init__(self, path: Path | str, session_id: str, *, prefix=(), workspace=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        metadata = {
            "type": "mini.session",
            "format": FORMAT,
            "mini_version": "2.4.6",
            "session_id": session_id,
        }
        if workspace is not None:
            metadata["workspace"] = deepcopy(workspace)
        self.append(metadata)
        self.prefix_messages = []
        for entry in prefix:
            if entry["type"] != "mini.session":
                self.append(entry)
            if entry["type"] == "mini.message" and entry["message"].get("role") != "exit":
                self.prefix_messages.append(deepcopy(entry["message"]))

    def append(self, entry: dict) -> None:
        self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def index_native(
    journal: Path,
    transcript: Path,
    slot: str,
    session_id: str,
    *,
    events: list[dict] | None = None,
    inherited_native: dict | None = None,
) -> list[NativePoint]:
    """Recovery points of one attempt: closed turns with an exact snapshot."""
    if slot != "mini-swe-agent":
        raise ValueError("Unsupported native history slot")
    if events is None:
        # Split as bytes, like every journal reader: str.splitlines() also
        # breaks at U+0085 and U+2028, which a tool's output keeps verbatim.
        events = journal_events(journal)
    return index_mini(journal, transcript, session_id, events=list(events), inherited_native=inherited_native)


def index_mini(
    journal: Path,
    transcript: Path,
    session_id: str,
    *,
    events: list[dict],
    inherited_native: dict | None = None,
) -> list:
    local_calls = [e["call_id"] for e in events if e.get("type") == "tool.started"]
    positions = {call: i for i, call in enumerate(local_calls, 1)}
    admitted = set(local_calls)
    rejection_events = {
        e["call_id"]: e
        for e in events
        if e.get("type") == "tool.rejected"
        and e.get("source") == "schema_validation"
        and e.get("executed") is False
        and e.get("call_id")
    }
    rejected_turns = {
        e["turn_id"]: e
        for e in events
        if e.get("type") == "mini.turn.rejected"
        and e.get("source") == "schema_validation"
        and e.get("executed") is False
        and e.get("turn_id")
    }
    admitted.update(rejection_events)
    inherited_rejections = {}
    if inherited_native:
        if inherited_native["slot"] != "mini-swe-agent":
            raise ValueError("Inherited native prefix uses a different slot")
        for entry in read_entries(read_prefix(inherited_native)):
            admitted.update(call["id"] for call in entry.get("message", {}).get("tool_calls") or [])
            if entry.get("type") == "mini.turn" and entry.get("rejected") is True:
                inherited_rejections[entry["turn_id"]] = entry["call_ids"]
    checkpoints = branch_checkpoints(journal, events=events)
    closed = closed_turns(events)
    points, pending, seen, group = [], set(), set(), []
    checksum, offset, message_step = hashlib.sha256(), 0, 0
    for line in Path(transcript).read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        row = json.loads(line)
        checksum.update(line)
        offset += len(line)
        kind = row.get("type")
        if kind == "mini.session":
            if row.get("format") not in FORMATS or row.get("mini_version") != "2.4.6":
                raise ValueError("Unsupported mini native history version")
        elif kind == "mini.message":
            message = row["message"]
            calls = message.get("tool_calls") or []
            if calls:
                if pending or group:
                    break
                group = [call["id"] for call in calls]
                if len(set(group)) != len(group) or not set(group) <= admitted or set(group) & seen:
                    break
                pending.update(group)
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in pending:
                    break
                if call_id in rejection_events and message.get("content") != rejection_events[call_id].get(
                    "feedback"
                ):
                    break
                pending.remove(call_id)
                seen.add(call_id)
        elif kind == "mini.turn":
            if pending or row.get("call_ids") != group:
                break
            message_step += 1
            is_rejection = (
                bool(set(group) & set(rejection_events))
                or row.get("turn_id") in rejected_turns
                or row.get("turn_id") in inherited_rejections
            )
            if is_rejection or row.get("rejected") is True:
                if row.get("rejected") is not True:
                    break
                tid = row.get("turn_id")
                if set(group) & set(local_calls):
                    break
                if tid in inherited_rejections:
                    if inherited_rejections[tid] != group:
                        break
                elif rejected_turns.get(tid, {}).get("call_ids") != group or any(
                    rejection_events.get(call, {}).get("turn_id") != tid for call in group
                ):
                    break
                group = []
                continue
            local = [call for call in group if call in positions]
            if local:
                depth = max(positions[call] for call in local)
                turn = closed.get(row.get("turn_id"), {})
                point = checkpoints.get(depth)
                if (
                    turn.get("call_ids") != group
                    or turn.get("step") != depth
                    or not set(local_calls[:depth]) <= seen
                    or set(local_calls[depth:]) & seen
                ):
                    break
                if point and point.turn_id == row["turn_id"] and point.session_ckpt == session_id:
                    points.append(
                        NativePoint(
                            message_step,
                            depth,
                            point.snapshot_id,
                            {
                                "slot": "mini-swe-agent",
                                "session_id": session_id,
                                "path": str(Path(transcript).resolve()),
                                "byte_length": offset,
                                "sha256": checksum.hexdigest(),
                                "cut": row["turn_id"],
                            },
                        )
                    )
            group = []
        else:
            raise ValueError("Unknown mini native record")
    return points


def reference_at(journal: Path, step: int, session_id: str) -> dict | None:
    events = journal_events(Path(journal))
    reference = next(
        (
            e
            for e in reversed(events)
            if e.get("type") == "session.ref"
            and e.get("native_session_id") == session_id
            and e.get("transcript_path")
        ),
        None,
    )
    if reference is None:
        return None
    inherited = next((e.get("inherited_native") for e in events if e.get("type") == "mini.restored"), None)
    points = index_mini(
        journal, reference["transcript_path"], session_id, events=events, inherited_native=inherited
    )
    return next((point.native for point in points if point.tool_depth == step), None)


def conversation_restore(journal_path, step: int, session_ref: str) -> str | None:
    """The exact native cut at ``step``, or None when there is none."""
    if not any(
        row.get("type") == "run.started" and row.get("slot") == "mini-swe-agent"
        for row in read_journal(journal_path)
    ):
        return None
    reference = reference_at(journal_path, step, session_ref)
    return reference["cut"] if reference else None


def available_branch_points(journal_path):
    """The same candidates, indexing each native session once per inspection."""
    events = read_journal(journal_path)
    points = branch_checkpoints(journal_path, events=events)
    if not any(row.get("type") == "run.started" and row.get("slot") == "mini-swe-agent" for row in events):
        return {}
    # Match reference_at's last reference for each session. A local index avoids
    # reparsing the entire transcript once for every checkpoint, without keeping
    # stale state when the journal or native history changes between inspections.
    references = {
        row.get("native_session_id"): row
        for row in events
        if row.get("type") == "session.ref" and row.get("transcript_path")
    }
    inherited = next(
        (row.get("inherited_native") for row in events if row.get("type") == "mini.restored"), None
    )
    cuts = {}
    for session_id in dict.fromkeys(point.session_ckpt for point in points.values()):
        reference = references.get(session_id)
        if reference is None:
            continue
        indexed = index_mini(
            journal_path, reference["transcript_path"], session_id, events=events, inherited_native=inherited
        )
        cuts[session_id] = {point.tool_depth: point.native["cut"] for point in indexed}
    return {step: point for step, point in points.items() if cuts.get(point.session_ckpt, {}).get(step)}


def actor_tools_at(journal, step: int) -> list[dict]:
    """Read the last actual tool declaration before the selected turn closed."""
    tools = None
    for event in read_journal(journal):
        if event.get("type") == "rollout.model_tools" and event.get("shape") == "chat/completions":
            tools = event.get("tools")
        if event.get("type") == "model.turn.completed" and event.get("step") == step:
            if tools is None:
                raise ValueError("No recorded actor tool schema at the selected checkpoint")
            return validate_mini_tools(tools)
    raise ValueError("No completed turn for the selected actor tool schema")
