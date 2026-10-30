"""mini's native history: the transcript that ``History`` writes, and the branch points in it.

A branch point is a closed model turn whose last executed tool call has an exact snapshot.
``index_mini`` finds the branch points of one transcript. ``reference_at`` gives the native
reference of the prefix that a branch resumes from, and ``sprout.contract.mini_history.materialize``
writes that prefix for a new attempt. The format is in ``sprout.contract.mini_history``.
"""

import hashlib
import json
import os
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sprout.contract.mini_history import (
    FORMAT,
    FORMATS,
    NativePoint,
    read_entries,
    read_prefix,
    validate_mini_tools,
)
from sprout.worker.agents.mini_swe.sandbox import Workspace
from sprout.worker.checkpoint.rollback import (
    Checkpoint,
    branch_checkpoints,
    closed_in_order,
    closed_turns,
    exact_snapshot,
)
from sprout.worker.run.events import TOOL_SKIPPED
from sprout.worker.run.journal import journal_events, read_journal


class History:
    """The native history of one mini session: a file of JSON lines that only grows.

    The file starts with a ``mini.session`` header, then the entries of ``prefix``. ``append``
    returns after its entry is on disk. Raises ``FileExistsError`` if ``path`` exists.
    """

    def __init__(
        self,
        path: Path | str,
        session_id: str,
        *,
        prefix: Iterable[dict[str, Any]] = (),
        workspace: Workspace | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        metadata: dict[str, Any] = {
            "type": "mini.session",
            "format": FORMAT,
            "mini_version": "2.4.6",
            "session_id": session_id,
        }
        if workspace is not None:
            metadata["workspace"] = deepcopy(workspace)
        self.append(metadata)
        #: The messages of ``prefix`` that the agent resumes from: every message except ``exit``.
        self.prefix_messages: list[dict[str, Any]] = []
        for entry in prefix:
            if entry["type"] != "mini.session":
                self.append(entry)
            if entry["type"] == "mini.message" and entry["message"].get("role") != "exit":
                self.prefix_messages.append(deepcopy(entry["message"]))

    def append(self, entry: dict[str, Any]) -> None:
        """Write ``entry`` as one JSON line, and return after it is on disk."""
        self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def index_native(
    journal: str | Path,
    transcript: str | Path,
    slot: str,
    session_id: str,
    *,
    events: list[dict[str, Any]] | None = None,
    inherited_native: dict[str, Any] | None = None,
) -> list[NativePoint]:
    """The recovery points of one attempt: its closed turns with an exact snapshot.

    ``events`` are the records of ``journal``, read from it when None. Raises ``ValueError`` for a
    ``slot`` other than ``mini-swe-agent`` (``index_mini``) and ``sprout-harness`` (``index_harness``).
    """
    if slot not in ("mini-swe-agent", "sprout-harness"):
        raise ValueError("Unsupported native history slot")
    if events is None:
        # journal_events splits the bytes. str.splitlines() would also break at U+0085 and
        # U+2028, which a tool's output keeps verbatim.
        events = journal_events(journal)
    if slot == "sprout-harness":
        from sprout.worker.agents.harness.history import index_harness

        return index_harness(
            journal, transcript, session_id, events=list(events), inherited_native=inherited_native
        )
    return index_mini(journal, transcript, session_id, events=list(events), inherited_native=inherited_native)


def index_mini(
    journal: str | Path,
    transcript: str | Path,
    session_id: str,
    *,
    events: list[dict[str, Any]],
    inherited_native: dict[str, Any] | None = None,
) -> list[NativePoint]:
    """The branch points in the transcript of ``session_id`` that the journal's ``events`` confirm, in order.

    Indexing stops at the first record that the journal does not confirm. ``inherited_native`` is the
    reference of the restored prefix, or None. Raises ``ValueError`` for an unknown format or record.
    """
    journaled = _journal_calls(events, inherited_native=inherited_native)
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
                if len(set(group)) != len(group) or not set(group) <= journaled.admitted or set(group) & seen:
                    break
                pending.update(group)
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in pending:
                    break
                if call_id in journaled.rejections and message.get("content") != journaled.rejections[
                    call_id
                ].get("feedback"):
                    break
                pending.remove(call_id)
                seen.add(call_id)
            # A response with a malformed envelope or with no tool call adds no mini.turn. mini's
            # FormatError feedback is then the only record of that model turn.
            if message.get("extra", {}).get("interrupt_type") == "FormatError":
                message_step += 1
        elif kind == "mini.turn":
            if pending or row.get("call_ids") != group:
                break
            message_step += 1
            if _is_rejection(row, group=group, journaled=journaled):
                if not _rejection_confirmed(row, group=group, journaled=journaled):
                    break
                group = []
                continue
            local = [call for call in group if call in journaled.positions]
            if local:
                depth = max(journaled.positions[call] for call in local)
                turn = closed.get(row.get("turn_id"), {})
                if not closed_in_order(
                    turn, group=group, depth=depth, local_calls=journaled.local, seen=seen
                ):
                    break
                snapshot_id = exact_snapshot(
                    checkpoints.get(depth), turn_id=row["turn_id"], session_id=session_id
                )
                if snapshot_id is not None:
                    points.append(
                        NativePoint(
                            message_step,
                            depth,
                            snapshot_id,
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


@dataclass(frozen=True)
class _JournalCalls:
    """The tool calls that a journal and its inherited prefix confirm, for ``index_mini`` to check against."""

    #: The id of each call that started, in order.
    local: list[str]
    #: The tool depth of each call that started: its position in ``local``, from 1.
    positions: dict[str, int]
    #: Each call id that the transcript can hold: started, refused, skipped or inherited.
    admitted: set[str]
    #: The schema refusal of each call that did not execute, by call id.
    rejections: dict[str, dict[str, Any]]
    #: The schema refusal of each turn that did not execute, by turn id.
    rejected_turns: dict[str | None, dict[str, Any]]
    #: The call ids of each refused turn of the inherited prefix, by turn id.
    inherited_rejections: dict[str, list[str]]


def _journal_calls(events: list[dict[str, Any]], *, inherited_native: dict[str, Any] | None) -> _JournalCalls:
    """The calls that ``events`` and the prefix ``inherited_native`` confirm.

    Raises ``ValueError`` when the prefix comes from another slot."""
    local_calls = [e["call_id"] for e in events if e.get("type") == "tool.started"]
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
    # A skipped call ran no command. closed_turns checks that the skip names the call's own turn.
    admitted.update(e["call_id"] for e in events if e.get("type") == TOOL_SKIPPED and e.get("call_id"))
    inherited_rejections: dict[str, list[str]] = {}
    if inherited_native:
        if inherited_native["slot"] != "mini-swe-agent":
            raise ValueError("Inherited native prefix uses a different slot")
        for entry in read_entries(read_prefix(inherited_native)):
            admitted.update(call["id"] for call in entry.get("message", {}).get("tool_calls") or [])
            if entry.get("type") == "mini.turn" and entry.get("rejected") is True:
                inherited_rejections[entry["turn_id"]] = entry["call_ids"]
    return _JournalCalls(
        local=local_calls,
        positions={call: i for i, call in enumerate(local_calls, 1)},
        admitted=admitted,
        rejections=rejection_events,
        rejected_turns=rejected_turns,
        inherited_rejections=inherited_rejections,
    )


def _is_rejection(row: dict[str, Any], *, group: list[str], journaled: _JournalCalls) -> bool:
    """Whether the ``mini.turn`` ``row`` closes a refused turn, by its own mark or by a recorded refusal."""
    return (
        bool(set(group) & set(journaled.rejections))
        or row.get("turn_id") in journaled.rejected_turns
        or row.get("turn_id") in journaled.inherited_rejections
        or row.get("rejected") is True
    )


def _rejection_confirmed(row: dict[str, Any], *, group: list[str], journaled: _JournalCalls) -> bool:
    """Whether ``row`` marks its turn refused, no call of ``group`` started, and a recorded refusal fits."""
    if row.get("rejected") is not True:
        return False
    tid = row.get("turn_id")
    if set(group) & set(journaled.local):
        return False
    if tid in journaled.inherited_rejections:
        return journaled.inherited_rejections[tid] == group
    return journaled.rejected_turns.get(tid, {}).get("call_ids") == group and all(
        journaled.rejections.get(call, {}).get("turn_id") == tid for call in group
    )


def reference_at(journal: str | Path, step: int, session_id: str) -> dict[str, Any] | None:
    """The native reference of the branch point at tool depth ``step`` of the session ``session_id``.

    The transcript is the one that the last ``session.ref`` of the session in ``journal`` names.
    None when the journal names no transcript for the session, or the transcript has no point at ``step``.
    """
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


def conversation_restore(journal_path: str | Path, step: int, session_ref: str) -> str | None:
    """The ``cut`` turn id of the native reference that ``reference_at`` gives for ``step``.

    None when no mini-swe-agent run started in the journal, or there is no such reference.
    """
    if not any(
        row.get("type") == "run.started" and row.get("slot") == "mini-swe-agent"
        for row in read_journal(journal_path)
    ):
        return None
    reference = reference_at(journal_path, step, session_ref)
    if not reference:
        return None
    cut: str = reference["cut"]
    return cut


def available_branch_points(journal_path: str | Path) -> dict[int, Checkpoint]:
    """Each checkpoint of ``branch_checkpoints`` by step, if its session's transcript has a cut there.

    Empty when no mini-swe-agent run started in the journal.
    """
    events = read_journal(journal_path)
    points = branch_checkpoints(journal_path, events=events)
    if not any(row.get("type") == "run.started" and row.get("slot") == "mini-swe-agent" for row in events):
        return {}
    # The last session.ref of each session wins, as in reference_at. Each transcript is indexed
    # once per call, not once per checkpoint, and nothing is cached between calls.
    references = {
        row.get("native_session_id"): row
        for row in events
        if row.get("type") == "session.ref" and row.get("transcript_path")
    }
    inherited = next(
        (row.get("inherited_native") for row in events if row.get("type") == "mini.restored"), None
    )
    cuts: dict[str | None, dict[int, str]] = {}
    for session_id in dict.fromkeys(point.session_ckpt for point in points.values()):
        reference = references.get(session_id)
        # branch_checkpoints keeps checkpoints that name their session only. The None test narrows the type.
        if reference is None or session_id is None:
            continue
        indexed = index_mini(
            journal_path, reference["transcript_path"], session_id, events=events, inherited_native=inherited
        )
        cuts[session_id] = {point.tool_depth: point.native["cut"] for point in indexed}
    return {step: point for step, point in points.items() if cuts.get(point.session_ckpt, {}).get(step)}


def actor_tools_at(journal: str | Path, step: int) -> list[dict[str, Any]]:
    """The actor's tool declaration when the model turn at ``step`` completed, checked as mini's bash tool.

    Raises ``ValueError`` when the journal has no completed turn at ``step``, no declaration before
    it, or a declaration that ``validate_mini_tools`` refuses.
    """
    tools = None
    for event in read_journal(journal):
        if event.get("type") == "rollout.model_tools" and event.get("shape") == "chat/completions":
            tools = event.get("tools")
        if event.get("type") == "model.turn.completed" and event.get("step") == step:
            if tools is None:
                raise ValueError("No recorded actor tool schema at the selected checkpoint")
            return validate_mini_tools(tools)
    raise ValueError("No completed turn for the selected actor tool schema")
