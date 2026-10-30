"""Branch points: journal -> (environment snapshot, native session ref) per closed turn.

An agent's state has two halves, and a branch needs both:

- **environment**: files and installed packages. ``sprout.worker.tools.access.ToolAccess``
  takes one snapshot per model turn. The last turn's is the rollout's final disk snapshot.
- **conversation**: mini-swe-agent's native history, named by its session id and cut at the
  same closed turn (``sprout.worker.agents.mini_swe.history``).

A **branch point** is a closed turn, that is the model's whole response and each of its tool
results, whose ``checkpoint.captured`` record names a snapshot and the session. Restoring the
snapshot under that conversation prefix gives an agent whose memory agrees with its disk.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sprout.worker.run.events import CHECKPOINT_CAPTURED, TOOL_FINISHED, TOOL_SKIPPED, TOOL_STARTED
from sprout.worker.run.journal import read_journal


@dataclass(frozen=True)
class Checkpoint:
    """One ``checkpoint.captured`` record: the snapshot after tool step ``step``, and the session.

    ``reason`` is ``captured``, ``clean`` (the previous capture still holds) or ``failed``.
    ``final`` marks the rollout's final disk snapshot.
    """

    step: int
    seq: int
    snapshot_id: str | None
    session_ckpt: str | None = None
    reason: str = "captured"
    turn_id: str | None = None
    final: bool = False
    delta_empty: bool | None = None

    def is_complete(self) -> bool:
        """Whether both halves, the snapshot and the native session, are present."""
        return bool(self.snapshot_id) and bool(self.session_ckpt)


def _records(
    journal_path: str | Path | None, events: Iterable[dict[str, Any]] | None
) -> Iterable[dict[str, Any]]:
    """``events`` if given, else the records of the journal at ``journal_path``."""
    if events is not None:
        return events
    if journal_path is None:
        raise TypeError("a journal path or its events are required")
    return read_journal(journal_path)


def load_checkpoints(
    journal_path: str | Path | None, *, events: Iterable[dict[str, Any]] | None = None
) -> list[Checkpoint]:
    """Every ``checkpoint.captured`` record, in journal order.

    ``events``, if given, are the journal's records, and ``journal_path`` is not read.
    """
    out: list[Checkpoint] = []
    for record in _records(journal_path, events):
        if record.get("type") != CHECKPOINT_CAPTURED:
            continue
        out.append(
            Checkpoint(
                step=record.get("step") or 0,
                seq=record.get("seq") or 0,
                snapshot_id=record.get("snapshot_id"),
                session_ckpt=record.get("session_ckpt"),
                reason=record.get("reason") or "captured",
                turn_id=record.get("turn_id"),
                final=bool(record.get("final")),
                delta_empty=record.get("delta_empty"),
            )
        )
    return out


def closed_turns(events: Sequence[dict[str, Any]]) -> dict[str | None, dict[str, Any]]:
    """``turn_id -> {step, call_ids}`` for each closed turn, at the step of its last executed call.

    A turn is closed when one or more of its calls ran, and each call finished or ``tool.skipped`` names
    it in that turn. The map groups the conversation only. ``branch_checkpoints`` gives the turns with
    a snapshot."""
    steps = {
        e["call_id"]: e["step"]
        for e in events
        if e.get("type") == TOOL_STARTED and e.get("call_id") and type(e.get("step")) is int
    }
    finished = {e.get("call_id") for e in events if e.get("type") == TOOL_FINISHED}
    skipped = {(e.get("turn_id"), e.get("call_id")) for e in events if e.get("type") == TOOL_SKIPPED}
    closed: dict[str | None, dict[str, Any]] = {}
    for event in events:
        if event.get("type") != "model.turn.completed":
            continue
        calls = event.get("call_ids") or []
        executed = [c for c in calls if (event.get("turn_id"), c) not in skipped]
        if (
            executed
            and set(executed) <= finished
            and set(executed) <= steps.keys()
            and event.get("step") == max(steps[c] for c in executed)
        ):
            closed[event.get("turn_id")] = {"step": event["step"], "call_ids": list(calls)}
    return closed


def branch_checkpoints(
    journal_path: str | Path | None, *, events: Iterable[dict[str, Any]] | None = None
) -> dict[int, Checkpoint]:
    """``step -> checkpoint`` for each closed turn that ends at ``step`` with a complete checkpoint.

    A ``clean`` checkpoint counts only while it names the latest capture. A ``failed`` one never
    counts. The final checkpoint does not count after an ``execution.uncertain`` event.
    """
    records = list(_records(journal_path, events))
    turns = closed_turns(records)
    eligible: dict[int, Checkpoint] = {}
    latest_capture: str | None = None
    for checkpoint in load_checkpoints(journal_path, events=records):
        uncertain = False
        if checkpoint.reason == "captured":
            latest_capture = checkpoint.snapshot_id
        elif checkpoint.reason != "clean" or checkpoint.snapshot_id != latest_capture:
            continue
        if checkpoint.final:
            uncertain = any(e.get("type") == "execution.uncertain" for e in records)
        if (
            not uncertain
            and checkpoint.is_complete()
            and checkpoint.turn_id in turns
            and turns[checkpoint.turn_id]["step"] == checkpoint.step
        ):
            eligible[checkpoint.step] = checkpoint
    return eligible


def closed_in_order(
    turn: dict[str, Any], *, group: list[str], depth: int, local_calls: list[str], seen: set[str]
) -> bool:
    """Whether the journal closed ``turn`` with the calls ``group`` at the tool depth ``depth``.

    It also needs ``seen`` to hold each call of ``local_calls`` up to ``depth``, and none after it."""
    return (
        turn.get("call_ids") == group
        and turn.get("step") == depth
        and set(local_calls[:depth]) <= seen
        and not set(local_calls[depth:]) & seen
    )


def exact_snapshot(checkpoint: Checkpoint | None, *, turn_id: str, session_id: str) -> str | None:
    """The snapshot of ``checkpoint`` if it is the checkpoint of ``turn_id`` of ``session_id``, else None."""
    if (
        checkpoint
        and checkpoint.snapshot_id is not None
        and checkpoint.turn_id == turn_id
        and checkpoint.session_ckpt == session_id
    ):
        return checkpoint.snapshot_id
    return None
