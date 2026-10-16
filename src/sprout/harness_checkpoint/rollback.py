"""Branch points: journal -> (environment snapshot, native session ref) per closed turn.

An agent's state has two halves and neither alone is enough to branch a run:

- **environment**: files, installed packages. One snapshot per model turn,
  taken by ``harness.tools.access.ToolAccess``; the last turn's is the rollout's
  final disk snapshot.
- **conversation**: mini-swe-agent's native history, addressed by its session
  id and cut at the same closed turn (``harness/agents/mini_swe/history.py``).

A **branch point** is a closed turn -- the model's whole response and every one
of its tool results -- whose ``checkpoint.captured`` record names a snapshot and
the session. Restoring the snapshot under that conversation prefix gives an
agent whose memory agrees with its disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from sprout.harness.core.events import CHECKPOINT_CAPTURED
from sprout.harness.core.journal import read_journal


@dataclass(frozen=True)
class Checkpoint:
    step: int
    seq: int
    snapshot_id: Optional[str]
    session_ckpt: Optional[str] = None
    reason: str = "captured"
    turn_id: Optional[str] = None
    final: bool = False
    delta_empty: Optional[bool] = None

    def is_complete(self) -> bool:
        """Both halves present."""
        return bool(self.snapshot_id) and bool(self.session_ckpt)


def load_checkpoints(journal_path, *, events=None) -> List[Checkpoint]:
    """Every ``checkpoint.captured`` record, in journal order."""
    out: List[Checkpoint] = []
    for record in read_journal(journal_path) if events is None else events:
        if record.get("type") != CHECKPOINT_CAPTURED:
            continue
        out.append(Checkpoint(
            step=record.get("step") or 0, seq=record.get("seq") or 0,
            snapshot_id=record.get("snapshot_id"), session_ckpt=record.get("session_ckpt"),
            reason=record.get("reason") or "captured", turn_id=record.get("turn_id"),
            final=bool(record.get("final")), delta_empty=record.get("delta_empty"),
        ))
    return out


def closed_turns(events) -> Dict[str, dict]:
    """turn_id -> {step, call_ids} for turns whose every call finished; ``step`` is
    the turn's last call.

    Conversation grouping only; snapshot availability is ``branch_checkpoints``.
    """
    steps = {e["call_id"]: e["step"] for e in events
             if e.get("type") == "tool.started" and e.get("call_id") and type(e.get("step")) is int}
    finished = {e.get("call_id") for e in events if e.get("type") == "tool.finished"}
    closed = {}
    for event in events:
        if event.get("type") != "model.turn.completed":
            continue
        calls = event.get("call_ids") or []
        if (calls and set(calls) <= finished and set(calls) <= steps.keys()
                and event.get("step") == max(steps[c] for c in calls)):
            closed[event.get("turn_id")] = {"step": event["step"], "call_ids": list(calls)}
    return closed


def branch_checkpoints(journal_path, *, events=None) -> Dict[int, Checkpoint]:
    """step -> the checkpoint of the closed turn that ends there."""
    events = list(read_journal(journal_path) if events is None else events)
    turns = closed_turns(events)
    eligible: Dict[int, Checkpoint] = {}
    latest_capture = None
    for checkpoint in load_checkpoints(journal_path, events=events):
        uncertain = False
        if checkpoint.reason == "captured":
            latest_capture = checkpoint.snapshot_id
        elif checkpoint.reason != "clean" or checkpoint.snapshot_id != latest_capture:
            continue
        if checkpoint.final:
            uncertain = any(e.get("type") == "execution.uncertain" for e in events)
        if (not uncertain and checkpoint.is_complete() and checkpoint.turn_id in turns
                and turns[checkpoint.turn_id]["step"] == checkpoint.step):
            eligible[checkpoint.step] = checkpoint
    return eligible


def uncaptured_turns(journal_path) -> List[str]:
    """Closed turns with no snapshot at all: checkpointing silently stopped."""
    events = list(read_journal(journal_path))
    covered = {c.turn_id for c in load_checkpoints(journal_path, events=events) if c.snapshot_id}
    return sorted(t for t in closed_turns(events) if t not in covered)

