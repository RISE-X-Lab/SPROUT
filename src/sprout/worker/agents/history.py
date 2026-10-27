"""Recovery points and tool schemas for the selected native agent slot."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from sprout.contract.mini_history import NativePoint
from sprout.worker.agents.mini_swe import history as mini
from sprout.worker.checkpoint.rollback import Checkpoint, branch_checkpoints
from sprout.worker.run.journal import journal_events, read_journal


def _slot(events: list[dict[str, Any]]) -> str | None:
    slots = {e["slot"] for e in events if e.get("type") == "run.started" and e.get("slot")}
    return next(iter(slots)) if len(slots) == 1 else None


def index_native(
    journal: str | Path,
    transcript: str | Path,
    slot: str,
    session_id: str,
    *,
    events: list[dict[str, Any]] | None = None,
    inherited_native: dict[str, Any] | None = None,
) -> list[NativePoint]:
    """Index only closed native turns that have matching disk checkpoints."""
    if slot == "opencollab-base":
        from sprout.worker.agents.opencollab.history import index_native as oc_index

        return oc_index(journal, transcript, session_id, events=events, inherited_native=inherited_native)
    return mini.index_native(
        journal, transcript, slot, session_id, events=events, inherited_native=inherited_native
    )


def reference_at(journal: str | Path, step: int, session_id: str) -> dict[str, Any] | None:
    """The exact native prefix of the checkpoint at tool depth ``step``."""
    if _slot(read_journal(journal)) == "opencollab-base":
        from sprout.worker.agents.opencollab.history import reference_at as oc_reference

        return oc_reference(journal, step, session_id)
    return mini.reference_at(journal, step, session_id)


def conversation_restore(journal: str | Path, step: int, session_id: str) -> str | None:
    """The native turn id at a supported actor's exact checkpoint."""
    if _slot(read_journal(journal)) != "opencollab-base":
        return mini.conversation_restore(journal, step, session_id)
    reference = reference_at(journal, step, session_id)
    return str(reference["cut"]) if reference else None


def available_branch_points(journal: str | Path) -> dict[int, Checkpoint]:
    """Only disk checkpoints whose complete conversation and SDK state can be restored."""
    events = journal_events(journal)
    if _slot(events) != "opencollab-base":
        return mini.available_branch_points(journal)
    references = {
        e["native_session_id"]: e
        for e in events
        if e.get("type") == "session.ref" and e.get("transcript_path")
    }
    points = branch_checkpoints(journal, events=events)
    valid: dict[int, Checkpoint] = {}
    for session_id, ref in references.items():
        for point in index_native(
            journal, ref["transcript_path"], "opencollab-base", session_id, events=events
        ):
            if point.tool_depth in points:
                valid[point.tool_depth] = points[point.tool_depth]
    return valid


def actor_tools_at(journal: str | Path, step: int) -> list[dict[str, Any]]:
    """The actor's actual tool schemas at the selected checkpoint."""
    events = read_journal(journal)
    if _slot(events) != "opencollab-base":
        return mini.actor_tools_at(journal, step)
    tools = None
    for event in events:
        if event.get("type") == "rollout.model_tools" and event.get("shape") == "chat/completions":
            tools = event.get("tools")
        if event.get("type") == "model.turn.completed" and event.get("step") == step:
            if not isinstance(tools, list) or not tools:
                raise ValueError("No recorded actor tool schema at the selected checkpoint")
            return deepcopy(tools)
    raise ValueError("No completed turn for the selected actor tool schema")
