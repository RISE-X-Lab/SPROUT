"""Journal -> ATIF v1.8 export.

ATIF (Agent Trajectory Interchange Format, Harbor RFC 0001) is the interchange
boundary, not our internal model: the journal keeps strictly more information
(seq, snapshot pairing, raw.* passthrough, pipeline verdicts). Conversion is
lossy by design and one-way.

Why bother: it is the only versioned trajectory format with existing tooling
(viewers, HF datasets, RL pipelines), and three of its fields map exactly onto
this stack's rollback semantics:

- ``is_copied_context``  steps replayed from a parent trajectory after a fork.
  The spec requires producers to set it and SFT consumers to filter it out --
  without it, a forked prefix appears K times in training data.
- ``continued_trajectory_ref``  resume lineage.
- ``metrics.cached_tokens``  subset of prompt_tokens (same convention as our
  Usage.cached_input_tokens).

Caveat: ATIF is single-vendor maintained. Keep it at the boundary so dropping it
costs one module.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

from sprout.harness_core import events as E
from sprout.harness_core.journal import read_journal

ATIF_VERSION = "ATIF-v1.8"


def journal_to_atif(
    records: List[dict],
    *,
    trajectory_id: Optional[str] = None,
    continued_from: Optional[str] = None,
    inherited: Optional[List[dict]] = None,
) -> dict:
    """Build an ATIF document from journal records.

    ``inherited``: records replayed from a parent trajectory (a fork's prefix).
    They are prepended and force-marked ``is_copied_context=True``, which makes
    the child document self-contained for training while still letting an SFT
    consumer filter the shared prefix instead of learning it once per branch.
    """
    inherited = inherited or []
    run_id = _first(records, "run_id") or ""
    started = _first_of_type(records, E.RUN_STARTED) or {}
    finished = _last_of_type(records, E.RUN_FINISHED) or {}
    session = _last_of_type(records, E.SESSION_REF) or {}

    steps: List[dict] = []
    pending_tools: Dict[str, dict] = {}
    current: Optional[dict] = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            steps.append(current)
            current = None

    copied_now = False

    def ensure(record: dict) -> dict:
        nonlocal current
        if current is None:
            current = {
                "step_id": len(steps) + 1,
                "timestamp": record.get("ts"),
                "source": "agent",
                "model_name": started.get("model"),
                "message": None,
                "reasoning_content": None,
                "tool_calls": [],
                "observation": None,
                "metrics": _empty_metrics(),
                "is_copied_context": copied_now,
            }
        return current

    tagged = [(r, True) for r in inherited] + [(r, False) for r in records]
    was_inherited = None
    for record, is_inherited in tagged:
        if was_inherited is True and not is_inherited:
            # The inherited prefix always ends a step. Without this flush the
            # parent's trailing partial step (its boundary rarely lands exactly
            # on a turn.completed) merges with the child's first one, and the
            # merged step inherits copied=True -- hiding the child's own work
            # from any consumer that filters copied context.
            flush()
        was_inherited = is_inherited
        copied_now = is_inherited
        etype = record.get("type")

        if etype == E.AGENT_THINKING:
            step = ensure(record)
            step["reasoning_content"] = _join(step.get("reasoning_content"), record.get("text"))

        elif etype == E.AGENT_MESSAGE:
            step = ensure(record)
            step["message"] = _join(step.get("message"), record.get("text"))

        elif etype == E.TOOL_STARTED:
            step = ensure(record)
            call = {
                "id": record.get("call_id"),
                "name": record.get("name"),
                "arguments": record.get("args") or {},
            }
            step["tool_calls"].append(call)
            pending_tools[str(record.get("call_id"))] = step

        elif etype == E.TOOL_FINISHED:
            owner = pending_tools.pop(str(record.get("call_id")), None) or ensure(record)
            owner["observation"] = _join(owner.get("observation"), record.get("output"))
            if record.get("status") == "error":
                owner.setdefault("extra", {})["tool_error"] = True

        elif etype == E.TURN_COMPLETED:
            step = ensure(record)
            step["metrics"] = _metrics(record.get("usage") or {})
            if record.get("rejected"):
                # The response was rejected before execution (a malformed or
                # schema-violating call, or none at all): no tool call or
                # observation belongs to this step, and that is not a gap.
                step.setdefault("extra", {})["rejected"] = True
            flush()

    flush()

    total = _empty_metrics()
    for step in steps:
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            total[key] += step["metrics"].get(key) or 0
        total["cost_usd"] = (total.get("cost_usd") or 0.0) + (step["metrics"].get("cost_usd") or 0.0)

    document = {
        "schema_version": ATIF_VERSION,
        "session_id": session.get("native_session_id") or run_id,
        "trajectory_id": trajectory_id or run_id,
        "agent": {
            "name": started.get("slot") or "unknown",
            "version": started.get("slot_version"),
            "model": started.get("model"),
        },
        "steps": steps or [_placeholder_step(started)],
        "final_metrics": total,
        "extra": {
            "harness": "sprout",
            "journal_schema_version": E.JOURNAL_SCHEMA_VERSION,
            "status": finished.get("status"),
            "usage": finished.get("usage"),
            "checkpoints": _checkpoints(records),
        },
    }
    if continued_from:
        document["continued_trajectory_ref"] = continued_from
    return document


def export_file(
    journal_path: Union[str, os.PathLike],
    **kwargs,
) -> dict:
    return journal_to_atif(read_journal(journal_path), **kwargs)


# --- helpers ---------------------------------------------------------------
def _empty_metrics() -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cost_usd": 0.0,
    }


def _metrics(usage: dict) -> dict:
    return {
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
        "cached_tokens": int(usage.get("cached_input_tokens") or 0),
        "cost_usd": float(usage.get("cost_usd") or 0.0),
    }


def _placeholder_step(started: dict) -> dict:
    """ATIF requires steps to be non-empty (min_length=1)."""
    return {
        "step_id": 1,
        "source": "system",
        "message": started.get("task_prompt") or "",
        "model_name": started.get("model"),
        "tool_calls": [],
        "metrics": _empty_metrics(),
        "is_copied_context": False,
    }


def _checkpoints(records: List[dict]) -> List[dict]:
    return [
        {
            "step": r.get("step"),
            "seq": r.get("seq"),
            "snapshot_id": r.get("snapshot_id"),
            "session_ckpt": r.get("session_ckpt"),
        }
        for r in records
        if r.get("type") == E.CHECKPOINT_CAPTURED
    ]


def _join(existing: Optional[str], addition: Optional[str]) -> Optional[str]:
    if not addition:
        return existing
    return addition if not existing else existing + "\n" + addition


def _first(records: List[dict], key: str):
    for record in records:
        if record.get(key):
            return record[key]
    return None


def _first_of_type(records: List[dict], etype: str) -> Optional[dict]:
    for record in records:
        if record.get("type") == etype:
            return record
    return None


def _last_of_type(records: List[dict], etype: str) -> Optional[dict]:
    found = None
    for record in records:
        if record.get("type") == etype:
            found = record
    return found
