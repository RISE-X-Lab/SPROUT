"""The export of a journal as an ATIF v1.8 trajectory document.

ATIF (Agent Trajectory Interchange Format, Harbor RFC 0001) is the export format, not the
internal model. The journal holds more, such as ``seq``, the snapshot of each turn and the
``raw.*`` events. So the export loses information, and there is no import.

Three ATIF fields carry the branching of this repository:

- ``is_copied_context`` marks the steps that a fork copied from its parent trajectory. An
  SFT consumer drops them, so it does not learn a shared prefix once for each branch.
- ``continued_trajectory_ref`` names the trajectory that a resumed one continues.
- ``metrics.cached_tokens`` is a subset of ``prompt_tokens``, like
  ``Usage.cached_input_tokens``.

ATIF is the only versioned trajectory format with existing tools, such as viewers, Hugging Face
datasets and RL pipelines. One vendor maintains it, so it stays in this one module.
"""

from __future__ import annotations

import os
from typing import Any

from sprout.worker.run import events as E
from sprout.worker.run.journal import read_journal

ATIF_VERSION = "ATIF-v1.8"


def journal_to_atif(
    records: list[dict[str, Any]],
    *,
    trajectory_id: str | None = None,
    continued_from: str | None = None,
    inherited: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The ATIF document of the journal ``records``, after the records ``inherited`` of a fork's parent.

    Each step of ``inherited`` has ``is_copied_context`` true, so the document stands alone and
    a consumer can still drop the shared prefix. ``continued_from`` is ``continued_trajectory_ref``.
    """
    run_id = _first(records, "run_id") or ""
    started = _first_of_type(records, E.RUN_STARTED) or {}
    finished = _last_of_type(records, E.RUN_FINISHED) or {}
    session = _last_of_type(records, E.SESSION_REF) or {}
    steps = _steps(records, inherited=inherited or [], model_name=started.get("model"))
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
        "final_metrics": _total_metrics(steps),
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


def _steps(
    records: list[dict[str, Any]], *, inherited: list[dict[str, Any]], model_name: Any
) -> list[dict[str, Any]]:
    """The ATIF steps of ``inherited`` and then ``records``, one for each model turn.

    A step of ``inherited`` has ``is_copied_context`` true. ``model_name`` is the model of each step."""
    steps: list[dict[str, Any]] = []
    pending_tools: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            steps.append(current)
            current = None

    copied_now = False

    def ensure(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal current
        if current is None:
            current = {
                "step_id": len(steps) + 1,
                "timestamp": record.get("ts"),
                "source": "agent",
                "model_name": model_name,
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
            # The inherited prefix ends a step. Otherwise its partial last step takes in the
            # child's first one, and a consumer that drops copied steps loses the child's work.
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
                # The response was rejected before execution. Its step has no tool call and
                # no observation, and that is not a gap.
                step.setdefault("extra", {})["rejected"] = True
            flush()

    flush()
    return steps


def _total_metrics(steps: list[dict[str, Any]]) -> dict[str, int | float]:
    """The sums of the metrics of ``steps``."""
    total = _empty_metrics()
    for step in steps:
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            total[key] += step["metrics"].get(key) or 0
        total["cost_usd"] = (total.get("cost_usd") or 0.0) + (step["metrics"].get("cost_usd") or 0.0)
    return total


def export_file(
    journal_path: str | os.PathLike[str],
    *,
    trajectory_id: str | None = None,
    continued_from: str | None = None,
    inherited: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The ATIF document of the journal at ``journal_path``, as :func:`journal_to_atif` makes it."""
    return journal_to_atif(
        read_journal(journal_path),
        trajectory_id=trajectory_id,
        continued_from=continued_from,
        inherited=inherited,
    )


# --- helpers ---------------------------------------------------------------
def _empty_metrics() -> dict[str, int | float]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cost_usd": 0.0,
    }


def _metrics(usage: dict[str, Any]) -> dict[str, int | float]:
    return {
        "prompt_tokens": int(usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("output_tokens") or 0),
        "cached_tokens": int(usage.get("cached_input_tokens") or 0),
        "cost_usd": float(usage.get("cost_usd") or 0.0),
    }


def _placeholder_step(started: dict[str, Any]) -> dict[str, Any]:
    """The one step of a document whose journal has none: ATIF requires at least one step."""
    return {
        "step_id": 1,
        "source": "system",
        "message": started.get("task_prompt") or "",
        "model_name": started.get("model"),
        "tool_calls": [],
        "metrics": _empty_metrics(),
        "is_copied_context": False,
    }


def _checkpoints(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _join(existing: str | None, addition: str | None) -> str | None:
    if not addition:
        return existing
    return addition if not existing else existing + "\n" + addition


def _first(records: list[dict[str, Any]], key: str) -> Any:
    """The first truthy value of ``key`` in ``records``, or None. Record values are untyped JSON."""
    for record in records:
        if record.get(key):
            return record[key]
    return None


def _first_of_type(records: list[dict[str, Any]], etype: str) -> dict[str, Any] | None:
    for record in records:
        if record.get("type") == etype:
            return record
    return None


def _last_of_type(records: list[dict[str, Any]], etype: str) -> dict[str, Any] | None:
    found = None
    for record in records:
        if record.get("type") == etype:
            found = record
    return found
