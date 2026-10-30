"""The training export of a mini-swe-agent or sprout-harness attempt, read from its native history.

The export holds the messages without branch hints, the tools that the model was
offered, and the provenance of each message that no policy sampled in the attempt.
Each tool result stays with the call that produced it. The result fields:
``docs/RUNSTORE.md`` and ``docs/RL_DRIVER.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sprout.contract.harness_history import MESSAGE_RECORDS, training_messages
from sprout.contract.messages import (
    INVALID_STOP_REASONS,
    LIMIT_STOP_REASONS,
    branch_origin,
    clean_messages,
    export_tools,
    failure_kind,
    is_truncated_result,
)
from sprout.worker.run.journal import journal_events


def export_messages(
    directory: Path, session_id: str | None, slot: str, *, allow_incomplete_tail: bool = False
) -> list[dict[str, Any]]:
    """The training messages of the native history of ``session_id`` under ``directory``.

    Raises ValueError without a session id, for a slot without a native history, or unless exactly
    one history exists. ``allow_incomplete_tail`` drops an unfinished last record.
    """
    if not isinstance(session_id, str) or not session_id or slot not in MESSAGE_RECORDS:
        raise ValueError("Missing or unsupported native session identity")
    files = list((directory / "native-home").glob(f"**/*{session_id}.jsonl"))
    if len(files) != 1:
        raise ValueError("Expected one native history for the attempt")
    data = files[0].read_bytes()
    if not data.endswith(b"\n"):
        if not allow_incomplete_tail:
            raise ValueError("Native history has an incomplete final record")
        data = data[: data.rfind(b"\n") + 1]
    entries = [json.loads(line) for line in data.splitlines() if line.strip()]
    # The slot writes its formatted observation before it closes the turn. The runtime's own
    # presentation in ``tool.finished`` differs, and the export does not use it.
    return clean_messages(training_messages(entries, slot=slot))


def provenance(
    directory: Path,
    session_id: str,
    messages: list[dict[str, Any]],
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Which of the exported ``messages`` no policy sampled in this job.

    ``prefix_messages`` counts the leading messages restored from ``recovery``. ``inserted_messages``
    indexes the assistant messages that a reviewer wrote, in this job or in an ancestor that it continues.
    """
    from sprout.contract.mini_history import read_entries, read_prefix

    files = list((directory / "native-home").glob(f"**/*{session_id}.jsonl"))
    if len(files) != 1:
        raise ValueError("Expected one native history for the attempt")
    inserted_calls = {
        call["id"]
        for entry in read_entries(files[0].read_bytes())
        if entry.get("type") in MESSAGE_RECORDS.values()
        and (entry["message"].get("extra") or {}).get("source") == "reviewer"
        for call in entry["message"].get("tool_calls") or []
    }
    inserted = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
        and {call["id"] for call in message.get("tool_calls") or []} & inserted_calls
    ]
    restored = 0
    if recovery:
        native = recovery["native"]
        prefix = clean_messages(
            training_messages(read_entries(read_prefix(native)), slot=native.get("slot", "mini-swe-agent"))
        )
        if messages[: len(prefix)] != prefix:
            raise ValueError("The exported history does not begin with the branch point's prefix")
        restored = len(prefix)
    return {"prefix_messages": restored, "inserted_messages": inserted}


def salvage_interrupted(
    result: dict[str, Any],
    directory: Path,
    point: dict[str, Any],
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The attempt ``result`` that failed for good, cut back to its last valid recovery ``point``.

    The attempt started from the point ``recovery``, if any. The salvaged result is ``truncated``, with
    ``stop_reason: "interrupted"`` and the failure under ``interrupted``. None if ``point`` cannot be rebuilt.
    """
    from sprout.contract.mini_history import read_entries, read_prefix

    try:
        events = journal_events(directory / "trajectory.jsonl")
        references = [
            e["native_session_id"]
            for e in events
            if e.get("type") == "session.ref" and e.get("native_session_id")
        ]
        session_id = result.get("native_session_id") or (references[-1] if references else None)
        if not isinstance(session_id, str):
            return None
        native = point["native"]
        messages = clean_messages(
            training_messages(read_entries(read_prefix(native)), slot=native.get("slot", "mini-swe-agent"))
        )
        if not any(message.get("role") == "assistant" for message in messages):
            return None
        usage = [e for e in events if e.get("type") == "rollout.usage"]
        salvaged = {
            **{key: value for key, value in result.items() if key not in {"error", "training_export_error"}},
            "status": "truncated",
            "stop_reason": "interrupted",
            "failure_kind": None,
            "final_snapshot_id": point["snapshot_id"],
            "native_session_id": session_id,
            "training_messages": messages,
            "training_tools": export_tools(events),
            "training_provenance": provenance(directory, session_id, messages, recovery),
            "rollout_usage": {key: usage[-1][key] for key in ("model_calls", "tool_calls")} if usage else {},
            "interrupted": {
                "status": result.get("status"),
                "error": result.get("error"),
                "failure_kind": result.get("failure_kind"),
                "point_id": point["id"],
                "tool_depth": point["tool_depth"],
                "message_step": point["message_step"],
            },
        }
        if recovery:
            salvaged["training_origin"] = branch_origin(recovery)
    except (ValueError, KeyError, TypeError, OSError, IndexError):
        return None
    return salvaged if is_truncated_result(salvaged) else None


def complete_message_result(
    result: dict[str, Any],
    directory: Path,
    slot: str,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A copy of the finished attempt's ``result`` with its training export.

    ``recovery`` is the point that the attempt started from, or None. A failed export sets
    ``training_export_error``.
    """
    result = dict(result)
    events = journal_events(directory / "trajectory.jsonl")
    # A spent budget, or an invalid attempt (``INVALID_STOP_REASONS``), ends at a known turn.
    stopped_by_limit = result.get("stop_reason") in LIMIT_STOP_REASONS | INVALID_STOP_REASONS
    unsafe = any(event.get("type") == "execution.uncertain" for event in events)
    if (
        stopped_by_limit
        and not unsafe
        and result.get("final_snapshot_id")
        and failure_kind(result) != "infrastructure"
    ):
        result.update(agent_status=result["status"], status="truncated", failure_kind=None)
    references = [
        e["native_session_id"]
        for e in events
        if e.get("type") == "session.ref" and e.get("native_session_id")
    ]
    session_id = result.get("native_session_id") or (references[-1] if references else None)
    result["native_session_id"] = session_id
    try:
        result["training_messages"] = export_messages(
            directory, session_id, slot, allow_incomplete_tail=is_truncated_result(result)
        )
        result["training_tools"] = export_tools(events)
        if recovery:
            result["training_origin"] = branch_origin(recovery)
        usage = [e for e in events if e.get("type") == "rollout.usage"]
        result["rollout_usage"] = (
            {key: usage[-1][key] for key in ("model_calls", "tool_calls")} if usage else {}
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        result["training_export_error"] = str(error)
        if slot in MESSAGE_RECORDS and is_truncated_result(result) and not unsafe:
            # After a timeout between two actions of one response, or during one call, part of the
            # response does not match the disk. Fall back to the last messages and snapshot that are proven.
            from sprout.contract.mini_history import read_entries, read_prefix
            from sprout.worker.agents.mini_swe.history import index_native

            files = list((directory / "native-home").glob(f"**/*{session_id}.jsonl"))
            try:
                points = (
                    index_native(
                        directory / "trajectory.jsonl",
                        files[0],
                        slot,
                        session_id,
                        events=events,
                        inherited_native=(recovery or {}).get("native"),
                    )
                    if len(files) == 1 and isinstance(session_id, str)
                    else []
                )
                if points:
                    point = points[-1]
                    result["training_messages"] = clean_messages(
                        training_messages(read_entries(read_prefix(point.native)), slot=slot)
                    )
                    result["training_tools"] = export_tools(events)
                    result["timeout_fallback"] = {
                        "tool_depth": point.tool_depth,
                        "message_step": point.message_step,
                        "native_prefix": point.native,
                        "discarded_final_snapshot_id": result["final_snapshot_id"],
                    }
                    result["final_snapshot_id"] = point.snapshot_id
                    result.pop("training_export_error", None)
                    if recovery:
                        result["training_origin"] = branch_origin(recovery)
                    usage = [e for e in events if e.get("type") == "rollout.usage"]
                    result["rollout_usage"] = (
                        {key: usage[-1][key] for key in ("model_calls", "tool_calls")} if usage else {}
                    )
            except (ValueError, KeyError, TypeError, OSError) as fallback_error:
                result["training_export_error"] += f"; native prefix recovery: {fallback_error}"
    if (
        isinstance(session_id, str)
        and result.get("training_messages") is not None
        and not result.get("training_export_error")
    ):
        try:
            result["training_provenance"] = provenance(
                directory, session_id, result["training_messages"], recovery
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            # Messages whose authorship cannot be told apart are not training data.
            result["training_export_error"] = f"training provenance: {error}"
    return result
