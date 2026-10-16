"""Export native histories for hint-free training, retaining tool causality."""

import json

from sprout.contract.messages import (
    LIMIT_STOP_REASONS,
    branch_origin,
    clean_messages,
    export_tools,
    failure_kind,
    is_truncated_result,
)
from sprout.worker.run.journal import journal_events


def export_messages(directory, session_id, slot, *, allow_incomplete_tail=False):
    if not isinstance(session_id, str) or not session_id or slot != "mini-swe-agent":
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
    from sprout.contract.mini_history import training_messages

    # mini writes its actual formatted observation before closing the turn.
    # Do not substitute the runtime's different presentation from tool.finished.
    return clean_messages(training_messages(entries))


def provenance(directory, session_id: str, messages: list[dict], branch_point: dict | None = None) -> dict:
    """Which exported messages no policy sampled in this job.

    ``prefix_messages`` counts the leading messages restored from the job's
    branch point: the parent's history, not this job's samples.
    ``inserted_messages`` indexes every assistant message written outside the
    actor -- an assistant-turn branch's turn, in this job or in an ancestor it
    continues. Its tool results are real, but training on it as the policy's
    output would teach what the policy never produced.
    """
    from sprout.contract.mini_history import read_entries, read_prefix, training_messages

    files = list((directory / "native-home").glob(f"**/*{session_id}.jsonl"))
    if len(files) != 1:
        raise ValueError("Expected one native history for the attempt")
    inserted_calls = {
        call["id"]
        for entry in read_entries(files[0].read_bytes())
        if entry.get("type") == "mini.message"
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
    if branch_point:
        prefix = clean_messages(training_messages(read_entries(read_prefix(branch_point["native"]))))
        if messages[: len(prefix)] != prefix:
            raise ValueError("The exported history does not begin with the branch point's prefix")
        restored = len(prefix)
    return {"prefix_messages": restored, "inserted_messages": inserted}


def salvage_interrupted(result: dict, directory, point: dict, branch_point=None) -> dict | None:
    """An attempt that failed for good, as the truncation at its last valid recovery point.

    A turn's snapshot is taken before the next turn's first call runs, so the
    messages up to ``point`` and its snapshot are a pair the policy produced,
    whatever happened afterwards -- a crash, a lost connection, an uncertain
    command, a stop. The result is ``truncated`` with ``stop_reason:
    "interrupted"`` and graded like a timeout; what went wrong moves under
    ``interrupted``. None when the pair cannot be rebuilt.
    """
    from sprout.contract.mini_history import read_entries, read_prefix, training_messages

    try:
        events = journal_events(directory / "trajectory.jsonl")
        references = [
            e["native_session_id"]
            for e in events
            if e.get("type") == "session.ref" and e.get("native_session_id")
        ]
        session_id = result.get("native_session_id") or (references[-1] if references else None)
        messages = clean_messages(training_messages(read_entries(read_prefix(point["native"]))))
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
            "training_provenance": provenance(directory, session_id, messages, branch_point),
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
        if branch_point:
            salvaged["training_origin"] = branch_origin(branch_point)
    except (ValueError, KeyError, TypeError, OSError, IndexError):
        return None
    return salvaged if is_truncated_result(salvaged) else None


def complete_message_result(result: dict, directory, slot: str, recovery=None, branch_point=None) -> dict:
    """Attach the training export to a finished attempt's result.

    ``recovery`` is the point this attempt started from, a retry's included;
    ``branch_point`` is the one the job itself branched from, if any.
    """
    result = dict(result)
    events = journal_events(directory / "trajectory.jsonl")
    stopped_by_limit = result.get("stop_reason") in LIMIT_STOP_REASONS
    unsafe = any(event.get("type") == "execution.uncertain" for event in events)
    if (
        stopped_by_limit
        and not unsafe
        and result.get("final_snapshot_id")
        and failure_kind(result) not in {"configuration", "infrastructure"}
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
        if branch_point:
            result["training_origin"] = branch_origin(branch_point)
        usage = [e for e in events if e.get("type") == "rollout.usage"]
        result["rollout_usage"] = (
            {key: usage[-1][key] for key in ("model_calls", "tool_calls")} if usage else {}
        )
    except (ValueError, KeyError, TypeError, OSError) as error:
        result["training_export_error"] = str(error)
        if slot == "mini-swe-agent" and is_truncated_result(result) and not unsafe:
            # A timeout between two actions in one response cannot train the
            # half response against the later disk. Keep a proven earlier pair.
            from sprout.contract.mini_history import read_entries, read_prefix, training_messages
            from sprout.worker.agents.mini_swe.history import index_mini

            files = list((directory / "native-home").glob(f"**/*{session_id}.jsonl"))
            try:
                points = (
                    index_mini(
                        directory / "trajectory.jsonl",
                        files[0],
                        session_id,
                        events=events,
                        inherited_native=(recovery or {}).get("native"),
                    )
                    if len(files) == 1
                    else []
                )
                if points:
                    point = points[-1]
                    result["training_messages"] = clean_messages(
                        training_messages(read_entries(read_prefix(point.native)))
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
                    if branch_point:
                        result["training_origin"] = branch_origin(branch_point)
                    usage = [e for e in events if e.get("type") == "rollout.usage"]
                    result["rollout_usage"] = (
                        {key: usage[-1][key] for key in ("model_calls", "tool_calls")} if usage else {}
                    )
            except (ValueError, KeyError, TypeError, OSError) as fallback_error:
                result["training_export_error"] += f"; mini prefix recovery: {fallback_error}"
    if result.get("training_messages") is not None and not result.get("training_export_error"):
        try:
            result["training_provenance"] = provenance(
                directory, session_id, result["training_messages"], branch_point
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            # Messages whose authorship cannot be told apart are not training data.
            result["training_export_error"] = f"training provenance: {error}"
    return result
