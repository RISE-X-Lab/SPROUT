"""The training-message format of an exported attempt.

It covers the hint markers, the cleanup that each exported history gets, the kind of a
failure, and what counts as a truncated result.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

#: The markers around a hint in a system or user message. ``clean_messages`` removes the
#: hint and its markers, so the actor reads the hint but training does not.
HINT_START = "<sprout_training_hint>"


HINT_END = "</sprout_training_hint>"


HINT_PATTERN = re.compile(re.escape(HINT_START) + r".*?" + re.escape(HINT_END), re.DOTALL)


def mark_hint(text: str) -> str:
    """``text`` between the hint markers. Raises ``ValueError`` if ``text`` contains a marker."""
    if HINT_START in text or HINT_END in text:
        raise ValueError("Branch hint contains reserved delimiters")
    return HINT_START + text + HINT_END


def clean_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The training copy of ``messages``, without hints in system and user messages.

    ``developer`` becomes ``system``, and a message that held only hints is dropped. Raises ``ValueError``
    for an unknown role, non-text content, a partial marker, an unmatched tool call or no assistant message.
    """
    cleaned = []
    for original in messages:
        message = deepcopy(original)
        role = message.get("role")
        if role == "developer":
            message["role"] = role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role {role!r}")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("Training export requires text content")
        if role in {"system", "user"}:
            content = HINT_PATTERN.sub("", content)
            if HINT_START in content or HINT_END in content:
                raise ValueError("Incomplete hint boundary")
            if content != message.get("content", "") and not content.strip():
                continue
            message["content"] = content
        cleaned.append(message)
    pending: set[str] = set()
    seen: set[str] = set()
    for message in cleaned:
        for call in message.get("tool_calls") or []:
            if call["id"] in seen:
                raise ValueError("Duplicate tool call")
            seen.add(call["id"])
            pending.add(call["id"])
        if message["role"] == "tool":
            if message.get("tool_call_id") not in pending:
                raise ValueError("Unmatched tool result")
            pending.remove(message["tool_call_id"])
    if pending:
        raise ValueError("Native history has incomplete tool calls")
    if not any(m["role"] == "assistant" for m in cleaned):
        raise ValueError("Native history has no assistant output")
    return cleaned


def export_tools(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The Chat Completions tool schemas that the model was offered, one for each tool name.

    Raises ``ValueError`` if a schema is not a Chat Completions function or changed during the trajectory.
    """
    schemas: dict[str, Any] = {}
    for event in events:
        if event.get("type") != "rollout.model_tools":
            continue
        for tool in event.get("tools", []):
            if event.get("shape") != "chat/completions" or tool.get("type") != "function":
                raise ValueError("Unsupported native tool schema for training")
            item = deepcopy(tool["function"])
            name = item["name"]
            if name in schemas and schemas[name] != item:
                raise ValueError("Tool schema changed during the trajectory")
            schemas[name] = item
    return [{"type": "function", "function": schema} for schema in schemas.values()]


#: The stop reasons of an attempt that its turn limit or its wall-time budget ended.
LIMIT_STOP_REASONS = frozenset({"max_turns_reached", "timeout"})


#: The stop reasons of a truncated result. ``interrupted`` marks a failed attempt that
#: ``sprout.worker.jobs.export.salvage_interrupted`` cut back to its last closed turn.
TRUNCATION_STOP_REASONS = LIMIT_STOP_REASONS | {"interrupted"}


#: Seconds that the worker adds to the process timeout of an attempt that exports training messages.
#: The rollout stops itself at its wall-time budget, and then writes the export in this time.
TERMINATION_GRACE_SECONDS = 120


def is_truncated_result(result: Mapping[str, Any]) -> bool:
    """Whether ``result`` is a truncated attempt with a verified cutoff.

    It has the status ``truncated``, a stop reason in ``TRUNCATION_STOP_REASONS``, a final snapshot id,
    and a failure kind that is not ``infrastructure``.
    """
    return (
        result.get("status") == "truncated"
        and result.get("stop_reason") in TRUNCATION_STOP_REASONS
        and isinstance(result.get("final_snapshot_id"), str)
        and bool(result["final_snapshot_id"])
        and failure_kind(result) != "infrastructure"
    )


def branch_origin(point: Mapping[str, Any]) -> dict[str, Any]:
    """The ``training_origin`` of a branch from ``point``: its job id, point id and tool depth."""
    return {"job_id": point.get("job_id"), "point_id": point["id"], "tool_depth": point["tool_depth"]}


def failure_kind(result: Mapping[str, Any]) -> str | None:
    """The kind of failure in ``result``: ``actor`` or ``infrastructure``.

    None if ``result`` completed. The status and markers in ``error`` decide first. Otherwise the
    result's own ``failure_kind`` decides, and the default is ``actor``.
    """
    if result.get("status") == "completed":
        return None
    message = str(result.get("error") or "").lower()
    if result.get("status") in {"timeout", "step_limit", "cost_limit", "budget_exhausted"}:
        return "actor"
    # Each marker is part of an error text that another module or service writes.
    if any(marker in message for marker in ("budget exhausted", "cost limit", "step limit", "maximum turns")):
        return "actor"
    if re.search(r"\b(?:502|503|504)\b", message) or any(
        marker in message
        for marker in (
            "transportclosederror",
            "connection reset",
            "connection refused",
            "stream stalled",
            "execution uncertain",
            "tool execution did not settle",
            "final snapshot failed",
            "readtimeout",
        )
    ):
        return "infrastructure"
    return result.get("failure_kind") or "actor"
