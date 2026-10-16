"""The training-message format.

It covers the hint markers, the cleanup every exported message gets, and what counts as a
truncated result.
"""

from __future__ import annotations

import re
from copy import deepcopy

HINT_START = "<sprout_training_hint>"


HINT_END = "</sprout_training_hint>"


HINT_PATTERN = re.compile(re.escape(HINT_START) + r".*?" + re.escape(HINT_END), re.DOTALL)


def mark_hint(text):
    if HINT_START in text or HINT_END in text:
        raise ValueError("Branch hint contains reserved delimiters")
    return HINT_START + text + HINT_END


def clean_messages(messages):
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
    pending, seen = set(), set()
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


def export_tools(events):
    """The Chat Completions tool schemas the model was actually offered."""
    schemas = {}
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


LIMIT_STOP_REASONS = frozenset({"max_turns_reached", "timeout"})


#: ``interrupted``: an attempt that failed after closed turns, cut back to the
#: last of them (``salvage_interrupted``).
TRUNCATION_STOP_REASONS = LIMIT_STOP_REASONS | {"interrupted"}


TERMINATION_GRACE_SECONDS = 120


def is_truncated_result(result: dict) -> bool:
    return (
        result.get("status") == "truncated"
        and result.get("stop_reason") in TRUNCATION_STOP_REASONS
        and isinstance(result.get("final_snapshot_id"), str)
        and bool(result["final_snapshot_id"])
        and failure_kind(result) not in {"configuration", "infrastructure"}
    )


def branch_origin(point: dict) -> dict:
    return {"job_id": point.get("job_id"), "point_id": point["id"], "tool_depth": point["tool_depth"]}


def failure_kind(result: dict) -> str | None:
    if result.get("status") == "completed":
        return None
    message = str(result.get("error") or "").lower()
    if any(marker in message for marker in ("checksum changed", "revision differs", "profile changed")):
        return "configuration"
    if result.get("status") in {"timeout", "step_limit", "cost_limit", "budget_exhausted"}:
        return "actor"
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
