"""OpenCollab native prefixes and reviewer turns, without importing the SDK."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

SLOT = "opencollab-base"
FORMAT = "sprout-opencollab-v1"


def training_messages(rows: Any) -> list[dict[str, Any]]:
    messages = []
    for row in rows:
        if row.get("type") == "oc.context":
            messages = deepcopy(row["messages"])
        elif row.get("type") == "oc.message":
            messages.append(deepcopy(row["message"]))
    return messages


def load_prefix(reference: Any) -> list[dict[str, Any]]:
    with Path(reference["path"]).open("rb") as stream:
        data = stream.read(reference.get("byte_length", -1))
    if not data.endswith(b"\n") or hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("OpenCollab native prefix checksum changed")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if (
        not rows
        or rows[0].get("format") != FORMAT
        or rows[0].get("profile") != "single2"
        or rows[-1].get("type") != "oc.turn"
    ):
        raise ValueError("OpenCollab restore requires a closed Single2 turn")
    return rows


def validate_assistant_turn(
    message: object, *, history: Any = (), tools: list[dict[str, Any]], allow_reasoning: bool = False
) -> dict[str, Any]:
    """Validate new reviewer calls against the recorded schemas, preserving mini's separate contract."""
    from jsonschema import Draft202012Validator, ValidationError

    fields = {"role", "content", "tool_calls"}
    if not isinstance(message, dict) or set(message) not in (
        fields,
        fields | {"reasoning_content"} if allow_reasoning else fields,
    ):
        raise ValueError("assistant_turn requires content and tool calls")
    content = message["content"]
    reasoning = message.get("reasoning_content")
    if message["role"] != "assistant" or not isinstance(content, str):
        raise ValueError("assistant_turn requires assistant role and text content")
    if reasoning is not None and (
        not allow_reasoning or not isinstance(reasoning, str) or not reasoning.strip() or "\n" in content
    ):
        raise ValueError("Invalid reviewer reasoning content")
    if not content.strip() and not reasoning:
        raise ValueError("assistant_turn requires non-empty content or reasoning")
    calls = message["tool_calls"]
    if not isinstance(calls, list) or not calls:
        raise ValueError("assistant_turn requires tool calls")
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
    used = {call["id"] for row in history for call in row.get("tool_calls") or []}
    for call in calls:
        if (
            not isinstance(call, dict)
            or set(call) != {"id", "type", "function"}
            or call["type"] != "function"
        ):
            raise ValueError("Invalid reviewer tool call")
        if not isinstance(call["id"], str) or not call["id"].strip() or call["id"] in used:
            raise ValueError("Reviewer tool call ids must be unique across history")
        used.add(call["id"])
        function = call["function"]
        if (
            not isinstance(function, dict)
            or set(function) != {"name", "arguments"}
            or function["name"] not in schemas
        ):
            raise ValueError("Reviewer call names an unavailable Single2 tool")
        if not isinstance(function["arguments"], str):
            raise ValueError("Reviewer tool arguments must be a JSON string")
        arguments = json.loads(function["arguments"])
        schema = deepcopy(schemas[function["name"]])
        schema["additionalProperties"] = False
        try:
            Draft202012Validator(schema).validate(arguments)
        except ValidationError as error:
            raise ValueError(
                "Reviewer arguments violate the recorded tool schema: " + error.message
            ) from error
    return cast(dict[str, Any], deepcopy(message))
