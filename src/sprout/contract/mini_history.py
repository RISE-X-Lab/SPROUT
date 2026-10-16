"""mini-swe-agent's native history.

It covers the format, how to read a prefix of a history, and what a valid inserted turn is.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path


def validate_mini_tools(tools: list[dict]) -> list[dict]:
    """Check the interface our mini executor implements, without rewriting history."""
    if not isinstance(tools, list) or len(tools) != 1 or not isinstance(tools[0], dict):
        raise ValueError("Mini requires exactly one recorded bash tool")
    tool = tools[0]
    function = tool.get("function")
    if tool.get("type") != "function" or not isinstance(function, dict) or function.get("name") != "bash":
        raise ValueError("Recorded actor tool must be the bash function")
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or not isinstance(parameters.get("properties"), dict):
        raise ValueError("Recorded bash tool requires an object parameter schema")
    properties = parameters["properties"]
    if (
        parameters.get("type") != "object"
        or set(properties) != {"command"}
        or not isinstance(properties["command"], dict)
        or properties["command"].get("type") != "string"
        or parameters.get("required") != ["command"]
    ):
        raise ValueError("Recorded bash tool must require exactly one string command")
    return deepcopy(tools)


FORMAT = "sprout-mini-v1"


FORMATS = {FORMAT, "ash-mini-v1"}  # the second: histories written before the rename


def read_entries(data: bytes) -> list[dict]:
    return [
        json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b"\n") and line.strip()
    ]


def training_messages(entries: list[dict]) -> list[dict]:
    messages = []
    for entry in entries:
        if entry.get("type") != "mini.message":
            continue
        native = entry["message"]
        if native.get("role") == "exit":
            continue
        message = {
            key: deepcopy(native[key])
            for key in ("role", "content", "tool_calls", "tool_call_id", "reasoning_content")
            if native.get(key) is not None
        }
        message.setdefault("content", "")
        if not isinstance(message["content"], str):
            raise ValueError("mini training requires text content")
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": call["type"],
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    },
                }
                for call in message["tool_calls"]
            ]
        messages.append(message)
    return messages


def load_prefix(reference: dict) -> list[dict]:
    path = Path(reference["path"])
    with path.open("rb") as stream:
        data = stream.read(reference.get("byte_length", -1))
    if not data.endswith(b"\n") or hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("mini native prefix checksum changed")
    entries = read_entries(data)
    if not entries or entries[-1].get("type") != "mini.turn":
        raise ValueError("mini restoration requires a closed model turn")
    metadata = next((e for e in entries if e.get("type") == "mini.session"), {})
    if metadata.get("format") not in FORMATS or metadata.get("mini_version") != "2.4.6":
        raise ValueError("Unsupported mini native history version")
    return entries


#: An assistant turn written outside the actor, as its author writes it; the
#: canonical rendering (``continuation.render_reasoning``) moves its text into
#: the one key ``REASONING_TURN_KEYS`` adds.
TURN_KEYS = frozenset({"role", "content", "tool_calls"})


REASONING_TURN_KEYS = TURN_KEYS | {"reasoning_content"}


def validate_assistant_turn(
    message: dict,
    *,
    history: Sequence[dict] = (),
    tools: list[dict] | None = None,
    allow_reasoning: bool = False,
) -> dict:
    """Return an owned copy; observations and provider thinking are never supplied.

    ``allow_reasoning`` admits the canonical rendering as well: the text as
    non-blank ``reasoning_content`` beside a ``content`` of at most one line,
    usually empty. Only mini's runtime passes it, since it runs a turn
    ``continuation.prepare`` may already have rendered; the turn its author
    writes is checked in its own form, without reasoning.
    """
    if tools is not None:
        tools = validate_mini_tools(tools)
    keys = set(message) if isinstance(message, dict) else None
    if allow_reasoning and keys == REASONING_TURN_KEYS:
        if (
            message["role"] != "assistant"
            or not isinstance(message["content"], str)
            or "\n" in message["content"]
        ):
            raise ValueError(
                "assistant_turn with reasoning_content requires assistant role "
                "and content of at most one line"
            )
        reasoning = message["reasoning_content"]
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("assistant_turn reasoning_content must be non-empty text")
    elif keys != TURN_KEYS:
        raise ValueError(
            "assistant_turn requires exactly role, content and tool_calls"
            + (", or those and reasoning_content" if allow_reasoning else "")
        )
    elif (
        message["role"] != "assistant"
        or not isinstance(message["content"], str)
        or not message["content"].strip()
    ):
        raise ValueError("assistant_turn requires assistant role and non-empty ordinary content")
    calls = message["tool_calls"]
    if not isinstance(calls, list) or not calls:
        raise ValueError("assistant_turn requires at least one bash tool call")
    used = {call["id"] for entry in history for call in entry.get("tool_calls") or []}
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
            raise ValueError("assistant_turn tool calls require id, type and function")
        identifier = call["id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in used:
            raise ValueError("assistant_turn tool call ids must be non-empty and unique across history")
        used.add(identifier)
        function = call["function"]
        if (
            call["type"] != "function"
            or not isinstance(function, dict)
            or set(function) != {"name", "arguments"}
            or not isinstance(function["arguments"], str)
        ):
            raise ValueError("assistant_turn supports only bash with JSON-string arguments")
        if function["name"] != "bash":
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) has forbidden tool name "
                f"{function['name']!r}; only bash is allowed. Return function.name='bash' "
                "with arguments containing exactly one string field, command. "
                "No calls from this plan have executed."
            )
        try:
            arguments = json.loads(function["arguments"])
        except ValueError as error:
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) bash arguments are not valid JSON: {error}"
            ) from error
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"command"}
            or not isinstance(arguments["command"], str)
            or not arguments["command"].strip()
        ):
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) bash arguments require exactly one non-empty string command"
            )
        if tools is not None:
            from jsonschema import Draft202012Validator, ValidationError

            try:
                Draft202012Validator(tools[0]["function"]["parameters"]).validate(arguments)
            except ValidationError as error:
                raise ValueError(
                    f"assistant_turn does not match the recorded actor tool schema: {error.message}"
                ) from error
    return deepcopy(message)


@dataclass(frozen=True)
class NativePoint:
    message_step: int
    tool_depth: int
    snapshot_id: str
    native: dict


def read_prefix(reference: dict) -> bytes:
    length = reference["byte_length"]
    if type(length) is not int or length <= 0:
        raise ValueError("Invalid native prefix length")
    with Path(reference["path"]).open("rb") as stream:
        data = stream.read(length)
    if len(data) != length or not data.endswith(b"\n"):
        raise ValueError("Native prefix missing or truncated")
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("Native prefix checksum changed")
    return data


def valid_prefix(reference: dict) -> bool:
    try:
        read_prefix(reference)
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def materialize(reference: dict, cwd: Path, destination: Path, native_home: Path) -> dict:
    data = read_prefix(reference)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "prefix.jsonl"
    path.write_bytes(data)
    if reference["slot"] != "mini-swe-agent":
        raise ValueError("Unsupported native history slot")
    return {"native_prefix": {"path": str(path.resolve()), "sha256": reference["sha256"]}}
