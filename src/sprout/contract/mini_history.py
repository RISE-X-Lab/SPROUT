"""mini-swe-agent's native history: its format, prefixes of it, and valid inserted turns.

A prefix reference is a dict that names a prefix of a history file by its ``path``, its
``byte_length`` and its ``sha256``. A reference that ``materialize`` reads also names its ``slot``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def validate_mini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """An owned copy of ``tools``, checked to be the one ``bash`` tool that mini runs.

    The tool takes exactly one required string, ``command``. Raises ``ValueError`` otherwise.
    The copy keeps the recorded schema as it is.
    """
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


#: The ``format`` that a new history records in its ``mini.session`` entry.
FORMAT = "sprout-mini-v1"


#: The formats that a reader accepts. ``ash-mini-v1`` is an older name of ``FORMAT``.
FORMATS = {FORMAT, "ash-mini-v1"}


def read_entries(data: bytes) -> list[dict[str, Any]]:
    """The JSON entries in ``data``, one for each non-blank line that ends with a newline."""
    return [
        json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b"\n") and line.strip()
    ]


def training_messages(entries: list[dict[str, Any]], *, record: str = "mini.message") -> list[dict[str, Any]]:
    """The chat messages of the ``record`` entries in ``entries``, with only the keys that training reads.

    mini's ``exit`` message is left out. Raises ``ValueError`` if a message has content that is not text.
    """
    messages = []
    for entry in entries:
        if entry.get("type") != record:
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


def load_prefix(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The entries of the prefix that ``reference`` names, checked for a branch to restore.

    Without ``byte_length``, the prefix is the whole file. Raises ``ValueError`` if the checksum differs,
    the prefix does not end at a closed model turn, or the format or the mini version is not supported.
    """
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


#: The keys of an assistant turn written outside the actor, in the form that its author writes.
TURN_KEYS = frozenset({"role", "content", "tool_calls"})


#: The keys of the same turn after ``continuation.render_reasoning`` moves its text to ``reasoning_content``.
REASONING_TURN_KEYS = TURN_KEYS | {"reasoning_content"}


def validate_assistant_turn(
    message: object,
    *,
    history: Sequence[Mapping[str, Any]] = (),
    tools: list[dict[str, Any]] | None = None,
    allow_reasoning: bool = False,
) -> dict[str, Any]:
    """An owned copy of ``message`` if it is a valid assistant turn. Raises ``ValueError`` if it is not.

    It has exactly ``TURN_KEYS``, non-blank ``content`` and one or more ``bash`` calls whose ids are new to
    ``history`` and whose arguments match ``tools``, if given. ``allow_reasoning`` adds the reasoning form.
    """
    if tools is not None:
        tools = validate_mini_tools(tools)
    keys = frozenset(message) if isinstance(message, dict) else None
    reasoning_form = allow_reasoning and keys == REASONING_TURN_KEYS
    if not isinstance(message, dict) or (keys != TURN_KEYS and not reasoning_form):
        raise ValueError(
            "assistant_turn requires exactly role, content and tool_calls"
            + (", or those and reasoning_content" if allow_reasoning else "")
        )
    if reasoning_form:
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
                f"assistant_turn.tool_calls[{index}] ({identifier}) "
                f"bash arguments are not valid JSON: {error}"
            ) from error
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"command"}
            or not isinstance(arguments["command"], str)
            or not arguments["command"].strip()
        ):
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) "
                "bash arguments require exactly one non-empty string command"
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
    """A closed model turn that a branch can start from.

    ``snapshot_id`` is the sandbox snapshot at the turn. ``native`` is the prefix reference of mini's
    history up to the turn.
    """

    message_step: int
    tool_depth: int
    snapshot_id: str
    native: dict[str, Any]


def read_prefix(reference: Mapping[str, Any]) -> bytes:
    """The bytes of the prefix that ``reference`` names.

    Raises ``ValueError`` if ``byte_length`` is not a positive int, the file has fewer bytes, the prefix
    does not end with a newline or its checksum differs. Raises ``OSError`` if the file cannot be read.
    """
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


def valid_prefix(reference: Mapping[str, Any]) -> bool:
    """Whether ``read_prefix`` accepts ``reference``. A missing file or a malformed reference gives False."""
    try:
        read_prefix(reference)
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def materialize(
    reference: Mapping[str, Any], cwd: Path, destination: Path, native_home: Path
) -> dict[str, Any]:
    """Copy the prefix that ``reference`` names to ``destination/prefix.jsonl``, and name the copy.

    The result is the ``RunSpec.extra`` entry for the copy. ``destination`` must not exist. Raises
    ``ValueError`` if ``read_prefix`` refuses ``reference`` or its ``slot`` is not ``mini-swe-agent``.
    """
    data = read_prefix(reference)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "prefix.jsonl"
    path.write_bytes(data)
    if reference["slot"] != "mini-swe-agent":
        raise ValueError("Unsupported native history slot")
    return {"native_prefix": {"path": str(path.resolve()), "sha256": reference["sha256"]}}
