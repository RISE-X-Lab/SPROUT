"""The mini bash contract shared by requests and reviewers, and the feedback an actor gets
when a response breaks it.
"""

import json
from typing import Any

from sprout.contract.mini_history import validate_mini_tools

#: Seconds an agent command runs before the runtime kills it, unless the run sets the mini
#: environment's ``timeout`` (``--command-timeout``). The bash tool has no timeout argument.
DEFAULT_COMMAND_TIMEOUT_S = 600


def mini_tool_schema(upstream: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The pinned mini 2.4.6 bash tool, as a chat/completions ``tools`` list.

    Raises ``ValueError`` if ``upstream``, the installed mini's tool, differs from it in more
    than ``additionalProperties``.
    """
    # The pinned mini 2.4.6 contract, written out: importing mini loads its host-global .env,
    # which must wait until the slot has set up its isolated config.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute"}
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    if upstream is not None:
        actual = validate_mini_tools([upstream])
        # Our executor already rejects extra arguments. Advertise that limit.
        actual[0]["function"]["parameters"]["additionalProperties"] = False
        if actual != tools:
            raise ValueError("Installed mini tool definition differs from the pinned actor contract")
    return tools


def malformed_tool_calls(calls: list[Any], history_ids: set[str]) -> list[dict[str, Any]]:
    """The structural faults of a response's raw tool calls, one ``{index, id, error}`` per faulty call.

    A call is faulty if it or its ``function`` is not an object, or if its ``type``, name, arguments
    or ``id`` is wrong. An ``id`` must be unique in the response and absent from ``history_ids``.
    """
    ids = [call.get("id") for call in calls if isinstance(call, dict)]
    problems: list[dict[str, Any]] = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            problems.append({"index": index, "id": None, "error": "the tool call is not an object"})
            continue
        faults = []
        # A missing or null type is the wire default, function, as litellm reads it.
        if call.get("type") not in (None, "function"):
            faults.append('type must be "function"')
        function = call.get("function")
        if not isinstance(function, dict):
            faults.append("function must be an object with name and arguments")
        else:
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                faults.append("function.name is missing or blank; the only tool is bash")
            # litellm serialises object arguments, and the schema check then reads the result.
            if not isinstance(function.get("arguments"), (str, dict)):
                faults.append("function.arguments must be a JSON-encoded string")
        identifier = call.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            faults.append("id is missing or blank")
        elif ids.count(identifier) > 1:
            faults.append("id repeats another call's id in this response")
        elif identifier in history_ids:
            faults.append("id was already used earlier in this conversation")
        if faults:
            problems.append(
                {
                    "index": index,
                    "id": identifier if isinstance(identifier, str) else None,
                    "error": "; ".join(faults),
                }
            )
    return problems


def _head(value: object, limit: int) -> str:
    """``value`` as text, cut to ``limit`` characters and a count of the rest."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else "%s …[%d more chars]" % (text[:limit], len(text) - limit)


def malformed_feedback(
    problems: list[dict[str, Any]], message: dict[str, Any], raw: dict[str, Any]
) -> dict[str, Any]:
    """The user-role rejection of a response with malformed tool calls.

    It quotes each rejected call, cut short, beside its fault, and names the shape a call must have.
    ``extra`` carries the problems and the raw response, as upstream does for its format errors.
    """
    calls = message.get("tool_calls") or []
    rejected = []
    for problem in problems:
        call = calls[problem["index"]]
        function = call.get("function") if isinstance(call, dict) else None
        function = function if isinstance(function, dict) else {}
        rejected.append(
            {**problem, "name": function.get("name"), "arguments": _head(function.get("arguments"), 2000)}
        )
    content = (
        "Tool call error: the previous response was rejected before execution because "
        "%d of its %d tool call(s) were malformed. None of them ran, and the response is "
        "not part of the conversation.\n\n<rejected_response>\n%s\n</rejected_response>\n\n"
        "Resend the response as one assistant message: your reasoning as text, then "
        "tool_calls in which every call has a unique non-empty string id not used earlier "
        'in this conversation, type "function", function.name "bash" and function.arguments '
        "as a JSON-encoded string with the single field command, for example "
        '{"command": "ls -la"}.'
        % (
            len(problems),
            len(calls),
            json.dumps(
                {"content": _head(message.get("content") or "", 1000), "rejected_tool_calls": rejected},
                ensure_ascii=False,
                indent=1,
            ),
        )
    )
    return {
        "role": "user",
        "content": content,
        "extra": {"interrupt_type": "FormatError", "malformed_tool_calls": problems, "response": raw},
    }


def tool_schema_feedback(
    message: dict[str, Any], errors: dict[str, str], tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One tool-role error for each call of ``message``. ``errors`` maps a call id to its schema fault.

    A call without a fault gets ``tool_batch_rejected``, because no call of the response runs.
    """
    observations = []
    for call in message["tool_calls"]:
        identifier = call["id"]
        error = {
            "type": "tool_schema_error" if identifier in errors else "tool_batch_rejected",
            "message": errors.get(identifier, "Another call in this response failed schema validation."),
        }
        content = json.dumps(
            {
                "error": error,
                "executed": False,
                "instruction": "No calls in this response were executed. Correct and resend the tool calls.",
                "available_tools": tools,
            },
            ensure_ascii=False,
        )
        observations.append({"role": "tool", "tool_call_id": identifier, "content": content})
    return observations
