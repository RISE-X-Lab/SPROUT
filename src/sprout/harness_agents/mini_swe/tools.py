"""The mini bash contract shared by actor prompts, requests and reviewers, and the feedback
an actor gets when a response breaks it.
"""

import json
import shlex
from copy import deepcopy

#: Seconds one agent command may run before the runtime kills it, unless the run
#: names another (``--command-timeout``, the mini environment's ``timeout``). The
#: model's bash tool has no timeout argument, so every command runs under this.
DEFAULT_COMMAND_TIMEOUT_S = 600


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


def mini_tool_schema(upstream: dict | None = None) -> list[dict]:
    # The pinned mini2.4.6 contract. Rendering a task must not import mini and
    # load its host-global .env before the slot establishes its isolated config.
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


def mini_tool_primer(workdir: str) -> str:
    tools = json.dumps(mini_tool_schema(), indent=2)
    example = json.dumps({"command": f"cd {shlex.quote(workdir)} && git status --short"})
    return (
        "## Your tool\n\n"
        "You have one function tool named `bash`. Its exact definition is:\n\n"
        f"```json\n{tools}\n```\n\n"
        "Arguments must contain only the string `command`. Use bash commands to "
        "inspect, edit and test files. Each call starts a fresh process; put cd "
        "and environment assignments inside the command. Execution timeouts are "
        "configured by the runner, not supplied as function arguments.\n"
        f"The default working directory is {workdir}.\n\n"
        f"Example bash arguments:\n```json\n{example}\n```\n"
    )


def malformed_tool_calls(calls: list, history_ids: set) -> list[dict]:
    """The structural faults of a response's raw tool calls, one record per faulty call.

    A fault is a call that is not an object; a ``type`` other than ``function``
    (a missing or null one is read as ``function``, the wire default, as litellm
    does); no function object; a missing or blank name; arguments that are
    neither a JSON string nor an object (litellm serialises an object, and the
    schema check then reads it); an id that is missing, blank, not a string,
    repeated in the response or already used in the conversation. The records
    are what the journal keeps and what the model is shown.
    """
    ids = [call.get("id") for call in calls if isinstance(call, dict)]
    problems = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            problems.append({"index": index, "id": None, "error": "the tool call is not an object"})
            continue
        faults = []
        if call.get("type") not in (None, "function"):
            faults.append('type must be "function"')
        function = call.get("function")
        if not isinstance(function, dict):
            faults.append("function must be an object with name and arguments")
        else:
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                faults.append("function.name is missing or blank; the only tool is bash")
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


def _head(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else "%s …[%d more chars]" % (text[:limit], len(text) - limit)


def malformed_feedback(problems: list[dict], message: dict, raw: dict) -> dict:
    """The user-role rejection of a response with malformed tool calls.

    The rejected calls are quoted as data, head-clipped, so the model sees what
    it sent beside what was wrong with it; the instruction names the shape every
    call must have. ``extra`` carries the problems for the journal and the raw
    response, as upstream does for its own format errors.
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


def tool_schema_feedback(message: dict, errors: dict[str, str], tools: list[dict]) -> list[dict]:
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
