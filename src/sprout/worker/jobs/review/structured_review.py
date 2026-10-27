"""The flat branch format that a reviewer writes, its JSON schemas, and its conversion to tool calls.

A flat branch carries ``content`` (the assistant's text) and ``commands`` (its bash commands, in
execution order). ``extract_assistant_turn_plan`` builds the ``assistant_turn`` from them, with call
ids of the form ``CALL_ID``, so the actor receives an ordinary assistant message. A branch in the
nested ``assistant_turn`` form is also accepted, so a recorded plan replays.
Why the format is flat: the Review section of ``docs/RUNSTORE.md``.
"""

import json
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from sprout.worker.jobs.review.branch_plan import extract_branch_plan


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


STRING = {"type": "string"}
COMMAND = _object({"command": STRING})
FUNCTION = _object({"name": {"type": "string", "enum": ["bash"]}, "arguments": COMMAND})
TOOL_CALL = _object({"id": STRING, "type": {"type": "string", "enum": ["function"]}, "function": FUNCTION})
ASSISTANT_TURN = _object(
    {
        "role": {"type": "string", "enum": ["assistant"]},
        "content": STRING,
        "tool_calls": {"type": "array", "items": TOOL_CALL},
    }
)
POINT = {"name": STRING, "base": STRING, "branch_step": {"type": "integer"}, "why": STRING}
#: What the reviewer writes: the continuation's text and its bash commands.
FLAT_BRANCH = _object(
    {**POINT, "content": STRING, "commands": {"type": "array", "minItems": 1, "items": STRING}}
)
#: mini ends the run when a command's output starts with this line. An inserted
#: turn that prints it leaves the policy nothing to sample after it.
SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
#: The wire form, which a flat branch is converted to. It is also accepted as input.
NESTED_BRANCH = _object({**POINT, "assistant_turn": ASSISTANT_TURN})
#: The outer envelope. Each branch is then checked against the form that its keys name.
REVIEW_SCHEMA = _object(
    {
        "synthesis": STRING,
        "branches": {"type": "array", "items": {"type": "object"}},
    }
)
#: The analyst's report on one failed attempt.
ANALYST_SCHEMA = _object(
    {
        "failure_reason": STRING,
        "lesson": STRING,
        "salvage": STRING,
        "branch_candidates": {
            "type": "array",
            "items": _object(
                {
                    "step": {"type": "integer"},
                    "why": STRING,
                }
            ),
        },
    }
)


def flat_review_schema(width: int, count_mode: str) -> dict[str, Any]:
    """The schema of a review of flat branches.

    The review has at most ``width`` branches, and exactly ``width`` if ``count_mode`` is ``fixed``."""
    schema = _object(
        {
            "synthesis": STRING,
            "branches": {
                "type": "array",
                "items": deepcopy(FLAT_BRANCH),
                "minItems": width if count_mode == "fixed" else 0,
                "maxItems": width,
            },
        }
    )
    return schema


#: The id of a call in an inserted turn: its round, its branch's position and its own position. Ids
#: differ between rounds, so a round-2 branch whose history holds round-1 calls repeats none of them.
CALL_ID = "reviewer-r{round_no}-b{branch}-c{call}"


def response_text(value: dict[str, Any]) -> str:
    """The text of the one message of a Chat Completions response, unchanged.

    The text is empty when the message makes tool calls or has no content. Raises ``ValueError``
    unless the response has exactly one choice, and its content is text or absent."""
    choices = value.get("choices") or []
    if len(choices) != 1:
        raise ValueError("Expected exactly one review response")
    message = choices[0].get("message") or {}
    if message.get("tool_calls"):
        return ""
    text = message.get("content")
    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError("Review response content must be text")
    return text


def _validate(schema: dict[str, Any], value: object, where: str) -> None:
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as error:
        path = ".".join(str(part) for part in error.absolute_path)
        raise ValueError(
            "Structured branch-plan field %s: %s"
            % (where + ("." + path if path else "") if where else (path or "<plan>"), error.message[:400])
        ) from error


def _assistant_turn(
    branch: dict[str, Any], *, round_no: int, index: int, command_tool: str = "bash"
) -> dict[str, Any]:
    """The wire message for a flat branch: its text, and one bash call per command."""
    calls = []
    for call, command in enumerate(branch["commands"], 1):
        if not command.strip():
            raise ValueError(
                "Structured branch-plan field branches.%d.commands.%d: a command must not be blank"
                % (index, call - 1)
            )
        calls.append(
            {
                "id": CALL_ID.format(round_no=round_no, branch=index + 1, call=call),
                "type": "function",
                "function": {"name": command_tool, "arguments": {"command": command}},
            }
        )
    if not branch["content"].strip():
        raise ValueError("Structured branch-plan field branches.%d.content: must not be blank" % index)
    return {"role": "assistant", "content": branch["content"], "tool_calls": calls}


def extract_assistant_turn_plan(
    text: str,
    *,
    round_no: int = 0,
    json_only: bool = False,
    tools_by_base: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """The plan in ``text`` with each branch in the nested form, and each call's arguments as a JSON string.

    A branch gives ``content`` and ``commands``, or a whole ``assistant_turn``. The calls built from
    ``commands`` get ids from ``round_no``. With ``json_only``, ``text`` is bare JSON, not a fenced block.
    Raises ``ValueError``, with the field named, for a plan that breaks a schema or a turn that submits."""
    plan = json.loads(text) if json_only else extract_branch_plan(text)
    _validate(REVIEW_SCHEMA, plan, "")
    result: dict[str, Any] = deepcopy(plan)
    for index, branch in enumerate(result["branches"]):
        where = "branches.%d" % index
        tools = (tools_by_base or {}).get(branch.get("base"), [])
        names = {tool["function"]["name"] for tool in tools}
        command_tool = "run_command" if "run_command" in names else "bash"
        flat = {"content", "commands"} & set(branch)
        if flat and "assistant_turn" in branch:
            raise ValueError(
                "Structured branch-plan field %s: write content and commands, not assistant_turn as well"
                % where
            )
        if flat:
            _validate(FLAT_BRANCH, branch, where)
            turn = _assistant_turn(branch, round_no=round_no, index=index, command_tool=command_tool)
            for key in ("content", "commands"):
                del branch[key]
            branch["assistant_turn"] = turn
        elif "assistant_turn" in branch:
            nested = deepcopy(NESTED_BRANCH)
            if names:
                function = nested["properties"]["assistant_turn"]["properties"]["tool_calls"]["items"][
                    "properties"
                ]["function"]["properties"]
                function["name"]["enum"] = sorted(names)
                function["arguments"] = {"type": "object"}
            _validate(nested, branch, where)
        else:
            raise ValueError("Structured branch-plan field %s: a branch needs content and commands" % where)
        for number, call in enumerate(branch["assistant_turn"]["tool_calls"]):
            arguments = call["function"]["arguments"]
            if SUBMIT_SENTINEL in json.dumps(arguments, ensure_ascii=False):
                raise ValueError(
                    "Structured branch-plan field %s.assistant_turn.tool_calls.%d: the inserted turn "
                    "must not submit (%s); the actor decides when the work is done"
                    % (where, number, SUBMIT_SENTINEL)
                )
            call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
    return result
