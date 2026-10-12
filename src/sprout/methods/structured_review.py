"""Fenced branch plans: a flat wire format the controller wraps into tool calls.

The reviewer once wrote each branch's continuation as the full Chat
Completions assistant message -- ``assistant_turn.tool_calls[].function
.arguments.command``, six levels deep. After a command of several thousand
characters (a heredoc with a whole script) the model had to close exactly
``}}}`` and miscounted often enough to lose rounds: in one TerminalBench run
23 of 79 reviewer replies were rejected, 19 of them for a brace count at a
tool call's end, and the model repeated the same miscount on every retry. A
branch now carries ``content`` (the assistant's text) and ``commands`` (the
bash commands, a JSON list of strings, in execution order); after each
command the model closes one string. The controller builds the tool calls,
minting ids of its own (``reviewer-r<round>-b<branch>-c<call>``), so the
actor still receives an ordinary assistant message and everything after the
parse -- ``validate_assistant_turn``, the journal, training export -- sees
the wire form it always did. A branch written in the old nested form is
still accepted, so a plan recorded before the change replays.
"""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator, ValidationError

from sprout.methods.branch_plan import extract_branch_plan


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


STRING = {"type": "string"}
COMMAND = _object({"command": STRING})
FUNCTION = _object({"name": {"type": "string", "enum": ["bash"]},
                    "arguments": COMMAND})
TOOL_CALL = _object({"id": STRING, "type": {"type": "string", "enum": ["function"]},
                    "function": FUNCTION})
ASSISTANT_TURN = _object({
    "role": {"type": "string", "enum": ["assistant"]}, "content": STRING,
    "tool_calls": {"type": "array", "items": TOOL_CALL},
})
POINT = {"name": STRING, "base": STRING, "branch_step": {"type": "integer"}, "why": STRING}
#: What the reviewer writes: the continuation's text and its bash commands.
FLAT_BRANCH = _object({**POINT, "content": STRING,
                       "commands": {"type": "array", "minItems": 1, "items": STRING}})
#: mini's submit command: the first line of a command's output ends the run. An
#: inserted turn that prints it leaves the policy nothing to sample after it.
SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
#: The wire form the controller builds from it, and still accepts as input.
NESTED_BRANCH = _object({**POINT, "assistant_turn": ASSISTANT_TURN})
#: The outer envelope; each branch is then checked against the form its keys name.
REVIEW_SCHEMA = _object({
    "synthesis": STRING,
    "branches": {"type": "array", "items": {"type": "object"}},
})
ANALYST_SCHEMA = _object({
    "failure_reason": STRING, "lesson": STRING, "salvage": STRING,
    "branch_candidates": {"type": "array", "items": _object({
        "step": {"type": "integer"}, "why": STRING,
    })},
})


def flat_review_schema(width: int, count_mode: str) -> dict:
    schema = _object({
        "synthesis": STRING,
        "branches": {"type": "array", "items": deepcopy(FLAT_BRANCH),
                     "minItems": width if count_mode == "fixed" else 0, "maxItems": width},
    })
    return schema

#: Controller-minted tool call ids: the round, the branch's position in the
#: plan and the call's position in the branch. Distinct across rounds, so a
#: round-2 branch whose history holds round-1 calls repeats none of them.
CALL_ID = "reviewer-r{round_no}-b{branch}-c{call}"


def response_text(value):
    """Pass provider text through unchanged; never wrap prose or invent a plan."""
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


def _validate(schema, value, where):
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as error:
        path = ".".join(str(part) for part in error.absolute_path)
        raise ValueError("Structured branch-plan field %s: %s" % (
            where + ("." + path if path else "") if where else (path or "<plan>"), error.message[:400])) from error


def _assistant_turn(branch, *, round_no, index):
    """The wire message for a flat branch: its text, one bash call per command."""
    calls = []
    for call, command in enumerate(branch["commands"], 1):
        if not command.strip():
            raise ValueError("Structured branch-plan field branches.%d.commands.%d: a command must not be blank"
                             % (index, call - 1))
        calls.append({"id": CALL_ID.format(round_no=round_no, branch=index + 1, call=call), "type": "function",
                      "function": {"name": "bash", "arguments": {"command": command}}})
    if not branch["content"].strip():
        raise ValueError("Structured branch-plan field branches.%d.content: must not be blank" % index)
    return {"role": "assistant", "content": branch["content"], "tool_calls": calls}


def extract_assistant_turn_plan(text, *, round_no=0, json_only=False):
    """Validate the wire plan and return it in the nested form, arguments serialized exactly once.

    Each branch is checked against the form its keys name: ``content`` +
    ``commands`` (what the prompt asks for; the controller builds the
    ``assistant_turn`` and mints its call ids from ``round_no``) or a full
    ``assistant_turn`` (the recorded form, still accepted). A branch mixing
    the two, or lacking both, is rejected with the field named.
    """
    plan = json.loads(text) if json_only else extract_branch_plan(text)
    _validate(REVIEW_SCHEMA, plan, "")
    result = deepcopy(plan)
    for index, branch in enumerate(result["branches"]):
        where = "branches.%d" % index
        flat = {"content", "commands"} & set(branch)
        if flat and "assistant_turn" in branch:
            raise ValueError("Structured branch-plan field %s: write content and commands, not assistant_turn as well"
                             % where)
        if flat:
            _validate(FLAT_BRANCH, branch, where)
            turn = _assistant_turn(branch, round_no=round_no, index=index)
            for key in ("content", "commands"):
                del branch[key]
            branch["assistant_turn"] = turn
        elif "assistant_turn" in branch:
            _validate(NESTED_BRANCH, branch, where)
        else:
            raise ValueError("Structured branch-plan field %s: a branch needs content and commands" % where)
        for number, call in enumerate(branch["assistant_turn"]["tool_calls"]):
            arguments = call["function"]["arguments"]
            if SUBMIT_SENTINEL in json.dumps(arguments, ensure_ascii=False):
                raise ValueError("Structured branch-plan field %s.assistant_turn.tool_calls.%d: the inserted turn "
                                 "must not submit (%s); the actor decides when the work is done"
                                 % (where, number, SUBMIT_SENTINEL))
            call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
    return result
