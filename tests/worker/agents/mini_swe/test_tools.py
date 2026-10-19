"""The mini bash contract: the faults of a response's raw tool calls, and the feedback."""

from __future__ import annotations

import json
from typing import Any

import pytest

from sprout.worker.agents.mini_swe.tools import (
    malformed_feedback,
    malformed_tool_calls,
    mini_tool_schema,
)
from tests.worker.agents.mini_swe.rollouts import BASH_TOOL

ABSENT = object()


def call(
    identifier: object = "call-1",
    *,
    type: object = "function",
    name: object = "bash",
    arguments: object = "{}",
) -> dict[str, Any]:
    """A raw tool call with these fields. ``ABSENT`` leaves a field out."""
    function = {
        key: value for key, value in (("name", name), ("arguments", arguments)) if value is not ABSENT
    }
    fields = (("id", identifier), ("type", type), ("function", function))
    return {key: value for key, value in fields if value is not ABSENT}


def fault(index: int, identifier: str | None, error: str) -> dict[str, Any]:
    return {"index": index, "id": identifier, "error": error}


@pytest.mark.parametrize(
    "calls, history, problems",
    [
        pytest.param([call()], set(), [], id="valid"),
        pytest.param([call(type=None)], set(), [], id="null-type-is-function"),
        pytest.param([call(type=ABSENT)], set(), [], id="absent-type-is-function"),
        pytest.param([call(arguments={"command": "ls"})], set(), [], id="object-arguments"),
        pytest.param([call(name="shell")], set(), [], id="unknown-name-is-a-schema-fault-only"),
        pytest.param(
            ["bash ls"], set(), [fault(0, None, "the tool call is not an object")], id="not-an-object"
        ),
        pytest.param(
            [call(type="tool")], set(), [fault(0, "call-1", 'type must be "function"')], id="wrong-type"
        ),
        pytest.param(
            [{"id": "call-1", "type": "function", "function": "bash"}],
            set(),
            [fault(0, "call-1", "function must be an object with name and arguments")],
            id="function-not-an-object",
        ),
        pytest.param(
            [call(name=" ")],
            set(),
            [fault(0, "call-1", "function.name is missing or blank; the only tool is bash")],
            id="blank-name",
        ),
        pytest.param(
            [call(name=ABSENT)],
            set(),
            [fault(0, "call-1", "function.name is missing or blank; the only tool is bash")],
            id="absent-name",
        ),
        pytest.param(
            [call(arguments=None)],
            set(),
            [fault(0, "call-1", "function.arguments must be a JSON-encoded string")],
            id="null-arguments",
        ),
        pytest.param([call(ABSENT)], set(), [fault(0, None, "id is missing or blank")], id="absent-id"),
        pytest.param([call("")], set(), [fault(0, "", "id is missing or blank")], id="blank-id"),
        pytest.param(
            [call("call-1"), call("call-1")],
            set(),
            [
                fault(0, "call-1", "id repeats another call's id in this response"),
                fault(1, "call-1", "id repeats another call's id in this response"),
            ],
            id="repeated-id",
        ),
        pytest.param(
            [call("call-1")],
            {"call-1"},
            [fault(0, "call-1", "id was already used earlier in this conversation")],
            id="id-from-history",
        ),
        pytest.param(
            [call(ABSENT, type="tool")],
            set(),
            [fault(0, None, 'type must be "function"; id is missing or blank')],
            id="two-faults",
        ),
    ],
)
def test_each_faulty_call_of_a_response_is_named_with_each_of_its_faults(calls, history, problems):
    assert malformed_tool_calls(calls, history) == problems


@pytest.mark.parametrize(
    "length, quoted",
    [(2000, "x" * 2000), (2001, "x" * 2000 + " \u2026[1 more chars]")],
    ids=["at-limit", "over-limit"],
)
def test_the_feedback_quotes_each_rejected_call_cut_to_2000_characters(length, quoted):
    message = {
        "role": "assistant",
        "content": "Run it.",
        "tool_calls": [call(ABSENT, arguments="x" * length)],
    }
    problems = [fault(0, None, "id is missing or blank")]

    feedback = malformed_feedback(problems, message, raw={"id": "chatcmpl-fixture"})

    assert rejected_response(feedback["content"]) == {
        "content": "Run it.",
        "rejected_tool_calls": [{**problems[0], "name": "bash", "arguments": quoted}],
    }


@pytest.mark.parametrize(
    "length, quoted",
    [(1000, "y" * 1000), (1001, "y" * 1000 + " \u2026[1 more chars]")],
    ids=["at-limit", "over-limit"],
)
def test_the_feedback_quotes_the_text_of_the_response_cut_to_1000_characters(length, quoted):
    message = {"role": "assistant", "content": "y" * length, "tool_calls": [call(ABSENT)]}
    problems = [fault(0, None, "id is missing or blank")]

    feedback = malformed_feedback(problems, message, raw={"id": "chatcmpl-fixture"})

    assert rejected_response(feedback["content"])["content"] == quoted


def rejected_response(content: str) -> Any:
    """The JSON between the ``rejected_response`` tags of a feedback message."""
    return json.loads(content.split("<rejected_response>\n")[1].split("\n</rejected_response>")[0])


def test_an_installed_mini_whose_bash_tool_differs_from_the_pinned_one_is_refused():
    upstream = json.loads(json.dumps(BASH_TOOL))
    upstream["function"]["description"] = "Execute a command"

    with pytest.raises(
        ValueError, match="^Installed mini tool definition differs from the pinned actor contract$"
    ):
        mini_tool_schema(upstream)
