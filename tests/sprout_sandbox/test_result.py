"""``ToolResult.from_response`` reads the result of one ``tools/call`` reply, and its command outcome."""

from __future__ import annotations

import json
import re

import pytest

from sprout_sandbox import ToolResult
from tests.support.paths import REPO

#: The Go file that defines the runtime's ``CommandOutcome``, the JSON that ``shell`` answers with.
OUTCOME_SOURCE = REPO / "runtime" / "tools" / "boundedlog.go"


def reply(text: str, *, is_error: bool = False) -> dict:
    """The ``result`` object of a reply whose one text slot holds ``text``."""
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def outcome_with_every_runtime_field(**values: object) -> dict:
    """A ``CommandOutcome`` with each JSON field of the runtime's Go struct: ``values``, else 0."""
    struct = re.search(r"type CommandOutcome struct \{(.*?)\n\}", OUTCOME_SOURCE.read_text(), re.S)
    assert struct is not None, "no CommandOutcome struct in %s" % OUTCOME_SOURCE
    return {name: values.get(name, 0) for name in re.findall(r'`json:"([a-z_]+)"`', struct.group(1))}


@pytest.mark.parametrize(
    "result, expected",
    [
        pytest.param(
            reply("hello", is_error=True), ToolResult(output="hello", is_error=True), id="text-slot"
        ),
        pytest.param({"isError": True}, ToolResult(output="", is_error=True), id="no-content"),
        pytest.param({"content": []}, ToolResult(output="", is_error=False), id="empty-content"),
    ],
)
def test_the_output_is_the_first_text_slot_and_the_error_flag_is_is_error(result, expected):
    assert ToolResult.from_response(result) == expected


@pytest.mark.parametrize(
    "outcome, expected",
    [
        pytest.param(
            {
                "stdout": "collected 42\n",
                "stderr": "AssertionError\n",
                "exit_code": 1,
                "running": False,
                "timed_out": True,
                "stdout_bytes": 13,
                "stderr_bytes": 419430400,
                "stdout_truncated": False,
                "stderr_truncated": True,
                "max_per_stream": 15728640,
            },
            {
                "exit_code": 1,
                "stdout": "collected 42\n",
                "stderr": "AssertionError\n",
                "running": False,
                "timed_out": True,
                "stdout_bytes": 13,
                "stderr_bytes": 419430400,
                "stdout_truncated": False,
                "stderr_truncated": True,
            },
            id="finished",
        ),
        pytest.param(
            {"stdout": "so far\n", "stderr": "", "exit_code": None, "running": True},
            {"exit_code": None, "stdout": "so far\n", "stderr": "", "running": True},
            id="still-running",
        ),
        pytest.param({"stdout": "", "stderr": ""}, {"stdout": "", "stderr": ""}, id="only-the-streams"),
    ],
)
def test_a_command_outcome_in_the_text_slot_fills_the_outcome_fields(outcome, expected):
    text = json.dumps(outcome)

    result = ToolResult.from_response(reply(text, is_error=True))

    assert result == ToolResult(output=text, is_error=True, **expected)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("hello_ci\n", id="plain-text"),
        pytest.param("{not json", id="malformed-json"),
        pytest.param('{"exit_code": 0}', id="no-stream"),
        pytest.param('{"stdout": "x", "exit_code": 0}', id="one-stream"),
        pytest.param(
            '{"stdout": "x", "stderr": "y", "name": "my-package"}', id="both-streams-and-a-foreign-key"
        ),
    ],
)
def test_text_that_is_not_a_command_outcome_is_plain_output(text):
    """A command can print JSON, such as ``cat package.json``. Read as an outcome, it would replace
    what the model sees of that output with empty streams."""
    assert ToolResult.from_response(reply(text)) == ToolResult(output=text, is_error=False)


def test_an_outcome_with_every_field_of_the_runtimes_command_outcome_is_read_as_one():
    """A field that the runtime adds and that the parser does not know turns each outcome into plain text."""
    outcome = outcome_with_every_runtime_field(stdout="out", stderr="err", exit_code=3)

    result = ToolResult.from_response(reply(json.dumps(outcome)))

    assert (result.stdout, result.stderr, result.exit_code) == ("out", "err", 3), sorted(outcome)
