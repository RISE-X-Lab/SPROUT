"""``validate_assistant_turn`` and ``validate_mini_tools``: a turn written outside the actor, and its tool.

A reviewer model writes the turn, so each refusal must name what to fix. mini runs a turn that passes.
"""

from __future__ import annotations

import json
import re

import pytest

from sprout.contract.mini_history import validate_assistant_turn, validate_mini_tools
from tests.contract.builders import REVIEWER_TEXT, assistant_turn, bash_call, bash_tools

FORM = "assistant_turn requires exactly role, content and tool_calls"
ONE_COMMAND = (
    "assistant_turn.tool_calls[0] (reviewer-call-1) "
    "bash arguments require exactly one non-empty string command"
)


def reasoning_turn(content: object = "", reasoning: object = REVIEWER_TEXT) -> dict[str, object]:
    """``assistant_turn()`` in the reasoning form, with its text in ``reasoning_content``."""
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": [bash_call()],
    }


def with_call(**changes: object) -> dict[str, object]:
    """``assistant_turn()`` whose one call has ``changes``."""
    return {"role": "assistant", "content": REVIEWER_TEXT, "tool_calls": [{**bash_call(), **changes}]}


def with_function(**changes: object) -> dict[str, object]:
    """``assistant_turn()`` whose one call's ``function`` has ``changes``."""
    call = bash_call()
    return with_call(function={**call["function"], **changes})


def test_a_valid_turn_comes_back_as_it_was_written_whatever_programs_its_command_runs():
    """Program names inside the command, such as ``python3``, are not tool names."""
    command = 'python3 -c "print(1)" && git status --short'

    turn = validate_assistant_turn(assistant_turn(command))

    assert turn == assistant_turn(command)


def test_the_checked_turn_shares_nothing_with_the_given_one():
    message = assistant_turn()

    validate_assistant_turn(message)["tool_calls"][0]["function"]["arguments"] = "{}"

    assert message == assistant_turn()


@pytest.mark.parametrize(
    "message, error",
    [
        pytest.param(7, FORM, id="number"),
        pytest.param({**assistant_turn(), "extra": {"actions": []}}, FORM, id="extra-key"),
        pytest.param({"role": "assistant", "content": REVIEWER_TEXT}, FORM, id="no-calls-key"),
        pytest.param(reasoning_turn(), FORM, id="reasoning-form"),
        pytest.param(
            {**assistant_turn(), "role": "user"},
            "assistant_turn requires assistant role and non-empty ordinary content",
            id="user-role",
        ),
        pytest.param(
            assistant_turn(content=" \n"),
            "assistant_turn requires assistant role and non-empty ordinary content",
            id="blank-content",
        ),
        pytest.param(
            {**assistant_turn(), "content": ["text"]},
            "assistant_turn requires assistant role and non-empty ordinary content",
            id="content-not-text",
        ),
        pytest.param(
            {**assistant_turn(), "tool_calls": []},
            "assistant_turn requires at least one bash tool call",
            id="no-call",
        ),
        pytest.param(
            {**assistant_turn(), "tool_calls": bash_call()},
            "assistant_turn requires at least one bash tool call",
            id="calls-not-a-list",
        ),
        pytest.param(
            with_call(index=0), "assistant_turn tool calls require id, type and function", id="extra-call-key"
        ),
        pytest.param(
            {**assistant_turn(), "tool_calls": [None]},
            "assistant_turn tool calls require id, type and function",
            id="null-call",
        ),
        pytest.param(
            with_call(id=" "),
            "assistant_turn tool call ids must be non-empty and unique across history",
            id="blank-id",
        ),
        pytest.param(
            with_call(id=7),
            "assistant_turn tool call ids must be non-empty and unique across history",
            id="id-not-text",
        ),
        pytest.param(
            {**assistant_turn(), "tool_calls": [bash_call("ls", "same"), bash_call("pwd", "same")]},
            "assistant_turn tool call ids must be non-empty and unique across history",
            id="id-twice-in-the-turn",
        ),
        pytest.param(
            with_call(type="custom"),
            "assistant_turn supports only bash with JSON-string arguments",
            id="custom-type",
        ),
        pytest.param(
            with_function(strict=True),
            "assistant_turn supports only bash with JSON-string arguments",
            id="extra-function-key",
        ),
        pytest.param(
            with_function(arguments={"command": "pwd"}),
            "assistant_turn supports only bash with JSON-string arguments",
            id="arguments-as-an-object",
        ),
        pytest.param(
            with_function(arguments="{invalid"),
            "assistant_turn.tool_calls[0] (reviewer-call-1) bash arguments are not valid JSON: ",
            id="arguments-not-json",
        ),
        pytest.param(
            # The arguments that a reviewer model wrote for the first Luna parent.
            with_function(
                arguments=json.dumps(
                    {"command": "git status --short --branch", "working_dir": "/app", "timeout": 120}
                )
            ),
            ONE_COMMAND,
            id="arguments-beside-the-command",
        ),
        pytest.param(
            with_function(arguments='{"command": 1}'),
            ONE_COMMAND,
            id="command-not-text",
        ),
        pytest.param(
            with_function(arguments='{"command": " "}'),
            ONE_COMMAND,
            id="blank-command",
        ),
        pytest.param(
            with_function(arguments="7"),
            ONE_COMMAND,
            id="arguments-a-number",
        ),
    ],
)
def test_a_turn_that_mini_cannot_run_is_refused(message, error):
    with pytest.raises(ValueError, match="^" + re.escape(error)):
        validate_assistant_turn(message)


@pytest.mark.parametrize("name", ["shell", "python", "apply_patch", "text_editor", "Bash", "mcp__bash"])
def test_a_call_of_another_tool_is_refused_with_what_the_reviewer_must_write_instead(name):
    message = with_function(name=name)

    with pytest.raises(ValueError) as refusal:
        validate_assistant_turn(message)

    assert str(refusal.value) == (
        f"assistant_turn.tool_calls[0] (reviewer-call-1) has forbidden tool name {name!r}; "
        "only bash is allowed. "
        "Return function.name='bash' with arguments containing exactly one string field, command. "
        "No calls from this plan have executed."
    )


def test_a_call_id_of_the_history_is_refused():
    history = [{"role": "assistant", "content": "", "tool_calls": [bash_call("ls", "parent-1")]}]

    with pytest.raises(ValueError, match="unique across history"):
        validate_assistant_turn(assistant_turn(identifier="parent-1"), history=history)


def test_a_turn_with_new_call_ids_continues_the_history():
    history = [
        {"role": "user", "content": "Fix the bug."},
        {"role": "assistant", "content": "", "tool_calls": [bash_call("ls", "parent-1")]},
        {"role": "tool", "tool_call_id": "parent-1", "content": "README"},
    ]

    turn = validate_assistant_turn(assistant_turn(identifier="inserted-1"), history=history)

    assert turn == assistant_turn(identifier="inserted-1")


def test_a_call_that_breaks_the_recorded_parameter_schema_is_refused():
    with pytest.raises(ValueError, match="^assistant_turn does not match the recorded actor tool schema: "):
        validate_assistant_turn(assistant_turn("printf wrong"), tools=bash_tools(pattern="^git "))


def test_a_call_that_fits_the_recorded_parameter_schema_is_accepted():
    turn = validate_assistant_turn(assistant_turn("git status"), tools=bash_tools(pattern="^git "))

    assert turn == assistant_turn("git status")


def test_a_recorded_tool_other_than_bash_refuses_every_turn():
    tools = bash_tools()
    tools[0]["function"]["name"] = "different_tool"

    with pytest.raises(ValueError, match="^Recorded actor tool must be the bash function$"):
        validate_assistant_turn(assistant_turn(), tools=tools)


@pytest.mark.parametrize("content", ["", "Checking the boundary."], ids=["no-content", "one-line"])
def test_the_reasoning_form_is_a_turn_where_reasoning_is_allowed(content):
    turn = validate_assistant_turn(reasoning_turn(content), allow_reasoning=True)

    assert turn == reasoning_turn(content)


def test_the_form_of_the_author_is_a_turn_where_reasoning_is_allowed():
    turn = validate_assistant_turn(assistant_turn(), allow_reasoning=True)

    assert turn == assistant_turn()


@pytest.mark.parametrize(
    "message, error",
    [
        pytest.param(
            reasoning_turn("Check the boundary.\nThen run the tests."),
            "assistant_turn with reasoning_content requires assistant role and content of at most one line",
            id="two-lines-of-content",
        ),
        pytest.param(
            reasoning_turn("\n"),
            "assistant_turn with reasoning_content requires assistant role and content of at most one line",
            id="newline-content",
        ),
        pytest.param(
            reasoning_turn(None),
            "assistant_turn with reasoning_content requires assistant role and content of at most one line",
            id="no-text-content",
        ),
        pytest.param(
            {**reasoning_turn(), "role": "user"},
            "assistant_turn with reasoning_content requires assistant role and content of at most one line",
            id="user-role",
        ),
        pytest.param(
            reasoning_turn("Checking.", reasoning=" \n "),
            "assistant_turn reasoning_content must be non-empty text",
            id="blank-reasoning",
        ),
        pytest.param(
            reasoning_turn(reasoning=None),
            "assistant_turn reasoning_content must be non-empty text",
            id="no-reasoning",
        ),
        pytest.param(
            reasoning_turn(reasoning=["why"]),
            "assistant_turn reasoning_content must be non-empty text",
            id="reasoning-not-text",
        ),
        pytest.param(
            {**assistant_turn(), "content": ""},
            "assistant_turn requires assistant role and non-empty ordinary content",
            id="author-form-without-content",
        ),
        pytest.param(
            {**reasoning_turn(), "extra": {"actions": []}},
            FORM + ", or those and reasoning_content",
            id="extra-key",
        ),
    ],
)
def test_a_reasoning_turn_without_reasoning_or_with_more_than_one_line_of_content_is_refused(message, error):
    with pytest.raises(ValueError, match="^%s$" % re.escape(error)):
        validate_assistant_turn(message, allow_reasoning=True)


def test_the_calls_of_a_reasoning_turn_meet_the_same_checks():
    turn = {
        **reasoning_turn(),
        "tool_calls": [{**bash_call(), "function": {"name": "shell", "arguments": "{}"}}],
    }

    with pytest.raises(ValueError, match="only bash is allowed"):
        validate_assistant_turn(turn, allow_reasoning=True)


def test_the_tool_of_mini_comes_back_as_recorded():
    """The recorded schema keeps its description and its ``additionalProperties``."""
    tools = bash_tools()

    checked = validate_mini_tools(tools)

    assert checked == bash_tools()


def test_the_checked_tool_list_shares_nothing_with_the_recorded_one():
    tools = bash_tools()

    validate_mini_tools(tools)[0]["function"]["parameters"]["required"].append("timeout")

    assert tools == bash_tools()


def tool_with(**parameters: object) -> list[dict[str, object]]:
    """``bash_tools()`` whose ``parameters`` have ``parameters``."""
    tools = bash_tools()
    tools[0]["function"]["parameters"] = {**tools[0]["function"]["parameters"], **parameters}
    return tools


@pytest.mark.parametrize(
    "tools, error",
    [
        pytest.param(
            {"tools": bash_tools()}, "Mini requires exactly one recorded bash tool", id="tools-in-an-object"
        ),
        pytest.param([], "Mini requires exactly one recorded bash tool", id="no-tool"),
        pytest.param(
            bash_tools() + bash_tools(), "Mini requires exactly one recorded bash tool", id="two-tools"
        ),
        pytest.param(["bash"], "Mini requires exactly one recorded bash tool", id="tool-not-an-object"),
        pytest.param(
            [{**bash_tools()[0], "type": "custom"}],
            "Recorded actor tool must be the bash function",
            id="custom-type",
        ),
        pytest.param(
            [{"type": "function", "function": "bash"}],
            "Recorded actor tool must be the bash function",
            id="function-not-an-object",
        ),
        pytest.param(
            [{"type": "function", "function": {"name": "bash"}}],
            "Recorded bash tool requires an object parameter schema",
            id="no-parameters",
        ),
        pytest.param(
            tool_with(properties=["command"]),
            "Recorded bash tool requires an object parameter schema",
            id="properties-not-an-object",
        ),
        pytest.param(
            tool_with(type="array"),
            "Recorded bash tool must require exactly one string command",
            id="array-parameters",
        ),
        pytest.param(
            tool_with(properties={"command": {"type": "string"}, "timeout": {"type": "integer"}}),
            "Recorded bash tool must require exactly one string command",
            id="second-property",
        ),
        pytest.param(
            tool_with(properties={"command": "string"}),
            "Recorded bash tool must require exactly one string command",
            id="command-schema-not-an-object",
        ),
        pytest.param(
            tool_with(properties={"command": {"type": "array"}}),
            "Recorded bash tool must require exactly one string command",
            id="command-not-text",
        ),
        pytest.param(
            tool_with(required=[]),
            "Recorded bash tool must require exactly one string command",
            id="command-optional",
        ),
    ],
)
def test_a_recorded_tool_list_other_than_the_one_bash_tool_of_mini_is_refused(tools, error):
    with pytest.raises(ValueError, match="^%s$" % re.escape(error)):
        validate_mini_tools(tools)
