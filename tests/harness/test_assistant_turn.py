from copy import deepcopy
import json

import pytest

from sprout.harness.agents.mini_swe.history import validate_assistant_turn


def assistant_turn(command="printf check", identifier="reviewer-call-1"):
    return {"role": "assistant", "content": "The existing branch suggests checking this boundary.",
            "tool_calls": [{"id": identifier, "type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"command": command})}}]}


@pytest.mark.parametrize("command", [
    "printf check",
    'python3 -c "print(1)" && git status --short',  # program names inside the command are not tool names
])
def test_turn_is_preserved_without_sharing_mutable_reviewer_input(command):
    message = assistant_turn(command)
    result = validate_assistant_turn(message)
    assert result == message and result is not message
    result["tool_calls"][0]["function"]["arguments"] = "{}"
    assert message == assistant_turn(command)


@pytest.mark.parametrize("change", [
    lambda m: m.update(role="user"),
    lambda m: m.update(content=""),
    lambda m: m.update(reasoning_content="native thinking"),
    lambda m: m.update(extra={"actions": [{"command": "unvalidated"}]}),
    lambda m: m.update(tool_calls=[]),
    lambda m: m["tool_calls"].append(deepcopy(m["tool_calls"][0])),
    lambda m: m["tool_calls"][0].update(id=""),
    lambda m: m["tool_calls"][0]["function"].update(name="shell"),
    lambda m: m["tool_calls"][0]["function"].update(arguments={"command": "pwd"}),
    lambda m: m["tool_calls"][0]["function"].update(arguments="{invalid"),
    lambda m: m["tool_calls"][0]["function"].update(arguments='{"command":"pwd","cwd":"/host"}'),
    lambda m: m["tool_calls"][0]["function"].update(arguments='{"command":1}'),
])
def test_invalid_synthetic_turn_is_rejected(change):
    message = assistant_turn()
    change(message)
    with pytest.raises(ValueError, match="assistant_turn"):
        validate_assistant_turn(message)


def rendered_turn(content="", reasoning="The existing branch suggests checking this boundary."):
    """``assistant_turn()`` in the canonical rendering: its text as reasoning."""
    return {"role": "assistant", "content": content, "reasoning_content": reasoning,
            "tool_calls": assistant_turn()["tool_calls"]}


@pytest.mark.parametrize("content", ["", "Checking the boundary."])
def test_the_rendered_form_is_accepted_only_where_reasoning_is_allowed(content):
    message = rendered_turn(content)
    result = validate_assistant_turn(message, allow_reasoning=True)
    assert result == message and result is not message
    result["tool_calls"][0]["id"] = "edited"
    assert message == rendered_turn(content)
    with pytest.raises(ValueError, match="assistant_turn requires exactly role, content and tool_calls$"):
        validate_assistant_turn(message)
    assert validate_assistant_turn(assistant_turn(), allow_reasoning=True) == assistant_turn(), (
        "the author's form is still a turn there")


@pytest.mark.parametrize("message", [
    pytest.param(rendered_turn("Check the boundary.\nThen run the tests."), id="multi-line-content"),
    pytest.param(rendered_turn("\n"), id="newline-content"),
    pytest.param(rendered_turn(None), id="no-text-content"),
    pytest.param(rendered_turn("Checking.", reasoning=" \n "), id="blank-reasoning"),
    pytest.param(rendered_turn(reasoning=None), id="null-reasoning"),
    pytest.param(rendered_turn(reasoning=["why"]), id="non-text-reasoning"),
    pytest.param(rendered_turn("", reasoning=""), id="both-blank"),
    pytest.param({**assistant_turn(), "content": ""}, id="blank-content-without-reasoning"),
    pytest.param({**rendered_turn(), "role": "user"}, id="user-role"),
    pytest.param({**rendered_turn(), "extra": {"actions": []}}, id="extra-key"),
])
def test_a_rendered_turn_with_no_reasoning_or_more_than_one_line_is_refused(message):
    with pytest.raises(ValueError, match="assistant_turn"):
        validate_assistant_turn(message, allow_reasoning=True)


def test_the_rendered_form_meets_the_same_call_checks():
    with pytest.raises(ValueError, match="unique across history"):
        validate_assistant_turn(rendered_turn(), history=[assistant_turn()], allow_reasoning=True)
    message = rendered_turn()
    message["tool_calls"][0]["function"]["name"] = "shell"
    with pytest.raises(ValueError, match="only bash is allowed"):
        validate_assistant_turn(message, allow_reasoning=True)


@pytest.mark.parametrize("name", ["shell", "python", "apply_patch", "text_editor", "Bash", "mcp__bash"])
def test_non_bash_name_is_reported_for_reviewer_correction(name):
    message = assistant_turn()
    message["tool_calls"][0]["function"]["name"] = name
    with pytest.raises(ValueError, match="only bash is allowed") as error:
        validate_assistant_turn(message)
    assert repr(name) in str(error.value)
    assert "tool_calls[0]" in str(error.value)
