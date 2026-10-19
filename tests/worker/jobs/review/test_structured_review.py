"""The flat branch that a reviewer writes, the tool calls built from it, and the reply that it comes in."""

from __future__ import annotations

import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS
from sprout.worker.jobs.review.structured_review import (
    extract_assistant_turn_plan,
    flat_review_schema,
    response_text,
)


def flat(**changes: Any) -> dict[str, Any]:
    """A flat branch with its text and one command, and ``changes``."""
    branch = {"name": "probe", "base": "parent", "branch_step": 2, "why": "Check the parser."}
    branch.update(content="I will check the parser.", commands=["pytest -q"])
    branch.update(changes)
    return branch


def flat_without(key: str) -> dict[str, Any]:
    branch = flat()
    del branch[key]
    return branch


def nested(**function: Any) -> dict[str, Any]:
    """A branch in the nested wire form, with one call whose function has ``function``."""
    call = {"id": "recorded-call", "type": "function", "function": function}
    turn = {"role": "assistant", "content": "I will check the parser.", "tool_calls": [call]}
    return {
        "name": "probe",
        "base": "parent",
        "branch_step": 2,
        "why": "Check the parser.",
        "assistant_turn": turn,
    }


def fenced(*branches: dict[str, Any], synthesis: str | None = "Probe first.") -> str:
    plan: dict[str, Any] = {"branches": list(branches)}
    if synthesis is not None:
        plan["synthesis"] = synthesis
    return "```branch-plan\n" + json.dumps(plan) + "\n```"


def bash(call_id: str, command: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def test_the_commands_of_flat_branches_become_bash_calls_named_by_round_branch_and_position():
    """The ids differ from round to round, so a branch of round 2 never repeats the id of a round-1 call."""
    text = fenced(flat(commands=["cd /app && ls", "pytest -q"]), flat(name="other", commands=["git diff"]))

    plan = extract_assistant_turn_plan(text, round_no=2)

    assert [branch["assistant_turn"] for branch in plan["branches"]] == [
        {
            "role": "assistant",
            "content": "I will check the parser.",
            "tool_calls": [
                bash("reviewer-r2-b1-c1", "cd /app && ls"),
                bash("reviewer-r2-b1-c2", "pytest -q"),
            ],
        },
        {
            "role": "assistant",
            "content": "I will check the parser.",
            "tool_calls": [bash("reviewer-r2-b2-c1", "git diff")],
        },
    ]
    assert set(plan["branches"][0]) == {"name", "base", "branch_step", "why", "assistant_turn"}


def test_a_command_with_quotes_newlines_and_unicode_reaches_the_call_unchanged():
    command = "python - <<'PY'\nprint(\"a \\\"quoted\\\" value\", '$(literal)', '中文')\nPY\n"

    plan = extract_assistant_turn_plan(fenced(flat(commands=[command])))

    arguments = plan["branches"][0]["assistant_turn"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"command": command}


def test_a_branch_in_the_nested_form_replays_with_its_ids_and_its_arguments_serialized_once():
    text = fenced(nested(name="bash", arguments={"command": "grep -rn 'x' . | head"}))

    plan = extract_assistant_turn_plan(text, round_no=3)

    assert plan["branches"][0]["assistant_turn"]["tool_calls"] == [
        bash("recorded-call", "grep -rn 'x' . | head")
    ]


@pytest.mark.parametrize(
    "branch, error",
    [
        pytest.param(flat(commands=[]), r"field branches\.0\.commands: ", id="no-command"),
        pytest.param(flat(commands=["ls", 3]), r"field branches\.0\.commands\.1: ", id="command-not-text"),
        pytest.param(flat(commands="ls"), r"field branches\.0\.commands: ", id="commands-not-a-list"),
        pytest.param(flat(branch_step=True), r"field branches\.0\.branch_step: ", id="step-not-an-integer"),
        pytest.param(
            flat_without("content"), r"field branches\.0: 'content' is a required property", id="no-content"
        ),
        pytest.param(flat(hint="Look closer."), r"field branches\.0: .*'hint' was unexpected", id="a-hint"),
        pytest.param(
            nested(name="bash", arguments={"command": "ls", "timeout": 30}),
            r"field branches\.0\.assistant_turn\.tool_calls\.0\.function\.arguments: "
            r".*'timeout' was unexpected",
            id="nested-arguments-beyond-the-command",
        ),
        pytest.param(
            flat(assistant_turn={"role": "assistant"}),
            r"field branches\.0: write content and commands, not assistant_turn as well",
            id="both-forms",
        ),
        pytest.param(
            {"name": "probe", "base": "parent", "branch_step": 2, "why": "Check."},
            r"field branches\.0: a branch needs content and commands",
            id="neither-form",
        ),
        pytest.param(
            nested(name="shell", arguments={"command": "ls"}),
            r"field branches\.0\.assistant_turn\.tool_calls\.0\.function\.name: ",
            id="nested-tool-not-bash",
        ),
        pytest.param(
            nested(name="bash", arguments='{"command": "ls"}'),
            r"field branches\.0\.assistant_turn\.tool_calls\.0\.function\.arguments: ",
            id="nested-arguments-already-a-string",
        ),
    ],
)
def test_a_branch_that_breaks_the_schema_is_refused_with_the_field_named_for_the_reviewer(branch, error):
    with pytest.raises(ValueError, match=r"^Structured branch-plan " + error):
        extract_assistant_turn_plan(fenced(branch))


def test_a_plan_without_its_synthesis_is_refused_as_a_whole():
    with pytest.raises(
        ValueError, match=r"^Structured branch-plan field <plan>: 'synthesis' is a required property"
    ):
        extract_assistant_turn_plan(fenced(flat(), synthesis=None))


@pytest.mark.parametrize(
    "branch, error",
    [
        (flat(commands=["ls", "   "]), r"field branches\.0\.commands\.1: a command must not be blank"),
        (flat(content=" \n"), r"field branches\.0\.content: must not be blank"),
    ],
)
def test_a_blank_command_or_text_is_refused(branch, error):
    with pytest.raises(ValueError, match=r"^Structured branch-plan " + error):
        extract_assistant_turn_plan(fenced(branch))


@pytest.mark.parametrize(
    "branch, call",
    [
        (flat(commands=["pytest -q", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]), 1),
        (flat(commands=["pytest -q && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]), 0),
        (nested(name="bash", arguments={"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}), 0),
    ],
)
def test_an_inserted_turn_that_submits_is_refused_with_its_call_named(branch, call):
    """A turn that submits leaves the policy no turn to sample after it."""
    expected = (
        rf"^Structured branch-plan field branches\.0\.assistant_turn\.tool_calls\.{call}: the inserted turn "
        r"must not submit \(COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\)"
    )

    with pytest.raises(ValueError, match=expected):
        extract_assistant_turn_plan(fenced(branch))


def test_a_bare_json_plan_is_refused_unless_json_only_is_asked_for():
    text = json.dumps({"synthesis": "Probe first.", "branches": [flat()]})

    with pytest.raises(ValueError, match="exactly one branch-plan block; found 0"):
        extract_assistant_turn_plan(text)
    assert extract_assistant_turn_plan(text, json_only=True)["branches"][0]["name"] == "probe"


def test_the_example_plan_in_the_assistant_turn_prompt_is_one_the_parser_accepts():
    """The reviewer copies the example's shape, so a shape that the parser refuses fails every review."""
    prompt = DEFAULT_PROMPTS.reviewer_assistant_turn.format(
        problem="Fix it.", reports="[]", count_rule="At most 2."
    )
    example = prompt.rsplit("```branch-plan\n", 1)[1].split("\n```", 1)[0].replace("<int>", "1")

    plan = extract_assistant_turn_plan("```branch-plan\n" + example + "\n```", round_no=1)

    assert [
        call["function"]["arguments"] for call in plan["branches"][0]["assistant_turn"]["tool_calls"]
    ] == [
        json.dumps({"command": "<bash command>"}),
        json.dumps({"command": "<next bash command>"}),
    ]


def reply(message: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": message}]}


@pytest.mark.parametrize(
    "content",
    ['```branch-plan\n{"synthesis": "x" "branches": []}\n```', "Prose first.\n```branch-plan\n{}\n```\nEnd."],
    ids=["invalid-json-is-not-repaired", "prose-around-the-fence-is-kept"],
)
def test_the_reply_text_is_handed_on_as_the_model_wrote_it(content):
    assert response_text(reply({"role": "assistant", "content": content})) == content


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant", "content": "plan", "tool_calls": [bash("call-1", "pwd")]},
        {"role": "assistant", "content": None},
    ],
    ids=["a-tool-call-is-no-plan", "no-content"],
)
def test_a_reply_without_text_reads_as_empty(message):
    assert response_text(reply(message)) == ""


@pytest.mark.parametrize(
    "response, error",
    [
        ({"choices": []}, "Expected exactly one review response"),
        (
            {"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]},
            "Expected exactly one review response",
        ),
        (reply({"role": "assistant", "content": [{"type": "text", "text": "plan"}]}), "content must be text"),
    ],
    ids=["no-choice", "two-choices", "content-parts"],
)
def test_a_reply_that_is_not_one_text_message_is_refused(response, error):
    with pytest.raises(ValueError, match=error):
        response_text(response)


@pytest.mark.parametrize(
    "count_mode, branches, valid",
    [
        ("fixed", 1, False),
        ("fixed", 2, True),
        ("fixed", 3, False),
        ("adaptive", 0, True),
        ("adaptive", 2, True),
        ("adaptive", 3, False),
    ],
)
def test_the_review_schema_holds_the_branch_count_to_the_width_and_the_count_mode(
    count_mode, branches, valid
):
    review = {"synthesis": "Probe first.", "branches": [flat()] * branches}

    errors = list(Draft202012Validator(flat_review_schema(2, count_mode)).iter_errors(review))

    assert (errors == []) is valid, errors
