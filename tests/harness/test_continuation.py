"""The one definition of what a branch adds after its restored history."""

from copy import deepcopy

import pytest

from sprout.harness.agents.mini_swe.continuation import (
    RENDERINGS, Continuation, prepare, render_branch_note, render_reasoning)
from sprout.harness.agents.mini_swe.history import validate_assistant_turn
from tests.harness.test_assistant_turn import assistant_turn


def prefix(*call_ids):
    """A restored native history whose assistant turns used these call ids."""
    entries = [{"type": "mini.session"},
               {"type": "mini.message", "message": {"role": "user", "content": "task"}}]
    for identifier in call_ids:
        entries.append({"type": "mini.message", "message": {
            "role": "assistant", "content": "", "tool_calls": [{"id": identifier}]}})
        entries.append({"type": "mini.message", "message": {
            "role": "tool", "tool_call_id": identifier, "content": "output"}})
        entries.append({"type": "mini.turn", "call_ids": [identifier]})
    return entries


def test_each_guidance_becomes_the_prompt_and_options_mini_reads():
    turn = assistant_turn("pytest -q", identifier="inserted-1")
    prompt, options = prepare(Continuation("assistant-turn", assistant_turn=turn), prefix("parent-1"))
    assert prompt == "" and options == {"assistant_turn": turn}
    assert options["assistant_turn"] is not turn, "mini gets its own copy"

    prompt, options = prepare(Continuation("none"), prefix("parent-1"))
    assert prompt == "" and options == {"resume_without_hint": True}

    prompt, options = prepare(Continuation("user-hint", hint="Check the boundary."), prefix("parent-1"),
                              commit_required=True)
    assert options == {}
    assert prompt == render_branch_note("Check the boundary.", commit_required=True)


def test_the_reasoning_rendering_moves_the_reviewers_text_into_the_turns_reasoning():
    turn = assistant_turn("pytest -q", identifier="inserted-1")
    original = deepcopy(turn)
    continuation = Continuation("assistant-turn", assistant_turn=turn, assistant_turn_rendering="reasoning")
    prompt, options = prepare(continuation, prefix("parent-1"))
    assert prompt == "" and options == {"assistant_turn": render_reasoning(turn)}
    assert options["assistant_turn"] == {"role": "assistant", "content": "", "reasoning_content": turn["content"],
                                         "tool_calls": turn["tool_calls"]}
    assert turn == original, "neither the validation nor the rendering edits the caller's turn"
    options["assistant_turn"]["tool_calls"][0]["id"] = "edited"
    assert turn == original, "mini gets its own copy"
    assert validate_assistant_turn(render_reasoning(turn), allow_reasoning=True) == render_reasoning(turn), (
        "the rendering is a turn mini's runtime runs")


@pytest.mark.parametrize("rendering", RENDERINGS)
def test_the_reviewer_writes_the_turn_in_its_own_form_whatever_the_rendering(rendering):
    rendered = render_reasoning(assistant_turn())
    with pytest.raises(ValueError, match="assistant_turn requires exactly role, content and tool_calls"):
        prepare(Continuation("assistant-turn", assistant_turn=rendered, assistant_turn_rendering=rendering), prefix())


@pytest.mark.parametrize("rendering", RENDERINGS)
def test_an_inserted_turn_may_not_reuse_a_call_id_of_the_history_it_continues(rendering):
    turn = assistant_turn(identifier="parent-1")
    with pytest.raises(ValueError, match="unique across history"):
        prepare(Continuation("assistant-turn", assistant_turn=turn, assistant_turn_rendering=rendering),
                prefix("parent-1"))


@pytest.mark.parametrize("rendering", RENDERINGS)
def test_the_recorded_tool_schema_is_applied_when_the_caller_has_it(rendering):
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string", "maxLength": 3}},
        "required": ["command"], "additionalProperties": False}}}]
    with pytest.raises(ValueError, match="recorded actor tool schema"):
        prepare(Continuation("assistant-turn", assistant_turn=assistant_turn("printf long"),
                             assistant_turn_rendering=rendering), prefix(), tools=tools)
    prepare(Continuation("assistant-turn", assistant_turn=assistant_turn("ls"),
                         assistant_turn_rendering=rendering), prefix(), tools=tools)


@pytest.mark.parametrize("value, message", [
    ({"guidance": "rewrite"}, "must be assistant-turn, user-hint or none"),
    ({}, "must be assistant-turn, user-hint or none"),
    ({"guidance": "assistant-turn"}, "needs its assistant_turn"),
    ({"guidance": "none", "assistant_turn": assistant_turn()}, "Only an assistant-turn branch"),
    ({"guidance": "user-hint"}, "non-empty hint"),
    ({"guidance": "user-hint", "hint": "  "}, "non-empty hint"),
    ({"guidance": "none", "hint": "x"}, "Only a user-hint branch"),
    ({"guidance": "none", "prompt": "x"}, "is an object with guidance"),
    ("none", "is an object with guidance"),
    ({"guidance": "none", "assistant_turn_rendering": "reasoning"},
     "Only an assistant-turn branch has an assistant_turn_rendering"),
    ({"guidance": "user-hint", "hint": "h", "assistant_turn_rendering": "reasoning"},
     "Only an assistant-turn branch has an assistant_turn_rendering"),
    ({"guidance": "assistant-turn", "assistant_turn": assistant_turn(), "assistant_turn_rendering": "think"},
     "assistant_turn_rendering must be content or reasoning"),
    ({"guidance": "assistant-turn", "assistant_turn": assistant_turn(), "assistant_turn_rendering": None},
     "assistant_turn_rendering must be content or reasoning"),
])
def test_a_continuation_that_cannot_be_honoured_is_refused(value, message):
    with pytest.raises(ValueError, match=message):
        Continuation.from_dict(value)


def test_the_wire_form_round_trips_without_sharing_the_callers_turn():
    body = {"guidance": "assistant-turn", "assistant_turn": assistant_turn()}
    original = deepcopy(body)
    continuation = Continuation.from_dict(body)
    body["assistant_turn"]["content"] = "edited after submission"
    assert continuation.to_dict() == original
    assert Continuation.from_dict({"guidance": "none"}).to_dict() == {"guidance": "none"}
    assert Continuation("user-hint", hint="h").to_dict() == {"guidance": "user-hint", "hint": "h"}


def test_a_rendering_round_trips_and_the_default_is_never_written():
    body = {"guidance": "assistant-turn", "assistant_turn": assistant_turn(), "assistant_turn_rendering": "reasoning"}
    continuation = Continuation.from_dict(body)
    assert continuation.assistant_turn_rendering == "reasoning"
    assert continuation.to_dict() == body and Continuation.from_dict(continuation.to_dict()) == continuation
    plain = {"guidance": "assistant-turn", "assistant_turn": assistant_turn()}
    assert Continuation.from_dict(plain).assistant_turn_rendering == "content"
    # An existing body keeps its bytes, and so its request hash and idempotency key.
    assert Continuation.from_dict({**plain, "assistant_turn_rendering": "content"}).to_dict() == plain
    assert Continuation.from_dict({"guidance": "none", "assistant_turn_rendering": "content"}).to_dict() == {
        "guidance": "none"}
    with pytest.raises(ValueError, match="Only an assistant-turn branch has"):
        Continuation("none", assistant_turn_rendering="reasoning")


def test_the_origin_records_what_the_actor_was_given():
    assert Continuation("none").origin() == {"branch_guidance": "none"}
    assert Continuation("assistant-turn", assistant_turn=assistant_turn()).origin() == {
        "branch_guidance": "assistant-turn", "assistant_turn_source": "reviewer"}
    assert Continuation("user-hint", hint="h").origin() == {
        "branch_guidance": "user-hint", "actor_hint": "h", "hint_delivery": "reviewer-direct"}
    rendered = Continuation("assistant-turn", assistant_turn=assistant_turn(), assistant_turn_rendering="reasoning")
    assert rendered.origin() == {"branch_guidance": "assistant-turn", "assistant_turn_source": "reviewer",
                                 "assistant_turn_rendering": "reasoning"}
    assert "assistant_turn_rendering" not in Continuation(
        "assistant-turn", assistant_turn=assistant_turn(), assistant_turn_rendering="content").origin()


@pytest.mark.parametrize("commit_required", [True, False])
def test_the_note_carries_the_hint_once_and_states_only_what_applies(commit_required):
    hint = "Inspect the relevant code and reproduce the issue."
    note = render_branch_note(hint, commit_required=commit_required)
    assert note.count(hint) == 1
    assert ("Commit your changes" in note) is commit_required
    with pytest.raises(ValueError):
        render_branch_note("  ")
