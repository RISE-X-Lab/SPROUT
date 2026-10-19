"""``Continuation`` and ``prepare``: what a branch adds after its restored history, at every entry point."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import pytest

from sprout.contract.continuation import Continuation, prepare, render_branch_note, render_reasoning
from sprout.contract.mini_history import validate_assistant_turn
from tests.contract.builders import REVIEWER_TEXT, assistant_turn, bash_call, bash_tools

HINT = "Check the boundary condition before editing."
OBJECT_SHAPE = (
    "A continuation is an object with guidance and, for its mode, "
    "assistant_turn (and optionally assistant_turn_rendering) or hint"
)


def restored_history(*call_ids: str) -> list[dict[str, Any]]:
    """mini's native history entries of a closed turn for each of ``call_ids``."""
    entries: list[dict[str, Any]] = [
        {"type": "mini.session"},
        {"type": "mini.message", "message": {"role": "user", "content": "Fix the bug."}},
    ]
    for identifier in call_ids:
        entries += [
            {
                "type": "mini.message",
                "message": {"role": "assistant", "content": "", "tool_calls": [bash_call("ls", identifier)]},
            },
            {
                "type": "mini.message",
                "message": {"role": "tool", "tool_call_id": identifier, "content": "output"},
            },
            {"type": "mini.turn", "call_ids": [identifier]},
        ]
    return entries


@pytest.mark.parametrize(
    "value, message",
    [
        pytest.param(
            {"guidance": "rewrite"}, "Branch guidance must be assistant-turn, user-hint or none", id="unknown"
        ),
        pytest.param({}, "Branch guidance must be assistant-turn, user-hint or none", id="no-guidance"),
        pytest.param(
            {"guidance": "assistant-turn"}, "An assistant-turn branch needs its assistant_turn", id="no-turn"
        ),
        pytest.param(
            {"guidance": "assistant-turn", "assistant_turn": "printf check"},
            "An assistant-turn branch needs its assistant_turn",
            id="turn-not-an-object",
        ),
        pytest.param(
            {"guidance": "none", "assistant_turn": assistant_turn()},
            "Only an assistant-turn branch carries an assistant turn",
            id="turn-without-assistant-turn",
        ),
        pytest.param({"guidance": "user-hint"}, "A user-hint branch needs a non-empty hint", id="no-hint"),
        pytest.param(
            {"guidance": "user-hint", "hint": " \n"}, "A user-hint branch needs a non-empty hint", id="blank"
        ),
        pytest.param(
            {"guidance": "user-hint", "hint": 7}, "A user-hint branch needs a non-empty hint", id="number"
        ),
        pytest.param(
            {"guidance": "none", "hint": HINT},
            "Only a user-hint branch carries a hint",
            id="hint-without-mode",
        ),
        pytest.param(
            {
                "guidance": "assistant-turn",
                "assistant_turn": assistant_turn(),
                "assistant_turn_rendering": "think",
            },
            "assistant_turn_rendering must be content or reasoning",
            id="unknown-rendering",
        ),
        pytest.param(
            {
                "guidance": "assistant-turn",
                "assistant_turn": assistant_turn(),
                "assistant_turn_rendering": None,
            },
            "assistant_turn_rendering must be content or reasoning",
            id="null-rendering",
        ),
        pytest.param(
            {"guidance": "none", "assistant_turn_rendering": "reasoning"},
            "Only an assistant-turn branch has an assistant_turn_rendering",
            id="rendering-without-turn",
        ),
        pytest.param(
            {"guidance": "user-hint", "hint": HINT, "assistant_turn_rendering": "reasoning"},
            "Only an assistant-turn branch has an assistant_turn_rendering",
            id="rendering-of-a-hint",
        ),
        pytest.param({"guidance": "none", "prompt": "Try again."}, OBJECT_SHAPE, id="unknown-key"),
        pytest.param("none", OBJECT_SHAPE, id="text"),
        pytest.param(None, OBJECT_SHAPE, id="absent"),
    ],
)
def test_a_continuation_that_does_not_fit_its_guidance_is_refused(value, message):
    with pytest.raises(ValueError, match="^%s$" % re.escape(message)):
        Continuation.from_dict(value)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"guidance": "none"}, id="none"),
        pytest.param({"guidance": "user-hint", "hint": HINT}, id="user-hint"),
        pytest.param({"guidance": "assistant-turn", "assistant_turn": assistant_turn()}, id="assistant-turn"),
        pytest.param(
            {
                "guidance": "assistant-turn",
                "assistant_turn": assistant_turn(),
                "assistant_turn_rendering": "reasoning",
            },
            id="reasoning-rendering",
        ),
    ],
)
def test_each_continuation_round_trips_through_its_json_object(body):
    continuation = Continuation.from_dict(body)

    assert continuation.to_dict() == body


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(
            {
                "guidance": "assistant-turn",
                "assistant_turn": assistant_turn(),
                "assistant_turn_rendering": "content",
            },
            {"guidance": "assistant-turn", "assistant_turn": assistant_turn()},
            id="assistant-turn",
        ),
        pytest.param(
            {"guidance": "none", "assistant_turn_rendering": "content"}, {"guidance": "none"}, id="none"
        ),
    ],
)
def test_the_default_rendering_stays_out_of_the_json_object(body, expected):
    """A body without the field keeps its bytes, and so its request hash and its idempotency key."""
    continuation = Continuation.from_dict(body)

    assert continuation.to_dict() == expected


def test_a_continuation_shares_its_turn_with_no_body():
    body = {"guidance": "assistant-turn", "assistant_turn": assistant_turn()}
    continuation = Continuation.from_dict(body)

    body["assistant_turn"]["content"] = "edited after submission"
    continuation.to_dict()["assistant_turn"]["content"] = "edited after reading"

    assert continuation.to_dict() == {"guidance": "assistant-turn", "assistant_turn": assistant_turn()}


@pytest.mark.parametrize(
    "continuation, origin",
    [
        pytest.param(Continuation("none"), {"branch_guidance": "none"}, id="none"),
        pytest.param(
            Continuation("user-hint", hint=HINT),
            {"branch_guidance": "user-hint", "actor_hint": HINT, "hint_delivery": "reviewer-direct"},
            id="user-hint",
        ),
        pytest.param(
            Continuation("assistant-turn", assistant_turn=assistant_turn()),
            {"branch_guidance": "assistant-turn", "assistant_turn_source": "reviewer"},
            id="assistant-turn",
        ),
        pytest.param(
            Continuation(
                "assistant-turn", assistant_turn=assistant_turn(), assistant_turn_rendering="reasoning"
            ),
            {
                "branch_guidance": "assistant-turn",
                "assistant_turn_source": "reviewer",
                "assistant_turn_rendering": "reasoning",
            },
            id="reasoning-rendering",
        ),
    ],
)
def test_the_origin_records_what_the_actor_is_given(continuation, origin):
    """A branch's ``fork.origin`` journal event holds these fields."""
    assert continuation.origin() == origin


def test_the_reasoning_rendering_moves_the_text_of_the_turn_into_its_reasoning():
    turn = assistant_turn("pytest -q", "inserted-1")

    rendered = render_reasoning(turn)

    assert rendered == {
        "role": "assistant",
        "content": "",
        "reasoning_content": REVIEWER_TEXT,
        "tool_calls": [bash_call("pytest -q", "inserted-1")],
    }


def test_the_reasoning_rendering_shares_no_call_with_its_turn():
    turn = assistant_turn("pytest -q", "inserted-1")

    render_reasoning(turn)["tool_calls"][0]["id"] = "edited"

    assert turn == assistant_turn("pytest -q", "inserted-1")


def test_the_reasoning_rendering_of_a_valid_turn_is_a_valid_reasoning_turn():
    """mini's runtime checks an inserted turn with ``allow_reasoning`` before it runs the turn."""
    rendered = render_reasoning(assistant_turn())

    checked = validate_assistant_turn(rendered, allow_reasoning=True)

    assert checked == rendered


def test_the_branch_note_starts_with_the_hint_verbatim():
    hint = "## Idea\n\nThe bug is in `parse()`; check the  empty case."

    note = render_branch_note(hint)

    assert note.startswith(hint + "\n\n")


@pytest.mark.parametrize(
    "commit_required, last_line",
    [
        (True, "Commit your changes when finished."),
        (False, "prior observations or treat a suggested cause as already established."),
    ],
)
def test_only_a_note_that_requires_a_commit_ends_with_the_commit_instruction(commit_required, last_line):
    note = render_branch_note(HINT, commit_required=commit_required)

    assert note.splitlines()[-1] == last_line


@pytest.mark.parametrize("hint", ["", " \n"])
def test_a_blank_hint_has_no_branch_note(hint):
    with pytest.raises(ValueError, match="^branch hint must be a non-empty string$"):
        render_branch_note(hint)


@pytest.mark.parametrize(
    "continuation, commit_required, expected",
    [
        pytest.param(Continuation("none"), False, ("", {"resume_without_hint": True}), id="none"),
        pytest.param(
            Continuation("user-hint", hint=HINT), False, (render_branch_note(HINT), {}), id="user-hint"
        ),
        pytest.param(
            Continuation("user-hint", hint=HINT),
            True,
            (render_branch_note(HINT, commit_required=True), {}),
            id="user-hint-with-commit",
        ),
        pytest.param(
            Continuation("assistant-turn", assistant_turn=assistant_turn("pytest -q", "inserted-1")),
            True,
            ("", {"assistant_turn": assistant_turn("pytest -q", "inserted-1")}),
            id="assistant-turn",
        ),
        pytest.param(
            Continuation(
                "assistant-turn",
                assistant_turn=assistant_turn("pytest -q", "inserted-1"),
                assistant_turn_rendering="reasoning",
            ),
            False,
            ("", {"assistant_turn": render_reasoning(assistant_turn("pytest -q", "inserted-1"))}),
            id="reasoning-rendering",
        ),
    ],
)
def test_each_guidance_gives_the_prompt_and_the_options_that_mini_reads(
    continuation, commit_required, expected
):
    """Only a user-hint branch appends a prompt. The options go into the branch's ``RunSpec.extra``."""
    result = prepare(continuation, restored_history("parent-1"), commit_required=commit_required)

    assert result == expected


@pytest.mark.parametrize("rendering", ["content", "reasoning"])
def test_an_inserted_turn_may_not_reuse_a_call_id_of_the_restored_history(rendering):
    continuation = Continuation(
        "assistant-turn",
        assistant_turn=assistant_turn(identifier="parent-1"),
        assistant_turn_rendering=rendering,
    )

    with pytest.raises(ValueError, match="unique across history"):
        prepare(continuation, restored_history("parent-1"))


@pytest.mark.parametrize("rendering", ["content", "reasoning"])
def test_the_author_writes_the_turn_in_its_own_form_whatever_the_rendering(rendering):
    continuation = Continuation(
        "assistant-turn",
        assistant_turn=render_reasoning(assistant_turn()),
        assistant_turn_rendering=rendering,
    )

    with pytest.raises(ValueError, match="^assistant_turn requires exactly role, content and tool_calls$"):
        prepare(continuation, restored_history())


def test_an_inserted_turn_that_breaks_the_recorded_tool_schema_is_refused():
    continuation = Continuation("assistant-turn", assistant_turn=assistant_turn("printf long"))

    with pytest.raises(ValueError, match="^assistant_turn does not match the recorded actor tool schema"):
        prepare(continuation, restored_history(), tools=bash_tools(maxLength=3))


def test_an_inserted_turn_that_fits_the_recorded_tool_schema_is_given_to_mini():
    continuation = Continuation("assistant-turn", assistant_turn=assistant_turn("ls"))

    result = prepare(continuation, restored_history(), tools=bash_tools(maxLength=3))

    assert result == ("", {"assistant_turn": assistant_turn("ls")})


def test_mini_gets_a_copy_of_the_turn_that_the_continuation_does_not_share():
    continuation = Continuation("assistant-turn", assistant_turn=assistant_turn())
    before = deepcopy(continuation.to_dict())

    _, options = prepare(continuation, restored_history())
    options["assistant_turn"]["tool_calls"][0]["id"] = "edited"

    assert continuation.to_dict() == before
