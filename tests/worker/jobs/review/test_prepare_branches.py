"""A reviewer's plan resolved against the recorded histories: exact checkpoints and cuts, or an error."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import training_messages
from sprout.worker.agents.mini_swe.history import available_branch_points
from sprout.worker.checkpoint.rollback import Checkpoint
from sprout.worker.jobs.review.branch_plan import ReviewerPlanError
from sprout.worker.jobs.review.branching import (
    Attempt,
    BranchChoice,
    Grade,
    outcome_from_journal,
    prepare_branches,
    reviewer_context,
    selected_prefix,
)
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS
from tests.worker.jobs.review.recorded import BASH_TOOL, WORKSPACE, record_attempt, turn


def recorded(directory: Path, name: str) -> Attempt:
    journal = record_attempt(directory / name, run=name)
    return Attempt(name, outcome_from_journal(journal, run_id=name), Grade())


@pytest.fixture
def parent(tmp_path: Path) -> Attempt:
    """A failed attempt with branch points at the steps 2, 3 and 4."""
    return recorded(tmp_path, "parent")


def checkpoints_of(*attempts: Attempt) -> dict[str, dict[int, Checkpoint]]:
    return {attempt.name: available_branch_points(attempt.outcome.journal_path) for attempt in attempts}


def resolve(
    plan: Any, *attempts: Attempt, guidance: str, limit: int = 2, count_mode: str = "adaptive"
) -> list[BranchChoice]:
    return prepare_branches(
        plan,
        limit=limit,
        round_no=1,
        attempts={attempt.name: attempt for attempt in attempts},
        checkpoints=checkpoints_of(*attempts),
        count_mode=count_mode,
        guidance_mode=guidance,
    )


def point(step: Any, **fields: Any) -> dict[str, Any]:
    return {
        "name": "lead",
        "base": "parent",
        "branch_step": step,
        "why": "Relevant work is present.",
        **fields,
    }


def reviewer_turn(command: str = "pytest -q", call_id: str = "reviewer-r1-b1-c1") -> dict[str, Any]:
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }
    return {"role": "assistant", "content": "I will run the tests first.", "tool_calls": [call]}


def resolved(choice: BranchChoice) -> tuple[Any, ...]:
    return (
        choice.run_name,
        choice.base.name,
        choice.checkpoint.step,
        choice.checkpoint.snapshot_id,
        choice.cut,
    )


def test_a_hint_branch_resolves_to_its_checkpoint_and_the_turn_that_cuts_its_conversation(parent):
    plan = {"branches": [point(2, hint="  Check the boundary.  "), point(4, name="late", hint="Retry.")]}

    choices = resolve(plan, parent, guidance="user-hint")

    assert [resolved(c) for c in choices] == [
        ("r1b1-lead", "parent", 2, "parent-snap-2", "turn-1"),
        ("r1b2-late", "parent", 4, "parent-snap-4", "turn-3"),
    ]
    assert [(c.hint, c.why, c.assistant_turn) for c in choices] == [
        ("  Check the boundary.  ", "Relevant work is present.", None),
        ("Retry.", "Relevant work is present.", None),
    ]


def test_a_point_only_branch_carries_no_hint_and_no_turn(parent):
    (choice,) = resolve({"branches": [point(3)]}, parent, guidance="none")

    assert (resolved(choice), choice.hint, choice.assistant_turn) == (
        ("r1b1-lead", "parent", 3, "parent-snap-3", "turn-2"),
        "",
        None,
    )


def test_an_assistant_turn_branch_carries_the_turn_that_the_actor_receives(parent):
    (choice,) = resolve(
        {"branches": [point(2, assistant_turn=reviewer_turn())]}, parent, guidance="assistant-turn"
    )

    assert (resolved(choice), choice.hint, choice.assistant_turn) == (
        ("r1b1-lead", "parent", 2, "parent-snap-2", "turn-1"),
        "",
        reviewer_turn(),
    )


def test_branches_may_share_a_point_or_take_different_bases(tmp_path, parent):
    child = recorded(tmp_path, "r1b1-first")
    plan = {"branches": [point(2), point(2, name="again"), point(3, name="child", base="r1b1-first")]}

    choices = resolve(plan, parent, child, guidance="none", limit=3)

    assert [resolved(c) for c in choices] == [
        ("r1b1-lead", "parent", 2, "parent-snap-2", "turn-1"),
        ("r1b2-again", "parent", 2, "parent-snap-2", "turn-1"),
        ("r1b3-child", "r1b1-first", 3, "r1b1-first-snap-3", "turn-2"),
    ]


@pytest.mark.parametrize(
    "guidance, count_mode, plan, error",
    [
        ("none", "adaptive", {"points": []}, "reviewer must return a branches list"),
        ("none", "adaptive", [point(2)], "reviewer must return a branches list"),
        (
            "none",
            "fixed",
            {"branches": [point(2)]},
            "fixed branch count requires exactly 2 branches; reviewer returned 1",
        ),
        ("none", "adaptive", {"branches": [point(2)] * 3}, "reviewer returned 3 branches above limit 2"),
        ("none", "adaptive", {"branches": ["parent@2"]}, "branch 1 is not an object"),
        (
            "none",
            "adaptive",
            {"branches": [point(2, base="elsewhere")]},
            "branch 1 names an unknown base 'elsewhere'",
        ),
        ("none", "adaptive", {"branches": [point(1)]}, "branch 1 has no eligible checkpoint at parent:1"),
        (
            "none",
            "adaptive",
            {"branches": [point(2, command="ls")]},
            "none branches may contain only point-selection fields",
        ),
        (
            "assistant-turn",
            "adaptive",
            {"branches": [point(2, hint="x", assistant_turn=reviewer_turn())]},
            "must not include a user hint",
        ),
        (
            "assistant-turn",
            "adaptive",
            {"branches": [point(2, assistant_turn=reviewer_turn(call_id="call-1"))]},
            r"branch 1 \(parent@2\): assistant_turn tool call ids must be non-empty and unique",
        ),
        (
            "assistant-turn",
            "adaptive",
            {
                "branches": [
                    point(
                        2,
                        assistant_turn=reviewer_turn(
                            "git add -A && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                        ),
                    )
                ]
            },
            r"^branch 1 \(parent@2\): a reviewer-authored turn never submits; "
            r"command 1 contains COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT$",
        ),
        (
            "user-hint",
            "adaptive",
            {"branches": [point(2, hint="x", assistant_turn=reviewer_turn())]},
            "assistant_turn requires",
        ),
        (
            "user-hint",
            "adaptive",
            {"branches": [point(2, hint=" ")]},
            "branch 1 must supply a non-empty hint",
        ),
        ("user-hint", "adaptive", {"branches": [point(2)]}, "branch 1 must supply a non-empty hint"),
    ],
    ids=[
        "no-branches",
        "a-list",
        "fixed-count-missed",
        "above-the-limit",
        "branch-not-an-object",
        "unknown-base",
        "a-step-inside-a-turn",
        "none-with-guidance",
        "assistant-turn-with-a-hint",
        "assistant-turn-reusing-a-call-id",
        "assistant-turn-that-submits",
        "user-hint-with-a-turn",
        "a-blank-hint",
        "no-hint",
    ],
)
def test_a_plan_the_reviewer_can_correct_is_refused_with_a_reviewer_plan_error(
    parent, guidance, count_mode, plan, error
):
    with pytest.raises(ReviewerPlanError, match=error):
        resolve(plan, parent, guidance=guidance, count_mode=count_mode)


def test_a_json_true_is_no_step_even_where_step_1_is_a_branch_point(tmp_path):
    """``True == 1`` in Python, and a reviewer's JSON can hold ``true``."""
    journal = record_attempt(tmp_path / "parent", [turn("ls"), turn("pytest -q")])
    one_call_turns = Attempt("parent", outcome_from_journal(journal, run_id="parent"), Grade())

    with pytest.raises(ReviewerPlanError, match="branch 1 has no eligible checkpoint at parent:True"):
        resolve({"branches": [point(True)]}, one_call_turns, guidance="none")


@pytest.mark.parametrize(
    "guidance, count_mode, error",
    [
        ("none", "exact", "unknown branch count mode: 'exact'"),
        ("prose", "adaptive", "unknown branch guidance mode: 'prose'"),
    ],
)
def test_an_unknown_mode_is_the_callers_error_not_the_reviewers(parent, guidance, count_mode, error):
    with pytest.raises(ValueError, match=error) as raised:
        resolve({"branches": []}, parent, guidance=guidance, count_mode=count_mode)

    assert not isinstance(raised.value, ReviewerPlanError)


@pytest.mark.parametrize(
    "guidance, branch, error",
    [
        ("user-hint", point(2, hint="Check it."), "branch 1 has no native conversation cut at parent:2"),
        ("none", point(2), "No exact mini prefix at the selected checkpoint"),
    ],
)
def test_a_checkpoint_without_its_session_is_an_error_of_the_run_not_of_the_reviewer(
    parent, guidance, branch, error
):
    checkpoints = {"parent": {2: Checkpoint(step=2, seq=1, snapshot_id="parent-snap-2", session_ckpt=None)}}

    with pytest.raises(ValueError, match=error) as raised:
        prepare_branches(
            {"branches": [branch]},
            limit=1,
            round_no=1,
            attempts={"parent": parent},
            checkpoints=checkpoints,
            guidance_mode=guidance,
        )

    assert not isinstance(raised.value, ReviewerPlanError)


def test_the_prefix_at_a_checkpoint_is_the_native_history_up_to_its_closed_turn(parent):
    journal = parent.outcome.journal_path

    prefix = selected_prefix(journal, available_branch_points(journal)[3])

    assert [e["type"] for e in prefix] == ["mini.session"] + ["mini.message"] * 5 + ["mini.turn"] + [
        "mini.message"
    ] * 2 + ["mini.turn"]
    assert prefix[-1] == {"type": "mini.turn", "turn_id": "turn-2", "call_ids": ["call-3"]}


@pytest.mark.parametrize(
    "checkpoint",
    [
        Checkpoint(step=3, seq=1, snapshot_id="parent-snap-3", session_ckpt=None),
        Checkpoint(step=1, seq=1, snapshot_id="parent-snap-1", session_ckpt="ses-1"),
    ],
    ids=["no-session", "no-closed-turn-at-the-step"],
)
def test_a_checkpoint_without_an_exact_prefix_is_refused(parent, checkpoint):
    with pytest.raises(ValueError, match="No exact mini prefix at the selected checkpoint"):
        selected_prefix(parent.outcome.journal_path, checkpoint)


def test_the_reviewer_reads_the_history_up_to_the_last_eligible_step_and_where_each_step_cuts_it(parent):
    journal = parent.outcome.journal_path
    points = available_branch_points(journal)

    context = reviewer_context(journal, {2: points[2], 3: points[3]})

    assert context == {
        "messages": training_messages(selected_prefix(journal, points[3])),
        "prefix_message_counts": {"2": 5, "3": 7},
        "tools": BASH_TOOL,
        "workspace": WORKSPACE,
    }
    assert [m["role"] for m in context["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "tool",
    ]


def test_without_an_eligible_step_the_reviewer_reads_no_history(parent):
    assert reviewer_context(parent.outcome.journal_path, {}) == {"messages": [], "prefix_message_counts": {}}


def test_a_step_that_the_history_does_not_close_a_turn_at_is_refused(parent):
    journal = parent.outcome.journal_path
    points = available_branch_points(journal)
    inside_a_turn = Checkpoint(step=1, seq=1, snapshot_id="parent-snap-1", session_ckpt="ses-1")

    with pytest.raises(ValueError, match="does not cover every eligible checkpoint"):
        reviewer_context(journal, {1: inside_a_turn, 4: points[4]})


def example_plan(prompt: str) -> dict[str, Any]:
    """The plan that a reviewer prompt shows as its example, with its placeholders filled for ``parent``."""
    marker = "```branch-plan\n" if "```branch-plan\n" in prompt else "Return ONLY a JSON object, no prose:"
    example = prompt.rsplit(marker, 1)[1].split("\n```", 1)[0]
    return json.loads(example.replace("<int>", "2").replace("<attempt name>", "parent"))


@pytest.mark.parametrize(
    "template, guidance",
    [(DEFAULT_PROMPTS.reviewer_point, "none"), (DEFAULT_PROMPTS.reviewer_user_hint, "user-hint")],
    ids=["point", "user-hint"],
)
def test_the_example_plan_of_each_packaged_reviewer_prompt_is_one_its_guidance_accepts(
    parent, template, guidance
):
    """The reviewer copies the example's fields, so a field that its guidance refuses fails every review."""
    prompt = template.format(problem="Fix it.", reports="[]", count_rule="At most 1.")

    (choice,) = resolve(example_plan(prompt), parent, guidance=guidance, limit=1)

    assert (choice.checkpoint.step, choice.assistant_turn) == (2, None)
