"""A reviewer request over its character budget: which messages each stage keeps, and how the stages apply."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from sprout.worker.jobs.review.branching import branch_count_rule
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS
from sprout.worker.jobs.review.request_size import (
    FLOOR,
    STAGES,
    Stage,
    Window,
    clip_messages,
    fit_review_prompt,
    omission_marker,
    reduce_row,
    window_messages,
)

REPORT = {"failure_reason": "The empty input is unhandled.", "lesson": "Handle it.", "salvage": "The parser."}


def history(turns: int, *, result_chars: int = 3000, reasoning_chars: int = 7000) -> dict[str, Any]:
    """A native history: the system and task prompts, then ``turns`` turns of one call and its result."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please solve this issue: " + "task " * 300},
    ]
    counts = {}
    for step in range(1, turns + 1):
        call = {
            "id": "c%d" % step,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": "cmd %d" % step})},
        }
        reasoning = ("think-%d " % step) * (reasoning_chars // 8)
        messages.append(
            {
                "role": "assistant",
                "content": "Step %d." % step,
                "reasoning_content": reasoning,
                "tool_calls": [call],
            }
        )
        output = ("out-%d " % step) * (result_chars // 6)
        messages.append(
            {"role": "tool", "tool_call_id": "c%d" % step, "content": json.dumps({"output": output})}
        )
        counts[str(step)] = len(messages)
    return {"messages": messages, "prefix_message_counts": counts, "tools": [], "workspace": {"cwd": "/app"}}


def row(name: str, round_no: int, turns: int, candidates: list[int]) -> dict[str, Any]:
    """A reviewer row: the analyst report on an attempt of ``turns`` steps, and its native history."""
    return {
        **REPORT,
        "steps": turns,
        "branch_candidates": [{"step": step, "why": "useful state"} for step in candidates],
        "name": name,
        "round": round_no,
        "grade": "official Harbor reward=0.0; resolved=False",
        "available_steps": list(range(1, turns + 1)),
        "native_history": history(turns),
    }


def shown(reduced: list[dict[str, Any]], original: list[dict[str, Any]]) -> list[int | tuple[int, int]]:
    """The position in ``original``, from 1, of each message kept, and the range of each marker."""
    return [tuple(m["positions"]) if m["role"] == "omitted" else original.index(m) + 1 for m in reduced]


@pytest.mark.parametrize(
    "tail, positions, omitted",
    [
        (4, [1, 2, (3, 36), *range(37, 45), (45, 78), 79, 80, 81, 82], 68),
        (0, [1, 2, (3, 36), *range(37, 45), (45, 82)], 72),
    ],
    ids=["with-a-tail", "the-gap-reaches-the-end"],
)
def test_a_window_keeps_the_task_the_messages_around_each_cut_and_the_tail_and_marks_each_gap(
    tail, positions, omitted
):
    native = history(40)
    cut = native["prefix_message_counts"]["20"]

    reduced, count = window_messages(native["messages"], [cut], Window(before=6, after=2, tail=tail))

    assert (shown(reduced, native["messages"]), count, cut) == (positions, omitted, 42)


def test_windows_that_overlap_leave_one_gap_between_them():
    native = history(30)

    reduced, _ = window_messages(native["messages"], [20, 22, 24], Window(before=4, after=2, tail=2))

    assert shown(reduced, native["messages"]) == [1, 2, (3, 16), *range(17, 27), (27, 60), 61, 62]


@pytest.mark.parametrize(
    "first, last, marker",
    [
        (
            3,
            36,
            "[34 messages omitted here to fit this request: positions 3-36 of the full history. "
            "prefix_message_counts count the full history and are still exact; "
            "the next message is position 37.]",
        ),
        (
            82,
            82,
            "[1 message omitted here to fit this request: "
            "positions 82-82 of the full history, through its end. "
            "prefix_message_counts count the full history and are still exact.]",
        ),
    ],
    ids=["a-gap-before-a-message", "a-gap-through-the-end"],
)
def test_a_marker_says_how_many_messages_it_stands_for_and_where_the_next_one_is(first, last, marker):
    assert omission_marker(first, last, 82) == {
        "role": "omitted",
        "omitted_messages": last - first + 1,
        "positions": [first, last],
        "content": marker,
    }


def test_clipping_keeps_the_head_the_tail_and_the_newlines_of_each_long_field_and_counts_the_fields():
    long_text = "head\n" * 4 + "tail\n" * 4  # 40 characters
    messages = [
        {"role": "user", "content": long_text},
        {
            "role": "assistant",
            "content": long_text,
            "reasoning_content": long_text,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": long_text}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": long_text},
        {"role": "tool", "tool_call_id": "c2", "content": "short"},
    ]
    before = deepcopy(messages)
    clipped_text = "head\nhead\nhead\nhead\n …[10 chars elided]… tail\ntail\n"

    clipped, count = clip_messages(messages, 30)

    assert (count, messages) == (3, before)
    assert clipped == [
        before[0],
        {
            **before[1],
            "reasoning_content": clipped_text,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": clipped_text}}
            ],
        },
        {**before[2], "content": clipped_text},
        before[3],
    ]


def test_a_reduced_row_keeps_its_report_and_its_prefix_counts_and_says_what_it_dropped():
    """The window keeps the task prompts, two messages up to the cut at step 10, one after it and the last.

    The candidate at step 99 has no cut. Four of the six messages kept are clipped."""
    original = row("r1b1-a", 1, 20, [10, 99])

    reduced, stats = reduce_row(original, Window(before=2, after=1, tail=1), 1000)

    native = reduced["native_history"]
    assert {key: value for key, value in reduced.items() if key != "native_history"} == {
        key: value for key, value in original.items() if key != "native_history"
    }
    assert (native["prefix_message_counts"], native["tools"], native["workspace"]) == (
        original["native_history"]["prefix_message_counts"],
        [],
        {"cwd": "/app"},
    )
    assert stats == {"messages": 42, "omitted": 36, "retained": 6, "clipped": 4}


@pytest.mark.parametrize(
    "native, window, clip",
    [
        (None, Window(1, 1, 1), 100),
        ({"messages": [], "prefix_message_counts": {}}, Window(1, 1, 1), 100),
        (history(3), None, None),
    ],
    ids=["no-history", "no-messages", "a-stage-that-does-nothing"],
)
def test_a_row_that_a_stage_cannot_reduce_is_returned_unchanged_without_counts(native, window, clip):
    original = {**REPORT, "name": "parent", "round": 0, "native_history": native}

    assert reduce_row(original, window, clip) == (original, {})


TEMPLATE = DEFAULT_PROMPTS.reviewer_point
COUNT_RULE = branch_count_rule("adaptive", 3)


def render(rows: list[dict[str, Any]]) -> str:
    return TEMPLATE.format(
        problem="Fix it.", reports=json.dumps(rows, indent=1, ensure_ascii=False), count_rule=COUNT_RULE
    )


def fit(rows: list[dict[str, Any]], budget: int) -> tuple[str, dict[str, Any]]:
    return fit_review_prompt(TEMPLATE, problem="Fix it.", rows=rows, count_rule=COUNT_RULE, budget=budget)


def prompt_at(rows: list[dict[str, Any]], stage: Stage) -> str:
    """The prompt after ``stage``: the round-0 history reduced by its parent window, the others by theirs."""
    return render(
        [reduce_row(r, stage.parent if r["round"] == 0 else stage.branches, stage.clip)[0] for r in rows]
    )


ROWS = [row("parent", 0, 60, [50, 30]), row("r1b1-a", 1, 70, [66]), row("r1b2-b", 1, 72, [70, 65])]


def test_a_request_within_its_budget_is_sent_unchanged():
    rows = [row("parent", 0, 6, [4]), row("r1b1-a", 1, 8, [7])]

    prompt, record = fit(rows, 10**7)

    assert (prompt, record) == (
        render(rows),
        {
            "budget": 10**7,
            "chars_before": len(prompt),
            "chars_after": len(prompt),
            "within_budget": True,
            "stage": None,
            "attempts": {},
        },
    )


def test_each_stage_of_the_reduction_gives_a_smaller_request_than_the_one_before():
    sizes = [len(render(ROWS))] + [len(prompt_at(ROWS, stage)) for stage in STAGES]

    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes), sizes


@pytest.mark.parametrize("stage", STAGES, ids=[stage.name for stage in STAGES])
def test_a_request_over_its_budget_is_reduced_by_the_first_stage_that_fits(stage):
    budget = len(prompt_at(ROWS, stage))

    prompt, record = fit(ROWS, budget)

    assert (prompt, record["stage"], record["within_budget"], record["chars_after"]) == (
        prompt_at(ROWS, stage),
        stage.name,
        True,
        budget,
    )


@pytest.mark.parametrize(
    "stage, attempts",
    [(STAGES[0], {"r1b1-a", "r1b2-b"}), (STAGES[1], {"parent", "r1b1-a", "r1b2-b"})],
    ids=["first-the-branches", "then-the-parent"],
)
def test_the_branch_histories_are_reduced_before_the_parents(stage, attempts):
    """A branch repeats the parent's prefix up to its cut."""
    _, record = fit(ROWS, len(prompt_at(ROWS, stage)))

    assert set(record["attempts"]) == attempts


def test_at_the_floor_each_history_is_one_marker_and_the_request_is_sent_even_over_its_budget():
    prompt, record = fit(ROWS, 1000)

    totals = {r["name"]: len(r["native_history"]["messages"]) for r in ROWS}
    floor = [
        {
            **r,
            "native_history": {
                **r["native_history"],
                "messages": [omission_marker(1, totals[r["name"]], totals[r["name"]])],
            },
        }
        for r in ROWS
    ]
    assert (prompt, record["stage"], record["within_budget"]) == (render(floor), FLOOR, False)
    assert record["attempts"] == {
        name: {"messages": total, "omitted": total, "retained": 0} for name, total in totals.items()
    }


@pytest.mark.parametrize(
    "template",
    [DEFAULT_PROMPTS.reviewer_assistant_turn, DEFAULT_PROMPTS.reviewer_point],
    ids=["assistant-turn", "point"],
)
def test_the_reviewer_prompts_explain_the_two_markers_that_a_reduction_writes(template):
    """Without the explanation, the reviewer reads a marker as a message of the attempt."""
    clipped, _ = clip_messages([{"role": "tool", "content": "x" * 40}], 30)

    assert 'role "%s" is a controller marker' % omission_marker(1, 2, 2)["role"] in " ".join(template.split())
    assert ("…[10 chars elided]…" in clipped[0]["content"], '"…[N chars elided]…"' in template) == (
        True,
        True,
    )
