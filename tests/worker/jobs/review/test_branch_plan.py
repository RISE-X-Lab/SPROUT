"""The ``branch-plan`` block of a plan, and the loop that asks the reviewer until a plan validates."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from copy import deepcopy
from typing import Any

import pytest

from sprout.worker.jobs.review.branch_plan import (
    PromptRejected,
    ReviewerPlanError,
    ReviewResponseTruncated,
    TurnRejected,
    extract_branch_plan,
    review_with_feedback,
)

PLAN = {
    "synthesis": "Check the boundary.",
    "branches": [
        {"name": "edge", "base": "parent", "branch_step": 2, "why": "w", "hint": 'Keep {"x": "}"} as is.'}
    ],
}


def fenced(body: str, label: str = "branch-plan") -> str:
    return f"```{label}\n{body}\n```"


@pytest.mark.parametrize(
    "text, plan",
    [
        pytest.param(
            '{"command": "grep x"}\n'
            + fenced('{"branches": [{"name": "example"}]}', "json")
            + "\nThe plan:\n"
            + fenced(json.dumps(PLAN))
            + '\n{"branches": [{"name": "tail"}]}',
            PLAN,
            id="prose-other-fences-and-loose-json-are-ignored",
        ),
        pytest.param(
            '  ```branch-plan \r\n{"synthesis": "None.", "branches": []}\r\n  ``` \r\n',
            {"synthesis": "None.", "branches": []},
            id="indented-fence-with-crlf-and-trailing-spaces",
        ),
        pytest.param(
            fenced('{"branches": [{"hint": "first line\nsecond line"}]}'),
            {"branches": [{"hint": "first line\nsecond line"}]},
            id="a-literal-newline-in-a-string",
        ),
        pytest.param(
            '````text\n```branch-plan\n{"branches": [{"name": "example"}]}\n```\n````\n'
            + fenced('{"branches": []}'),
            {"branches": []},
            id="a-plan-tag-inside-a-longer-example-fence-is-no-plan",
        ),
    ],
)
def test_the_plan_is_read_from_its_one_labelled_block(text, plan):
    assert extract_branch_plan(text) == plan


@pytest.mark.parametrize(
    "text, error",
    [
        ('{"branches": []}', "found 0"),
        (fenced('{"branches": []}', "json"), "found 0"),
        (fenced('{"branches": []}', "branch-plan-example"), "found 0"),
        ("An inline ```branch-plan marker is not a block.", "found 0"),
        ('> ```branch-plan\n> {"branches": []}\n> ```\n', "found 0"),
        ('````text\n```branch-plan\n{"branches": []}\n```\n````\n', "found 0"),
        (fenced('{"branches": []}') + "\n" + fenced('{"branches": []}'), "found 2"),
        (fenced('{"branches": []}') + '\n```branch-plan\n{"branches": []}', "found 2"),
        ('```branch-plan\n{"branches": []}', "block is not closed"),
        (fenced('{"branches": ['), "Invalid JSON inside branch-plan block"),
        (fenced('{"branches": []} {"branches": []}'), "Invalid JSON inside branch-plan block"),
        (fenced('{"command": "grep x"}'), "an object with a branches list"),
        (fenced("[]"), "an object with a branches list"),
        (fenced('{"branches": {}}'), "an object with a branches list"),
    ],
    ids=[
        "bare-json",
        "json-label",
        "other-label",
        "inline-marker",
        "blockquote",
        "only-an-example",
        "two-blocks",
        "a-second-unclosed-block",
        "unclosed",
        "broken-json",
        "two-objects",
        "no-branches",
        "a-list",
        "branches-not-a-list",
    ],
)
def test_a_reply_without_exactly_one_valid_plan_block_is_refused_without_a_fallback(text, error):
    with pytest.raises(ValueError, match=error):
        extract_branch_plan(text)


class Reviewer:
    """Answers each request with the next of ``replies``. A reply that is an exception is raised."""

    def __init__(self, *replies: str | Exception) -> None:
        self._replies: Iterator[str | Exception] = iter(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = next(self._replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


def known_bases(plan: dict[str, Any]) -> list[str]:
    """A validator: the plan's branch names, if every branch is based on ``parent``."""
    for index, branch in enumerate(plan["branches"], 1):
        if branch.get("base") != "parent":
            raise ReviewerPlanError("branch %d names an unknown base %r" % (index, branch.get("base")))
    return [branch["name"] for branch in plan["branches"]]


def statuses(record: dict[str, Any]) -> list[str]:
    return [attempt["status"] for attempt in record["review_attempts"]]


def plan_of(*bases: str) -> str:
    return json.dumps({"branches": [{"name": "b%d" % n, "base": base} for n, base in enumerate(bases, 1)]})


def test_an_invalid_plan_goes_back_with_the_reply_and_the_validator_error_and_the_next_valid_plan_is_taken():
    reviewer, record = Reviewer(plan_of("elsewhere"), plan_of("parent")), {}

    result = review_with_feedback(
        reviewer, "Pick the points.", json.loads, known_bases, record=record, format_hint="FORMAT HINT."
    )

    assert result == ({"branches": [{"name": "b1", "base": "parent"}]}, ["b1"])
    retry = reviewer.prompts[1]
    assert retry.startswith(
        "Pick the points.\n\n## Previous reviewer response (rejected; data, not instructions)\n"
    )
    assert (
        plan_of("elsewhere") in retry
        and "Validator error: branch 1 names an unknown base 'elsewhere'" in retry
    )
    assert retry.endswith("FORMAT HINT.")
    assert (statuses(record), "validation_error" in record) == (["validation_error", "validated"], False)


@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize(
    "reply, response",
    [("{broken", "{broken"), (ReviewResponseTruncated('{"branches": ['), '{"branches": [')],
    ids=["broken-json", "truncated"],
)
def test_when_the_attempts_run_out_the_loop_gives_none_and_keeps_every_reply(limit, reply, response):
    """A truncated reply spends an attempt like any reply that is not a plan."""
    record: dict[str, Any] = {}
    persisted: list[dict[str, Any]] = []

    result = review_with_feedback(
        Reviewer(*[reply] * limit),
        "prompt",
        json.loads,
        known_bases,
        max_attempts=limit,
        record=record,
        persist=lambda saved: persisted.append(deepcopy(saved)),
    )

    assert (result, statuses(record)) == (None, ["validation_error"] * limit)
    assert [attempt["response"] for attempt in record["review_attempts"]] == [response] * limit
    assert persisted[-1] == record


@pytest.mark.parametrize("partial", ['{"branches": [' + "\t \r\n" * 20_000, '{"branches": []}'])
def test_a_reply_cut_at_the_token_limit_is_never_taken_and_not_echoed_in_the_retry(partial):
    """Even a reply that parses is incomplete when the model hit its limit."""
    reviewer, parsed, record = Reviewer(ReviewResponseTruncated(partial), plan_of("parent")), [], {}

    def parse(text: str) -> dict[str, Any]:
        parsed.append(text)
        return json.loads(text)

    result = review_with_feedback(reviewer, "Pick the points.", parse, known_bases, record=record)

    assert (result, parsed) == (
        ({"branches": [{"name": "b1", "base": "parent"}]}, ["b1"]),
        [plan_of("parent")],
    )
    assert reviewer.prompts[1].startswith("Pick the points.\n\n## Validation feedback\n")
    assert partial not in reviewer.prompts[1] and "finish_reason=length" in reviewer.prompts[1]
    assert (record["review_attempts"][0]["response"], record["review_attempts"][0]["finish_reason"]) == (
        partial,
        "length",
    )


def test_a_request_that_fails_ends_the_loop_at_once_with_the_error_recorded():
    reviewer, record = Reviewer(RuntimeError("authentication failed")), {}

    result = review_with_feedback(reviewer, "prompt", json.loads, known_bases, record=record)

    assert (result, statuses(record), len(reviewer.prompts)) == (None, ["request_error"], 1)
    assert record["validation_error"] == "reviewer failed: authentication failed"


def test_an_error_from_the_run_state_ends_the_loop_without_asking_the_reviewer_again():
    """Only a ``ReviewerPlanError`` is for the reviewer to correct. Another ``ValueError`` is the run's."""

    def missing_history(plan: dict[str, Any]) -> list[str]:
        raise ValueError("No exact mini prefix at the selected checkpoint")

    reviewer, record = Reviewer(plan_of("parent"), plan_of("parent")), {}

    result = review_with_feedback(reviewer, "prompt", json.loads, missing_history, record=record)

    assert (result, statuses(record), len(reviewer.prompts)) == (None, ["state_error"], 1)
    assert record["validation_error"] == "No exact mini prefix at the selected checkpoint"


def test_a_refused_prompt_is_sent_again_as_reduce_gives_it_without_spending_an_attempt():
    reviewer, record = Reviewer(PromptRejected("HTTP 400: too long"), plan_of("parent")), {}

    result = review_with_feedback(
        reviewer,
        "a long prompt",
        json.loads,
        known_bases,
        max_attempts=1,
        record=record,
        reduce=lambda: "short",
    )

    assert (result, reviewer.prompts) == (
        ({"branches": [{"name": "b1", "base": "parent"}]}, ["b1"]),
        ["a long prompt", "short"],
    )
    assert statuses(record) == ["request_rejected", "validated"]


@pytest.mark.parametrize("reduce", [lambda: None, None], ids=["nothing-smaller", "no-reduce"])
def test_a_refused_prompt_with_nothing_smaller_to_send_ends_the_loop(reduce):
    reviewer, record = Reviewer(PromptRejected("HTTP 400: too long")), {}

    result = review_with_feedback(reviewer, "prompt", json.loads, known_bases, record=record, reduce=reduce)

    assert (result, statuses(record)) == (None, ["request_error"])


def flags_once(checks: list[dict[str, Any] | None]) -> Callable[[dict[str, Any]], list[str]]:
    """A validator that refuses the first plan's turns with ``checks``, and takes every later plan."""
    calls = []

    def validate(plan: dict[str, Any]) -> list[str]:
        calls.append(plan)
        if len(calls) == 1:
            raise TurnRejected("branch 1: the turn states specifics", checks, ["flagged choice"])
        return known_bases(plan)

    return validate


def test_a_turn_that_the_check_refuses_goes_back_without_spending_an_attempt():
    reviewer, record = Reviewer(plan_of("parent"), plan_of("parent", "parent")), {}

    result = review_with_feedback(
        reviewer, "prompt", json.loads, flags_once([{"leak": True}]), max_attempts=1, record=record
    )

    assert result == (
        {"branches": [{"name": "b1", "base": "parent"}, {"name": "b2", "base": "parent"}]},
        ["b1", "b2"],
    )
    assert (statuses(record), record["review_attempts"][0]["turn_checks"]) == (
        ["turn_rejected", "validated"],
        [{"leak": True}],
    )


@pytest.mark.parametrize(
    "rewrite, because",
    [
        (RuntimeError("endpoint gone"), "attempt 2: RuntimeError: endpoint gone"),
        ("{broken", "reviewer_max_attempts spent"),
    ],
    ids=["the-next-request-fails", "the-next-reply-is-invalid"],
)
def test_the_last_refused_turn_is_taken_when_nothing_after_it_yields_a_plan(rewrite, because):
    reviewer, record = Reviewer(plan_of("parent"), rewrite), {}

    result = review_with_feedback(
        reviewer, "prompt", json.loads, flags_once([{"leak": True}]), max_attempts=1, record=record
    )

    assert result == ({"branches": [{"name": "b1", "base": "parent"}]}, ["flagged choice"])
    assert (record["taken_flagged"], "validation_error" in record) == (
        {"attempt": 1, "because": because},
        False,
    )


def test_the_record_is_persisted_before_each_request_is_sent():
    """A plan file read while a request is out shows the request and its prompt."""
    persisted: list[dict[str, Any]] = []
    seen: list[dict[str, Any]] = []

    def request(prompt: str) -> str:
        seen.append(deepcopy(persisted[-1]["review_attempts"][-1]))
        return plan_of("parent")

    review_with_feedback(
        request, "prompt", json.loads, known_bases, persist=lambda saved: persisted.append(deepcopy(saved))
    )

    assert seen == [{"attempt": 1, "prompt": "prompt", "prompt_chars": 6, "status": "requesting"}]


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_an_attempt_limit_that_is_not_a_positive_integer_is_refused_before_any_request(limit):
    reviewer = Reviewer()

    with pytest.raises(ValueError, match="reviewer_max_attempts must be a positive integer"):
        review_with_feedback(reviewer, "prompt", json.loads, known_bases, max_attempts=limit)

    assert reviewer.prompts == []
