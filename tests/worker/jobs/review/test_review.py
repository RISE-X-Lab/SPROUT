"""``branching.review``: what the reviewer is offered and sent, and the plan file that records it all."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.agents.mini_swe.history import available_branch_points
from sprout.worker.jobs.review.branch_plan import PromptRejected
from sprout.worker.jobs.review.branching import (
    TURN_REJECTED,
    Attempt,
    BranchChoice,
    Grade,
    outcome_from_journal,
    review,
    reviewer_context,
)
from sprout.worker.jobs.review.prompts import Prompts
from sprout.worker.jobs.review.request_size import FLOOR
from sprout.worker.jobs.review.structured_review import flat_review_schema
from tests.worker.jobs.review.plain_prompts import plain_prompts, reports_in
from tests.worker.jobs.review.recorded import TURNS, record_attempt, turn

REPORT = {
    "failure_reason": "r",
    "lesson": "l",
    "salvage": "s",
    "branch_candidates": [{"step": 3, "why": "w"}],
}


class Reviewer:
    """A plain transport: the next of ``replies`` answers each request. An error among them is raised."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies, self.prompts = list(replies), []

    def __call__(self, model: str, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class StructuredReviewer(Reviewer):
    """A transport to an endpoint that takes JSON schemas. It records each schema and its name."""

    supports_json_schema = True

    def __init__(self, *replies: str | Exception, overhead: int = 1000) -> None:
        super().__init__(*replies)
        self.overhead, self.schemas, self.plain = overhead, [], []

    def json(self, model: str, prompt: str, *, schema: dict[str, Any], name: str) -> str:
        self.schemas.append((schema, name))
        return super().__call__(model, prompt)

    def json_overhead(self, schema: dict[str, Any], name: str) -> int:
        return self.overhead

    def __call__(self, model: str, prompt: str) -> str:
        self.plain.append(prompt)
        return super().__call__(model, prompt)


def attempt(
    directory: Path, name: str = "parent", *, round_no: int = 0, hint: str = "", turns: Any = TURNS
) -> Attempt:
    journal = record_attempt(directory / name, turns, run=name)
    return Attempt(
        name, outcome_from_journal(journal, run_id=name), Grade(reward=0.0), hint=hint, round_no=round_no
    )


def report(steps: int = 4) -> dict[str, Any]:
    """An analyst report on ``steps`` tool steps, with the sizing record that the analyst adds."""
    return {**REPORT, "steps": steps, "prompt_budget": {"budget": 1, "chars": 1}}


def fenced(*steps: int, base: str = "parent", **fields: Any) -> str:
    """A point plan in its block: a branch of ``base`` at each of ``steps``, with ``fields``."""
    branches = [
        {"name": "p%d" % step, "base": base, "branch_step": step, "why": "w", **fields} for step in steps
    ]
    return "```branch-plan\n" + json.dumps({"synthesis": "s", "branches": branches}) + "\n```"


def flat_turn(step: int) -> dict[str, Any]:
    return {
        "name": "probe",
        "base": "parent",
        "branch_step": step,
        "why": "w",
        "content": "I will check.",
        "commands": ["pytest -q"],
    }


def ask(
    transport: Any,
    attempts: list[Attempt],
    plan_path: Path,
    prompts: Prompts,
    *,
    cases: dict[str, dict[str, Any]] | None = None,
    guidance: str = "none",
    **settings: Any,
) -> list[BranchChoice]:
    return review(
        transport,
        "fixture",
        "Fix it.",
        attempts,
        cases if cases is not None else {a.name: report() for a in attempts},
        round_no=1,
        width=2,
        plan_path=plan_path,
        guidance_mode=guidance,
        prompts=prompts,
        **settings,
    )


def plan_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def statuses(plan: dict[str, Any]) -> list[str]:
    return [a["status"] for a in plan["review_attempts"]]


@pytest.mark.parametrize(
    "steps, eligible, offered",
    [(4, None, [2, 3, 4]), (3, None, [2, 3]), (4, {"parent": {3, 9}}, [3])],
    ids=["every-branch-point", "only-the-steps-the-report-covers", "only-the-eligible-steps"],
)
def test_the_reviewer_is_offered_the_branch_points_that_the_report_covers_and_eligible_allows(
    tmp_path, steps, eligible, offered
):
    reviewer = Reviewer(fenced(offered[0]))

    ask(
        reviewer,
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        cases={"parent": report(steps)},
        eligible=eligible,
    )

    assert (
        reports_in(reviewer.prompts[0])[0]["available_steps"],
        plan_file(tmp_path / "plan.json")["available_steps"],
    ) == (offered, {"parent": offered})


def test_without_a_step_to_offer_the_reviewer_is_not_asked(tmp_path):
    reviewer = Reviewer()

    choices = ask(
        reviewer,
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        cases={"parent": report(1)},
    )

    plan = plan_file(tmp_path / "plan.json")
    assert (choices, reviewer.prompts, plan["review"]) == ([], [], None)
    assert plan["validation_error"].startswith("no eligible snapshot/session pairs")


def test_the_plan_file_records_what_the_reviewer_read_and_the_branches_it_chose(tmp_path):
    prompts = plain_prompts(tmp_path / "p.yaml")

    ask(
        Reviewer(fenced(3, 3)),
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        prompts,
        count_mode="fixed",
        reviewer_max_attempts=2,
    )

    plan = plan_file(tmp_path / "plan.json")
    assert {
        key: plan[key] for key in ("available_steps", "branch_limit", "branch_count_mode", "branch_guidance")
    } == {
        "available_steps": {"parent": [2, 3, 4]},
        "branch_limit": 2,
        "branch_count_mode": "fixed",
        "branch_guidance": "none",
    }
    assert (plan["reports"], plan["reviewer_max_attempts"], plan["structured_review"], statuses(plan)) == (
        {"parent": report()},
        2,
        False,
        ["validated"],
    )
    assert plan["prompts"] == {"path": str(prompts.path), "sha256": prompts.sha256}
    point = {
        "base": "parent",
        "branch_step": 3,
        "snapshot_id": "parent-snap-3",
        "conversation_cut": "turn-2",
        "branch_guidance": "none",
    }
    assert plan["selected_branches"] == [{"run_name": "r1b1-p3", **point}, {"run_name": "r1b2-p3", **point}]


def test_the_reviewer_reads_each_report_without_the_analysts_sizing_and_with_the_attempts_history(tmp_path):
    parent = attempt(tmp_path)
    reviewer = Reviewer(fenced(3))

    ask(reviewer, [parent], tmp_path / "plan.json", plain_prompts(tmp_path / "p.yaml"))

    journal = parent.outcome.journal_path
    assert reports_in(reviewer.prompts[0]) == [
        {
            **REPORT,
            "steps": 4,
            "name": "parent",
            "round": 0,
            "grade": "official Harbor reward=0.0; resolved=False",
            "available_steps": [2, 3, 4],
            "assistant_turn_given": None,
            "native_history": reviewer_context(journal, available_branch_points(journal)),
        }
    ]


def test_a_hint_review_shows_each_attempts_hint_and_no_history(tmp_path):
    attempts = [attempt(tmp_path), attempt(tmp_path, "r1b1-lead", round_no=1, hint="Check the boundary.")]
    reviewer = Reviewer(
        json.dumps(
            {"branches": [{"name": "n", "base": "parent", "branch_step": 2, "why": "w", "hint": "Retry."}]}
        )
    )

    (choice,) = ask(
        reviewer, attempts, tmp_path / "plan.json", plain_prompts(tmp_path / "p.yaml"), guidance="user-hint"
    )

    rows = reports_in(reviewer.prompts[0])
    assert [(r["name"], r["hint_given"], "native_history" in r) for r in rows] == [
        ("parent", None, False),
        ("r1b1-lead", "Check the boundary.", False),
    ]
    assert (choice.hint, plan_file(tmp_path / "plan.json")["selected_branches"][0]["hint"]) == (
        "Retry.",
        "Retry.",
    )


@pytest.mark.parametrize("count_mode, rule", [("adaptive", "AT MOST 2"), ("fixed", "EXACTLY 2")])
def test_the_reviewer_is_sent_the_count_rule_of_its_mode(tmp_path, count_mode, rule):
    reviewer = Reviewer(fenced(2, 3))

    ask(
        reviewer,
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        count_mode=count_mode,
    )

    assert reviewer.prompts[0].endswith("\nCOUNT " + rule)


def test_a_fixed_review_refuses_a_plan_with_another_count_without_padding_it(tmp_path):
    choices = ask(
        Reviewer(fenced(2)),
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        count_mode="fixed",
        reviewer_max_attempts=1,
    )

    plan = plan_file(tmp_path / "plan.json")
    assert (choices, plan["validation_error"]) == (
        [],
        "fixed branch count requires exactly 2 branches; reviewer returned 1",
    )


def test_an_empty_plan_is_taken_without_asking_again(tmp_path):
    reviewer = Reviewer(fenced())

    choices = ask(
        reviewer,
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        reviewer_max_attempts=3,
    )

    assert (choices, len(reviewer.prompts), plan_file(tmp_path / "plan.json")["selected_branches"]) == (
        [],
        1,
        [],
    )


def test_an_assistant_turn_review_on_a_structured_endpoint_asks_for_the_flat_schema_within_the_budget(
    tmp_path,
):
    reviewer = StructuredReviewer(
        json.dumps({"synthesis": "s", "branches": [flat_turn(3), flat_turn(4)]}), overhead=1000
    )

    choices = ask(
        reviewer,
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        guidance="assistant-turn",
        count_mode="fixed",
        prompt_chars=200_000,
    )

    plan = plan_file(tmp_path / "plan.json")
    assert (reviewer.schemas, reviewer.plain, [c.checkpoint.step for c in choices]) == (
        [(flat_review_schema(2, "fixed"), "sprout_branches")],
        [],
        [3, 4],
    )
    assert (plan["structured_review"], plan["structured_output"]) == (
        True,
        {"schema": flat_review_schema(2, "fixed"), "request_budget": 200_000, "overhead_chars": 1000},
    )
    assert plan["prompt_budget"]["budget"] == 199_000


@pytest.mark.parametrize(
    "guidance, reply",
    [
        ("none", fenced(3)),
        (
            "user-hint",
            json.dumps(
                {"branches": [{"name": "n", "base": "parent", "branch_step": 3, "why": "w", "hint": "h"}]}
            ),
        ),
    ],
)
def test_only_an_assistant_turn_review_asks_a_structured_endpoint_for_a_schema(tmp_path, guidance, reply):
    reviewer = StructuredReviewer(reply)

    ask(
        reviewer,
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        guidance=guidance,
    )

    assert (
        reviewer.schemas,
        len(reviewer.plain),
        plan_file(tmp_path / "plan.json")["structured_review"],
    ) == ([], 1, False)


def big_attempts(directory: Path) -> list[Attempt]:
    """A parent and two branches, each of 30 steps whose outputs fill the native history."""
    turns = [turn("cmd %d" % n, outputs=["o" * 3000]) for n in range(30)]
    return [
        attempt(directory, "parent", turns=turns),
        attempt(directory, "r1b1-a", round_no=1, turns=turns),
        attempt(directory, "r1b2-b", round_no=1, turns=turns),
    ]


def big_cases(attempts: list[Attempt]) -> dict[str, dict[str, Any]]:
    return {a.name: {**report(30), "branch_candidates": [{"step": 20, "why": "w"}]} for a in attempts}


def test_a_refused_request_is_sent_again_within_half_the_budget_and_below_the_refused_size(tmp_path):
    attempts = big_attempts(tmp_path)
    reviewer = Reviewer(PromptRejected("HTTP 400 Bad Request: too long"), fenced(20))

    choices = ask(
        reviewer,
        attempts,
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        cases=big_cases(attempts),
        prompt_chars=300_000,
    )

    plan, (first, second) = plan_file(tmp_path / "plan.json"), reviewer.prompts
    assert (len(choices), statuses(plan), len(first) > 150_000) == (
        1,
        ["request_rejected", "validated"],
        True,
    )
    sizing = plan["prompt_budget"]
    assert (sizing["budget"], sizing["chars_after"], sizing["halvings"]) == (150_000, len(second), 1)
    assert [(fit["budget"], fit["chars_after"]) for fit in sizing["refused"]] == [(300_000, len(first))]


def test_a_refused_request_far_below_its_budget_is_sent_again_below_its_own_size(tmp_path):
    """A coarse ladder of stages would otherwise send the refused prompt again."""
    attempts = big_attempts(tmp_path)
    reviewer = Reviewer(PromptRejected("HTTP 400 Bad Request: too long"), fenced(20))

    ask(
        reviewer,
        attempts,
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        cases=big_cases(attempts),
        prompt_chars=4_000_000,
    )

    plan, (first, second) = plan_file(tmp_path / "plan.json"), reviewer.prompts
    assert (len(first) < 2_000_000, plan["prompt_budget"]["budget"], len(second) < len(first)) == (
        True,
        len(first) - 1,
        True,
    )


def test_a_request_refused_after_two_halvings_ends_the_review_with_the_refusal_recorded(tmp_path):
    attempts = big_attempts(tmp_path)
    reviewer = Reviewer(*[PromptRejected("HTTP 400 Bad Request: too long")] * 3)

    choices = ask(
        reviewer,
        attempts,
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        cases=big_cases(attempts),
        prompt_chars=300_000,
    )

    plan = plan_file(tmp_path / "plan.json")
    assert (choices, statuses(plan), plan["prompt_budget"]["halvings"]) == (
        [],
        ["request_rejected", "request_rejected", "request_error"],
        2,
    )
    assert plan["validation_error"] == "reviewer failed: HTTP 400 Bad Request: too long"
    assert len(reviewer.prompts[0]) > len(reviewer.prompts[1]) > len(reviewer.prompts[2])


def test_a_refusal_at_the_floor_ends_the_review_without_another_request(tmp_path):
    attempts = big_attempts(tmp_path)
    reviewer = Reviewer(PromptRejected("HTTP 400 Bad Request: too long"))

    ask(
        reviewer,
        attempts,
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        cases=big_cases(attempts),
        prompt_chars=1000,
    )

    plan = plan_file(tmp_path / "plan.json")
    assert (plan["prompt_budget"]["stage"], statuses(plan), len(reviewer.prompts)) == (
        FLOOR,
        ["request_error"],
        1,
    )


def test_a_correction_keeps_its_feedback_when_its_refused_prompt_is_sent_smaller(tmp_path):
    attempts = big_attempts(tmp_path)
    reviewer = Reviewer(fenced(99), PromptRejected("HTTP 400 Bad Request: too long"), fenced(20))

    ask(
        reviewer,
        attempts,
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        cases=big_cases(attempts),
        prompt_chars=300_000,
        reviewer_max_attempts=2,
    )

    assert statuses(plan_file(tmp_path / "plan.json")) == [
        "validation_error",
        "request_rejected",
        "validated",
    ]
    assert "Validator error: branch 1 has no eligible checkpoint at parent:99" in reviewer.prompts[2]


def fenced_turn(step: int, command: str) -> str:
    branch = {**flat_turn(step), "commands": [command]}
    return "```branch-plan\n" + json.dumps({"synthesis": "s", "branches": [branch]}) + "\n```"


def flags(command: str) -> Any:
    """A turn check that flags each turn that runs ``command``."""

    def check(choice: BranchChoice) -> dict[str, Any]:
        assert choice.assistant_turn is not None
        return {"leak": command in choice.assistant_turn["tool_calls"][0]["function"]["arguments"]}

    return check


def test_a_flagged_turn_goes_back_naming_only_its_branch_and_a_clean_rewrite_is_taken_with_its_check(
    tmp_path,
):
    reviewer = Reviewer(fenced_turn(3, "cat SECRET"), fenced_turn(3, "pytest -q"))

    (choice,) = ask(
        reviewer,
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        guidance="assistant-turn",
        turn_check=flags("SECRET"),
        turn_resamples=2,
        reviewer_max_attempts=1,
    )

    plan = plan_file(tmp_path / "plan.json")
    assert (choice.turn_check, statuses(plan), plan["turn_rejections"], plan["turn_resamples"]) == (
        {"leak": False},
        ["turn_rejected", "validated"],
        1,
        2,
    )
    feedback = plan["review_attempts"][0]["feedback"]
    assert TURN_REJECTED.format(branches="1") in feedback and "SECRET" not in feedback
    assert plan["selected_branches"][0]["turn_check"] == {"leak": False}


def test_a_turn_still_flagged_when_its_resamples_run_out_is_taken_with_its_check(tmp_path):
    reviewer = Reviewer(fenced_turn(3, "cat SECRET"), fenced_turn(3, "cat SECRET"))

    (choice,) = ask(
        reviewer,
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        guidance="assistant-turn",
        turn_check=flags("SECRET"),
        turn_resamples=1,
    )

    plan = plan_file(tmp_path / "plan.json")
    assert (choice.turn_check, statuses(plan), plan["turn_rejections"]) == (
        {"leak": True},
        ["turn_rejected", "validated"],
        1,
    )


def test_without_a_turn_check_no_turn_carries_a_check(tmp_path):
    (choice,) = ask(
        Reviewer(fenced_turn(3, "pytest -q")),
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        guidance="assistant-turn",
    )

    plan = plan_file(tmp_path / "plan.json")
    assert (choice.turn_check, "turn_resamples" in plan, "turn_check" in plan["selected_branches"][0]) == (
        None,
        False,
        False,
    )


@pytest.mark.parametrize("resamples", [-1, 1.0, True])
def test_a_resample_count_that_is_not_a_non_negative_integer_is_refused(tmp_path, resamples):
    with pytest.raises(ValueError, match="turn_resamples must be a non-negative integer"):
        ask(
            Reviewer(),
            [attempt(tmp_path)],
            tmp_path / "plan.json",
            plain_prompts(tmp_path / "p.yaml"),
            guidance="assistant-turn",
            turn_check=flags("x"),
            turn_resamples=resamples,
        )


@pytest.mark.parametrize(
    "guidance, rejected, valid, hint",
    [
        ("assistant-turn", fenced_turn(9, "pytest -q"), fenced_turn(3, "pytest -q"), "FORMAT HINT."),
        ("none", fenced(9), fenced(3), ""),
    ],
)
def test_only_an_assistant_turn_correction_repeats_the_shape_of_a_turn(
    tmp_path, guidance, rejected, valid, hint
):
    """A rejected turn most often has the wrong shape. A point plan has no turn to shape."""
    reviewer = Reviewer(rejected, valid)

    ask(
        reviewer,
        [attempt(tmp_path)],
        tmp_path / "plan.json",
        plain_prompts(tmp_path / "p.yaml"),
        guidance=guidance,
    )

    assert plan_file(tmp_path / "plan.json")["review_attempts"][0]["feedback"].endswith(
        "Do not add explanations instead of the plan.\n" + hint
    )
