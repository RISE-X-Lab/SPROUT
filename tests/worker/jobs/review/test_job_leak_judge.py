"""A review with ``leak_judge``: each repair turn is checked before it is taken. A flagged one goes back."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.jobs.review.branching import TURN_REJECTED
from sprout.worker.jobs.review.job import jev_judge, run_review
from sprout.worker.jobs.review.leak.judge import HiddenText, JevError, LeakJudgement
from tests.worker.jobs.review.review_model import HIDDEN, ReviewModel, failed_roots, repair_plan, review_spec

FIXTURE = "FIXTURE-ABC-123"
LEAKING = f"python -c \"print(convert('{FIXTURE}'))\""


class Judge:
    """Jev for one review: a turn with ``FIXTURE`` scores 0.95, any other 0.1. It records what it read."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error, self.seen, self.closed = error, [], 0

    def judge(
        self, turn: dict[str, Any], *, task_statement: str, hidden: HiddenText, visible: str
    ) -> LeakJudgement:
        self.seen.append({"task_statement": task_statement, "hidden": hidden, "visible": visible})
        if self.error is not None:
            raise self.error
        score = 0.95 if FIXTURE in json.dumps(turn) else 0.1
        return LeakJudgement(score, ((FIXTURE, score),), "jev-1.13.0")

    def close(self) -> None:
        self.closed += 1


def judged(
    tmp_path: Path, judge: Judge, *, replies: list[Any], last: str = "pytest -q", **changes: Any
) -> tuple[dict[str, Any], ReviewModel, dict[str, Any]]:
    """A review of one point and one sample, whose parent's grade names a test patch that holds ``FIXTURE``.

    The sample answers with ``replies`` first, then proposes ``last``.
    Returns the result, the model and the plan."""
    root = tmp_path / "artifacts"
    (tmp_path / "test.patch").write_text(
        f"+++ b/tests/test_convert.py\n+    assert convert('{FIXTURE}') == 'ok'\n"
    )
    parents = failed_roots(root, 1, **HIDDEN, test_patch=str(tmp_path / "test.patch"))
    model = ReviewModel(
        points={"points": [("root-1", 2)]}, proposals=last, replies={"point1-candidate1": replies}
    )
    result = run_review(
        review_spec(root, parents, points=1, candidates=1, **changes),
        tmp_path / "review",
        transport_factory=model.factory,
        judge_factory=lambda config: judge,
    )
    return result, model, json.loads((tmp_path / "review" / "plan-point1-candidate1.json").read_text())


def statuses(plan: dict[str, Any]) -> list[str]:
    return [a["status"] for a in plan["review_attempts"]]


def commands(candidate: dict[str, Any]) -> list[str]:
    return [
        json.loads(c["function"]["arguments"])["command"] for c in candidate["assistant_turn"]["tool_calls"]
    ]


def test_a_flagged_turn_goes_back_told_only_that_it_states_what_the_actor_could_not_know(tmp_path):
    """The flagged spans come from the hidden tests, and naming them would show the reviewer more of them."""
    result, model, plan = judged(
        tmp_path,
        Judge(),
        replies=[repair_plan("root-1", 2, LEAKING)],
        leak_judge={"resamples": 2},
        max_attempts=1,
    )

    (candidate,) = result["points"][0]["candidates"]
    assert (
        commands(candidate),
        candidate["leak_judge"]["leak"],
        candidate["leak_judge"]["jev"]["score"],
    ) == (["pytest -q"], False, 0.1)
    assert (statuses(plan), plan["turn_rejections"]) == (["turn_rejected", "validated"], 1)
    feedback = plan["review_attempts"][0]["feedback"]
    assert TURN_REJECTED.format(branches="1") in feedback and FIXTURE not in feedback
    assert FIXTURE not in model.prompts_of("point1-candidate1")[1].split("## Validation feedback")[1]


def test_the_judge_reads_the_test_patch_the_verdict_the_task_and_h_s_and_is_closed_after_its_call(tmp_path):
    judge = Judge()

    judged(tmp_path, judge, replies=[], leak_judge={})

    (seen,) = judge.seen
    assert (seen["task_statement"], seen["hidden"]) == (
        "Fix the failing test.",
        HiddenText(
            f"+++ b/tests/test_convert.py\n+    assert convert('{FIXTURE}') == 'ok'\n",
            ("tests/test_hidden.py::test_secret_behaviour",),
            "target tests passed (FAIL_TO_PASS): 50%",
        ),
    )
    assert ("echo early > answer" in seen["visible"], "echo late" in seen["visible"], judge.closed) == (
        True,
        False,
        1,
    )


def test_a_turn_still_flagged_when_its_resamples_run_out_is_taken_with_its_check(tmp_path):
    result, _, plan = judged(tmp_path, Judge(), replies=[], last=LEAKING, leak_judge={"resamples": 1})

    (candidate,) = result["points"][0]["candidates"]
    assert (commands(candidate), candidate["leak_judge"]["leak"], candidate["leak_judge"]["jev"]["leak"]) == (
        [LEAKING],
        True,
        True,
    )
    assert (statuses(plan), plan["turn_rejections"], plan["turn_resamples"]) == (
        ["turn_rejected", "validated"],
        1,
        1,
    )


def test_an_empty_leak_judge_config_sends_a_flagged_turn_back_twice(tmp_path):
    _, _, plan = judged(tmp_path, Judge(), replies=[], last=LEAKING, leak_judge={})

    assert (plan["turn_resamples"], statuses(plan)) == (2, ["turn_rejected", "turn_rejected", "validated"])


def test_a_hidden_test_name_is_flagged_even_when_jev_finds_nothing(tmp_path):
    result, _, plan = judged(
        tmp_path, Judge(), replies=[], last="pytest -q -k test_secret_behaviour", leak_judge={"resamples": 0}
    )

    (candidate,) = result["points"][0]["candidates"]
    assert (
        candidate["leak_judge"]["leak"],
        candidate["leak_judge"]["leak_terms"],
        candidate["leak_terms"],
    ) == (
        True,
        ["test_secret_behaviour"],
        ["test_secret_behaviour"],
    )
    assert (candidate["leak_judge"]["jev"]["leak"], statuses(plan)) == (False, ["validated"])


@pytest.mark.parametrize(
    "error",
    [JevError("HTTP 503"), TypeError("float() argument must be a string or a real number, not 'NoneType'")],
    ids=["jev-unavailable", "an-answer-of-the-wrong-shape"],
)
def test_a_judge_that_fails_leaves_the_turn_unchecked_and_sends_nothing_back(tmp_path, error):
    result, _, plan = judged(tmp_path, Judge(error=error), replies=[], leak_judge={})

    (candidate,) = result["points"][0]["candidates"]
    assert (candidate["leak_judge"]["leak"], candidate["leak_judge"]["jev"], statuses(plan)) == (
        None,
        {"error": f"{type(error).__name__}: {error}"},
        ["validated"],
    )


@pytest.mark.parametrize(
    "rewrite",
    [RuntimeError("endpoint gone"), "no plan here"],
    ids=["the-rewrite-request-fails", "the-rewrite-is-invalid"],
)
def test_a_rewrite_that_yields_no_plan_falls_back_to_the_flagged_turn(tmp_path, rewrite):
    result, _, plan = judged(
        tmp_path,
        Judge(),
        replies=[repair_plan("root-1", 2, LEAKING), rewrite],
        leak_judge={"resamples": 2},
        max_attempts=1,
    )

    (candidate,) = result["points"][0]["candidates"]
    assert (commands(candidate), candidate["leak_judge"]["leak"]) == ([LEAKING], True)
    assert (plan["taken_flagged"]["attempt"], "validation_error" in plan) == (1, False)


def test_without_a_leak_judge_no_judge_is_made_and_no_candidate_carries_a_check(tmp_path):
    root = tmp_path / "artifacts"
    model = ReviewModel(points={"points": [("root-1", 2)]}, proposals=LEAKING)

    def no_judge(config: dict[str, Any]) -> Any:
        raise AssertionError("the review made a judge")

    result = run_review(
        review_spec(root, failed_roots(root, 1, **HIDDEN), points=1, candidates=1),
        tmp_path / "review",
        transport_factory=model.factory,
        judge_factory=no_judge,
    )

    plan = json.loads((tmp_path / "review" / "plan-point1-candidate1.json").read_text())
    assert set(result["points"][0]["candidates"][0]) == {
        "assistant_turn",
        "why",
        "multiplicity",
        "leak_terms",
    }
    assert ("turn_resamples" in plan, "turn_check" in plan["selected_branches"][0]) == (False, False)


def test_the_judge_of_a_review_takes_its_key_from_the_variable_that_the_config_names(monkeypatch):
    monkeypatch.setenv("TS_KEY", " typesafe-key ")

    judge = jev_judge({"api_key_env": "TS_KEY", "threshold": 0.8, "model": "jev-1.13.0"})

    assert (judge.key, judge.threshold, judge.model) == ("typesafe-key", 0.8, "jev-1.13.0")
    judge.close()


def test_the_judge_of_a_review_gives_up_on_a_request_after_three_tries_of_30_seconds(monkeypatch):
    """The review runs under its own deadline, so each request of the judge is bounded."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-key")

    judge = jev_judge({})

    assert (judge.timeout_s, judge.max_attempts) == (30.0, 3)
    judge.close()


def test_a_judge_whose_key_variable_is_unset_is_refused_naming_the_variable(monkeypatch):
    monkeypatch.delenv("TS_KEY", raising=False)

    with pytest.raises(ValueError, match=r"leak_judge needs the TypeSafe key in the worker's \$TS_KEY"):
        jev_judge({"api_key_env": "TS_KEY"})
