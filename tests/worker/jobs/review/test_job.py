"""The review job as a worker runs it: what it returns for each guidance, and what it refuses to run.

The review model is ``ReviewModel``. The analyst and the reviewer read the short templates of
``plain_prompts``, which the spec names as a review's ``prompts`` names a file.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.messages import mark_hint
from sprout.worker.jobs.review.job import grade_of, run_review
from tests.worker.jobs.review.plain_prompts import prompts_record, reports_in
from tests.worker.jobs.review.recorded import record_attempt, turn
from tests.worker.jobs.review.review_model import HIDDEN, ReviewModel, failed_roots, parent, review_spec

LEAKED = ["test_hidden", "test_hidden.py", "tests/test_hidden.py"]


def bash_turn(command: str, call_id: str = "reviewer-r1-b1-c1") -> dict[str, Any]:
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }
    return {"role": "assistant", "content": "I will check the test before editing.", "tool_calls": [call]}


def candidate(command: str, multiplicity: int, leak_terms: list[str] | None) -> dict[str, Any]:
    return {
        "assistant_turn": bash_turn(command),
        "why": "check first",
        "multiplicity": multiplicity,
        "leak_terms": leak_terms,
    }


def by_sample(*points: list[str]) -> dict[str, str]:
    """The command of each sample at each point of a review over every failed attempt."""
    return {
        f"point{p}-candidate{s}": command
        for p, commands in enumerate(points, 1)
        for s, command in enumerate(commands, 1)
    }


def test_an_assistant_turn_review_gives_each_point_its_samples_merged_by_their_commands(tmp_path):
    """Proposals with the same commands merge into the first one, which is the one executed and trained."""
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1, **HIDDEN), prompts=prompts_record(tmp_path))
    model = ReviewModel(
        points={"points": [("root-1", 2), ("root-1", 3)]},
        proposals=by_sample(["pytest -q", "pytest -q", "ls"], ["git diff", "ls", "git diff"]),
    )

    result = run_review(spec, tmp_path / "review", transport_factory=model.factory)

    common = {
        "base": "root-1",
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "why": "a decision",
        "samples": 3,
    }
    assert {key: value for key, value in result.items() if key != "reports"} == {
        "status": "completed",
        "guidance": "assistant-turn",
        "points": [
            {
                **common,
                "run_name": "r1b1-point-2",
                "step": 2,
                "snapshot_id": "root-1-snap-2",
                "conversation_cut": "turn-1",
                "candidates": [candidate("pytest -q", 2, []), candidate("ls", 1, [])],
            },
            {
                **common,
                "run_name": "r1b2-point-3",
                "step": 3,
                "snapshot_id": "root-1-snap-3",
                "conversation_cut": "turn-2",
                "candidates": [candidate("git diff", 2, []), candidate("ls", 1, [])],
            },
        ],
        "analysed": ["root-1"],
        "review_requests": 1 + 1 + 2 * 3,
    }
    assert {key: value for key, value in result["reports"]["root-1"].items() if key != "prompt_budget"} == {
        "failure_reason": "edited the wrong file",
        "lesson": "read the test first",
        "salvage": "the exploration",
        "branch_candidates": [{"step": 2, "why": "before the edit"}],
        "steps": 4,
    }


def test_a_candidate_that_names_a_hidden_test_is_flagged_and_still_returned(tmp_path):
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1, **HIDDEN), points=1, candidates=2)
    model = ReviewModel(
        points={"points": [("root-1", 2)]},
        proposals=by_sample(["pytest -q", "pytest -q -k test_secret_behaviour"]),
    )

    result = run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert result["points"][0]["candidates"] == [
        candidate("pytest -q", 1, []),
        candidate("pytest -q -k test_secret_behaviour", 1, ["test_secret_behaviour"]),
    ]


def test_without_the_hidden_facts_in_the_grade_no_candidate_is_checked(tmp_path):
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1), points=1, candidates=1)
    model = ReviewModel(points={"points": [("root-1", 2)]}, proposals="pytest -q -k test_secret_behaviour")

    result = run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert result["points"][0]["candidates"] == [candidate("pytest -q -k test_secret_behaviour", 1, None)]


def test_a_turn_is_checked_against_the_hint_free_history_that_its_own_point_keeps(tmp_path):
    """The hint of a branch names the hidden test file, and the actor opens the file at its second turn.

    Training drops the hint, so the same repair leaks at step 1 and not at step 2."""
    root = tmp_path / "artifacts"
    turns = [
        turn("cat answer"),
        turn("cat tests/test_hidden.py"),
        turn("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
    ]
    record_attempt(
        root / "job-2" / "attempt-2",
        turns,
        run="branch-1",
        task=mark_hint("Start from tests/test_hidden.py."),
    )
    spec = review_spec(root, [parent("branch-1", "job-2", "attempt-2", **HIDDEN)], candidates=1)
    model = ReviewModel(
        points={"points": [("branch-1", 1), ("branch-1", 2)]}, proposals="pytest -q tests/test_hidden.py"
    )

    result = run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert [(p["step"], p["candidates"][0]["leak_terms"]) for p in result["points"]] == [(1, LEAKED), (2, [])]


def test_a_user_hint_review_writes_a_hint_at_each_point_and_no_candidates(tmp_path):
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1), guidance="user-hint", candidates=1)
    model = ReviewModel(points={"points": [("root-1", 2)]}, hint="Check the boundary.")

    result = run_review(spec, tmp_path / "review", transport_factory=model.factory)

    (point,) = result["points"]
    assert (point["step"], point["hint"], point["candidates"], "samples" in point) == (
        2,
        "Check the boundary.",
        [],
        False,
    )
    assert model.calls == ["analyst-1", "points"]


def test_a_user_hint_review_shows_the_reviewer_the_hint_that_each_parent_was_given(tmp_path):
    """The hint and its outcome are then in one place, and the reviewer does not repeat a spent direction."""
    root = tmp_path / "artifacts"
    parents = failed_roots(root, 2)
    parents[1]["hint"] = "Check the boundary."
    spec = review_spec(root, parents, guidance="user-hint", candidates=1, prompts=prompts_record(tmp_path))
    model = ReviewModel(points={"points": [("root-1", 2)]}, hint="Retry.")

    run_review(spec, tmp_path / "review", transport_factory=model.factory)

    rows = reports_in(model.prompts_of("points")[0])
    assert [(row["name"], row["hint_given"]) for row in rows] == [
        ("root-1", None),
        ("root-2", "Check the boundary."),
    ]


def test_a_review_without_guidance_picks_the_points_alone(tmp_path):
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1), guidance="none", candidates=1)
    model = ReviewModel(points={"points": [("root-1", 2), ("root-1", 4)]})

    result = run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert [(p["step"], p["candidates"], "samples" in p, "hint" in p) for p in result["points"]] == [
        (2, [], False, False),
        (4, [], False, False),
    ]
    assert (model.calls, result["review_requests"]) == (["analyst-1", "points"], 2)


def test_with_points_per_attempt_each_failed_attempt_is_reviewed_alone_and_checked_against_its_own_grade(
    tmp_path,
):
    root = tmp_path / "artifacts"
    parents = failed_roots(root, 2)
    parents[0]["grade"].update(HIDDEN)
    spec = review_spec(
        root, parents, points=1, points_per_attempt=True, candidates=1, prompts=prompts_record(tmp_path)
    )
    model = ReviewModel(points={"points-1": [("root-1", 2)], "points-2": [("root-2", 3)]})

    result = run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert [(p["base"], p["step"], p["candidates"][0]["leak_terms"]) for p in result["points"]] == [
        ("root-1", 2, []),
        ("root-2", 3, None),
    ]
    assert [
        [row["name"] for row in reports_in(model.prompts_of(review)[0])]
        for review in ("points-1", "points-2")
    ] == [["root-1"], ["root-2"]]
    assert sorted(model.made) == [
        "analyst-1",
        "analyst-2",
        "point1.1-candidate1",
        "point2.1-candidate1",
        "points-1",
        "points-2",
    ]
    assert sorted(path.name for path in (tmp_path / "review").glob("plan-*.json")) == [
        "plan-point1.1-candidate1.json",
        "plan-point2.1-candidate1.json",
        "plan-points-1.json",
        "plan-points-2.json",
    ]


def test_each_sample_is_offered_its_own_point_only(tmp_path):
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1), candidates=1)
    model = ReviewModel(points={"points": [("root-1", 2), ("root-1", 4)]})

    run_review(spec, tmp_path / "review", transport_factory=model.factory)

    plans = [json.loads((tmp_path / "review" / f"plan-point{p}-candidate1.json").read_text()) for p in (1, 2)]
    assert [(plan["available_steps"], plan["selected_branches"][0]["branch_step"]) for plan in plans] == [
        ({"root-1": [2]}, 2),
        ({"root-1": [4]}, 4),
    ]


def test_resolved_attempts_are_not_analysed(tmp_path):
    root = tmp_path / "artifacts"
    parents = failed_roots(root, 2)
    parents[0]["grade"] = {"resolved": True, "progress": 1.0}
    model = ReviewModel(points={"points": [("root-2", 2)]})

    result = run_review(
        review_spec(root, parents, guidance="none", candidates=1),
        tmp_path / "review",
        transport_factory=model.factory,
    )

    assert (result["analysed"], list(result["reports"]), [p["base"] for p in result["points"]]) == (
        ["root-2"],
        ["root-2"],
        ["root-2"],
    )


def never_called(directory: Path, **settings: Any) -> Any:
    raise AssertionError("the review made a model call")


def test_a_review_of_attempts_that_all_resolved_asks_the_model_nothing(tmp_path):
    root = tmp_path / "artifacts"
    parents = failed_roots(root, 1)
    parents[0]["grade"] = {"resolved": True, "progress": 1.0}

    result = run_review(review_spec(root, parents), tmp_path / "review", transport_factory=never_called)

    assert result == {
        "status": "completed",
        "guidance": "assistant-turn",
        "points": [],
        "reports": {},
        "reason": "every attempt resolved the task; nothing to repair",
    }


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"artifact_root": None}, "The worker must supply artifact_root to a review"),
        ({"api_key_env": None}, "A review needs api_key_env"),
        ({"guidance": "prose"}, "Review guidance must be assistant-turn, none or user-hint"),
        ({"leak_judge": {}}, r"leak_judge needs the TypeSafe key in the worker's \$TYPESAFE_API_KEY"),
    ],
    ids=["no-artifact-root", "no-key-variable", "an-invalid-spec", "no-typesafe-key"],
)
def test_a_review_that_cannot_run_is_refused_before_any_model_call(tmp_path, monkeypatch, changes, error):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1), **changes)

    with pytest.raises(ValueError, match=error):
        run_review(spec, tmp_path / "review", transport_factory=never_called)


def test_a_parent_without_a_journal_is_refused_by_name_before_any_model_call(tmp_path):
    spec = review_spec(tmp_path / "artifacts", [parent("root-9", "job-9", "attempt-9")])

    with pytest.raises(ValueError, match="Review parent root-9 has no journal at "):
        run_review(spec, tmp_path / "review", transport_factory=never_called)


def test_a_review_whose_prompts_file_changed_since_it_was_submitted_is_refused_before_any_model_call(
    tmp_path,
):
    root = tmp_path / "artifacts"
    record = prompts_record(tmp_path)
    Path(record["path"]).write_text(Path(record["path"]).read_text() + "# edited\n")

    with pytest.raises(ValueError, match="the prompts file changed after the review was submitted"):
        run_review(
            review_spec(root, failed_roots(root, 1), prompts=record),
            tmp_path / "review",
            transport_factory=never_called,
        )


@pytest.mark.parametrize("request_timeout_s, timeout_s", [(90, 90), (None, 600)])
def test_each_call_gets_a_transport_of_its_own_with_the_reviews_model_settings(
    tmp_path, request_timeout_s, timeout_s
):
    root = tmp_path / "artifacts"
    spec = review_spec(
        root,
        failed_roots(root, 1),
        guidance="none",
        candidates=1,
        max_output_tokens=2048,
        reasoning_effort="low",
        temperature=0.5,
        timeout_s=600,
        request_timeout_s=request_timeout_s,
        prompts=prompts_record(tmp_path),
    )
    model = ReviewModel(points={"points": [("root-1", 2)]})

    run_review(spec, tmp_path / "review", transport_factory=model.factory)

    stops = [model.settings[call].pop("stop") for call in ("analyst-1", "points")]
    assert model.settings == {
        call: {
            "model": "fixture",
            "model_endpoint": "http://127.0.0.1:1",
            "api_key_env": "REVIEW_KEY",
            "max_output_tokens": 2048,
            "timeout_s": timeout_s,
            "reasoning_effort": "low",
            "temperature": 0.5,
            "json_directive": "\nJSON ONLY.",
            "supports_json_schema": False,
        }
        for call in ("analyst-1", "points")
    }
    assert (isinstance(stops[0], threading.Event), stops[0] is not stops[1]) == (True, True)


def test_each_call_reads_the_templates_of_the_prompts_file_that_the_spec_names(tmp_path):
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1), points=1, candidates=1, prompts=prompts_record(tmp_path))
    model = ReviewModel(points={"points": [("root-1", 2)]})

    run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert [(call, prompt.split(" ", 1)[0]) for call, _, prompt in model.requests] == [
        ("analyst-1", "PROBLEM"),
        ("points", "POINTS"),
        ("point1-candidate1", "TURNS"),
    ]
    assert json.loads((tmp_path / "review" / "plan-points.json").read_text())["prompts"] == prompts_record(
        tmp_path
    )


def test_a_structured_review_asks_the_analyst_for_json_and_a_point_review_for_its_fenced_plan(tmp_path):
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1), guidance="none", candidates=1, structured=True)
    model = ReviewModel(points={"points": [("root-1", 2)]})

    run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert [(call, kind) for call, kind, _ in model.requests] == [("analyst-1", "json"), ("points", "text")]
    assert model.settings["analyst-1"]["supports_json_schema"] is True


def test_every_request_of_a_review_is_held_to_its_prompt_chars(tmp_path):
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 1), guidance="none", candidates=1, prompt_chars=123_456)

    run_review(
        spec, tmp_path / "review", transport_factory=ReviewModel(points={"points": [("root-1", 2)]}).factory
    )

    plan = json.loads((tmp_path / "review" / "plan-points.json").read_text())
    assert (plan["prompt_budget"]["budget"], plan["reports"]["root-1"]["prompt_budget"]["budget"]) == (
        123_456,
        123_456,
    )


@pytest.mark.parametrize(
    "failed_steps, offered",
    [((), "[2, 3, 4]"), ((3,), "[2, 4]")],
    ids=["every-closed-turn", "a-failed-snapshot"],
)
def test_the_analyst_is_offered_only_the_steps_where_a_branch_can_start(tmp_path, failed_steps, offered):
    """Step 1 is inside a turn of two calls. A step whose snapshot failed is never replaced by another."""
    root = tmp_path / "artifacts"
    record_attempt(root / "job-1" / "attempt-1", run="root-1", failed_steps=failed_steps)
    spec = review_spec(
        root,
        [parent("root-1", "job-1", "attempt-1")],
        guidance="none",
        candidates=1,
        prompts=prompts_record(tmp_path),
    )
    model = ReviewModel(points={"points": [("root-1", 2)]})

    run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert model.prompts_of("analyst-1")[0].endswith(f"\nSTEPS {offered} OF 1..4 AT MOST 2")


def test_the_analyst_reads_the_patch_the_progress_the_failed_tests_the_log_tail_and_the_error_output(
    tmp_path,
):
    (tmp_path / "model.patch").write_text("diff --git a/src/module.py b/src/module.py\n")
    (tmp_path / "test.log").write_text("collected 2 items\n" + "x" * 20_000 + "\nFAILED t::a")
    statuses = {"t::a": "FAILED", "t::b": "PASSED", "t::c": "ERROR"}
    record = {"resolved": False, "progress": 0.25, "reason": "tests_failed", "test_statuses": statuses}
    record.update(
        model_patch=str(tmp_path / "model.patch"),
        log=str(tmp_path / "test.log"),
        error_output="E" * 4001,
        error="exit 1",
    )

    grade = grade_of(record)

    assert (grade.resolved, grade.reward, grade.patch, grade.error) == (
        False,
        0.0,
        "diff --git a/src/module.py b/src/module.py\n",
        "exit 1",
    )
    assert grade.detail == (
        "target tests passed (FAIL_TO_PASS): 25%\nreason: tests_failed\n"
        "test statuses: 3 recorded, 2 not passed\n  t::a: FAILED\n  t::c: ERROR\n"
        "test log (tail):\n" + ("x" * 20_000 + "\nFAILED t::a")[-20_000:] + "\nerror output:\n" + "E" * 4000
    )


def test_the_analyst_reads_at_most_40_tests_that_did_not_pass(tmp_path):
    statuses = {"t::%02d" % n: "FAILED" for n in range(41)}

    grade = grade_of({"resolved": False, "test_statuses": statuses})

    assert grade.detail == "test statuses: 41 recorded, 41 not passed\n" + "\n".join(
        "  t::%02d: FAILED" % n for n in range(40)
    )


def test_the_analyst_never_reads_the_hidden_facts_of_a_grade(tmp_path):
    """The leak checks read them. A prompt that showed them would teach the reviewer the hidden tests."""
    (tmp_path / "test.patch").write_text("+    assert convert('SECRET-FIXTURE') == 'ok'\n")
    record = {"resolved": False, "progress": 0.0, "touched_test_paths": ["tests/test_touched.py"], **HIDDEN}

    grade = grade_of({**record, "test_patch": str(tmp_path / "test.patch")})

    assert (grade.patch, grade.detail, grade.summary()) == (
        "",
        "target tests passed (FAIL_TO_PASS): 0%",
        "official Harbor reward=0.0; resolved=False",
    )


def test_no_request_of_a_review_shows_the_hidden_facts_of_a_grade(tmp_path):
    """The attempt never showed them, so no analyst, point review or sample may read them."""
    root = tmp_path / "artifacts"
    (tmp_path / "test.patch").write_text("+    assert convert('SECRET-FIXTURE') == 'ok'\n")
    facts = {**HIDDEN, "test_patch": str(tmp_path / "test.patch"), "test_statuses": {"t::a": "FAILED"}}
    model = ReviewModel(points={"points": [("root-1", 2)]})

    run_review(
        review_spec(root, failed_roots(root, 1, **facts), points=1, candidates=1),
        tmp_path / "review",
        transport_factory=model.factory,
    )

    sent = "\n".join(prompt for _, _, prompt in model.requests)
    hidden = ("test_secret_behaviour", "test_hidden", "SECRET-FIXTURE", str(tmp_path / "test.patch"))
    assert ("t::a: FAILED" in sent, [fact for fact in hidden if fact in sent]) == (True, [])


@pytest.mark.parametrize("count_mode, rule", [("adaptive", "AT MOST 2"), ("fixed", "EXACTLY 2")])
def test_the_point_review_counts_its_points_as_the_spec_says(tmp_path, count_mode, rule):
    root = tmp_path / "artifacts"
    spec = review_spec(
        root,
        failed_roots(root, 1),
        guidance="none",
        candidates=1,
        count_mode=count_mode,
        prompts=prompts_record(tmp_path),
    )
    model = ReviewModel(points={"points": [("root-1", 2), ("root-1", 3)]})

    run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert model.prompts_of("points")[0].endswith("\nCOUNT " + rule)
