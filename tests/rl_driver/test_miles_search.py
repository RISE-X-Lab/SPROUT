"""A Miles search request: its roots, the review of the failed ones, and the branches where the review points.

The Run Store is ``FakeRunStore``, and the test plays its worker and its review job: a review's result is
written here by hand. A test that takes the ``clock`` fixture runs at ``NOW``. docs/RL_DRIVER.md
("Search") gives the labels of the roots, the shapes of the review and the fields of each trajectory.

The job ids follow the order of submission: the roots' actors, their grades, the review, the branches'
actors, their grades, and the review of the next round.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sprout.rl_driver.messages import MessageAdapter, internal_id
from sprout.rl_driver.server import create_app
from tests.rl_driver.conftest import NOW
from tests.rl_driver.fake_runstore import FakeRunStore
from tests.rl_driver.miles import (
    REVIEW_CONFIG,
    adapter_config,
    exported,
    failed,
    miles_adapter,
    run_review,
    run_samples,
    search_request,
    succeeded,
    verdict,
)

#: What a grade says about the task's hidden tests, beside its verdict.
HIDDEN = {"fail_to_pass": ["tests/test_hidden.py::test_secret"], "test_patch_paths": ["tests/test_hidden.py"]}


def search_adapter(fake: FakeRunStore, directory: Path, **review: Any) -> MessageAdapter:
    """An adapter whose ``review`` section is ``REVIEW_CONFIG`` with ``review``."""
    config = adapter_config(review={**REVIEW_CONFIG, **review})
    return miles_adapter(fake, directory / "ledger.sqlite3", config)


def repair_turn(call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "Run the tests before the edit.",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pytest -q"}'},
            }
        ],
    }


def candidate(call_id: str, **fields: Any) -> dict[str, Any]:
    """A repair turn that a review proposes, with ``fields`` such as its ``multiplicity``."""
    return {"assistant_turn": repair_turn(call_id), "why": "check first", "leak_terms": [], **fields}


def review_plan(job_id: str, *candidates: dict[str, Any], step: int = 2) -> dict[str, Any]:
    """A review's result: one point at tool step ``step`` of the attempt of ``job_id``, and ``candidates``."""
    point = {
        "job_id": job_id,
        "attempt_id": "attempt-" + job_id,
        "step": step,
        "why": "before the edit",
        "samples": 4,
        "candidates": list(candidates),
    }
    return {"status": "completed", "points": [point]}


def publish_point(
    fake: FakeRunStore, job_id: str, point_id: str, *, step: int = 2, message_step: int = 1
) -> None:
    """Give the attempt of ``job_id`` an available recovery point ``point_id`` at tool step ``step``."""
    fake.publish(
        job_id,
        tools=[{"depth": depth, "response": {"output": "ok"}} for depth in range(1, step + 1)],
        points=[
            {
                "id": point_id,
                "tool_depth": step,
                "message_step": message_step,
                "snapshot_id": "snap-" + point_id,
                "available": True,
            }
        ],
    )


def search_view(adapter: MessageAdapter) -> dict[str, Any]:
    """What the execution view says of the search: its state, the roots' label and counts, and its error."""
    search = adapter.execution("rollout-1")["search"]
    return {key: search.get(key) for key in ("state", "outcome", "shape", "root_counts", "error")}


def reviews(fake: FakeRunStore) -> list[tuple[str, int, list[str]]]:
    """The guidance, the candidates and the parent jobs of each review job, in submission order."""
    specs = [fake.request(job_id)["spec"] for job_id in fake.ids("review")]
    return [
        (spec["guidance"], spec["candidates"], [parent["job_id"] for parent in spec["parents"]])
        for spec in specs
    ]


def groups(result: dict[str, Any]) -> list[str | None]:
    return [trajectory["group"] for trajectory in result["trajectories"]]


def searches(result: dict[str, Any]) -> list[dict[str, Any] | None]:
    """The ``metadata.search`` of each trajectory of the Miles result, in slot order."""
    return [trajectory["metadata"]["search"] for trajectory in result["trajectories"]]


def label(
    state: str, outcome: str, shape: str, counts: tuple[int, int, int], error: str | None
) -> dict[str, Any]:
    """A ``search_view``: the roots' ``counts`` are planned, graded and resolved."""
    root_counts = dict(zip(("planned", "graded", "resolved"), counts, strict=True))
    return {"state": state, "outcome": outcome, "shape": shape, "root_counts": root_counts, "error": error}


def repair_continuation(call_id: str) -> dict[str, Any]:
    """How a branch inserts the repair turn ``call_id``: as the turn's reasoning."""
    return {
        "guidance": "assistant-turn",
        "assistant_turn": repair_turn(call_id),
        "assistant_turn_rendering": "reasoning",
    }


def branch_post(
    parent: str, continuation: dict[str, Any], *, point_id: str, max_turns: int
) -> tuple[str, Any]:
    """A branch's post, with the deadline ``wall_time_seconds`` after ``NOW``."""
    budget = {"max_turns": max_turns, "deadline_at": NOW + 900.0}
    return (parent, {"point_id": point_id, "continuation": continuation, "overrides": {}, "budget": budget})


def test_the_failed_roots_are_reviewed_in_one_job_with_the_reviewers_settings_and_what_it_may_read(
    fake_run_store, tmp_path
):
    """The review reads each verdict, and the hidden tests for its leak check, never what the diff touched."""
    adapter = search_adapter(fake_run_store, tmp_path)
    adapter.submit(
        search_request(
            slot_count=14, roots=2, candidates=2, student_continuations=2, candidate_continuations=2
        )
    )
    grade = verdict(
        False, log="2 failed", touched_test_paths=["tests/test_hidden.py"], duration_s=12, **HIDDEN
    )

    run_samples(adapter, fake_run_store, actors=2 * [succeeded(exported())], grades=2 * [succeeded(grade)])

    seen = {"resolved": False, "progress": 0.0, "log": "2 failed", **HIDDEN}
    assert fake_run_store.request("job-4") == {
        "kind": "review",
        "profile": "review",
        "spec": {
            "problem": "Fix the parser",
            "model": "policy",
            "model_endpoint": "http://model.test:30000",
            "api_key_env": None,
            "temperature": 0.7,
            "max_output_tokens": 4096,
            "guidance": "assistant-turn",
            "points": 1,
            "points_per_attempt": True,
            "candidates": 2,
            "timeout_s": 600.0,
            "parents": [
                {
                    "name": internal_id("slot-0"),
                    "job_id": "job-0",
                    "attempt_id": "attempt-job-0",
                    "grade": seen,
                },
                {
                    "name": internal_id("slot-1"),
                    "job_id": "job-1",
                    "attempt_id": "attempt-job-1",
                    "grade": seen,
                },
            ],
        },
        "context": {
            "rl_driver": {"rollout_job_id": internal_id("rollout-1"), "phase": "review", "outcome": "FF"}
        },
        "max_infra_retries": 0,
        "parent_point": None,
        "affinity": internal_id("rollout-1"),
    }


UNEXPORTED = succeeded(exported(training_export_error="Native history has an incomplete final record"))
LOST_GRADE = failed("sandbox did not start")
RESOLVED, UNRESOLVED = succeeded(verdict(True)), succeeded(verdict(False))


# Two roots: job-0 and job-1. A missing root has no verdict: its grade failed, or its export did.
@pytest.mark.parametrize(
    "actors, grades, tf_students, search, review",
    [
        pytest.param(
            [succeeded(exported())] * 2,
            [RESOLVED, RESOLVED],
            2,
            label("done", "TT", "none", (2, 2, 2), "every root resolved the task"),
            [],
            id="TT",
        ),
        pytest.param(
            [succeeded(exported())] * 2,
            [RESOLVED, LOST_GRADE],
            2,
            label(
                "done",
                "TT_partial",
                "none",
                (2, 1, 1),
                "no gradable failed root: every graded root resolved, 1 grade(s) missing",
            ),
            [],
            id="TT_partial",
        ),
        pytest.param(
            [succeeded(exported())] * 2,
            [LOST_GRADE, LOST_GRADE],
            2,
            label("done", "ungraded", "none", (2, 0, 0), "no root could be graded"),
            [],
            id="ungraded",
        ),
        pytest.param(
            [succeeded(exported())] * 2,
            [RESOLVED, UNRESOLVED],
            2,
            label("reviewing", "TF", "points_only", (2, 2, 1), None),
            [("none", 1, ["job-1"])],
            id="TF",
        ),
        pytest.param(
            [succeeded(exported())] * 2,
            [RESOLVED, UNRESOLVED],
            0,
            label(
                "done",
                "TF",
                "none",
                (2, 2, 1),
                "tf_student_continuations is 0: a task whose roots split is not branched",
            ),
            [],
            id="TF-without-students",
        ),
        pytest.param(
            [succeeded(exported())] * 2,
            [LOST_GRADE, UNRESOLVED],
            2,
            label("reviewing", "FF_partial", "points_only", (2, 1, 0), None),
            [("none", 1, ["job-1"])],
            id="FF_partial-lost-grade",
        ),
        pytest.param(
            [UNEXPORTED, succeeded(exported())],
            [UNRESOLVED, UNRESOLVED],
            2,
            label("reviewing", "FF_partial", "points_only", (2, 1, 0), None),
            [("none", 1, ["job-1"])],
            id="FF_partial-unexported",
        ),
        pytest.param(
            [succeeded(exported())] * 2,
            [UNRESOLVED, UNRESOLVED],
            2,
            label("reviewing", "FF", "full", (2, 2, 0), None),
            [("assistant-turn", 2, ["job-0", "job-1"])],
            id="FF",
        ),
    ],
)
def test_the_roots_label_decides_whether_and_how_the_failed_roots_are_reviewed(
    fake_run_store, tmp_path, actors, grades, tf_students, search, review
):
    adapter = search_adapter(fake_run_store, tmp_path)
    adapter.submit(search_request(slot_count=8, roots=2, candidates=2, tf_student_continuations=tf_students))

    run_samples(adapter, fake_run_store, actors=actors, grades=grades)

    assert (search_view(adapter), reviews(fake_run_store)) == (search, review)


@pytest.mark.parametrize(
    "grades, outcome, counts",
    [
        ([RESOLVED, RESOLVED], "TT", {"planned": 2, "graded": 2, "resolved": 2}),
        ([RESOLVED, LOST_GRADE], "TT_partial", {"planned": 2, "graded": 1, "resolved": 1}),
        ([LOST_GRADE, LOST_GRADE], "ungraded", {"planned": 2, "graded": 0, "resolved": 0}),
    ],
)
def test_each_root_trajectory_carries_the_roots_label_and_counts(
    fake_run_store, tmp_path, grades, outcome, counts
):
    """Miles counts the roots that it receives, and refuses a group whose label or counts differ."""
    adapter = search_adapter(fake_run_store, tmp_path)
    adapter.submit(search_request(slot_count=6, roots=2))
    run_samples(adapter, fake_run_store, actors=2 * [succeeded(exported())], grades=grades)

    result = adapter.get("rollout-1")

    root = {"kind": "root", "outcome": outcome, "root_counts": counts}
    assert (groups(result), searches(result)) == (["root", "root"], [root, root])


@pytest.mark.parametrize(
    "review, error, skipped",
    [
        pytest.param(failed("model endpoint refused"), "model endpoint refused", None, id="review-failed"),
        pytest.param(
            succeeded({"status": "completed", "points": []}),
            "the review proposed no branch that could be made",
            [],
            id="no-point",
        ),
        pytest.param(
            succeeded(review_plan("job-1", candidate("repair-1"))),
            "the review proposed no branch that could be made",
            [{"point": 1, "step": 2, "reason": "not a root of this group"}],
            id="point-in-another-job",
        ),
        pytest.param(
            succeeded(review_plan("job-0", candidate("repair-1"), step=5)),
            "the review proposed no branch that could be made",
            [{"point": 1, "step": 5, "reason": "no available recovery point at that step"}],
            id="point-without-snapshot",
        ),
    ],
)
def test_a_review_that_fails_or_proposes_no_usable_point_returns_the_roots_alone(
    fake_run_store, tmp_path, review, error, skipped
):
    """Unfilled slots are not failures: the group completes with fewer samples than its maximum."""
    adapter = search_adapter(fake_run_store, tmp_path)
    adapter.submit(search_request(slot_count=3))
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[UNRESOLVED])
    publish_point(fake_run_store, "job-0", "point-2")

    run_review(adapter, fake_run_store, review)

    result = adapter.get("rollout-1")
    search = adapter.execution("rollout-1")["search"]
    assert (result["status"], result["max_samples"], groups(result)) == ("completed", 3, ["root"])
    assert (search["state"], search["error"], search.get("skipped")) == ("done", error, skipped)


def branch_full_point(fake: FakeRunStore, directory: Path) -> MessageAdapter:
    """One failed root, then a review that proposes two repair turns at its point ``point-2``.

    The search runs one student and two repairs of each turn there: slots 1 to 5.
    """
    adapter = search_adapter(fake, directory)
    adapter.submit(search_request(slot_count=6, candidates=2, candidate_continuations=2))
    run_samples(adapter, fake, actors=[succeeded(exported())], grades=[UNRESOLVED])
    publish_point(fake, "job-0", "point-2", step=2, message_step=1)
    plan = review_plan("job-0", candidate("repair-1", multiplicity=2), candidate("repair-2", leak_terms=None))
    run_review(adapter, fake, succeeded(plan))
    return adapter


def test_the_reviews_point_becomes_a_student_and_the_repairs_of_each_turn_with_the_turns_left(
    fake_run_store, tmp_path, clock
):
    """A branch has the turns that its root had left at the point, and its own wall time from now."""
    branch_full_point(fake_run_store, tmp_path)

    assert fake_run_store.branch_posts == [
        branch_post("job-0", {"guidance": "none"}, point_id="point-2", max_turns=10 - 1),
        branch_post("job-0", repair_continuation("repair-1"), point_id="point-2", max_turns=10 - 1),
        branch_post("job-0", repair_continuation("repair-1"), point_id="point-2", max_turns=10 - 1),
        branch_post("job-0", repair_continuation("repair-2"), point_id="point-2", max_turns=10 - 1),
        branch_post("job-0", repair_continuation("repair-2"), point_id="point-2", max_turns=10 - 1),
    ]


def test_the_branches_take_the_free_slots_in_the_order_planned_and_name_their_root(fake_run_store, tmp_path):
    adapter = branch_full_point(fake_run_store, tmp_path)
    run_samples(adapter, fake_run_store, actors=5 * [succeeded(exported())], grades=5 * [UNRESOLVED])

    result = adapter.get("rollout-1")

    assert [(t["sample_slot_id"], t["group"], t["parent_branch_id"]) for t in result["trajectories"]] == [
        ("slot-0", "root", None),
        ("slot-1", "student:point-2", "job-0"),
        ("slot-2", "repair:point-2:1", "job-0"),
        ("slot-3", "repair:point-2:1", "job-0"),
        ("slot-4", "repair:point-2:2", "job-0"),
        ("slot-5", "repair:point-2:2", "job-0"),
    ]
    assert (result["status"], result["actual_samples"], result["search_branches"]) == ("completed", 6, 5)


def test_a_branch_trajectory_carries_its_point_and_a_repair_its_candidate(fake_run_store, tmp_path):
    adapter = branch_full_point(fake_run_store, tmp_path)
    grades = [UNRESOLVED, RESOLVED, UNRESOLVED, UNRESOLVED, UNRESOLVED]
    run_samples(adapter, fake_run_store, actors=5 * [succeeded(exported())], grades=grades)

    result = adapter.get("rollout-1")

    student = {
        "kind": "student",
        "point": 1,
        "point_id": "point-2",
        "step": 2,
        "message_step": 1,
        "parent_job_id": "job-0",
        "outcome": "FF",
        "root_counts": {"planned": 1, "graded": 1, "resolved": 0},
    }
    repair = {
        **student,
        "kind": "repair",
        "candidate": 1,
        "multiplicity": 2,
        "samples": 4,
        "round": 1,
        "leak_terms": [],
        "assistant_turn_rendering": "reasoning",
    }
    second_turn = {**repair, "candidate": 2, "multiplicity": 1, "leak_terms": None}
    assert searches(result)[1:] == [student, repair, repair, second_turn, second_turn]
    assert [trajectory["reward"] for trajectory in result["trajectories"]] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_split_roots_run_unaided_students_alone_at_the_failed_roots_point(fake_run_store, tmp_path):
    """The review of split roots picks points alone: the repair turns of its result are not run."""
    adapter = search_adapter(fake_run_store, tmp_path)
    adapter.submit(search_request(slot_count=8, roots=2, candidates=2, tf_student_continuations=3))
    run_samples(adapter, fake_run_store, actors=2 * [succeeded(exported())], grades=[RESOLVED, UNRESOLVED])
    publish_point(fake_run_store, "job-1", "point-2")

    run_review(
        adapter, fake_run_store, succeeded(review_plan("job-1", candidate("repair-1"), candidate("repair-2")))
    )

    assert [(parent, body["continuation"]) for parent, body in fake_run_store.branch_posts] == 3 * [
        ("job-1", {"guidance": "none"})
    ]


def test_a_lost_review_reply_before_a_cancel_is_reconciled_by_its_key_and_the_group_ends(
    fake_run_store, tmp_path
):
    adapter = search_adapter(fake_run_store, tmp_path)
    adapter.submit(search_request(slot_count=3))
    adapter.driver.tick()
    fake_run_store.finish("job-0", result=exported())
    adapter.driver.tick()
    fake_run_store.finish("job-1", result=verdict(False))
    fake_run_store.lose_next_reply()
    adapter.driver.tick()
    adapter.release("rollout-1")
    adapter.driver.tick()
    group = internal_id("rollout-1")

    with TestClient(create_app(adapter.driver, None, background=False)) as http:
        http.post(
            f"/execution-groups/{group}/samples/search/reconcile-submission",
            json={"phase": "review", "job_id": "job-2"},
        )
        adapter.driver.tick()

    view = adapter.driver.get(group)
    assert (view["status"], view["search"]["state"], fake_run_store.jobs["job-2"]["state"]) == (
        "cancelled",
        "done",
        "cancelled",
    )
    assert [path for path, _ in fake_run_store.posts] == 3 * ["/v1/jobs"], "the review was posted once"


def test_the_execution_view_of_a_search_leaves_out_the_review_bodies(fake_run_store, tmp_path):
    adapter = through_round_one(fake_run_store, tmp_path, repair_rounds=2, grades=[UNRESOLVED, UNRESOLVED])

    search = adapter.execution("rollout-1")["search"]

    assert (
        {"review_templates", "review_template"} & set(search),
        "submission" in search["review"],
        "submission" in search["chain_reviews"][0],
    ) == (set(), False, False)


def through_round_one(
    fake: FakeRunStore, directory: Path, *, repair_rounds: int, grades: list[dict[str, Any]], **review: Any
) -> MessageAdapter:
    """A failed root ``job-0``, then a student ``job-3`` and a repair ``job-4`` at its point ``point-2``.

    ``grades`` settle the two branches. The repair's turn has the leak terms ``[]``. A later round's
    review, if any, is ``job-7``.
    """
    adapter = search_adapter(fake, directory, **review)
    adapter.submit(search_request(slot_count=2 + repair_rounds, repair_rounds=repair_rounds))
    run_samples(adapter, fake, actors=[succeeded(exported())], grades=[UNRESOLVED])
    publish_point(fake, "job-0", "point-2", step=2, message_step=1)
    run_review(adapter, fake, succeeded(review_plan("job-0", candidate("repair-1"))))
    run_samples(adapter, fake, actors=2 * [succeeded(exported())], grades=grades)
    return adapter


def test_a_failed_repair_is_reviewed_on_its_own_in_the_next_round(fake_run_store, tmp_path):
    through_round_one(fake_run_store, tmp_path, repair_rounds=3, grades=[UNRESOLVED, UNRESOLVED])

    review = fake_run_store.request("job-7")

    spec = review["spec"]
    assert ([parent["job_id"] for parent in spec["parents"]], spec["points"], spec["candidates"]) == (
        ["job-4"],
        1,
        1,
    )
    assert (spec["round_no"], review["context"]["rl_driver"]) == (
        2,
        {"rollout_job_id": internal_id("rollout-1"), "phase": "review", "outcome": "FF", "round": 2},
    )


def test_a_next_round_repair_continues_its_parent_repair_with_the_turns_that_it_had_left(
    fake_run_store, tmp_path, clock
):
    adapter = through_round_one(fake_run_store, tmp_path, repair_rounds=2, grades=[UNRESOLVED, UNRESOLVED])
    publish_point(fake_run_store, "job-4", "point-r2", step=3, message_step=2)

    run_review(adapter, fake_run_store, succeeded(review_plan("job-4", candidate("repair-r2"), step=3)))

    assert fake_run_store.branch_posts[-1] == branch_post(
        "job-4", repair_continuation("repair-r2"), point_id="point-r2", max_turns=10 - 1 - 2
    )


def test_a_chain_repair_belongs_to_its_root_point_and_carries_the_leak_terms_of_every_turn(
    fake_run_store, tmp_path
):
    adapter = through_round_one(fake_run_store, tmp_path, repair_rounds=3, grades=[UNRESOLVED, UNRESOLVED])
    publish_point(fake_run_store, "job-4", "point-r2", step=3, message_step=2)
    run_review(
        adapter,
        fake_run_store,
        succeeded(review_plan("job-4", candidate("r2", leak_terms=["test_a"]), step=3)),
    )
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[UNRESOLVED])
    publish_point(fake_run_store, "job-8", "point-r3", step=4, message_step=3)
    plan = review_plan("job-8", candidate("r3", leak_terms=["test_b"]), step=4)
    run_review(adapter, fake_run_store, succeeded(plan))
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[RESOLVED])

    result = adapter.get("rollout-1")

    assert groups(result) == [
        "root",
        "student:point-2",
        "repair:point-2:1",
        "repair:point-2:2",
        "repair:point-2:3",
    ]
    assert searches(result)[-1] == {
        "kind": "repair",
        "point": 1,
        "point_id": "point-2",
        "step": 2,
        "message_step": 1,
        "parent_job_id": "job-0",
        "outcome": "FF",
        "root_counts": {"planned": 1, "graded": 1, "resolved": 0},
        "round": 3,
        "chain_parent_job_id": "job-8",
        "chain_point_id": "point-r3",
        "chain_step": 4,
        "candidate": 3,
        "multiplicity": 1,
        "samples": 4,
        "leak_terms": ["test_a", "test_b"],
        "assistant_turn_rendering": "reasoning",
    }


@pytest.mark.parametrize(
    "grades",
    [
        pytest.param([RESOLVED, UNRESOLVED], id="student-resolved"),
        pytest.param([UNRESOLVED, succeeded(verdict(True, touched_test_paths=[]))], id="repair-resolved"),
        pytest.param([LOST_GRADE, UNRESOLVED], id="no-graded-student"),
    ],
)
def test_no_next_round_follows_a_point_that_resolved_or_had_no_graded_student(
    fake_run_store, tmp_path, grades
):
    """The trainer credits a repair only at a point where a student was graded and nothing resolved."""
    adapter = through_round_one(fake_run_store, tmp_path, repair_rounds=3, grades=grades)

    search = adapter.execution("rollout-1")["search"]

    assert (fake_run_store.ids("review"), search["state"], search.get("chain_end")) == (
        ["job-2"],
        "done",
        "after round 1: no unresolved point with a failed repair to review",
    )


def test_a_repair_that_resolved_by_changing_the_hidden_tests_is_reviewed_in_the_next_round(
    fake_run_store, tmp_path
):
    touched = succeeded(verdict(True, touched_test_paths=["tests/test_hidden.py"]))

    through_round_one(fake_run_store, tmp_path, repair_rounds=2, grades=[UNRESOLVED, touched])

    assert reviews(fake_run_store)[1:] == [("assistant-turn", 1, ["job-4"])]


def test_the_leak_judge_and_the_test_patch_that_the_grade_applied_reach_the_review(fake_run_store, tmp_path):
    adapter = search_adapter(fake_run_store, tmp_path, leak_judge={"threshold": 0.8, "resamples": 1})
    adapter.submit(search_request(slot_count=3))

    run_samples(
        adapter,
        fake_run_store,
        actors=[succeeded(exported())],
        grades=[succeeded(verdict(False, test_patch="/path/to/grade/test.patch"))],
    )

    spec = fake_run_store.request("job-2")["spec"]
    assert (spec["leak_judge"], spec["parents"][0]["grade"]["test_patch"]) == (
        {"threshold": 0.8, "resamples": 1},
        "/path/to/grade/test.patch",
    )


def test_a_repair_carries_the_leak_check_of_every_turn_on_its_chain_in_round_order(fake_run_store, tmp_path):
    flagged = {"leak": True, "leak_terms": [], "jev": {"score": 0.93, "leak": True}}
    clean = {"leak": False, "leak_terms": [], "jev": {"score": 0.1, "leak": False}}
    adapter = search_adapter(fake_run_store, tmp_path, leak_judge={"threshold": 0.8})
    adapter.submit(search_request(slot_count=4, repair_rounds=2))
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[UNRESOLVED])
    publish_point(fake_run_store, "job-0", "point-2")
    run_review(adapter, fake_run_store, succeeded(review_plan("job-0", candidate("r1", leak_judge=flagged))))
    run_samples(adapter, fake_run_store, actors=2 * [succeeded(exported())], grades=[UNRESOLVED, UNRESOLVED])
    publish_point(fake_run_store, "job-4", "point-r2", step=3, message_step=2)
    run_review(
        adapter, fake_run_store, succeeded(review_plan("job-4", candidate("r2", leak_judge=clean), step=3))
    )
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[RESOLVED])

    student, first_repair, second_repair = searches(adapter.get("rollout-1"))[1:]

    assert ["leak_judge" in student, first_repair["leak_judge"], second_repair["leak_judge"]] == [
        False,
        [flagged],
        [flagged, clean],
    ]


def test_without_a_leak_judge_neither_the_review_nor_a_repair_carries_one(fake_run_store, tmp_path):
    adapter = through_round_one(fake_run_store, tmp_path, repair_rounds=1, grades=[UNRESOLVED, RESOLVED])

    repair = adapter.get("rollout-1")["trajectories"][2]["metadata"]["search"]

    assert ("leak_judge" in fake_run_store.request("job-2")["spec"], "leak_judge" in repair) == (False, False)


@pytest.mark.parametrize(
    "roots, slot_count, grades, outcome",
    [
        pytest.param(1, 3, [UNRESOLVED], "FF", id="FF"),
        pytest.param(2, 6, [RESOLVED, UNRESOLVED], "TF", id="TF"),
    ],
)
def test_a_search_stored_before_outcome_keyed_review_ends_at_its_labelled_roots(
    fake_run_store, tmp_path, roots, slot_count, grades, outcome
):
    adapter = search_adapter(fake_run_store, tmp_path)
    adapter.submit(search_request(slot_count=slot_count, roots=roots))
    group = internal_id("rollout-1")
    document = adapter.driver.ledger.get(group)["document"]
    # The format without outcome-keyed review: one review template, and no tf_student_continuations.
    document["search"]["review_template"] = document["search"].pop("review_templates")["full"]
    del document["search"]["config"]["tf_student_continuations"]
    adapter.driver.ledger.save(group, document)

    run_samples(adapter, fake_run_store, actors=roots * [succeeded(exported())], grades=grades)

    search = search_view(adapter)
    assert (fake_run_store.ids("review"), search["state"], search["outcome"], search["error"]) == (
        [],
        "done",
        outcome,
        "search document predates outcome-keyed review",
    )


# A round takes its review_seconds (600) and wall_time_seconds (900), and each round but the last the
# grade_wall_seconds (600) of its grades. The finalization time covers the grades of the last round.
@pytest.mark.parametrize(
    "repair_rounds, slot_count, search_seconds",
    [(1, 3, 600.0 + 900.0), (3, 5, 3 * (600.0 + 900.0) + 2 * 600.0)],
)
def test_the_group_deadline_leaves_time_for_every_rounds_review_branches_and_grades(
    fake_run_store, tmp_path, clock, repair_rounds, slot_count, search_seconds
):
    adapter = search_adapter(fake_run_store, tmp_path)

    adapter.submit(search_request(slot_count=slot_count, repair_rounds=repair_rounds))

    view = adapter.execution("rollout-1")
    assert (view["execution_deadline_at"], view["deadline_at"]) == (
        NOW + 900.0,
        NOW + 900.0 + 1800.0 + search_seconds,
    )
