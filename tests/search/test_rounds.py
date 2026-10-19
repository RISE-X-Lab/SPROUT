"""A group's search: the label of its roots, the review of the failed ones, the branches where it points.

The coordinator runs the search of a ``search_group`` document on a ``FakeRunStore``. The test plays
the worker of each job, the review's included: its result is the plan that the review job returns.
docs/RL_DRIVER.md describes the search. Later repair rounds are in test_chain.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from sprout.search.rounds import REVIEW_SLOT
from tests.search.documents import (
    GROUP,
    NO_VERDICT,
    REVIEW,
    coordinator_for,
    exported,
    lose,
    review_point,
    reviewed,
    search_group,
    search_of,
    settle_branches,
    settle_roots,
    stored,
    turn,
    verdict,
)
from tests.search.fake_runstore import FakeRunStore

REPAIR_1 = turn("pytest -q", call_id="repair-1")
REPAIR_2 = turn("sed -n 1,40p src/parser.py", call_id="repair-2")
#: What a leak judge reports about a turn that it found clean.
CLEAN = {"leak": False, "leak_terms": [], "jev": {"score": 0.1, "leak": False}}


def parent(index: int) -> dict[str, Any]:
    """What a review reads of the failed root ``root-<index>``: its job ``job-<index>`` and its verdict."""
    return {
        "name": "root-%d" % index,
        "job_id": "job-%d" % index,
        "attempt_id": "attempt-job-%d" % index,
        "grade": {"resolved": False, "progress": 0.0},
    }


def reviews(fake: FakeRunStore) -> list[tuple[str, int, list[dict[str, Any]], str]]:
    """The guidance, the candidates, the parents and the outcome of each review job, in order."""
    return [
        (
            body["spec"]["guidance"],
            body["spec"]["candidates"],
            body["spec"]["parents"],
            body["context"]["eval"]["outcome"],
        )
        for body in fake.submitted("review")
    ]


def branch_budgets(fake: FakeRunStore) -> list[tuple[str, dict[str, Any]]]:
    """The point and the budget of each branch, in order."""
    return [(body["point_id"], body["budget"]) for _, body in fake.branches()]


def counts(planned: int, graded: int, resolved: int) -> dict[str, int]:
    return {"planned": planned, "graded": graded, "resolved": resolved}


TF_STUDENTS = "tf_student_continuations is 0: a task whose roots split is not branched"


@pytest.mark.parametrize(
    "grades, exports_nothing, outcome, root_counts, error",
    [
        ((verdict(True),), (), "TT", counts(1, 1, 1), "every root resolved the task"),
        (
            (verdict(True), NO_VERDICT),
            (),
            "TT_partial",
            counts(2, 1, 1),
            "no gradable failed root: every graded root resolved, 1 grade(s) missing",
        ),
        ((NO_VERDICT,), (), "ungraded", counts(1, 0, 0), "no root could be graded"),
        ((verdict(True), verdict(False)), (), "TF", counts(2, 2, 1), TF_STUDENTS),
        ((verdict(False), verdict(False)), (0,), "FF_partial", counts(2, 1, 0), TF_STUDENTS),
    ],
    ids=["TT", "TT_partial", "ungraded", "TF without students", "FF_partial without students"],
)
def test_roots_that_call_for_no_review_end_the_search_with_the_reason(
    fake_run_store, groups, grades, exports_nothing, outcome, root_counts, error
):
    stored(groups, search_group(roots=len(grades), tf_student_continuations=0))
    coordinator = coordinator_for(fake_run_store, groups)

    settle_roots(fake_run_store, coordinator, *grades, exports_nothing=exports_nothing)

    search = search_of(groups)
    assert (
        search["state"],
        search["outcome"],
        search["shape"],
        search["root_counts"],
        search.get("error"),
        fake_run_store.job_ids("review"),
    ) == ("done", outcome, "none", root_counts, error, [])


def test_the_failed_roots_are_reviewed_in_one_review_job_of_the_full_shape(fake_run_store, groups):
    """It reads each failed root's attempt and verdict, and picks the points of each root on its own."""
    stored(groups, search_group(roots=2, candidates=2))
    coordinator = coordinator_for(fake_run_store, groups)

    settle_roots(fake_run_store, coordinator, verdict(False), verdict(False))

    assert fake_run_store.submitted("review") == [
        {
            "kind": "review",
            "profile": "review",
            "max_infra_retries": 0,
            "spec": {
                **REVIEW,
                "guidance": "assistant-turn",
                "points": 1,
                "points_per_attempt": True,
                "candidates": 2,
                "timeout_s": 600.0,
                "parents": [parent(0), parent(1)],
            },
            "context": {"eval": {"rollout_job_id": "group", "phase": "review", "outcome": "FF"}},
            "parent_point": None,
            "affinity": "group",
        }
    ]


@pytest.mark.parametrize(
    "grades, exports_nothing, outcome, shape, guidance, candidates, parents",
    [
        ((verdict(False), verdict(False)), (), "FF", "full", "assistant-turn", 2, [parent(0), parent(1)]),
        ((verdict(True), verdict(False)), (), "TF", "points_only", "none", 1, [parent(1)]),
        ((NO_VERDICT, verdict(False)), (), "FF_partial", "points_only", "none", 1, [parent(1)]),
        ((verdict(False), verdict(False)), (0,), "FF_partial", "points_only", "none", 1, [parent(1)]),
    ],
    ids=["FF", "TF", "FF_partial, no verdict", "FF_partial, nothing exported"],
)
def test_the_label_of_the_roots_decides_the_shape_of_their_review_and_a_missing_root_is_not_reviewed(
    fake_run_store, groups, grades, exports_nothing, outcome, shape, guidance, candidates, parents
):
    stored(groups, search_group(roots=2, candidates=2, tf_student_continuations=2))
    coordinator = coordinator_for(fake_run_store, groups)

    settle_roots(fake_run_store, coordinator, *grades, exports_nothing=exports_nothing)

    search = search_of(groups)
    assert (search["outcome"], search["shape"], reviews(fake_run_store)) == (
        outcome,
        shape,
        [(guidance, candidates, parents, outcome)],
    )


def test_a_review_reads_the_facts_of_a_grade_but_not_which_hidden_test_files_the_diff_touched(
    fake_run_store, groups
):
    """The analyst reads the verdict and its evidence, the leak checks the hidden tests. The trainer
    alone reads ``touched_test_paths``. A key that no review reads stays out too."""
    stored(groups, search_group())
    coordinator = coordinator_for(fake_run_store, groups)
    facts = {
        "test_statuses": {"tests/test_parser.py::test_parses": "FAILED"},
        "model_patch": "/path/to/model.patch",
        "log": "/path/to/test.log",
        "reason": "tests_failed",
        "error_output": "AssertionError",
        "fail_to_pass": ["tests/test_hidden.py::test_secret_behaviour"],
        "test_patch_paths": ["tests/test_hidden.py"],
        "test_patch": "/path/to/test.patch",
    }

    settle_roots(
        fake_run_store,
        coordinator,
        verdict(False, **facts, touched_test_paths=["tests/test_hidden.py"], expected_pass_share=0.5),
    )

    assert reviews(fake_run_store) == [
        ("assistant-turn", 1, [{**parent(0), "grade": {"resolved": False, "progress": 0.0, **facts}}], "FF")
    ]


def test_a_cancelled_group_ends_its_search_at_the_roots_without_a_review(fake_run_store, groups):
    stored(groups, search_group())
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()
    fake_run_store.finish("job-1", verdict(False))
    groups.release(GROUP)

    coordinator.tick()

    search = search_of(groups)
    assert (
        search["state"],
        search["error"],
        fake_run_store.job_ids("review"),
        groups.get(GROUP)["document"]["status"],
    ) == (
        "done",
        None,
        [],
        "cancelled",
    )


def test_a_search_document_from_before_outcome_keyed_review_ends_after_its_roots_are_labelled(
    fake_run_store, groups
):
    """Such a document has one review template and no ``tf_student_continuations`` (docs/RL_DRIVER.md)."""
    legacy = search_group()
    legacy["search"]["review_template"] = legacy["search"].pop("review_templates")["full"]
    del legacy["search"]["config"]["tf_student_continuations"]
    stored(groups, legacy)
    coordinator = coordinator_for(fake_run_store, groups)

    settle_roots(fake_run_store, coordinator, verdict(False))

    search = search_of(groups)
    assert (search["state"], search["outcome"], search.get("error"), fake_run_store.job_ids("review")) == (
        "done",
        "FF",
        "search document predates outcome-keyed review",
        [],
    )


@pytest.mark.parametrize(
    "state, result, error",
    [
        ("failed", {"status": "error", "error": "model endpoint refused"}, "model endpoint refused"),
        ("cancelled", {"status": "cancelled"}, "review cancelled"),
    ],
)
def test_a_review_that_does_not_succeed_ends_the_search_and_the_group_keeps_its_roots(
    fake_run_store, groups, state, result, error
):
    stored(groups, search_group())
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    fake_run_store.finish("job-2", result, state=state)

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    assert (
        settled["search"]["state"],
        settled["search"]["error"],
        settled["status"],
        len(settled["samples"]),
    ) == (
        "done",
        error,
        "completed",
        1,
    )


def test_a_full_review_branches_its_students_then_the_repairs_of_each_turn_at_each_point(
    fake_run_store, groups, clock
):
    """Each branch starts at the root's recovery point at the review's step, with the turns that the
    root had left there and a deadline of its own. A repair inserts its turn as the turn's reasoning."""
    stored(groups, search_group(candidates=2, candidate_continuations=2))
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    fake_run_store.finish(
        "job-2",
        reviewed(
            review_point(
                "job-0", step=2, candidates=({"assistant_turn": REPAIR_1}, {"assistant_turn": REPAIR_2})
            )
        ),
    )

    coordinator.tick()

    branch = {
        "point_id": "job-0-step2",
        "overrides": {},
        "budget": {"max_turns": 10 - 1, "deadline_at": 1000.0 + 900.0},
    }
    repair_1 = {
        "guidance": "assistant-turn",
        "assistant_turn": REPAIR_1,
        "assistant_turn_rendering": "reasoning",
    }
    repair_2 = {
        "guidance": "assistant-turn",
        "assistant_turn": REPAIR_2,
        "assistant_turn_rendering": "reasoning",
    }
    assert fake_run_store.branches() == [
        ("job-0", {**branch, "continuation": {"guidance": "none"}}),
        ("job-0", {**branch, "continuation": repair_1}),
        ("job-0", {**branch, "continuation": repair_1}),
        ("job-0", {**branch, "continuation": repair_2}),
        ("job-0", {**branch, "continuation": repair_2}),
    ]


def test_each_branch_carries_its_point_its_kind_and_the_label_of_the_roots(fake_run_store, groups):
    """The trainer groups the branches by ``group`` and checks the label of each against the roots it gets.
    A repair also carries what the review found in its turn: its leak terms and, if run, the judge's check."""
    stored(groups, search_group(candidates=2))
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    candidates = (
        {"assistant_turn": REPAIR_1, "multiplicity": 3, "leak_terms": [], "leak_judge": CLEAN},
        {"assistant_turn": REPAIR_2, "leak_terms": None},
    )
    fake_run_store.finish("job-2", reviewed(review_point("job-0", step=2, candidates=candidates)))

    coordinator.tick()

    about = {
        "point": 1,
        "point_id": "job-0-step2",
        "step": 2,
        "message_step": 1,
        "parent_job_id": "job-0",
        "outcome": "FF",
        "root_counts": counts(1, 1, 0),
    }
    repair = {**about, "kind": "repair", "samples": 2, "round": 1, "assistant_turn_rendering": "reasoning"}
    _root, student, repair_1, repair_2 = groups.get(GROUP)["document"]["samples"]
    assert [student["search"], repair_1["search"], repair_2["search"]] == [
        {**about, "kind": "student", "group": "student:job-0-step2"},
        {
            **repair,
            "group": "repair:job-0-step2:1",
            "candidate": 1,
            "multiplicity": 3,
            "leak_terms": [],
            "leak_judge": [CLEAN],
        },
        {**repair, "group": "repair:job-0-step2:2", "candidate": 2, "multiplicity": 1, "leak_terms": None},
    ]


def test_a_points_only_review_branches_unaided_students_alone(fake_run_store, groups, clock):
    """The plan of a points-only review may still hold repair turns. A split task runs none of them."""
    stored(groups, search_group(roots=2, tf_student_continuations=2))
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(True), verdict(False))
    fake_run_store.finish(
        "job-4", reviewed(review_point("job-1", candidates=({"assistant_turn": REPAIR_1},)))
    )

    coordinator.tick()

    student = {
        "point_id": "job-1-step2",
        "continuation": {"guidance": "none"},
        "overrides": {},
        "budget": {"max_turns": 9, "deadline_at": 1900.0},
    }
    assert fake_run_store.branches() == [("job-1", student), ("job-1", student)]


def test_a_review_point_that_no_branch_can_start_from_is_skipped_with_the_reason(
    fake_run_store, groups, clock
):
    """``job-1`` is the root's grade, not a root. The root has no recovery point after tool call 7."""
    stored(groups, search_group())
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    plan = reviewed(
        review_point("job-1", step=2), review_point("job-0", step=7), review_point("job-0", step=3)
    )
    fake_run_store.finish("job-2", plan)

    coordinator.tick()

    assert (search_of(groups)["skipped"], branch_budgets(fake_run_store)) == (
        [
            {"point": 1, "step": 2, "reason": "not a root of this group"},
            {"point": 2, "step": 7, "reason": "no available recovery point at that step"},
        ],
        [("job-0-step3", {"max_turns": 10 - 2, "deadline_at": 1000.0 + 900.0})],
    )


def test_a_review_point_in_a_job_that_the_run_store_does_not_know_is_skipped(fake_run_store, groups):
    stored(groups, search_group())
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    fake_run_store.finish("job-2", reviewed(review_point("job-404"), review_point("job-0")))

    coordinator.tick()

    search = search_of(groups)
    assert (search.get("skipped"), search.get("last_error"), len(fake_run_store.branches())) == (
        [{"point": 1, "step": 2, "reason": "not a root of this group"}],
        None,
        1,
    )


def test_branches_beyond_the_free_slots_are_skipped_and_the_others_fill_the_slots_in_order(
    fake_run_store, groups
):
    stored(groups, search_group(slots=2))
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    fake_run_store.finish(
        "job-2", reviewed(review_point("job-0", candidates=({"assistant_turn": REPAIR_1},) * 2))
    )

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    _root, first, second = settled["samples"]
    assert (
        (first["sample_slot_id"], first["search"]["kind"]),
        (second["sample_slot_id"], second["search"]["kind"]),
        settled["search"]["skipped"],
        settled["search"]["free_slots"],
    ) == (("slot-1", "student"), ("slot-2", "repair"), [{"point": 1, "reason": "no free slot"}], [])


def test_a_review_that_proposes_no_branch_that_can_be_made_ends_the_search(fake_run_store, groups):
    stored(groups, search_group())
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    fake_run_store.finish("job-2", reviewed())

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    assert (settled["search"]["state"], settled["search"]["error"], settled["ready"], settled["status"]) == (
        "done",
        "the review proposed no branch that could be made",
        True,
        "completed",
    )


@pytest.mark.parametrize("parent_max_turns, max_turns", [(3, 2), (2, 1), (1, 1)])
def test_a_branch_gets_the_turns_that_its_root_had_left_at_the_point_and_at_least_one(
    fake_run_store, groups, clock, parent_max_turns, max_turns
):
    """The point is at message step 1."""
    stored(groups, search_group(parent_max_turns=parent_max_turns))
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    fake_run_store.finish("job-2", reviewed(review_point("job-0", step=2)))

    coordinator.tick()

    assert branch_budgets(fake_run_store) == [
        ("job-0-step2", {"max_turns": max_turns, "deadline_at": 1900.0})
    ]


def test_a_search_group_is_ready_once_its_branches_and_their_grades_settle(fake_run_store, groups):
    """A branch is graded from its own final snapshot, with the grade template of its root."""
    stored(groups, search_group())
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    fake_run_store.finish(
        "job-2", reviewed(review_point("job-0", candidates=({"assistant_turn": REPAIR_1},)))
    )
    coordinator.tick()

    settle_branches(fake_run_store, coordinator, verdict(False), verdict(True))

    settled = groups.get(GROUP)
    root_grade, student_grade, repair_grade = fake_run_store.submitted("grade")
    assert (
        settled["document"]["search"]["state"],
        settled["terminal"],
        settled["document"]["status"],
        (student_grade["spec"], repair_grade["spec"]),
    ) == (
        "done",
        True,
        "completed",
        ({**root_grade["spec"], "snapshot_id": "ABC"}, {**root_grade["spec"], "snapshot_id": "ABC"}),
    )


def test_a_review_whose_reply_was_lost_before_a_cancel_is_attached_by_its_slot_and_stopped(
    fake_run_store, groups
):
    """The review is the phase ``review`` of the slot ``REVIEW_SLOT``. Its submission is not posted again."""
    stored(groups, search_group())
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()
    fake_run_store.finish("job-1", verdict(False))
    fake_run_store.lose_next_reply()
    coordinator.tick()
    groups.release(GROUP)
    coordinator.tick()

    coordinator.reconcile_submission(GROUP, sample_id=REVIEW_SLOT, phase="review", job_id="job-2")
    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    assert (
        fake_run_store.job("job-2")["state"],
        settled["search"]["state"],
        settled["status"],
        fake_run_store.requests.count(("POST", "jobs")),
    ) == ("cancelled", "done", "cancelled", 3)


def test_a_failed_read_while_the_branches_are_planned_is_retried_without_repeating_a_point(
    fake_run_store, groups
):
    stored(groups, search_group(roots=2))
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False), verdict(False))
    fake_run_store.finish("job-4", reviewed(review_point("job-0"), review_point("job-1")))
    fake_run_store.fail_next("GET", "recovery-points", status=503, after=1)

    coordinator.tick()
    coordinator.tick()

    about = {"step": 2, "message_step": 1, "outcome": "FF", "root_counts": counts(2, 2, 0)}
    about.update(snapshot_id="AB", why="before the edit", candidates=0)
    assert search_of(groups)["points"] == [
        {**about, "point": 1, "point_id": "job-0-step2", "parent_job_id": "job-0"},
        {**about, "point": 2, "point_id": "job-1-step2", "parent_job_id": "job-1"},
    ]


def test_a_search_group_that_runs_again_starts_its_search_again_with_every_branch_slot_free(
    fake_run_store, groups
):
    """A branch takes the first free slot, so the free slots of the new run are those of the group's start."""
    initial = search_group()
    stored(groups, initial)
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    fake_run_store.finish(
        "job-2", reviewed(review_point("job-0", candidates=({"assistant_turn": REPAIR_1},)))
    )
    coordinator.tick()
    lose(fake_run_store, "job-3")

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    (earlier,) = settled["earlier_runs"]
    assert (
        settled["samples"],
        settled["search"],
        [sample["sample_slot_id"] for sample in earlier["samples"]],
        (earlier["search"]["state"], earlier["search"]["free_slots"]),
    ) == (initial["samples"], initial["search"], ["root-0", "slot-1", "slot-2"], ("branching", []))


def test_the_roots_of_a_run_that_lost_a_job_are_not_reviewed(fake_run_store, groups):
    """The group runs again from its roots. A review of the roots of the lost run would be wasted."""
    stored(groups, search_group(roots=2, tf_student_continuations=1))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()
    fake_run_store.finish("job-2", verdict(False))
    lose(fake_run_store, "job-1")

    coordinator.tick()

    assert (fake_run_store.job_ids("review"), groups.get(GROUP)["document"]["affinity"]) == ([], "group-r1")


def test_a_review_that_lost_its_data_runs_the_group_again_and_does_not_end_its_search(fake_run_store, groups):
    stored(groups, search_group())
    coordinator = coordinator_for(fake_run_store, groups)
    settle_roots(fake_run_store, coordinator, verdict(False))
    lose(fake_run_store, "job-2")

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    assert (settled["ready"], settled["affinity"], settled["search"]["state"]) == (False, "group-r1", "roots")
