"""The later repair rounds of a search: a failed repair is reviewed alone, and its chain grows a turn.

Each search here has one root, one point and one repair turn in each of ``repair_rounds`` rounds. The
root fails, its review picks its step 2, and round 1 makes a student and a repair there. The jobs are
``job-0`` (the root), ``job-1`` (its grade), ``job-2`` (its review), ``job-3`` and ``job-4`` (the
student and the repair), ``job-5`` and ``job-6`` (their grades) and ``job-7`` (the review of round 2).
"""

from __future__ import annotations

from typing import Any

import pytest

from sprout.eval.groups import FileGroups
from sprout.search.coordinator import Coordinator
from tests.search.documents import (
    GROUP,
    NO_VERDICT,
    REVIEW,
    coordinator_for,
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
ROUND_1_END = "after round 1: no unresolved point with a failed repair to review"


def first_round(
    fake: FakeRunStore, groups: FileGroups, *, rounds: int = 3, candidate: dict[str, Any] | None = None
) -> Coordinator:
    """A coordinator whose search has run its root and the review of the root: the branches of round 1
    are submitted. The review's one repair turn is ``candidate``, by default ``REPAIR_1`` alone."""
    stored(groups, search_group(repair_rounds=rounds))
    coordinator = coordinator_for(fake, groups)
    settle_roots(fake, coordinator, verdict(False))
    fake.finish(
        "job-2", reviewed(review_point("job-0", candidates=(candidate or {"assistant_turn": REPAIR_1},)))
    )
    coordinator.tick()
    return coordinator


def test_a_failed_repair_at_an_unresolved_point_is_reviewed_alone_in_the_next_round(fake_run_store, groups):
    """The chain review picks one point in the repair's history and writes one more turn there."""
    coordinator = first_round(fake_run_store, groups)

    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))

    search = search_of(groups)
    chain_reviews = fake_run_store.submitted("review")[1:]
    assert (search["state"], search.get("round"), chain_reviews) == (
        "chaining",
        2,
        [
            {
                "kind": "review",
                "profile": "review",
                "max_infra_retries": 0,
                "spec": {
                    **REVIEW,
                    "guidance": "assistant-turn",
                    "points": 1,
                    "points_per_attempt": True,
                    "candidates": 1,
                    "timeout_s": 600.0,
                    "round_no": 2,
                    "parents": [
                        {
                            "name": "slot-2",
                            "job_id": "job-4",
                            "attempt_id": "attempt-job-4",
                            "grade": {"resolved": False, "progress": 0.0},
                        }
                    ],
                },
                "context": {
                    "eval": {"rollout_job_id": "group", "phase": "review", "outcome": "FF", "round": 2}
                },
                "parent_point": None,
                "affinity": "group",
            }
        ],
    )


def test_a_chain_branch_continues_its_parent_repair_and_belongs_to_the_point_of_the_root(
    fake_run_store, groups, clock
):
    """Its turn limit is what the parent repair had left at the chosen point. The trainer groups it with the
    repairs of the root's point, numbered on from the candidates of the earlier rounds."""
    coordinator = first_round(fake_run_store, groups)
    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))
    fake_run_store.finish(
        "job-7", reviewed(review_point("job-4", step=3, candidates=({"assistant_turn": REPAIR_2},)))
    )

    coordinator.tick()

    *_, chain = groups.get(GROUP)["document"]["samples"]
    assert (fake_run_store.branches()[-1], chain["search"]) == (
        (
            "job-4",
            {
                "point_id": "job-4-step3",
                "continuation": {
                    "guidance": "assistant-turn",
                    "assistant_turn": REPAIR_2,
                    "assistant_turn_rendering": "reasoning",
                },
                "overrides": {},
                "budget": {"max_turns": (10 - 1) - 2, "deadline_at": 1000.0 + 900.0},
            },
        ),
        {
            "kind": "repair",
            "group": "repair:job-0-step2:2",
            "point": 1,
            "point_id": "job-0-step2",
            "step": 2,
            "message_step": 1,
            "parent_job_id": "job-0",
            "outcome": "FF",
            "root_counts": {"planned": 1, "graded": 1, "resolved": 0},
            "round": 2,
            "chain_parent_job_id": "job-4",
            "chain_point_id": "job-4-step3",
            "chain_step": 3,
            "candidate": 2,
            "multiplicity": 1,
            "samples": 2,
            "leak_terms": None,
            "assistant_turn_rendering": "reasoning",
        },
    )


@pytest.mark.parametrize(
    "earlier, later, merged",
    [
        ([], ["tests/test_hidden.py"], ["tests/test_hidden.py"]),
        (["a_hidden.py"], ["b_hidden.py", "a_hidden.py"], ["a_hidden.py", "b_hidden.py"]),
        (None, [], None),
        ([], None, None),
    ],
    ids=[
        "a new term",
        "the union, earlier terms first",
        "an earlier turn unchecked",
        "a later turn unchecked",
    ],
)
def test_a_chain_repair_carries_the_leak_terms_of_every_turn_on_its_chain(
    fake_run_store, groups, earlier, later, merged
):
    """None says that a turn had nothing to check against, so the chain cannot be called clean."""
    coordinator = first_round(
        fake_run_store, groups, candidate={"assistant_turn": REPAIR_1, "leak_terms": earlier}
    )
    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))
    candidate = {"assistant_turn": REPAIR_2, "leak_terms": later}
    fake_run_store.finish("job-7", reviewed(review_point("job-4", step=3, candidates=(candidate,))))

    coordinator.tick()

    *_, chain = groups.get(GROUP)["document"]["samples"]
    assert chain["search"]["leak_terms"] == merged


def test_a_chain_repair_carries_the_leak_judges_check_of_every_turn_on_its_chain_in_round_order(
    fake_run_store, groups
):
    flagged = {"leak": True, "leak_terms": [], "jev": {"score": 0.93, "leak": True}}
    clean = {"leak": False, "leak_terms": [], "jev": {"score": 0.1, "leak": False}}
    coordinator = first_round(
        fake_run_store, groups, candidate={"assistant_turn": REPAIR_1, "leak_judge": flagged}
    )
    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))
    candidate = {"assistant_turn": REPAIR_2, "leak_judge": clean}
    fake_run_store.finish("job-7", reviewed(review_point("job-4", step=3, candidates=(candidate,))))

    coordinator.tick()

    *_, chain = groups.get(GROUP)["document"]["samples"]
    assert chain["search"]["leak_judge"] == [flagged, clean]


@pytest.mark.parametrize(
    "student, repair",
    [
        (verdict(True), verdict(False)),
        (verdict(False), verdict(True, touched_test_paths=[])),
        (NO_VERDICT, verdict(False)),
    ],
    ids=["a student resolved", "a repair resolved", "no student was graded"],
)
def test_a_point_that_resolved_or_has_no_graded_student_is_not_reviewed_again(
    fake_run_store, groups, student, repair
):
    """The trainer credits a repair only at a point where a student was graded and none resolved."""
    coordinator = first_round(fake_run_store, groups)

    settle_branches(fake_run_store, coordinator, student, repair)

    search = search_of(groups)
    assert (search["state"], search.get("chain_end"), len(fake_run_store.job_ids("review"))) == (
        "done",
        ROUND_1_END,
        1,
    )


@pytest.mark.parametrize(
    "repair",
    [verdict(True, touched_test_paths=["tests/test_hidden.py"]), verdict(True)],
    ids=["it changed a hidden test file", "it cannot say"],
)
def test_a_repair_that_resolved_by_changing_the_hidden_tests_does_not_resolve_its_point(
    fake_run_store, groups, repair
):
    coordinator = first_round(fake_run_store, groups)

    settle_branches(fake_run_store, coordinator, verdict(False), repair)

    assert (search_of(groups)["state"], len(fake_run_store.job_ids("review"))) == ("chaining", 2)


def test_the_last_repair_round_ends_the_search(fake_run_store, groups):
    coordinator = first_round(fake_run_store, groups, rounds=2)
    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))
    fake_run_store.finish(
        "job-7", reviewed(review_point("job-4", step=3, candidates=({"assistant_turn": REPAIR_2},)))
    )
    coordinator.tick()

    settle_branches(fake_run_store, coordinator, verdict(False))

    settled = groups.get(GROUP)
    search = settled["document"]["search"]
    assert (search["state"], search["round"], len(fake_run_store.job_ids("review")), settled["terminal"]) == (
        "done",
        2,
        2,
        True,
    )


def test_a_chain_review_that_does_not_succeed_ends_the_search_with_its_chain_error(fake_run_store, groups):
    coordinator = first_round(fake_run_store, groups)
    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))
    fake_run_store.finish("job-7", {"status": "error", "error": "model endpoint refused"}, state="failed")

    coordinator.tick()

    search = search_of(groups)
    assert (search["state"], search.get("chain_error"), search.get("error")) == (
        "done",
        "model endpoint refused",
        None,
    )


def test_a_chain_review_that_proposes_no_branch_that_can_be_made_ends_the_search(fake_run_store, groups):
    coordinator = first_round(fake_run_store, groups)
    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))
    fake_run_store.finish(
        "job-7", reviewed(review_point("job-0", candidates=({"assistant_turn": REPAIR_2},)))
    )

    coordinator.tick()

    search = search_of(groups)
    assert (search["state"], search.get("skipped"), search.get("chain_error")) == (
        "done",
        [{"round": 2, "step": 2, "reason": "not a repair of this group"}],
        "round 2's review proposed no branch that could be made",
    )


def test_a_failed_read_while_a_round_is_planned_is_retried_without_repeating_a_skipped_point(
    fake_run_store, groups
):
    """The first point is not a repair, so the search skips it before the read for the second point fails."""
    coordinator = first_round(fake_run_store, groups)
    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))
    plan = reviewed(
        review_point("job-0"), review_point("job-4", step=3, candidates=({"assistant_turn": REPAIR_2},))
    )
    fake_run_store.finish("job-7", plan)
    fake_run_store.fail_next("GET", "recovery-points", status=503)

    coordinator.tick()
    coordinator.tick()

    search = search_of(groups)
    assert (search["state"], search["skipped"]) == (
        "branching",
        [{"round": 2, "step": 2, "reason": "not a repair of this group"}],
    )


def test_a_chain_review_whose_reply_was_lost_before_a_cancel_is_attached_by_its_slot(fake_run_store, groups):
    """The review of round r is the phase ``review`` of the slot ``search-r<r>`` (docs/RL_DRIVER.md)."""
    coordinator = first_round(fake_run_store, groups)
    fake_run_store.lose_next_reply(after=2)
    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))
    groups.release(GROUP)
    coordinator.tick()

    coordinator.reconcile_submission(GROUP, sample_id="search-r2", phase="review", job_id="job-7")
    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    assert (
        fake_run_store.job("job-7")["state"],
        settled["search"]["state"],
        settled["status"],
        fake_run_store.requests.count(("POST", "jobs")),
    ) == ("cancelled", "done", "cancelled", 6)


def test_a_chain_review_that_lost_its_data_runs_the_group_again_and_does_not_end_its_search(
    fake_run_store, groups
):
    coordinator = first_round(fake_run_store, groups)
    settle_branches(fake_run_store, coordinator, verdict(False), verdict(False))
    lose(fake_run_store, "job-7")

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    assert (settled["ready"], settled["affinity"], settled["search"]["state"]) == (False, "group-r1", "roots")
