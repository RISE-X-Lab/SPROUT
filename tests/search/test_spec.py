"""What a search asks for (``SearchSpec``), and the search document that a group starts it from.

Miles sends the ``search`` object, allocates the slots of a request by ``SearchSpec.branches()``, and sets
the deadline of a group with ``search_seconds``. docs/RL_DRIVER.md gives the formulas.
"""

from __future__ import annotations

import math

import pytest

from sprout.search.spec import SearchSpec, search_document
from tests.search.documents import REVIEW, SEARCH


def spec(**changes: object) -> SearchSpec:
    """The ``SearchSpec`` of ``SEARCH`` with ``changes``."""
    return SearchSpec.from_dict({**SEARCH, **changes})


@pytest.mark.parametrize(
    "value",
    [
        {key: value for key, value in SEARCH.items() if key != "tf_student_continuations"},
        {**SEARCH, "rounds": 2},
        [SEARCH],
    ],
    ids=["a missing field", "an unknown field", "not an object"],
)
def test_a_search_object_without_exactly_the_fields_of_a_search_spec_is_refused(value):
    with pytest.raises(ValueError, match="search names roots, points, candidates, .* and wall_time_seconds"):
        SearchSpec.from_dict(value)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("roots", 0, "search.roots must be positive"),
        ("points", 0, "search.points must be positive"),
        ("student_continuations", 0, "search.student_continuations must be positive"),
        ("repair_rounds", 0, "search.repair_rounds must be positive"),
        ("roots", True, "search.roots must be a non-negative integer"),
        ("candidates", -1, "search.candidates must be a non-negative integer"),
        ("candidate_continuations", 1.0, "search.candidate_continuations must be a non-negative integer"),
        ("tf_student_continuations", True, "search.tf_student_continuations must be a non-negative integer"),
        ("review_seconds", 0, "search.review_seconds must be finite and positive"),
        ("wall_time_seconds", -1.0, "search.wall_time_seconds must be finite and positive"),
        ("review_seconds", math.inf, "search.review_seconds must be finite and positive"),
        ("review_seconds", math.nan, "search.review_seconds must be finite and positive"),
        ("review_seconds", "600", "search.review_seconds must be finite and positive"),
        ("review_seconds", True, "search.review_seconds must be finite and positive"),
    ],
)
def test_a_search_field_out_of_its_range_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        spec(**{field: value})


@pytest.mark.parametrize("candidates, candidate_continuations", [(0, 1), (2, 0)])
def test_repair_turns_and_their_continuations_are_both_zero_or_both_positive(
    candidates, candidate_continuations
):
    with pytest.raises(ValueError, match="both zero or both positive"):
        spec(candidates=candidates, candidate_continuations=candidate_continuations)


@pytest.mark.parametrize("candidates, candidate_continuations", [(2, 1), (1, 2)])
def test_more_than_one_repair_round_takes_one_repair_turn_and_one_continuation(
    candidates, candidate_continuations
):
    with pytest.raises(ValueError, match="more than one search.repair_rounds takes one candidate"):
        spec(repair_rounds=2, candidates=candidates, candidate_continuations=candidate_continuations)


@pytest.mark.parametrize(
    "changes, branches",
    [
        (
            {
                "roots": 2,
                "candidates": 2,
                "student_continuations": 2,
                "candidate_continuations": 2,
                "tf_student_continuations": 2,
            },
            12,
        ),
        (
            {
                "roots": 2,
                "points": 2,
                "candidates": 0,
                "candidate_continuations": 0,
                "tf_student_continuations": 5,
            },
            10,
        ),
        ({"roots": 2, "repair_rounds": 3, "tf_student_continuations": 2}, 8),
        ({"roots": 1, "tf_student_continuations": 9}, 2),
    ],
    ids=["every root failed", "all but one root failed", "three repair rounds", "one root"],
)
def test_branches_is_the_most_branches_that_any_label_of_the_roots_can_make(changes, branches):
    """``points * max(roots * (students + candidates * continuations * rounds), (roots - 1) * tf)``."""
    assert spec(**changes).branches() == branches


@pytest.mark.parametrize("repair_rounds, seconds", [(1, 600.0 + 900.0), (3, 3 * (600.0 + 900.0) + 2 * 300.0)])
def test_search_seconds_is_each_rounds_review_and_branches_and_the_grades_of_all_rounds_but_the_last(
    repair_rounds, seconds
):
    """The grades of the last round belong to the finalization time of the group."""
    searching = spec(repair_rounds=repair_rounds, review_seconds=600.0, wall_time_seconds=900.0)

    assert searching.search_seconds(300.0) == seconds


def document(searching: SearchSpec) -> dict[str, object]:
    return search_document(
        config=searching,
        review=REVIEW,
        review_profile="review",
        parent_max_turns=10,
        root_slots=["root-0"],
        free_slots=["slot-1", "slot-2"],
    )


def test_a_search_document_starts_at_its_roots_with_a_review_template_for_each_shape_that_reviews():
    """Each template lacks only the parents. A review job is not retried, and it picks the points of each
    failed root on its own. A points-only review writes no repair turn."""
    review = {
        **REVIEW,
        "guidance": "assistant-turn",
        "points": 1,
        "points_per_attempt": True,
        "candidates": 2,
        "timeout_s": 600.0,
    }

    started = document(spec(candidates=2))

    assert started == {
        "config": {**SEARCH, "candidates": 2},
        "state": "roots",
        "review": None,
        "points": [],
        "review_templates": {
            "full": {"kind": "review", "profile": "review", "max_infra_retries": 0, "spec": review},
            "points_only": {
                "kind": "review",
                "profile": "review",
                "max_infra_retries": 0,
                "spec": {**review, "guidance": "none", "candidates": 1},
            },
        },
        "parent_max_turns": 10,
        "root_slots": ["root-0"],
        "free_slots": ["slot-1", "slot-2"],
    }


def test_a_search_of_more_than_one_repair_round_has_a_chain_template_for_one_point_and_one_turn():
    templates = document(spec(repair_rounds=3))["review_templates"]

    chain = templates.get("chain", {}).get("spec", {})
    assert (sorted(templates), chain.get("points"), chain.get("candidates")) == (
        ["chain", "full", "points_only"],
        1,
        1,
    )


def test_a_search_without_repair_turns_reviews_for_points_alone_under_the_full_shape():
    full = document(spec(candidates=0, candidate_continuations=0))["review_templates"]["full"]

    assert (full["spec"]["guidance"], full["spec"]["candidates"]) == ("none", 1)


@pytest.mark.parametrize(
    "review, error, message",
    [
        ({**REVIEW, "model_endpoint": ""}, ValueError, "A review needs model_endpoint"),
        ({**REVIEW, "temperature": 3}, ValueError, "Review temperature must be between 0 and 2"),
        ({**REVIEW, "rounds": 2}, TypeError, "rounds"),
    ],
    ids=["no model endpoint", "a setting out of range", "an unknown field"],
)
def test_a_review_that_no_worker_can_run_is_refused_when_the_search_document_is_made(review, error, message):
    """Not later, when the roots of the group have finished."""
    with pytest.raises(error, match=message):
        search_document(
            config=spec(),
            review=review,
            review_profile="review",
            parent_max_turns=10,
            root_slots=["root-0"],
            free_slots=["slot-1"],
        )
