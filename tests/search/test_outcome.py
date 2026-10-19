"""What the roots of a search came to, and which samples have a trajectory and a verdict.

Miles recomputes the label of the roots with its own copy of ``search_outcome`` and refuses a group whose
label differs, so the label table is a contract with Miles. The table of docs/RL_DRIVER.md gives the
shape that follows each label.
"""

from __future__ import annotations

from typing import Any

import pytest

from sprout.contract.messages import mark_hint
from sprout.search.outcome import export_problem, root_verdict, search_outcome, search_shape, zero_reason
from tests.search.documents import NO_VERDICT, exported, verdict


@pytest.mark.parametrize(
    "planned, graded, resolved, label",
    [
        (1, 0, 0, "ungraded"),
        (2, 0, 0, "ungraded"),
        (1, 1, 1, "TT"),
        (2, 2, 2, "TT"),
        (3, 2, 2, "TT_partial"),
        (2, 2, 1, "TF"),
        (3, 2, 1, "TF"),
        (1, 1, 0, "FF"),
        (2, 2, 0, "FF"),
        (2, 1, 0, "FF_partial"),
    ],
)
def test_the_label_of_the_roots_follows_from_how_many_were_planned_graded_and_resolved(
    planned, graded, resolved, label
):
    assert search_outcome(planned, graded, resolved) == label


@pytest.mark.parametrize(
    "planned, graded, resolved",
    [(1, 2, 0), (2, 1, 2), (2, 2, -1), (2, 2, True), (2.0, 2, 0)],
    ids=["graded above planned", "resolved above graded", "negative", "a bool", "a float"],
)
def test_counts_that_no_roots_can_have_get_no_label(planned, graded, resolved):
    with pytest.raises(ValueError, match="expected 0 <= resolved <= graded <= planned"):
        search_outcome(planned, graded, resolved)


@pytest.mark.parametrize(
    "label, tf_students, shape",
    [
        ("TT", 2, "none"),
        ("TT_partial", 2, "none"),
        ("ungraded", 2, "none"),
        ("TF", 2, "points_only"),
        ("FF_partial", 2, "points_only"),
        ("FF", 2, "full"),
        ("TF", 0, "none"),
        ("FF_partial", 0, "none"),
        ("FF", 0, "full"),
    ],
)
def test_the_label_decides_what_the_search_samples_next(label, tf_students, shape):
    """A points-only label samples nothing when no unaided student would run from its points."""
    assert search_shape(label, {"tf_student_continuations": tf_students}) == shape


def sample(*, actor: dict[str, Any] | None = None, grade: dict[str, Any] | None = None) -> dict[str, Any]:
    """A sample of a group document whose actor succeeded with ``exported()`` and whose grade resolved."""
    return {
        "actor": actor or {"state": "succeeded", "result": exported()},
        "grade": grade or {"state": "succeeded", "result": verdict(True)},
    }


def succeeded(**changes: Any) -> dict[str, Any]:
    """An actor that succeeded with ``exported(**changes)``."""
    return {"state": "succeeded", "result": exported(**changes)}


HINTED = [
    {"role": "user", "content": mark_hint("check the parser")},
    {"role": "assistant", "content": "Fixed."},
]
TRUNCATED = {"status": "truncated", "stop_reason": "max_turns_reached"}


@pytest.mark.parametrize(
    "actor, problem",
    [
        ({"state": "failed", "result": exported()}, "Actor did not complete"),
        (succeeded(**TRUNCATED), "Truncated actor lacks a verified cutoff result"),
        (
            succeeded(training_export_error="Native history has an incomplete final record"),
            "Native history has an incomplete final record",
        ),
        (succeeded(training_messages=None), "Execution has no exported native messages"),
        (succeeded(training_messages=[]), "Execution has no exported native messages"),
        (
            succeeded(training_provenance=None),
            "Execution does not say which exported messages the policy sampled",
        ),
        (succeeded(training_messages=HINTED), "Exported messages are not in their cleaned training form"),
        (
            succeeded(training_messages=[{"role": "robot", "content": "beep"}]),
            "Unsupported message role 'robot'",
        ),
        ({"state": "succeeded"}, "Execution has no exported native messages"),
    ],
    ids=[
        "actor failed",
        "truncated without a cutoff",
        "export failed",
        "no messages",
        "empty messages",
        "no provenance",
        "a hint left in",
        "a history that cannot be cleaned",
        "no result",
    ],
)
def test_a_sample_without_a_trajectory_to_train_on_says_why(actor, problem):
    assert export_problem(sample(actor=actor)) == problem


def test_a_sample_without_a_planned_grade_has_no_trajectory():
    assert export_problem({"actor": succeeded(), "grade": None}) == "Sample has no grade"


@pytest.mark.parametrize(
    "actor",
    [succeeded(), succeeded(**TRUNCATED, final_snapshot_id="FINAL")],
    ids=["completed", "truncated with a verified cutoff"],
)
def test_a_sample_whose_actor_exported_its_cleaned_history_has_a_trajectory(actor):
    assert export_problem(sample(actor=actor)) is None


@pytest.mark.parametrize(
    "grade, reason",
    [
        ({"state": "succeeded", "result": verdict(True)}, None),
        ({"state": "succeeded", "result": verdict(False)}, None),
        ({"state": "succeeded", "result": {"status": "completed", "resolved": None}}, "grade_no_verdict"),
        ({"state": "succeeded", "result": {"status": "completed", "resolved": 1}}, "grade_no_verdict"),
        ({"state": "failed", "result": NO_VERDICT}, "grade_failed"),
        ({"state": "expired", "result": None}, "grade_expired"),
        ({"state": "cancelled", "result": None}, "grade_cancelled"),
        ({"state": "skipped"}, "grade_skipped"),
    ],
)
def test_a_graded_sample_without_a_verdict_names_the_reason_that_its_reward_is_missing(grade, reason):
    """Each reason is a ``metadata.zero_reason`` of the Miles wire (docs/RL_DRIVER.md)."""
    assert zero_reason(sample(grade=grade)) == reason


@pytest.mark.parametrize(
    "actor, grade, resolved",
    [
        (succeeded(), {"state": "succeeded", "result": verdict(True)}, True),
        (succeeded(), {"state": "succeeded", "result": verdict(False)}, False),
        (succeeded(), {"state": "failed", "result": NO_VERDICT}, None),
        (
            succeeded(training_export_error="no history"),
            {"state": "succeeded", "result": verdict(True)},
            None,
        ),
    ],
    ids=["resolved", "failed", "no verdict", "no trajectory"],
)
def test_a_root_counts_only_with_a_trajectory_and_a_verdict(actor, grade, resolved):
    """A caller that counts the roots that it exports by ``root_verdict`` counts the roots of the label."""
    assert root_verdict(sample(actor=actor, grade=grade)) is resolved
