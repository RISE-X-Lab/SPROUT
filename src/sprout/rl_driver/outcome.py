"""What a search's roots came to, and what the search samples after them.

The roots of a search task are labelled by three counts (``search_outcome``):
how many were planned, how many were graded -- exported as a trajectory whose
grade gave a verdict -- and how many of those resolved. A root whose actor
failed, whose export is unusable, or whose grade failed, expired or gave no
verdict is missing: neither resolved nor failed, it counts as neither.

The label decides what the driver samples next (``SHAPES``, ``search_shape``)::

    TT          every root graded, every one resolved      none
    TT_partial  every graded root resolved, some missing   none
    ungraded    no root graded                             none
    TF          some resolved, some failed                 points_only
    FF_partial  none resolved, some failed, some missing   points_only
    FF          every root graded, none resolved           full

``full`` is a review that picks points in each failed root and writes repair
turns there, then students (nothing added) and repairs (a turn inserted) from
each point; ``points_only`` a review that picks the points only, then
``tf_student_continuations`` students from each, or nothing at all when that
is 0; ``none`` ends the search at its roots (``DONE_REASONS`` says why). Miles
labels the roots it is sent with the same function
(``miles.rollout.sprout.gsml.search_outcome``) and refuses a group whose
annotations say otherwise; only an FF task earns its λ credit.

The predicates on one sample of a driver document are shared by the search and
the export (``messages.MessageAdapter._export``), so the roots the driver
counts are the roots the trainer is sent: ``export_problem`` says why a sample
has no trajectory, ``grade_verdict`` and ``zero_reason`` whether its grade
gave a verdict, ``root_verdict`` both at once. Every function is pure.
"""

from __future__ import annotations

from collections.abc import Mapping

from sprout.runstore.messages import clean_messages, is_truncated_result

#: What the search samples after roots with each label. New shapes -- roots
#: that confirm a split, children of a resolved root -- would be added here.
SHAPES = {
    "TT": "none",
    "TT_partial": "none",
    "ungraded": "none",
    "TF": "points_only",
    "FF_partial": "points_only",
    "FF": "full",
}

_NO_STUDENTS = "tf_student_continuations is 0: a task whose roots split is not branched"
#: Why a search whose shape is ``none`` ends at its roots, by label (``done_reason``
#: fills in ``missing``). A points-only label has that shape only when no
#: student would run from its points.
DONE_REASONS = {
    "TT": "every root resolved the task",
    "TT_partial": "no gradable failed root: every graded root resolved, {missing} grade(s) missing",
    "ungraded": "no root could be graded",
    "TF": _NO_STUDENTS,
    "FF_partial": _NO_STUDENTS,
}


def search_outcome(planned: int, graded: int, resolved: int) -> str:
    """The label of a search task's roots: planned, graded (a verdict came back) and resolved."""
    if (
        any(type(count) is not int for count in (planned, graded, resolved))
        or not 0 <= resolved <= graded <= planned
    ):
        raise ValueError(
            f"expected 0 <= resolved <= graded <= planned, got {planned=}, {graded=}, {resolved=}"
        )
    if graded == 0:
        return "ungraded"
    if resolved == graded:
        return "TT" if graded == planned else "TT_partial"
    if resolved >= 1:
        return "TF"
    return "FF" if graded == planned else "FF_partial"


def search_shape(outcome: str, config: Mapping) -> str:
    """What the search samples after roots labelled ``outcome``, under the request's search ``config``.

    ``SHAPES[outcome]``, except that a points-only shape with no
    ``tf_student_continuations`` to run samples nothing, not even its review.
    """
    shape = SHAPES[outcome]
    if shape == "points_only" and config["tf_student_continuations"] == 0:
        return "none"
    return shape


def done_reason(outcome: str, root_counts: Mapping) -> str:
    """``DONE_REASONS[outcome]`` for roots with these counts."""
    return DONE_REASONS[outcome].format(missing=root_counts["planned"] - root_counts["graded"])


def export_problem(sample: dict) -> str | None:
    """Why a driver sample exports no trajectory, or None when it exports one.

    The export's checks, in order, each with the reason a failed sample is
    returned with: the actor completed, a truncated one left a verified
    cutoff, the worker exported the native history in its cleaned training
    form and said which of it the policy sampled, and a grade was planned.
    Whether that grade gave a verdict is not one of them: a sample whose grade
    did not is still exported, marked missing (``zero_reason``).
    """
    try:
        actor = sample["actor"]
        if actor["state"] != "succeeded":
            return "Actor did not complete"
        output = actor.get("result") or {}
        if output.get("status") == "truncated" and not is_truncated_result(output):
            return "Truncated actor lacks a verified cutoff result"
        if output.get("training_export_error"):
            return str(output["training_export_error"])
        messages = output.get("training_messages")
        if not isinstance(messages, list) or not messages:
            return "Execution has no exported native messages"
        if not isinstance(output.get("training_provenance"), dict):
            return "Execution does not say which exported messages the policy sampled"
        if clean_messages(messages) != messages:
            # The provenance indexes these messages; nothing may shift them.
            return "Exported messages are not in their cleaned training form"
    except (ValueError, KeyError, TypeError) as error:  # a history clean_messages refuses, a malformed record
        return str(error)
    if not sample.get("grade"):
        return "Sample has no grade"
    return None


def grade_verdict(sample: dict) -> dict | None:
    """The sample's grade result when it is a verdict -- the grade succeeded and says
    whether the task resolved (a bool ``resolved``) -- and None otherwise."""
    grade = sample.get("grade") or {}
    result = grade.get("result") or {}
    if grade.get("state") != "succeeded" or type(result.get("resolved")) is not bool:
        return None
    return result


def zero_reason(sample: dict) -> str | None:
    """Why a graded sample has no verdict, or None when it has one.

    ``grade_no_verdict`` for a grade that succeeded without saying whether the
    task resolved, ``grade_<state>`` for one that did not succeed: failed,
    expired, cancelled or skipped. The trainer counts such a sample as
    missing, never as a 0.
    """
    if grade_verdict(sample) is not None:
        return None
    state = sample["grade"]["state"]
    return "grade_no_verdict" if state == "succeeded" else f"grade_{state}"


def root_verdict(sample: dict) -> bool | None:
    """Whether a root resolved the task; None when it is missing (no trajectory, or no verdict).

    The driver counts and reviews roots by this, and the trainer, from the
    trajectories it is sent, counts the same ones.
    """
    if export_problem(sample) is not None:
        return None
    verdict = grade_verdict(sample)
    return None if verdict is None else verdict["resolved"]
