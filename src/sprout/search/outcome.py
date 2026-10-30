"""What a search's roots came to, and what the search samples after them.

``search_outcome`` labels the roots by three counts: how many were planned, how
many were graded (exported, with a grade that gave a verdict) and how many of
those resolved. A missing root counts as neither resolved nor failed. The label
decides what the search samples next (``SHAPES``, ``search_shape``)::

    TT          every root graded, every one resolved      none
    TT_partial  every graded root resolved, some missing   none
    ungraded    no root graded                             none
    TF          some resolved, some failed                 points_only
    FF_partial  none resolved, some failed, some missing   points_only
    FF          every root graded, none resolved           full

The predicates on one sample of a group document (``export_problem``,
``grade_verdict``, ``zero_reason``, ``root_verdict``) decide both which roots
the search counts and which samples a caller exports, so the two agree. Every
function is pure. docs/RL_DRIVER.md describes what each shape samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sprout.contract.messages import INVALID_STOP_REASONS, clean_messages, is_truncated_result

#: What the search samples after roots with each label.
SHAPES = {
    "TT": "none",
    "TT_partial": "none",
    "ungraded": "none",
    "TF": "points_only",
    "FF_partial": "points_only",
    "FF": "full",
}

_NO_STUDENTS = "tf_student_continuations is 0: a task whose roots split is not branched"
#: Why a search of shape ``none`` ends at its roots, by label. ``done_reason`` fills in ``{missing}``.
#: A points-only label has shape ``none`` only when no student would run from its points.
DONE_REASONS = {
    "TT": "every root resolved the task",
    "TT_partial": "no gradable failed root: every graded root resolved, {missing} grade(s) missing",
    "ungraded": "no root could be graded",
    "TF": _NO_STUDENTS,
    "FF_partial": _NO_STUDENTS,
}


# Miles recomputes this label from the roots it is sent and refuses a group whose label
# differs: a change here needs the same change in miles.rollout.sprout.gsml.search_outcome.
def search_outcome(planned: int, graded: int, resolved: int) -> str:
    """The label of a search task's roots: planned, graded (a verdict came back) and resolved.

    Raises ValueError unless the counts are ints with ``0 <= resolved <= graded <= planned``.
    """
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


def search_shape(outcome: str, config: Mapping[str, Any]) -> str:
    """What the search samples after roots labelled ``outcome``, under the request's search ``config``.

    ``SHAPES[outcome]``, except that a points-only shape with no
    ``tf_student_continuations`` to run samples nothing, not even its review.
    """
    shape = SHAPES[outcome]
    if shape == "points_only" and config["tf_student_continuations"] == 0:
        return "none"
    return shape


def done_reason(outcome: str, root_counts: Mapping[str, int]) -> str:
    """``DONE_REASONS[outcome]`` for roots with these counts."""
    return DONE_REASONS[outcome].format(missing=root_counts["planned"] - root_counts["graded"])


def export_problem(sample: dict[str, Any]) -> str | None:
    """Why a group's sample exports no trajectory, or None when it exports one.

    The checks, in order: the actor completed, a truncated one left a verified cutoff, the worker exported
    the cleaned native history with its provenance, and a grade was planned. A grade need not give a verdict.
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
            # The provenance indexes these messages, so nothing may shift them.
            return "Exported messages are not in their cleaned training form"
    except (ValueError, KeyError, TypeError) as error:  # A history clean_messages refuses, or a bad record.
        return str(error)
    if not sample.get("grade"):
        return "Sample has no grade"
    return None


def invalid_reason(sample: dict[str, Any]) -> str | None:
    """The stop reason of an actor that ended as an invalid attempt (``INVALID_STOP_REASONS``), else None.

    No verifier grades such an attempt: it failed, and its reward is ``INVALID_REWARD``."""
    actor = sample.get("actor") or {}
    reason = (actor.get("result") or {}).get("stop_reason")
    return reason if actor.get("state") == "succeeded" and reason in INVALID_STOP_REASONS else None


def grade_verdict(sample: dict[str, Any]) -> dict[str, Any] | None:
    """The sample's grade result when it is a verdict, and None otherwise.

    A verdict is the result of a grade that succeeded and says whether the task resolved (a bool
    ``resolved``). An invalid attempt has the verdict ``{"resolved": False, "invalid": <stop reason>}``.
    """
    invalid = invalid_reason(sample)
    if invalid is not None:
        return {"resolved": False, "invalid": invalid}
    grade = sample.get("grade") or {}
    result = grade.get("result") or {}
    if grade.get("state") != "succeeded" or type(result.get("resolved")) is not bool:
        return None
    return result


def zero_reason(sample: dict[str, Any]) -> str | None:
    """Why a graded sample has no verdict, or None when it has one. Such a sample is missing, not a 0.

    ``grade_no_verdict`` for a grade that succeeded without saying whether the task resolved,
    ``grade_<state>`` for one that did not succeed: failed, expired, cancelled or skipped.
    """
    if grade_verdict(sample) is not None:
        return None
    state = sample["grade"]["state"]
    return "grade_no_verdict" if state == "succeeded" else f"grade_{state}"


def root_verdict(sample: dict[str, Any]) -> bool | None:
    """Whether a root resolved the task, or None when it is missing: no trajectory, or no verdict.

    The search counts and reviews roots by this. A caller that counts the
    roots it exports by it counts the same ones.
    """
    if export_problem(sample) is not None:
        return None
    verdict = grade_verdict(sample)
    return None if verdict is None else verdict["resolved"]
