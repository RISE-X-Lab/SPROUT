"""A group request is checked before any job exists, and a stage's key stays the same on every try."""

from __future__ import annotations

from typing import Any

import pytest

from sprout.search.group import Submitter, identifier, validate_request
from tests.search.documents import GRADE, REVIEW, RUN, run

IN_PLACE = {
    "profile": "grade",
    "spec": {"benchmark": "harbor", "task_dir": "/path/to/task"},
    "in_place": True,
}
BRANCH = {"job_id": "job-0", "point_id": "job-0-step2", "continuation": {"guidance": "none"}}
REVIEW_JOB = {
    "profile": "review",
    "spec": {
        **REVIEW,
        "parents": [
            {"name": "root", "job_id": "job-x", "attempt_id": "attempt-x", "grade": {"resolved": False}}
        ],
    },
}
#: What ``JobSpec.validate`` adds to a request body that names only its kind, profile and spec.
JOB_DEFAULTS = {"context": {}, "max_infra_retries": 2, "parent_point": None}


def request(*samples: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """A group request of ``samples``, by default one graded run, with ``fields`` changed."""
    return {
        "rollout_job_id": "group",
        "prompt_group_id": "prompt",
        "samples": list(samples) or [run("slot-0")],
        **fields,
    }


def test_the_key_of_a_stage_is_the_submitters_prefix_and_a_hash_that_stored_groups_depend_on():
    """A restarted coordinator finds the jobs of a stored group by these keys. New keys submit them again.

    ``affinity`` is the key of the group's current run, which the RL driver sets to the group id."""
    submitter = Submitter(context_key="eval", key_prefix="eval/")

    key = submitter.key(affinity="group", sample_id="slot-0", phase="actor")

    assert key == "eval/3381537af98613b317f6eeaf70496bec7f9c5de35e9412b570598511acb6a5cd"


@pytest.mark.parametrize("value", ["a", "A" * 200, "0lead", "job-1_2.3:4"])
def test_an_identifier_is_1_to_200_url_safe_characters_that_start_with_a_letter_or_a_digit(value):
    assert identifier(value, "rollout_job_id") == value


@pytest.mark.parametrize(
    "value", ["", "a" * 201, "-lead", "_lead", ".lead", "has/slash", "has space", "a\n", 7, None]
)
def test_a_value_that_is_not_an_identifier_is_refused_with_its_name(value):
    with pytest.raises(ValueError, match="rollout_job_id must be a URL-safe identifier of 1..200 characters"):
        identifier(value, "rollout_job_id")


def runs(count: int) -> list[dict[str, Any]]:
    """``count`` graded runs, in the slots ``slot-0``, ``slot-1`` and so on."""
    return [run("slot-%d" % index) for index in range(count)]


def branch_sample(**changes: Any) -> dict[str, Any]:
    return {"sample_slot_id": "slot-0", "branch": {**BRANCH, **changes}}


SELECTED = "A grade's snapshot_id or in_place_job is selected from its actor"


@pytest.mark.parametrize(
    "body, message",
    [
        (request(protocol_version=1), "Invalid group fields"),
        (request(rollout_job_id="has/slash"), "rollout_job_id must be a URL-safe identifier"),
        (request(prompt_group_id=7), "prompt_group_id must be a URL-safe identifier"),
        (request(context=["fixture"]), "context must be an object"),
        (
            request(context={"api_key": "sk-fixture"}),
            "Use a worker-side \\$env reference instead of inline credentials",
        ),
        (request(samples=[]), "samples must contain 1..1000 entries"),
        (
            request(samples=runs(1001)),
            "samples must contain 1..1000 entries",
        ),
        (request(run("slot-0"), run("slot-0")), "sample_slot_id values must be unique"),
        (
            request({**run("slot-0"), "sample_slot_id": "has space"}),
            "sample_slot_id must be a URL-safe identifier",
        ),
        (request({**run("slot-0"), "notes": "fixture"}), "Invalid sample fields"),
        (request({"sample_slot_id": "slot-0"}), "Each sample requires exactly one run, branch or review"),
        (
            request({**run("slot-0"), "review": REVIEW_JOB}),
            "Each sample requires exactly one run, branch or review",
        ),
        (request({"sample_slot_id": "slot-0", "run": [RUN]}), "run must be a JobSpec object"),
        (
            request({"sample_slot_id": "slot-0", "run": {**REVIEW_JOB, "kind": "review"}}),
            "run must be a rollout job from",
        ),
        (
            request(
                {
                    "sample_slot_id": "slot-0",
                    "run": {**RUN, "parent_point": "p", "continuation": {"guidance": "none"}},
                }
            ),
            "run must be a rollout job from scratch; use branch for continuation",
        ),
        (
            request({"sample_slot_id": "slot-0", "review": REVIEW_JOB, "grade": GRADE}),
            "A review sample has no grade",
        ),
        (
            request({"sample_slot_id": "slot-0", "review": {**RUN, "kind": "rollout"}}),
            "review must be a review job from scratch",
        ),
        (request(branch_sample(prompt="Fix it")), "Invalid branch fields"),
        (request(branch_sample(job_id="has/slash")), "branch.job_id must be a URL-safe identifier"),
        (request(branch_sample(continuation=None)), "A continuation is an object"),
        (request(branch_sample(overrides={"prompt": "Fix it"})), "Invalid branch overrides fields"),
        (request(branch_sample(budget={"turns": 3})), "A branch budget sets max_turns and/or deadline_at"),
        (request(run("slot-0", grade={"profile": "grade"})), "grade must be a GradeSpec job template"),
        (request(run("slot-0", grade={**IN_PLACE, "in_place": "yes"})), "grade in_place must be a boolean"),
        (request(run("slot-0", grade={**GRADE, "spec": {**GRADE["spec"], "snapshot_id": "ABC"}})), SELECTED),
        (
            request(run("slot-0", grade={**IN_PLACE, "spec": {**IN_PLACE["spec"], "in_place_job": "job-0"}})),
            SELECTED,
        ),
        (
            request(run("slot-0", grade={**GRADE, "kind": "review"})),
            "grade must be an independent grading job",
        ),
        (
            request(run("slot-0", grade={**GRADE, "in_place": True})),
            "task_dir and in_place_job belong to harbor grades",
        ),
        (
            request({"sample_slot_id": "slot-0", "run": {**RUN, "affinity": "host-a"}}),
            "The coordinator sets the affinity of a run",
        ),
        (
            request({"sample_slot_id": "slot-0", "review": {**REVIEW_JOB, "affinity": "host-a"}}),
            "The coordinator sets the affinity of a review",
        ),
        (
            request(run("slot-0", grade={**GRADE, "affinity": "host-a"})),
            "The coordinator sets the affinity of a grade",
        ),
    ],
    ids=[
        "an unknown group field",
        "a bad group id",
        "a bad prompt group id",
        "a context that is not an object",
        "an inline credential",
        "no sample",
        "1001 samples",
        "a repeated slot",
        "a bad slot id",
        "an unknown sample field",
        "no actor",
        "two actors",
        "a run that is not an object",
        "a run of another kind",
        "a run that continues a point",
        "a graded review",
        "a review of another kind",
        "an unknown branch field",
        "a bad branch job id",
        "a branch without a continuation",
        "a branch override of the prompt",
        "a bad branch budget",
        "a grade without a spec",
        "an in_place that is not a bool",
        "a grade that names its snapshot",
        "a grade that names its job",
        "a grade of another kind",
        "an in-place SWE-rebench grade",
        "a run with an affinity",
        "a review with an affinity",
        "a grade with an affinity",
    ],
)
def test_a_request_that_the_coordinator_cannot_run_is_refused_before_any_job(body, message):
    with pytest.raises(ValueError, match=message):
        validate_request(body)


@pytest.mark.parametrize("samples", [1, 1000])
def test_a_group_has_from_1_to_1000_samples(samples):
    checked = validate_request(request(samples=runs(samples)))

    assert len(checked["samples"]) == samples


def test_a_checked_request_is_the_form_that_a_caller_stores_and_compares_on_a_resubmission():
    """A grade template leaves its source to its actor. Only an in-place template keeps ``in_place``."""
    body = request(
        run("slot-0"), run("slot-1", grade=IN_PLACE), {"sample_slot_id": "slot-2", "branch": BRANCH}
    )

    checked = validate_request(body)

    rollout = {"kind": "rollout", **RUN, **JOB_DEFAULTS}
    assert checked == {
        "rollout_job_id": "group",
        "prompt_group_id": "prompt",
        "context": {},
        "samples": [
            {"sample_slot_id": "slot-0", "run": rollout, "grade": {"kind": "grade", **GRADE, **JOB_DEFAULTS}},
            {
                "sample_slot_id": "slot-1",
                "run": rollout,
                "grade": {
                    "kind": "grade",
                    "profile": "grade",
                    "spec": {"benchmark": "harbor", "task_dir": "/path/to/task"},
                    **JOB_DEFAULTS,
                    "in_place": True,
                },
            },
            {"sample_slot_id": "slot-2", "branch": {**BRANCH, "overrides": {}}},
        ],
    }
