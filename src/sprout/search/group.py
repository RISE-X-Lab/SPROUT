"""A group request, the samples a coordinator runs as Run Store jobs, and its group document.

Each sample of a request runs a rollout from scratch (``run``), continues a
job from one of its recovery points (``branch``) or reviews finished attempts
(``review``). A run or a branch may carry a grade job template, which grades
the actor's final snapshot, or with ``in_place`` the sandbox that the actor
keeps. ``validate_request`` checks a request before anything is submitted.
``initial_document`` is the group document before any job exists: the JSON
state that a coordinator advances and its group store keeps. ``planned_again``
gives one of its samples as it was before any job, for a group that runs again.
``Submitter`` names who submits the jobs.
"""

import hashlib
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sprout.contract.continuation import Continuation
from sprout.runstore.specs import (
    BRANCH_OVERRIDES,
    JobSpec,
    canonical,
    no_credentials,
    validate_branch_budget,
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")


@dataclass(frozen=True)
class Submitter:
    """Who submits a group's jobs, as the Run Store records it.

    Each run, grade and review job carries its record in ``context[context_key]``, and each idempotency key
    starts with ``key_prefix``. Submitters with different prefixes never share a job, even for one group id.
    """

    context_key: str
    key_prefix: str

    def key(self, *, affinity: str, sample_id: str, phase: str) -> str:
        """The idempotency key of one stage of a sample, in the group whose affinity key is ``affinity``.

        It is the same on every try to submit the stage.
        """
        return self.key_prefix + hashlib.sha256(canonical([affinity, sample_id, phase]).encode()).hexdigest()


def identifier(value: object, name: str) -> str:
    """``value`` when it is a URL-safe identifier of 1 to 200 characters. Raises ValueError otherwise."""
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a URL-safe identifier of 1..200 characters")
    return value


def _object_fields(value: object, allowed: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError(f"Invalid {name} fields; allowed: {sorted(allowed)}")


#: The fields of a sample that name its actor: a sample has exactly one of them.
_ACTORS = ("run", "branch", "review")
#: What stands in for the grade source that the actor decides while a template is validated.
_PENDING = "pending-actor"


def validate_request(body: dict[str, Any]) -> dict[str, Any]:
    """A checked copy of a group request: ``rollout_job_id``, ``prompt_group_id``, ``context``, ``samples``.

    Raises ValueError for a missing or malformed field, a repeated slot id, a sample without exactly one
    of ``run``, ``branch`` and ``review``, a graded review, a grade template that names its own
    ``snapshot_id`` or ``in_place_job``, a job with an ``affinity``, or an inline credential."""
    _object_fields(body, {"rollout_job_id", "prompt_group_id", "context", "samples"}, "group")
    result = deepcopy(body)
    identifier(result.get("rollout_job_id"), "rollout_job_id")
    identifier(result.get("prompt_group_id"), "prompt_group_id")
    result.setdefault("context", {})
    if not isinstance(result["context"], dict):
        raise ValueError("context must be an object")
    samples = result.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 1000:
        raise ValueError("samples must contain 1..1000 entries")
    seen: set[str] = set()
    for sample in samples:
        _object_fields(sample, {"sample_slot_id", *_ACTORS, "grade"}, "sample")
        slot = identifier(sample.get("sample_slot_id"), "sample_slot_id")
        if slot in seen:
            raise ValueError("sample_slot_id values must be unique")
        seen.add(slot)
        if sum(name in sample for name in _ACTORS) != 1:
            raise ValueError("Each sample requires exactly one run, branch or review")
        if "run" in sample:
            sample["run"] = _job(sample["run"], kind="rollout", name="run")
        elif "review" in sample:
            if "grade" in sample:
                raise ValueError("A review sample has no grade")
            sample["review"] = _job(sample["review"], kind="review", name="review")
        else:
            _check_branch(sample["branch"])
        if "grade" in sample:
            sample["grade"] = _grade_template(sample["grade"])
    no_credentials(result)
    canonical(result)
    return result


def _job(value: object, *, kind: str, name: str) -> dict[str, Any]:
    """A checked JobSpec body of ``kind`` that starts from scratch."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JobSpec object")
    job = JobSpec.from_dict({"kind": kind, **value})
    if job.kind != kind or job.parent_point:
        raise ValueError(f"{name} must be a {kind} job from scratch; use branch for continuation")
    _no_affinity(job, name=name)
    return job.validate()


def _no_affinity(job: JobSpec, *, name: str) -> None:
    if job.affinity is not None:
        raise ValueError(f"The coordinator sets the affinity of a {name}")


def _check_branch(branch: Any) -> None:
    """Check a sample's ``branch``, an object of the request's JSON, and normalize it in place."""
    _object_fields(branch, {"job_id", "point_id", "continuation", "budget", "overrides"}, "branch")
    identifier(branch.get("job_id"), "branch.job_id")
    identifier(branch.get("point_id"), "branch.point_id")
    branch["continuation"] = Continuation.from_dict(branch.get("continuation")).to_dict()
    if "budget" in branch:
        branch["budget"] = validate_branch_budget(branch["budget"])
    branch.setdefault("overrides", {})
    _object_fields(branch["overrides"], set(BRANCH_OVERRIDES), "branch overrides")


def _grade_template(grade: object) -> dict[str, Any]:
    """A checked grade job template, without the source of the grade, which its actor decides.

    ``in_place: true`` grades the sandbox that the actor keeps (``GradeSpec.in_place_job``), and
    otherwise the grade runs from the actor's final snapshot (``GradeSpec.snapshot_id``).
    """
    if not isinstance(grade, dict) or not isinstance(grade.get("spec"), dict):
        raise ValueError("grade must be a GradeSpec job template")
    in_place = grade.get("in_place", False)
    if type(in_place) is not bool:
        raise ValueError("grade in_place must be a boolean")
    if {"snapshot_id", "in_place_job"} & set(grade["spec"]):
        raise ValueError("A grade's snapshot_id or in_place_job is selected from its actor")
    source = "in_place_job" if in_place else "snapshot_id"
    job = JobSpec.from_dict(
        {
            **{key: value for key, value in grade.items() if key != "in_place"},
            "kind": "grade",
            "spec": {**grade["spec"], source: _PENDING},
        }
    )
    if grade.get("kind", "grade") != "grade" or job.parent_point:
        raise ValueError("grade must be an independent grading job")
    _no_affinity(job, name="grade")
    template = job.validate()
    del template["spec"][source]
    # Only an in-place template carries the key: a validated request is compared whole on resubmission.
    return {**template, "in_place": True} if in_place else template


def initial_document(request: dict[str, Any], *, context_key: str, affinity: str) -> dict[str, Any]:
    """The group document of a validated request before any job exists, whose jobs carry the key ``affinity``.

    The job context of each run and each review gets the coordinator's record under ``context_key``. Groups
    of one search can share a key. Then their samples need distinct ids, because a stage's idempotency key
    comes from the key, the sample id and the phase (``Submitter.key``)."""
    samples: list[dict[str, Any]] = []
    for spec in request["samples"]:
        operation = next(name for name in _ACTORS if name in spec)
        if operation == "branch":
            origin = {key: spec["branch"][key] for key in ("job_id", "point_id")}
            actor = _planned(operation, submission=spec["branch"], origin=origin)
        else:
            job = deepcopy(spec[operation])
            job["context"][context_key] = {
                "rollout_job_id": request["rollout_job_id"],
                "prompt_group_id": request["prompt_group_id"],
                "sample_slot_id": spec["sample_slot_id"],
                "context": request["context"],
            }
            actor = _planned(operation, submission=job)
        samples.append(
            {
                "sample_slot_id": spec["sample_slot_id"],
                "actor": actor,
                "grade": _planned("grade", template=spec["grade"]) if "grade" in spec else None,
            }
        )
    return {
        "rollout_job_id": request["rollout_job_id"],
        "prompt_group_id": request["prompt_group_id"],
        "affinity": affinity,
        "status": "queued",
        "context": request["context"],
        "ready": False,
        "samples": samples,
    }


def planned_again(sample: dict[str, Any]) -> dict[str, Any]:
    """A copy of a sample of ``initial_document`` with its actor and its grade as they were before any job."""
    actor, grade = sample["actor"], sample["grade"]
    kept = {key: deepcopy(actor[key]) for key in ("submission", "origin") if key in actor}
    return {
        "sample_slot_id": sample["sample_slot_id"],
        "actor": _planned(actor["operation"], **kept),
        "grade": None if grade is None else _planned("grade", template=deepcopy(grade["template"])),
    }


def _planned(operation: str, **fields: Any) -> dict[str, Any]:
    """A stage of ``operation`` that has no job yet, with ``fields``."""
    return {"state": "planned", "job_id": None, "attempt_id": None, "operation": operation, **fields}
