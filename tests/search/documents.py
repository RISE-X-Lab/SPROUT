"""Group requests, group documents and coordinators for the tests of ``sprout.search``.

Each function builds the smallest valid value, so that a test names only the values that its behavior
depends on. The jobs of these groups run on a ``FakeRunStore``, and their documents live in a
``FileGroups`` store.
"""

from __future__ import annotations

import json
import threading
from copy import deepcopy
from typing import Any

from sprout.contract.messages import clean_messages
from sprout.eval.groups import FileGroups
from sprout.search.coordinator import Coordinator
from sprout.search.group import Submitter, initial_document, validate_request
from sprout.search.spec import SearchSpec, search_document
from tests.search.fake_runstore import FakeRunStore

SUBMITTER = Submitter(context_key="eval", key_prefix="eval/")
GROUP = "group"
#: A rollout from scratch. Its rollout contract lets a branch of it set its own budget.
RUN = {
    "profile": "actor",
    "spec": {
        "prompt": "Fix the parser",
        "slot": "mini-swe-agent",
        "sandbox_image": "task-image",
        "timeout_s": 30,
        "extra": {"rollout_contract": {"max_turns": 10}},
    },
}
#: A grade template: the SWE-rebench grade of the actor's final snapshot.
GRADE = {
    "profile": "grade",
    "spec": {
        "benchmark": "swe-rebench-v2",
        "instance_id": "task",
        "dataset_path": "/path/to/task.json",
        "dataset_sha256": "123",
        "grader_revision": "sha256:abc",
        "parser_path": "/path/to/log_parsers.py",
        "image": "docker.io/swerebenchv2/task",
        "timeout_s": 60,
    },
}
#: The ``ReviewSpec`` fields that a search does not set.
REVIEW = {"problem": "Fix the parser", "model": "policy", "model_endpoint": "http://model/v1"}
#: A search of one root: one point, one student and one repair turn with one repair, one round.
SEARCH = {
    "roots": 1,
    "points": 1,
    "candidates": 1,
    "student_continuations": 1,
    "candidate_continuations": 1,
    "tf_student_continuations": 0,
    "repair_rounds": 1,
    "review_seconds": 600.0,
    "wall_time_seconds": 900.0,
}


def run(slot: str, *, grade: dict[str, Any] | None = GRADE) -> dict[str, Any]:
    """A sample that runs ``RUN`` and, unless ``grade`` is None, grades it with that template."""
    sample: dict[str, Any] = {"sample_slot_id": slot, "run": deepcopy(RUN)}
    if grade is not None:
        sample["grade"] = deepcopy(grade)
    return sample


def document(
    *samples: dict[str, Any], group_id: str = GROUP, affinity: str | None = None, **fields: Any
) -> dict[str, Any]:
    """The initial document of the group of ``samples`` that ``SUBMITTER`` makes, with ``fields`` added.

    Its key is ``affinity``, by default the group id, as the RL driver gives it."""
    request = validate_request(
        {"rollout_job_id": group_id, "prompt_group_id": "prompt", "samples": list(samples)}
    )
    initial = initial_document(request, context_key=SUBMITTER.context_key, affinity=affinity or group_id)
    return {**initial, **fields}


def search_group(*, slots: int | None = None, parent_max_turns: int = 10, **config: Any) -> dict[str, Any]:
    """The document of a search with ``SEARCH`` changed by ``config``: a graded run in each root slot.

    The roots take the slots ``root-0``, ``root-1`` and so on, and the branches the free slots
    ``slot-1``, ``slot-2`` and so on, ``slots`` of them, by default ``SearchSpec.branches()``."""
    spec = SearchSpec.from_dict({**SEARCH, **config})
    roots = ["root-%d" % index for index in range(spec.roots)]
    free = ["slot-%d" % index for index in range(1, 1 + (spec.branches() if slots is None else slots))]
    search = search_document(
        config=spec,
        review=REVIEW,
        review_profile="review",
        parent_max_turns=parent_max_turns,
        root_slots=roots,
        free_slots=free,
    )
    return document(*(run(slot) for slot in roots), search=search)


def stored(groups: FileGroups, document: dict[str, Any]) -> str:
    """Put ``document`` in ``groups`` as an active group, and return its id."""
    groups.create(document["rollout_job_id"], document)
    return str(document["rollout_job_id"])


def coordinator_for(
    fake: FakeRunStore,
    groups: FileGroups,
    *,
    stopping: threading.Event | None = None,
    runs_again: bool = True,
) -> Coordinator:
    """A coordinator of ``groups`` over the client of ``fake``, for ``SUBMITTER``."""
    return Coordinator(
        client=fake.client,
        groups=groups,
        submitter=SUBMITTER,
        stopping=stopping or threading.Event(),
        runs_again=runs_again,
    )


class Clock:
    """A ``time.time`` that a test sets: it reads ``now``, in seconds."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def exported(**changes: Any) -> dict[str, Any]:
    """The result of a rollout that completed and exported its history for training, with ``changes``."""
    messages = [
        {"role": "user", "content": "Fix the parser"},
        {"role": "assistant", "content": "Fixed."},
    ]
    return {
        "status": "completed",
        "training_messages": clean_messages(messages),
        "training_provenance": {"prefix_messages": 0, "inserted_messages": []},
        **changes,
    }


def verdict(resolved: bool, **facts: Any) -> dict[str, Any]:
    """The result of a grade that says whether the task resolved, with the grade's ``facts``."""
    return {"status": "completed", "resolved": resolved, "progress": 1.0 if resolved else 0.0, **facts}


def stage(groups: FileGroups, slot: str, phase: str, *, group_id: str = GROUP) -> dict[str, Any]:
    """The stage ``phase`` (``actor`` or ``grade``) of the sample ``slot``, as the group store keeps it."""
    samples = groups.get(group_id)["document"]["samples"]
    found: dict[str, Any] = next(sample for sample in samples if sample["sample_slot_id"] == slot)[phase]
    return found


def search_of(groups: FileGroups, *, group_id: str = GROUP) -> dict[str, Any]:
    """The group's ``search``, as the group store keeps it."""
    found: dict[str, Any] = groups.get(group_id)["document"]["search"]
    return found


#: The result of a grade that failed: its sample has no verdict.
NO_VERDICT = {"status": "error", "error": "sandbox did not start"}
#: The result of a job that the Run Store failed because the host that held its data was lost.
DATA_LOST = {"reason": "data_lost", "error": "every worker on the host of the job's key is dead"}


def lose(fake: FakeRunStore, job_id: str) -> None:
    """End the job as the Run Store ends each unfinished job of a lost host: failed, with ``DATA_LOST``."""
    fake.finish(job_id, DATA_LOST, state="failed")


def _finish_grade(fake: FakeRunStore, job_id: str, result: dict[str, Any]) -> None:
    fake.finish(job_id, result, state="failed" if result["status"] == "error" else "succeeded")


def settle_roots(
    fake: FakeRunStore,
    coordinator: Coordinator,
    *grades: dict[str, Any],
    exports_nothing: tuple[int, ...] = (),
) -> None:
    """Run the roots of a search group through their actors and grades, with a tick after each step.

    Root ``i`` exports its history, unless ``i`` is in ``exports_nothing``, and its grade ends with
    ``grades[i]``: a ``verdict`` or ``NO_VERDICT``."""
    coordinator.tick()
    for index, job_id in enumerate(fake.job_ids("rollout")):
        problem = {"training_export_error": "Native history has an incomplete final record"}
        fake.finish_rollout(job_id, exported(**problem) if index in exports_nothing else exported())
    coordinator.tick()
    for job_id, result in zip(fake.job_ids("grade"), grades, strict=True):
        _finish_grade(fake, job_id, result)
    coordinator.tick()


def settle_branches(fake: FakeRunStore, coordinator: Coordinator, *grades: dict[str, Any]) -> None:
    """Finish each queued rollout as an exported branch, then end the new grades with ``grades``, in order.

    It ticks after each step."""
    branches = [job_id for job_id in fake.job_ids("rollout") if fake.job(job_id)["state"] == "queued"]
    for job_id in branches:
        fake.finish_rollout(job_id, exported())
    coordinator.tick()
    pending = [job_id for job_id in fake.job_ids("grade") if fake.job(job_id)["state"] == "queued"]
    for job_id, result in zip(pending, grades, strict=True):
        _finish_grade(fake, job_id, result)
    coordinator.tick()


def turn(command: str, *, call_id: str) -> dict[str, Any]:
    """A repair turn as a reviewer writes it: a text, then one ``bash`` call of ``command``."""
    arguments = json.dumps({"command": command})
    call = {"id": call_id, "type": "function", "function": {"name": "bash", "arguments": arguments}}
    return {"role": "assistant", "content": "Run the tests first.", "tool_calls": [call]}


def review_point(
    job_id: str, *, step: int = 2, candidates: tuple[dict[str, Any], ...] = ()
) -> dict[str, Any]:
    """The point that a review picks after tool call ``step`` of ``job_id``'s attempt, with ``candidates``."""
    return {
        "job_id": job_id,
        "attempt_id": "attempt-" + job_id,
        "step": step,
        "why": "before the edit",
        "samples": 2,
        "candidates": list(candidates),
    }


def reviewed(*points: dict[str, Any]) -> dict[str, Any]:
    """The result of a review that picked ``points``."""
    return {"status": "completed", "points": list(points)}
