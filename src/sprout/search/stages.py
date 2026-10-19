"""One stage of a group's sample: its Run Store job, submitted once and followed to a final state.

A stage is a JSON object in a group document: the actor (a run, a branch or a
review), the grade, or a search's review. ``Stages.advance`` moves one stage one
step: it submits the job under its idempotency key, polls it, stops it when the
group is cancelled, and gives up a grade that runs longer than the document's
``grade_wall_seconds``. Each job carries the document's ``affinity`` key, by
which the Run Store runs the jobs of a group on one host. Each change is saved
to the group store before the next remote call, so a retry after a lost reply
sends the same key and makes no second job.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import httpx

from sprout.runstore.specs import JobSpec
from sprout.search.client import RunStoreClient
from sprout.search.group import Submitter

if TYPE_CHECKING:
    from sprout.search.coordinator import GroupStore

#: The states that a stage keeps once it reaches them.
DONE = frozenset({"succeeded", "failed", "cancelled", "skipped", "expired"})


class Stages:
    """Submits and follows the stages of the group documents in one group store."""

    def __init__(self, *, client: RunStoreClient, groups: GroupStore, submitter: Submitter) -> None:
        self._client = client
        self._groups = groups
        self._submitter = submitter

    def save(self, document: dict[str, Any]) -> None:
        """Store ``document`` in the group store, as terminal when it is ``ready``."""
        self._groups.save(document["rollout_job_id"], document, terminal=document["ready"])

    def cancelled(self, document: dict[str, Any]) -> bool:
        """Whether someone requested the cancellation of the document's group."""
        return bool(self._groups.get(document["rollout_job_id"])["cancel_requested"])

    def advance(self, document: dict[str, Any], sample: dict[str, Any], phase: str) -> None:
        """Move ``sample[phase]`` one step: expire it, hold it, submit it or poll it.

        Raises ``httpx.HTTPError`` when a Run Store call fails, after the stage records what is known. A
        submission that may have arrived is marked ``submission_uncertain``, and a retry sends the same key.
        A grade whose source cannot be proved fails, without an exception."""
        stage = sample[phase]
        if stage is None or stage["state"] in DONE:
            return
        wall = document.get("grade_wall_seconds")
        if phase == "grade" and wall is not None and time.time() >= stage.get("started_at", math.inf) + wall:
            self._expire(document, stage, wall)
            return
        if not stage["job_id"]:
            if self.cancelled(document):
                self._hold(document, sample, phase)
                return
            if phase == "grade" and "submission" not in stage:
                try:
                    self._prepare_grade(document, sample)
                except ValueError as error:
                    stage.update(state="failed", error=str(error))
                    self.save(document)
                    return
            self._dispatch(document, sample, phase)
        if stage["job_id"]:
            self._poll(document, stage)

    def attach(self, document: dict[str, Any], *, stage: dict[str, Any], key: str, job_id: str) -> None:
        """Give ``stage`` the job ``job_id`` and its state, when that job was submitted under ``key``.

        Submits nothing and changes no job. Raises ValueError when the job's idempotency key is not
        ``key``, and ``httpx.HTTPError`` when the Run Store cannot return the job.
        """
        job = self._client.get(job_id)
        if job.get("idempotency_key") != key:
            raise ValueError("Job does not match this submission's idempotency key")
        stage.update(
            job_id=job["id"],
            state=job["state"],
            attempt_id=job.get("active_attempt"),
            phase=job.get("phase"),
            error=job.get("error"),
        )
        if job["state"] in DONE:
            stage["result"] = job.get("result")
        stage.pop("last_error", None)
        stage.pop("submission_uncertain", None)
        self.save(document)

    def _key(self, document: dict[str, Any], sample: dict[str, Any], phase: str) -> str:
        return self._submitter.key(
            affinity=document["affinity"], sample_id=sample["sample_slot_id"], phase=phase
        )

    def _expire(self, document: dict[str, Any], stage: dict[str, Any], wall: float) -> None:
        # Nothing reads the verdict of an expired grade. The stop request makes its worker end the job
        # and reclaim its sandboxes.
        stage.update(state="expired", error=f"Grading did not settle within {wall:g} s")
        self.save(document)
        try:
            self._client.stop(stage["job_id"])
        except httpx.HTTPError as error:
            stage["last_error"] = f"Could not stop the expired grade: {error}"
            self.save(document)

    def _hold(self, document: dict[str, Any], sample: dict[str, Any], phase: str) -> None:
        """Settle a cancelled group's stage that has no job: never submitted, or submitted with no reply."""
        stage = sample[phase]
        if stage["state"] == "planned":
            stage["state"] = "cancelled"
        else:
            # Do not resubmit: if the first submission never arrived, a retry makes a new job after the
            # cancellation request.
            stage["last_error"] = (
                "Submission acknowledgement unknown; inspect Run Store using the idempotency key"
            )
            stage["idempotency_key"] = self._key(document, sample, phase)
        self.save(document)

    def _dispatch(self, document: dict[str, Any], sample: dict[str, Any], phase: str) -> None:
        stage = sample[phase]
        if stage["state"] == "planned":
            if self.cancelled(document):
                stage["state"] = "cancelled"
                self.save(document)
                return
            stage["state"] = "submitting"
            self.save(document)  # The intent is durable before the remote side effect.
        key = self._key(document, sample, phase)
        try:
            if stage["operation"] == "branch":
                # A branch takes the affinity key of the job that it continues, where its snapshot is.
                branch = stage["submission"]
                job_id = self._client.branch(
                    branch["job_id"],
                    branch["point_id"],
                    idempotency_key=key,
                    continuation=branch["continuation"],
                    budget=branch.get("budget"),
                    **branch["overrides"],
                )
            else:
                job = JobSpec.from_dict({**stage["submission"], "affinity": document["affinity"]})
                job_id = self._client.submit(job, key)
        except httpx.HTTPStatusError as error:
            if error.response.status_code in {400, 404, 409, 422} and not stage.get("submission_uncertain"):
                stage.update(state="failed", error=error.response.text)
                self.save(document)
                return
            stage["submission_uncertain"] = True
            raise
        except httpx.RequestError:
            stage["submission_uncertain"] = True
            raise
        stage.update(job_id=job_id, state="queued")
        stage.pop("submission_uncertain", None)
        stage.pop("last_error", None)
        self.save(document)

    def _poll(self, document: dict[str, Any], stage: dict[str, Any]) -> None:
        job = self._client.get(stage["job_id"])
        if job["state"] in {"queued", "running"} and self.cancelled(document):
            # Nothing reads a cancelled group's results. The stop request cancels a queued job, and it
            # stops a running one, whose worker reclaims its sandboxes.
            self._client.stop(stage["job_id"])
            job = self._client.get(stage["job_id"])
        stage.update(
            state=job["state"],
            attempt_id=job.get("active_attempt"),
            phase=job.get("phase"),
            error=job.get("error"),
        )
        if job["state"] != "queued":
            # When the job left the queue: a grade's wall time excludes its time in the queue.
            stage.setdefault("started_at", time.time())
        if job["state"] in DONE:
            stage["result"] = job.get("result")
        stage.pop("last_error", None)
        self.save(document)

    def _prepare_grade(self, document: dict[str, Any], sample: dict[str, Any]) -> None:
        """Fill in the grade's submission from its template, with its source: the actor's kept sandbox
        (an ``in_place`` template) or the actor's final snapshot. Raises ValueError for no source."""
        actor, grade = sample["actor"], sample["grade"]
        attempt = actor["attempt_id"]
        if not attempt:
            raise ValueError("Completed actor has no attempt identity")
        submission = deepcopy(grade["template"])
        record = {
            "rollout_job_id": document["rollout_job_id"],
            "sample_slot_id": sample["sample_slot_id"],
            "actor_job_id": actor["job_id"],
            "actor_attempt_id": attempt,
        }
        snapshot_id = None
        if submission.pop("in_place", False):
            submission["spec"]["in_place_job"] = actor["job_id"]
        else:
            point = self._final_point(actor, attempt=attempt)
            snapshot_id = submission["spec"]["snapshot_id"] = point["snapshot_id"]
            record["point_id"] = point["id"]
        submission["context"][self._submitter.context_key] = record
        grade["submission"] = JobSpec.from_dict(submission).validate()
        if snapshot_id is not None:
            grade["snapshot_id"] = snapshot_id
        self.save(document)

    def _final_point(self, actor: dict[str, Any], *, attempt: str) -> dict[str, Any]:
        """The recovery point at the actor's final state. Raises ValueError when it cannot be proved."""
        output = actor.get("result") or {}
        if output.get("final_snapshot_id"):
            # A rollout that finalized has a final disk snapshot. It also covers a text-only last
            # turn and a run with no tool calls.
            return {"id": None, "snapshot_id": output["final_snapshot_id"]}
        tools = self._client.all_tools(actor["job_id"], attempt_id=attempt)
        if not tools or any(tool.get("response") is None for tool in tools):
            raise ValueError("Cannot grade without a complete final tool boundary")
        final_depth = tools[-1]["depth"]
        if [tool["depth"] for tool in tools] != list(range(1, final_depth + 1)):
            raise ValueError("Final tool prefix has a gap")
        points = self._client.points(actor["job_id"], attempt_id=attempt)
        candidates = [
            point for point in points if point.get("available") is True and point["tool_depth"] == final_depth
        ]
        if not candidates:
            raise ValueError("Final message has no available recovery point; refusing an older snapshot")
        return max(candidates, key=lambda item: item["message_step"])
