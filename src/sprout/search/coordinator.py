"""The search coordinator: runs the active groups of a group store as Run Store jobs.

A caller makes a group document (``group.validate_request`` and
``group.initial_document``, with its affinity key, and with ``spec.search_document``
as its ``search`` for a search), puts it in its ``GroupStore``, and calls
``Coordinator.tick`` until the document is ``ready``. Each tick submits what is due,
polls what runs, advances the search and sets the document's ``status``. A group whose
job lost its data with its host runs again from its roots (``rerun``), unless the
caller runs its search again itself. The coordinator runs no job itself. After a
restart it follows the jobs that the stored documents name, and it never makes a
second job for one stage.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol

import httpx

from sprout.search.client import RunStoreClient
from sprout.search.group import Submitter
from sprout.search.rerun import give_up, lost, run_again
from sprout.search.rounds import REVIEW_SLOT, Search
from sprout.search.stages import DONE, Stages


class GroupStore(Protocol):
    """Where a coordinator keeps its group documents, by group id (the document's ``rollout_job_id``).

    It keeps a JSON document, a ``cancel_requested`` flag and a terminal flag for each group. A store
    that outlives the process lets a new coordinator resume its groups. The coordinator's calls never overlap.
    """

    def active(self) -> list[str]:
        """The ids of the groups that are not terminal, oldest first."""
        ...

    def get(self, group_id: str) -> dict[str, Any]:
        """The group's ``document`` and ``cancel_requested``. Raises KeyError for an unknown group."""
        ...

    def save(self, group_id: str, document: dict[str, Any], *, terminal: bool = False) -> None:
        """Replace the group's document. A terminal group is no longer active."""
        ...

    def release(self, group_id: str) -> None:
        """Request the cancellation of an active group."""
        ...


class Coordinator:
    """Submits, follows and searches the groups of one group store over one Run Store client.

    With ``runs_again`` false, a group whose job lost its data fails at once (``rerun.give_up``), for a
    caller that runs its search again itself. Otherwise the group runs again (``rerun.run_again``).
    """

    def __init__(
        self,
        *,
        client: RunStoreClient,
        groups: GroupStore,
        submitter: Submitter,
        stopping: threading.Event,
        runs_again: bool = True,
    ) -> None:
        self._groups = groups
        self._submitter = submitter
        self._stopping = stopping
        self._runs_again = runs_again
        self._tick_lock = threading.Lock()
        self._init_stages(client=client)
        self._init_search(client=client)

    def _init_stages(self, *, client: RunStoreClient) -> None:
        self._stages = Stages(client=client, groups=self._groups, submitter=self._submitter)

    def _init_search(self, *, client: RunStoreClient) -> None:
        self._search = Search(
            client=client,
            stages=self._stages,
            context_key=self._submitter.context_key,
            stopping=self._stopping,
        )

    def tick(self) -> None:
        """Advance every active group once, oldest first. Returns at once while another tick runs.

        It requests the cancellation of each group past its ``deadline_at``, and runs again or fails each
        group whose job lost its data. A failed Run Store call is kept as the ``last_error`` of its stage or
        search, the tick goes on, and the next tick tries again. ``stopping`` ends it early."""
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            for group_id in self._groups.active():
                if self._stopping.is_set():
                    break
                document = self._groups.get(group_id)["document"]
                if document.get("deadline_at") is not None and time.time() >= document["deadline_at"]:
                    self._groups.release(group_id)
                if not self._advance_samples(document):
                    return
                if document.get("search") and not self._lost(document):
                    # After the samples, so that a stage that just settled counts in this tick. The
                    # search also dispatches the branches that it plans in this tick.
                    self._advance_search(document)
                # Again after the search, which polls its reviews and the branches that it dispatched.
                if self._lost(document):
                    document = run_again(document) if self._runs_again else give_up(document)
                self._settle(document)
        finally:
            self._tick_lock.release()

    def reconcile_submission(self, group_id: str, *, sample_id: str, phase: str, job_id: str) -> None:
        """Attach the accepted job to a stage whose reply was lost: ``submitting``, with no job id.

        The job must have the stage's idempotency key. A review is phase ``review`` of ``REVIEW_SLOT`` or
        ``chain_review_slot(round)``. Raises KeyError for an unknown group or sample, and ValueError for a
        bad phase, a stage that is not an unacknowledged submission, or a job with another key."""
        if phase not in {"actor", "grade", "review"}:
            raise ValueError("phase must be actor, grade or review")
        with self._tick_lock:
            document = self._groups.get(group_id)["document"]
            stage = _stage(document, sample_id=sample_id, phase=phase)
            if stage is None or stage["state"] != "submitting" or stage["job_id"]:
                raise ValueError("Only an unacknowledged submission can be reconciled")
            key = self._submitter.key(affinity=document["affinity"], sample_id=sample_id, phase=phase)
            self._stages.attach(document, stage=stage, key=key, job_id=job_id)

    def _advance_samples(self, document: dict[str, Any]) -> bool:
        """Advance each sample's actor, then its grade. False when ``stopping`` cut the pass short."""
        for sample in document["samples"]:
            for phase in ("actor", "grade"):
                if self._stopping.is_set():
                    return False
                stage = sample[phase]
                if stage is None:
                    continue
                if phase == "grade" and sample["actor"]["state"] != "succeeded":
                    if sample["actor"]["state"] in DONE:
                        stage.update(state="skipped", error="Actor did not succeed")
                    continue
                try:
                    self._stages.advance(document, sample, phase)
                except (httpx.HTTPError, ValueError, KeyError) as error:
                    # Do not turn a polling failure into a new execution.
                    stage["last_error"] = str(error)
                    self._stages.save(document)
        return True

    def _advance_search(self, document: dict[str, Any]) -> None:
        try:
            self._search.advance(document)
        except (httpx.HTTPError, ValueError, KeyError) as error:
            document["search"]["last_error"] = str(error)
            self._stages.save(document)

    def _lost(self, document: dict[str, Any]) -> bool:
        """Whether a job of the group's current run lost its data, and nobody cancelled the group."""
        return lost(document) and not self._stages.cancelled(document)

    def _settle(self, document: dict[str, Any]) -> None:
        """Set the document's ``ready`` and ``status`` from its stages and its search, and save it.

        A group with an ``error`` does not run again after a lost host: it is ready, and failed."""
        stages = [
            sample[phase]
            for sample in document["samples"]
            for phase in ("actor", "grade")
            if sample[phase] is not None
        ]
        failed = document.get("error") is not None
        settled = not document.get("search") or document["search"]["state"] == "done"
        document["ready"] = failed or (settled and all(stage["state"] in DONE for stage in stages))
        if document["ready"]:
            document["status"] = (
                "cancelled"
                if self._stages.cancelled(document)
                else "failed"
                if failed or any(stage["state"] != "succeeded" for stage in stages)
                else "completed"
            )
        else:
            document["status"] = "running"
        self._stages.save(document)


def _stage(document: dict[str, Any], *, sample_id: str, phase: str) -> dict[str, Any] | None:
    """The stage ``phase`` of sample ``sample_id``, or a search's review. None when it is not made yet.

    Raises KeyError for an unknown sample or review slot.
    """
    if phase == "review":
        search = document.get("search") or {}
        reviews: dict[str, dict[str, Any] | None] = {
            REVIEW_SLOT: search.get("review"),
            **{stage["slot"]: stage for stage in search.get("chain_reviews", [])},
        }
        if sample_id not in reviews:
            raise KeyError(sample_id)
        return reviews[sample_id]
    sample = next((s for s in document["samples"] if s["sample_slot_id"] == sample_id), None)
    if sample is None:
        raise KeyError(sample_id)
    stage: dict[str, Any] | None = sample[phase]
    return stage
