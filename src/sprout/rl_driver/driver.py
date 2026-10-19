"""The RL driver's groups: kept in the ledger, run by the search coordinator.

The driver checks each execution group, stores it in the ledger, and lets the search
coordinator (``sprout.search.coordinator.Coordinator``) submit and follow its jobs. A view
of a group leaves out the submission bodies. The Run Store's workers run every agent and
grade job.
"""

import threading
from copy import deepcopy
from typing import Any

from sprout.rl_driver.ledger import Ledger
from sprout.search.client import RunStoreClient
from sprout.search.coordinator import Coordinator
from sprout.search.group import Submitter, initial_document, validate_request

#: The names the driver's jobs carry in the Run Store. A restarted driver finds the jobs that
#: its stored groups submitted by these keys, so a changed name would submit them again.
SUBMITTER = Submitter(context_key="rl_driver", key_prefix="rl-driver/")


class Driver:
    """Runs the execution groups of one ledger as jobs of one Run Store.

    ``tick()`` advances the groups. A change through this class sets ``wakeup`` to ask the
    poller for an early tick. ``stopping`` tells the poller and a running tick to stop.
    """

    def __init__(self, client: RunStoreClient, ledger: Ledger) -> None:
        self.ledger = ledger
        self.wakeup = threading.Event()
        self.stopping = threading.Event()
        self._init_coordinator(client)

    def _init_coordinator(self, client: RunStoreClient) -> None:
        self._coordinator = Coordinator(
            client=client, groups=self.ledger, submitter=SUBMITTER, stopping=self.stopping
        )

    def submit(
        self,
        body: dict[str, Any],
        *,
        source_request: dict[str, Any] | None = None,
        extra_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Check the execution group ``body``, record it, and return its view (``get``).

        A resubmission must repeat ``source_request``, by default the checked group. ``extra_document`` adds
        fields to the document. Raises ValueError for an invalid group, and ``Conflict`` for a reused id.
        """
        request = validate_request(body)
        # The jobs of a Miles group carry its id as their affinity key (ADR-0007).
        initial = initial_document(
            request, context_key=SUBMITTER.context_key, affinity=request["rollout_job_id"]
        )
        document = {**initial, **(extra_document or {})}
        row = self.ledger.create(request["rollout_job_id"], source_request or request, document)
        self.wakeup.set()
        return self._view(row)

    @staticmethod
    def _view(row: dict[str, Any]) -> dict[str, Any]:
        document: dict[str, Any] = deepcopy(row["document"])
        document["cancel_requested"] = row["cancel_requested"]
        document["acknowledged_at"] = row["acknowledged_at"]
        if row["cancel_requested"] and not row["terminal"]:
            document["status"] = "cancelling"
        # Submission bodies can hold long prompts, and the ledger keeps them. A view gives only the
        # references to the jobs.
        for run in [document, *document.get("earlier_runs", [])]:
            _drop_submissions(run)
        return document

    def get(self, group_id: str) -> dict[str, Any]:
        """The group's view: its document without the submission bodies. Raises KeyError for an unknown group.

        The view adds ``cancel_requested`` and ``acknowledged_at``. An active group with a cancellation
        request has the ``status`` ``cancelling``.
        """
        return self._view(self.ledger.get(group_id))

    def release(self, group_id: str) -> dict[str, Any]:
        """Release the group (``Ledger.release``), and return its id, ``status`` and ``acknowledged_at``."""
        self.ledger.release(group_id)
        self.wakeup.set()
        view = self.get(group_id)
        return {key: view[key] for key in ("rollout_job_id", "status", "acknowledged_at")}

    def tick(self) -> None:
        """Advance every active group once. A failed Run Store call for one group does not stop the others."""
        self._coordinator.tick()

    def reconcile_submission(self, group_id: str, sample_id: str, phase: str, job_id: str) -> dict[str, Any]:
        """Attach the accepted job ``job_id`` to a stage whose submission reply was lost, and return the view.

        The job must carry the stage's idempotency key. The call submits no job and changes no job's state.
        ``Coordinator.reconcile_submission`` gives the phases, the review slots and the errors.
        """
        self._coordinator.reconcile_submission(group_id, sample_id=sample_id, phase=phase, job_id=job_id)
        self.wakeup.set()
        return self.get(group_id)


def _drop_submissions(run: dict[str, Any]) -> None:
    """Remove the submission bodies and the templates from the samples and the search of one run."""
    for sample in run["samples"]:
        for phase in ("actor", "grade"):
            stage = sample[phase]
            if stage:
                stage.pop("submission", None)
                stage.pop("template", None)
    search = run.get("search")
    if search:
        search.pop("review_templates", None)
        # The one review template of a search document that predates outcome-keyed review.
        search.pop("review_template", None)
        for stage in [search.get("review"), *search.get("chain_reviews", [])]:
            if stage:
                stage.pop("submission", None)
