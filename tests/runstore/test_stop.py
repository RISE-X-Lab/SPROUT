"""Stopping a job: a stop request cancels a queued job, and a running one reads it at its next progress write.

These tests use the PostgreSQL test database (``run_store``), and skip without it.
"""

from __future__ import annotations

from typing import Any

import pytest

from sprout.runstore.store import Conflict, Store
from tests.runstore.builders import job
from tests.support.runstore import registered


def projection(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: row[key] for key in keys}


def claimed_job(store: Store) -> dict[str, Any]:
    """A rollout that ``worker-1`` claimed."""
    registered(store, "worker-1")
    store.submit(job("rollout"), "key-1")
    claimed = store.claim("worker-1")
    assert claimed is not None
    return claimed


def test_a_stop_request_cancels_a_queued_job(run_store):
    submitted = run_store.submit(job("rollout"), "key-1")

    stopped = run_store.request_stop(submitted["id"])

    assert projection(stopped, "state", "phase") == {"state": "cancelled", "phase": "cancelled"}


def test_a_stop_request_leaves_a_running_job_running_until_its_worker_ends_it(run_store):
    """Only the worker can say that its attempt stopped."""
    claimed = claimed_job(run_store)

    stopped = run_store.request_stop(claimed["id"])

    assert (stopped["state"], stopped["stop_requested_at"] is not None) == ("running", True)


def test_the_next_progress_write_of_a_running_job_reports_its_stop_request(run_store):
    claimed = claimed_job(run_store)
    before = run_store.progress(claimed["id"], claimed["lease_token"])
    run_store.request_stop(claimed["id"])

    after = run_store.progress(claimed["id"], claimed["lease_token"])

    assert (before, after) == (False, True)


def test_a_repeated_stop_request_keeps_the_time_of_the_first(run_store):
    claimed = claimed_job(run_store)
    first = run_store.request_stop(claimed["id"])["stop_requested_at"]

    again = run_store.request_stop(claimed["id"])["stop_requested_at"]

    assert again == first


def test_a_stop_request_leaves_a_finished_job_as_it_is(run_store):
    claimed = claimed_job(run_store)
    run_store.finish(claimed["id"], claimed["lease_token"], {"status": "completed"})
    before = run_store.get(claimed["id"])

    stopped = run_store.request_stop(claimed["id"])

    assert stopped == before


def test_a_stop_request_for_an_unknown_job_raises_key_error(run_store):
    with pytest.raises(KeyError, match="no-such-job"):
        run_store.request_stop("no-such-job")


def test_cancel_cancels_a_queued_job(run_store):
    submitted = run_store.submit(job("rollout"), "key-1")

    run_store.cancel_queued(submitted["id"])

    assert projection(run_store.get(submitted["id"]), "state", "phase") == {
        "state": "cancelled",
        "phase": "cancelled",
    }


def test_cancel_refuses_a_running_job(run_store):
    claimed = claimed_job(run_store)

    with pytest.raises(
        Conflict, match="^Only a queued job can be cancelled; a running one needs a stop request$"
    ):
        run_store.cancel_queued(claimed["id"])

    assert run_store.get(claimed["id"])["state"] == "running"


def test_cancel_refuses_an_unknown_job(run_store):
    with pytest.raises(
        Conflict, match="^Only a queued job can be cancelled; a running one needs a stop request$"
    ):
        run_store.cancel_queued("no-such-job")
