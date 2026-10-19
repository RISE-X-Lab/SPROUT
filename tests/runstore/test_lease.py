"""The lease of a claimed job: progress writes, fencing, and the transactions that write the lease.

These tests use the PostgreSQL test database (``run_store``), and skip without it.
"""

from __future__ import annotations

from typing import Any

import pytest

from sprout.runstore.index import Index
from sprout.runstore.store import Fenced, Store
from tests.runstore.builders import job, payload_for
from tests.support.runstore import registered


def claimed_job(store: Store) -> dict[str, Any]:
    """A rollout that ``worker-1`` claimed."""
    registered(store, "worker-1")
    store.submit(job("rollout"), "key-1")
    claimed = store.claim("worker-1")
    assert claimed is not None
    return claimed


def expire_lease(store: Store, job_id: str) -> None:
    """End the lease of ``job_id`` a second ago, as for a worker that stopped reporting heartbeats."""
    with store.transaction() as cursor:
        cursor.execute(
            "UPDATE rs_jobs SET lease_until=clock_timestamp() - interval '1 second' WHERE id=%s", (job_id,)
        )


# Progress


def test_a_progress_write_leaves_the_lease_to_the_heartbeat_of_the_worker(run_store):
    """Only the worker's heartbeat says that the worker lives, so the attempt must not extend the lease."""
    claimed = claimed_job(run_store)

    run_store.progress(claimed["id"], claimed["lease_token"], phase="rollout")

    assert run_store.get(claimed["id"])["lease_until"] == claimed["lease_until"]


def test_a_progress_write_with_a_phase_replaces_the_phase_and_one_without_keeps_it(run_store):
    claimed = claimed_job(run_store)

    run_store.progress(claimed["id"], claimed["lease_token"], phase="running-agent")
    run_store.progress(claimed["id"], claimed["lease_token"])

    assert run_store.get(claimed["id"])["phase"] == "running-agent"


def test_progress_writes_merge_their_execution_into_the_attempts_key_by_key(run_store):
    claimed = claimed_job(run_store)

    run_store.progress(claimed["id"], claimed["lease_token"], execution={"process": {"pid": 10}})
    run_store.progress(claimed["id"], claimed["lease_token"], execution={"scope": {"run_id": "run-1"}})
    run_store.progress(claimed["id"], claimed["lease_token"], execution={"process": {"pid": 11}})

    assert run_store.attempts(claimed["id"])[0]["execution"] == {
        "process": {"pid": 11},
        "scope": {"run_id": "run-1"},
    }


# Fencing

#: Each write that takes a lease token, called with the token of ``claimed``.
WRITES = {
    "progress": lambda store, claimed: store.progress(claimed["id"], claimed["lease_token"]),
    "finish": lambda store, claimed: store.finish(
        claimed["id"], claimed["lease_token"], {"status": "completed"}
    ),
    "release": lambda store, claimed: store.release(
        claimed["id"], claimed["lease_token"], {"status": "error", "stop_reason": "worker_shutdown"}
    ),
    "append_events": lambda store, claimed: store.append_events(
        claimed["id"], claimed["lease_token"], [{"seq": 1}]
    ),
    "freeze_payload": lambda store, claimed: store.freeze_payload(
        claimed["id"], claimed["lease_token"], payload_for(claimed)
    ),
    "project": lambda store, claimed: Index(store).project(claimed["id"], claimed["lease_token"], {}, []),
}


@pytest.mark.parametrize("write", WRITES.values(), ids=WRITES.keys())
def test_each_write_with_the_token_of_an_expired_lease_is_fenced(run_store, write):
    claimed = claimed_job(run_store)
    expire_lease(run_store, claimed["id"])

    with pytest.raises(Fenced, match="Attempt no longer owns a live lease"):
        write(run_store, claimed)


def test_a_write_with_a_token_that_never_owned_the_lease_is_fenced(run_store):
    claimed = claimed_job(run_store)

    with pytest.raises(Fenced):
        run_store.progress(claimed["id"], "not-the-lease-token")


def test_a_finished_attempt_cannot_overwrite_its_result(run_store):
    claimed = claimed_job(run_store)
    run_store.finish(claimed["id"], claimed["lease_token"], {"final_text": "first"})

    with pytest.raises(Fenced):
        run_store.finish(claimed["id"], claimed["lease_token"], {"final_text": "stale"})

    assert Store(run_store.dsn).get(claimed["id"])["result"] == {"final_text": "first"}


# Transactions


def test_a_transaction_that_raises_changes_nothing(run_store):
    submitted = run_store.submit(job("rollout"), "key-1")

    with pytest.raises(RuntimeError), run_store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET phase='changed' WHERE id=%s", (submitted["id"],))
        raise RuntimeError("the block failed")

    assert run_store.get(submitted["id"])["phase"] == "queued"


def test_a_transaction_bounds_each_wait_on_the_database(run_store):
    """A dead or locked database fails a worker's call within seconds, instead of hanging its claim loop."""
    with run_store.transaction() as cursor:
        cursor.execute("SHOW statement_timeout")
        statement_timeout = cursor.fetchone()["statement_timeout"]
        connection = cursor.connection.get_dsn_parameters()

    assert (statement_timeout, connection["connect_timeout"], connection["tcp_user_timeout"]) == (
        "5s",
        "5",
        "10000",
    )
