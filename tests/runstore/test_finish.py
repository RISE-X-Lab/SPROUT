"""Ending an attempt: the job's next state, its result, the rerun of an infrastructure failure, and a release.

These tests use the PostgreSQL test database (``run_store``), and skip without it.
"""

from __future__ import annotations

from typing import Any

import pytest

from sprout.contract.mini_history import NativePoint
from sprout.runstore.index import Index
from sprout.runstore.store import Store
from tests.runstore.builders import job, stored_job
from tests.support.runstore import registered

INFRASTRUCTURE_FAILURE = {
    "status": "error",
    "failure_kind": "infrastructure",
    "error": "HTTP 502 Bad Gateway",
}
#: The result of an attempt that the worker's shutdown stopped, as the worker records it.
WORKER_SHUTDOWN = {
    "status": "error",
    "failure_kind": "infrastructure",
    "stop_reason": "worker_shutdown",
    "error": "AttemptStopped: worker_shutdown",
}


def claimed_job(
    store: Store, *, kind: str = "rollout", attempts_left: bool = True, stop_requested: bool = False
) -> dict[str, Any]:
    """A job of ``kind`` that a worker claimed for its first attempt.

    With ``attempts_left``, one more attempt remains after this one. Otherwise this attempt is the last.
    """
    registered(store, "worker-1")
    store.submit(stored_job(store, kind, max_infra_retries=1 if attempts_left else 0), "key-1")
    claimed = store.claim("worker-1")
    assert claimed is not None
    if stop_requested:
        store.request_stop(claimed["id"])
    return claimed


def result_for(rerun: bool) -> dict[str, Any]:
    return INFRASTRUCTURE_FAILURE if rerun else {"status": "completed"}


def state_and_phase(store: Store, job_id: str) -> tuple[str, str]:
    row = store.get(job_id)
    return row["state"], row["phase"]


def projection(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: row[key] for key in keys}


def seconds_between(later: Any, earlier: Any) -> int:
    return round((later - earlier).total_seconds())


def ready_now(store: Store, job_id: str) -> None:
    """End the backoff of the queued ``job_id`` now, so that the next claim takes it."""
    with store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET ready_at=clock_timestamp() WHERE id=%s", (job_id,))


# 3-way covering array (covering_array.py) of the finish() parameters, from the worked example of the
# testing skill. The requeue condition joins three of them, so t is 3.
@pytest.mark.parametrize(
    "state,rerun,kind,attempts_left,stop_requested",
    [
        ("succeeded", False, "rollout", False, False),
        ("succeeded", False, "grade", True, True),
        ("succeeded", False, "review", False, True),
        ("failed", False, "rollout", True, False),
        ("failed", False, "grade", False, False),
        ("failed", False, "review", True, True),
        ("failed", True, "rollout", False, True),
        ("failed", True, "grade", True, False),
        ("failed", True, "review", False, False),
        ("succeeded", False, "rollout", True, True),
        ("succeeded", False, "grade", False, False),
        ("succeeded", False, "review", True, False),
        ("failed", True, "rollout", True, False),
        ("failed", True, "grade", False, True),
        ("failed", True, "review", True, True),
    ],
)
def test_finish_requeues_only_a_rerun_failure_with_attempts_left_and_no_stop(
    run_store, state, rerun, kind, attempts_left, stop_requested
):
    claimed = claimed_job(run_store, kind=kind, attempts_left=attempts_left, stop_requested=stop_requested)

    run_store.finish(claimed["id"], claimed["lease_token"], result_for(rerun), state=state, rerun=rerun)

    requeued = rerun and attempts_left and not stop_requested
    expected = ("queued", "retry_wait") if requeued else (state, state)
    assert state_and_phase(run_store, claimed["id"]) == expected


@pytest.mark.parametrize("state", ["queued", "running", "cancelled", "quarantined"])
def test_finish_refuses_a_state_that_does_not_end_an_attempt(run_store, state):
    claimed = claimed_job(run_store)

    with pytest.raises(ValueError, match="Invalid terminal outcome"):
        run_store.finish(claimed["id"], claimed["lease_token"], {"status": "completed"}, state=state)


@pytest.mark.parametrize(
    "state,result",
    [
        ("succeeded", INFRASTRUCTURE_FAILURE),
        ("failed", {"status": "error", "failure_kind": "actor", "error": "No patch produced"}),
        ("failed", {"status": "error", "error": "HTTP 502 Bad Gateway"}),
    ],
    ids=["succeeded", "actor-failure", "unclassified-failure"],
)
def test_only_a_failed_infrastructure_attempt_runs_again(run_store, state, result):
    claimed = claimed_job(run_store, kind="grade")

    with pytest.raises(ValueError, match="^Only an infrastructure failure runs again$"):
        run_store.finish(claimed["id"], claimed["lease_token"], result, state=state, rerun=True)


def test_an_infrastructure_failure_of_a_rollout_runs_it_again_without_a_recovery_point(run_store):
    """Each attempt starts from the job's spec, so a rerun needs nothing from the attempt before it."""
    claimed = claimed_job(run_store, kind="rollout")

    run_store.finish(
        claimed["id"], claimed["lease_token"], INFRASTRUCTURE_FAILURE, state="failed", rerun=True
    )

    assert state_and_phase(run_store, claimed["id"]) == ("queued", "retry_wait")


def test_finish_records_the_result_on_the_job_and_its_attempt_and_releases_the_lease(run_store):
    claimed = claimed_job(run_store, kind="grade")

    run_store.finish(claimed["id"], claimed["lease_token"], INFRASTRUCTURE_FAILURE, state="failed")

    stored = Store(run_store.dsn)
    finished, attempt = stored.get(claimed["id"]), stored.attempts(claimed["id"])[0]
    assert projection(finished, "result", "error", "lease_token", "lease_until") == {
        "result": INFRASTRUCTURE_FAILURE,
        "error": "HTTP 502 Bad Gateway",
        "lease_token": None,
        "lease_until": None,
    }
    assert projection(attempt, "state", "result", "error", "execution") == {
        "state": "failed",
        "result": INFRASTRUCTURE_FAILURE,
        "error": "HTTP 502 Bad Gateway",
        "execution": {},
    }


def test_a_result_and_an_error_with_a_nul_character_are_kept_with_a_replacement_character(run_store):
    """A rollout's result holds its tool results, which can carry U+0000; PostgreSQL cannot hold it."""
    claimed = claimed_job(run_store, kind="rollout")
    result = {"status": "error", "error": "stdout: DIRC\x00", "messages": [{"content": "LASF\x00"}]}

    run_store.finish(claimed["id"], claimed["lease_token"], result, state="failed")

    kept = {"status": "error", "error": "stdout: DIRC\ufffd", "messages": [{"content": "LASF\ufffd"}]}
    finished, attempt = run_store.get(claimed["id"]), run_store.attempts(claimed["id"])[0]
    assert projection(finished, "result", "error") == {"result": kept, "error": "stdout: DIRC\ufffd"}
    assert projection(attempt, "result", "error") == {"result": kept, "error": "stdout: DIRC\ufffd"}


def test_a_rerun_job_waits_30_seconds_before_a_claim_takes_it_again(run_store):
    registered(run_store, "worker-2")
    claimed = claimed_job(run_store, kind="grade")

    run_store.finish(
        claimed["id"], claimed["lease_token"], INFRASTRUCTURE_FAILURE, state="failed", rerun=True
    )

    waiting = run_store.get(claimed["id"])
    assert (seconds_between(waiting["ready_at"], waiting["updated_at"]), run_store.claim("worker-2")) == (
        30,
        None,
    )


def test_the_wait_before_a_rerun_grows_by_30_seconds_with_each_attempt(run_store):
    registered(run_store, "worker-1")
    run_store.submit(job("grade", max_infra_retries=2), "key-1")
    first = run_store.claim("worker-1")
    run_store.finish(first["id"], first["lease_token"], INFRASTRUCTURE_FAILURE, state="failed", rerun=True)
    # Make the first wait end now, so that the second attempt starts without a 30 s pause.
    ready_now(run_store, first["id"])
    second = run_store.claim("worker-1")

    run_store.finish(second["id"], second["lease_token"], INFRASTRUCTURE_FAILURE, state="failed", rerun=True)

    waiting = run_store.get(second["id"])
    assert (waiting["attempt_count"], seconds_between(waiting["ready_at"], waiting["updated_at"])) == (2, 60)


def indexed(store: Store, claimed: dict[str, Any], *, call_id: str = "call-1") -> None:
    """Give the attempt of ``claimed`` an event, an indexed ``ls`` call and a recovery point after it."""
    call = {"seq": 1, "type": "tool.started", "call_id": call_id, "name": "bash", "args": {"command": "ls"}}
    store.append_events(claimed["id"], claimed["lease_token"], [call])
    native = {"slot": "mini-swe-agent", "path": "/path/to/native.jsonl", "byte_length": 1, "sha256": "0" * 64}
    point = NativePoint(message_step=1, tool_depth=1, snapshot_id="snap-1", native=native)
    Index(store).project(claimed["id"], claimed["lease_token"], {"context": {}}, [call], [point])


def prefix_nodes(store: Store) -> int:
    with store.transaction() as cursor:
        cursor.execute("SELECT count(*) AS nodes FROM rs_prefix_nodes")
        return int(cursor.fetchone()["nodes"])


def test_an_attempt_that_runs_again_ends_lost_and_the_run_store_forgets_its_records(run_store):
    """The next attempt starts from the spec. A branch from a point of the lost attempt must find no point."""
    claimed = claimed_job(run_store)
    indexed(run_store, claimed)
    index, attempt_id = Index(run_store), claimed["active_attempt"]

    run_store.finish(
        claimed["id"], claimed["lease_token"], INFRASTRUCTURE_FAILURE, state="failed", rerun=True
    )

    [attempt] = run_store.attempts(claimed["id"])
    assert projection(attempt, "state", "error") == {"state": "lost", "error": "HTTP 502 Bad Gateway"}
    assert (run_store.events(attempt_id), index.tools(attempt_id), index.points(attempt_id)) == ([], [], [])
    assert prefix_nodes(run_store) == 0


def test_a_forgotten_attempt_leaves_the_prefix_nodes_that_another_attempt_shares(run_store):
    """Two attempts with one scope and the same first call share its node, by its digest."""
    registered(run_store, "worker-1")
    run_store.submit(job("rollout"), "kept")
    kept = run_store.claim("worker-1")
    indexed(run_store, kept)
    run_store.submit(job("rollout"), "lost")
    lost = run_store.claim("worker-1")
    indexed(run_store, lost)

    run_store.finish(lost["id"], lost["lease_token"], INFRASTRUCTURE_FAILURE, state="failed", rerun=True)

    tools = Index(run_store).tools(kept["active_attempt"])
    assert ([tool["call_id"] for tool in tools], prefix_nodes(run_store)) == (["call-1"], 1)


def test_a_job_that_runs_out_of_attempts_fails_with_the_error_of_each_attempt_that_counted(run_store):
    """A released attempt ran nothing that counts, so its error is not among them."""
    registered(run_store, "worker-1")
    run_store.submit(job("grade", max_infra_retries=1), "key-1")
    first = run_store.claim("worker-1")
    run_store.finish(
        first["id"],
        first["lease_token"],
        {**INFRASTRUCTURE_FAILURE, "error": "first"},
        state="failed",
        rerun=True,
    )
    ready_now(run_store, first["id"])
    released = run_store.claim("worker-1")
    run_store.release(released["id"], released["lease_token"], WORKER_SHUTDOWN)
    last = run_store.claim("worker-1")

    run_store.finish(
        last["id"],
        last["lease_token"],
        {**INFRASTRUCTURE_FAILURE, "error": "last"},
        state="failed",
        rerun=True,
    )

    failed = run_store.get(first["id"])
    states = [attempt["state"] for attempt in run_store.attempts(first["id"])]
    assert (failed["state"], failed["result"]["attempt_errors"], states) == (
        "failed",
        ["first", "last"],
        ["lost", "released", "failed"],
    )


# Release


@pytest.mark.parametrize("kind", ["rollout", "grade", "review"])
def test_a_released_job_is_claimed_again_at_once_and_its_attempt_does_not_count(run_store, kind):
    """The job has one attempt. A released attempt neither waits for a backoff nor uses that attempt."""
    registered(run_store, "worker-1", "worker-2")
    run_store.submit(stored_job(run_store, kind, max_infra_retries=0), "key-1")
    first = run_store.claim("worker-1")
    run_store.release(first["id"], first["lease_token"], WORKER_SHUTDOWN)

    again = run_store.claim("worker-2")

    assert again is not None, "the released job is not ready for a claim"
    assert projection(again, "id", "attempt_count") == {"id": first["id"], "attempt_count": 1}


def test_a_release_ends_the_attempt_as_released_and_queues_the_job_without_its_lease(run_store):
    claimed = claimed_job(run_store)

    run_store.release(claimed["id"], claimed["lease_token"], WORKER_SHUTDOWN)

    released, attempt = run_store.get(claimed["id"]), run_store.attempts(claimed["id"])[0]
    assert projection(released, "state", "phase", "attempt_count", "lease_token", "lease_until") == {
        "state": "queued",
        "phase": "released",
        "attempt_count": 0,
        "lease_token": None,
        "lease_until": None,
    }
    assert projection(attempt, "state", "result", "error") == {
        "state": "released",
        "result": WORKER_SHUTDOWN,
        "error": "AttemptStopped: worker_shutdown",
    }


def test_a_release_forgets_the_records_of_the_attempt_that_a_shutdown_stopped(run_store):
    """The job runs again from its spec, as after a lost attempt."""
    claimed = claimed_job(run_store)
    indexed(run_store, claimed)
    index, attempt_id = Index(run_store), claimed["active_attempt"]

    run_store.release(claimed["id"], claimed["lease_token"], WORKER_SHUTDOWN)

    assert (run_store.events(attempt_id), index.tools(attempt_id), index.points(attempt_id)) == ([], [], [])


def test_the_attempt_after_a_release_takes_the_next_number(run_store):
    """The released attempt keeps its number, so the next claim must not reuse it."""
    registered(run_store, "worker-2")
    claimed = claimed_job(run_store)
    run_store.release(claimed["id"], claimed["lease_token"], WORKER_SHUTDOWN)

    run_store.claim("worker-2")

    numbers = [(attempt["number"], attempt["state"]) for attempt in run_store.attempts(claimed["id"])]
    assert numbers == [(1, "released"), (2, "running")]


def test_a_released_job_with_a_stop_request_is_cancelled_instead_of_queued(run_store):
    """A stop request cancels a queued job, so a release must not queue a job that was asked to stop."""
    claimed = claimed_job(run_store, stop_requested=True)

    run_store.release(claimed["id"], claimed["lease_token"], WORKER_SHUTDOWN)

    assert state_and_phase(run_store, claimed["id"]) == ("cancelled", "cancelled")
