"""Which job a worker's claim takes: by the host of its affinity key, its bound worker and its grader.

These tests use the PostgreSQL test database (``run_store``), and skip without it.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Thread
from typing import Any

import pytest

from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Conflict, Store
from tests.runstore.builders import TASK_DIR, job, rollout_spec, swerebench_grade_spec
from tests.support.runstore import registered, stopped
from tests.support.waiting import wait_until

#: Fixed deadlines, so that no test depends on the clock: one long past, one far ahead.
PAST, FUTURE = "2000-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00"


def projection(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: row[key] for key in keys}


def kept_rollout(store: Store, keep_until: str) -> str:
    """The id of a finished rollout whose sandbox ``rollout-worker`` keeps until ``keep_until``."""
    registered(store, "rollout-worker")
    rollout = store.submit(job("rollout", keep_sandbox=True), "kept-rollout")
    claimed = store.claim("rollout-worker")
    result = {"status": "completed", "kept_sandbox": {"sandbox_id": "sandbox-1", "keep_until": keep_until}}
    store.finish(rollout["id"], claimed["lease_token"], result)
    return str(rollout["id"])


def bound_grade(store: Store, rollout_id: str) -> str:
    """The id of a queued grade in the sandbox that the rollout ``rollout_id`` kept."""
    grade = JobSpec("grade", {"benchmark": "harbor", "task_dir": TASK_DIR, "in_place_job": rollout_id})
    return str(store.submit(grade, "bound-grade")["id"])


# Unbound jobs


def test_a_claim_leases_the_job_to_the_worker_and_opens_its_first_attempt(run_store):
    registered(run_store, "worker-1")
    submitted = run_store.submit(job("rollout"), "key-1")

    claimed = run_store.claim("worker-1")

    attempt = run_store.attempts(submitted["id"])[0]
    assert projection(claimed, "id", "state", "phase", "attempt_count", "worker_id") == {
        "id": submitted["id"],
        "state": "running",
        "phase": "claimed",
        "attempt_count": 1,
        "worker_id": "worker-1",
    }
    assert projection(attempt, "id", "number", "worker_id", "lease_token") == {
        "id": claimed["active_attempt"],
        "number": 1,
        "worker_id": "worker-1",
        "lease_token": claimed["lease_token"],
    }


def test_claims_take_queued_jobs_in_submission_order(run_store):
    registered(run_store, "worker-1", "worker-2")
    first = run_store.submit(job("rollout"), "key-1")["id"]
    second = run_store.submit(job("rollout"), "key-2")["id"]
    third = run_store.submit(job("rollout"), "key-3")["id"]

    claims = [
        run_store.claim("worker-1")["id"],
        run_store.claim("worker-2")["id"],
        run_store.claim("worker-1")["id"],
    ]

    assert claims == [first, second, third]


def test_a_claim_finds_nothing_once_each_job_is_claimed(run_store):
    registered(run_store, "worker-1", "worker-2")
    run_store.submit(job("rollout"), "key-1")
    run_store.claim("worker-1")

    assert run_store.claim("worker-2") is None


def test_a_claim_takes_only_the_kinds_that_the_worker_runs(run_store):
    registered(run_store, "grader")
    run_store.submit(job("rollout"), "rollout-1")
    grade = run_store.submit(job("grade"), "grade-1")

    assert run_store.claim("grader", kinds=("grade",))["id"] == grade["id"]


def test_a_claim_leases_the_job_for_600_s_as_a_heartbeat_of_the_worker_renews_it(run_store):
    """A worker that dies before its first heartbeat still leaves its jobs to the sweep after 600 s."""
    registered(run_store, "worker-1")
    run_store.submit(job("rollout"), "key-1")

    claimed = run_store.claim("worker-1")

    assert round((claimed["lease_until"] - claimed["updated_at"]).total_seconds()) == 600


def test_a_claim_skips_a_job_that_another_claim_holds_instead_of_waiting_for_it(run_store):
    """Without the skip, the claim waits until the statement timeout fails it."""
    registered(run_store, "worker-1")
    held = run_store.submit(job("rollout"), "key-1")
    free = run_store.submit(job("rollout"), "key-2")

    with run_store.transaction() as cursor:
        cursor.execute("SELECT id FROM rs_jobs WHERE id=%s FOR UPDATE", (held["id"],))
        claimed = run_store.claim("worker-1")

    assert claimed["id"] == free["id"]


# Bound grades


def test_only_the_bound_worker_claims_a_bound_grade_before_keep_until(run_store):
    registered(run_store, "other-worker")
    grade_id = bound_grade(run_store, kept_rollout(run_store, FUTURE))

    other, bound = run_store.claim("other-worker"), run_store.claim("rollout-worker")

    assert (other, bound["id"]) == (None, grade_id)


def test_the_bound_worker_claims_its_grade_before_jobs_queued_earlier(run_store):
    """A bound grade that waits behind other jobs can outlive its sandbox."""
    rollout_id = kept_rollout(run_store, FUTURE)
    earlier = run_store.submit(job("rollout"), "earlier")
    grade_id = bound_grade(run_store, rollout_id)

    claims = [run_store.claim("rollout-worker")["id"], run_store.claim("rollout-worker")["id"]]

    assert claims == [grade_id, earlier["id"]]


def test_a_claim_settles_a_bound_grade_past_keep_until_as_sandbox_lost_and_takes_the_next_job(run_store):
    registered(run_store, "other-worker")
    grade_id = bound_grade(run_store, kept_rollout(run_store, PAST))
    rollout = run_store.submit(job("rollout"), "unbound")

    claimed = run_store.claim("other-worker")

    settled = run_store.get(grade_id)
    assert (claimed["id"], projection(settled, "state", "phase", "result", "attempt_count")) == (
        rollout["id"],
        {
            "state": "succeeded",
            "phase": "succeeded",
            "result": {"reward": 0.0, "reason": "sandbox_lost"},
            "attempt_count": 0,
        },
    )


# Grader matching

#: The checksums of the dataset and the parser that the grades below name.
DATASET, PARSER = "0" * 64, "sha256:" + "1" * 64


def swerebench_grade(store: Store) -> str:
    """The id of a queued ``swe-rebench-v2`` grade of the dataset DATASET with the parser PARSER."""
    spec = swerebench_grade_spec(dataset_sha256=DATASET, grader_revision=PARSER)
    return str(store.submit(JobSpec("grade", spec, affinity="group-1"), "grade")["id"])


@pytest.mark.parametrize(
    "grades, claims",
    [
        pytest.param([(DATASET, PARSER)], True, id="both"),
        pytest.param([(DATASET, "sha256:" + "2" * 64)], False, id="other-parser"),
        pytest.param([("3" * 64, PARSER)], False, id="other-dataset"),
        pytest.param([(DATASET, "sha256:" + "2" * 64), ("3" * 64, PARSER)], False, id="each-in-another-pair"),
        pytest.param([], False, id="none"),
    ],
)
def test_a_swerebench_grade_goes_only_to_a_worker_that_has_its_dataset_and_parser_as_one_pair(
    run_store, grades, claims
):
    """Another worker would read files with other checksums, so the grade could not be made."""
    run_store.register_worker("grader", host="host-1", session="session-1", grades=grades)
    grade_id = swerebench_grade(run_store)

    claimed = run_store.claim("grader")

    assert (claimed is not None and claimed["id"] == grade_id) is claims


def test_a_harbor_grade_goes_to_a_worker_without_grades(run_store):
    """A harbor grade reads its task directory, which no checksum pins."""
    run_store.register_worker("grader", host="host-1", session="session-1")
    grade = run_store.submit(job("grade"), "grade")

    assert run_store.claim("grader")["id"] == grade["id"]


def test_a_session_that_takes_over_a_worker_id_claims_by_its_own_grades(run_store):
    """A worker that starts again checks its profiles again, and its files can have changed."""
    run_store.register_worker("grader", host="host-1", session="old", grades=[(DATASET, PARSER)])
    with run_store.transaction() as cursor:
        cursor.execute("UPDATE rs_workers SET seen_at=clock_timestamp() - interval '610 seconds'")
    run_store.register_worker("grader", host="host-1", session="new")
    swerebench_grade(run_store)

    assert run_store.claim("grader") is None


# Live workers


def silent(store: Store, worker_id: str) -> None:
    """Register ``worker_id``, and make its last heartbeat 610 s old."""
    registered(store, worker_id)
    with store.transaction() as cursor:
        cursor.execute(
            "UPDATE rs_workers SET seen_at=clock_timestamp() - interval '610 seconds' WHERE worker_id=%s",
            (worker_id,),
        )


def registered_and_stopped(store: Store, worker_id: str) -> None:
    registered(store, worker_id)
    stopped(store, worker_id)


#: The steps that leave ``worker-1`` not live: without a heartbeat in the last 600 s, or stopped.
NOT_LIVE: dict[str, Callable[[Store], object]] = {
    "never-registered": lambda store: None,
    "silent-for-610-s": lambda store: silent(store, "worker-1"),
    "stopped": lambda store: registered_and_stopped(store, "worker-1"),
}


@pytest.mark.parametrize("step", NOT_LIVE.values(), ids=NOT_LIVE.keys())
def test_a_worker_that_is_not_live_is_refused_a_claim(run_store, step):
    """The claim needs the worker's host and grades, and the sweep would take a dead worker's job."""
    submitted = run_store.submit(job("rollout"), "key-1")
    step(run_store)

    with pytest.raises(
        Conflict, match="^Worker worker-1 is not live: it stopped, or it has no heartbeat in 600 s$"
    ):
        run_store.claim("worker-1")

    assert run_store.get(submitted["id"])["state"] == "queued"


# Affinity keys


def of_group(key: str, name: str) -> JobSpec:
    """A rollout of the affinity key ``key``, whose prompt ``name`` makes it a request of its own."""
    return JobSpec("rollout", rollout_spec(prompt=name), affinity=key)


def test_a_job_of_a_bound_key_goes_only_to_a_worker_on_the_host_of_the_first_claim_of_the_key(run_store):
    """A later job of a group reads the data that the earlier jobs of the group left on that host."""
    registered(run_store, "worker-a", "other-on-a", host="host-a")
    registered(run_store, "worker-b", host="host-b")
    run_store.submit(of_group("group-1", "root"), "root")
    run_store.claim("worker-a")
    later = run_store.submit(of_group("group-1", "branch"), "branch")

    claims = (run_store.claim("worker-b"), run_store.claim("other-on-a")["id"])

    assert claims == (None, later["id"])


def test_on_its_host_a_claim_takes_the_jobs_of_keys_bound_there_before_jobs_queued_earlier(run_store):
    """A group that started on the host goes on before the host starts another one."""
    registered(run_store, "worker-a", host="host-a")
    run_store.submit(of_group("group-1", "root"), "root")
    run_store.claim("worker-a")
    earlier = run_store.submit(of_group("group-2", "root"), "earlier")
    later = run_store.submit(of_group("group-1", "branch"), "branch")

    claims = [run_store.claim("worker-a")["id"], run_store.claim("worker-a")["id"]]

    assert claims == [later["id"], earlier["id"]]


def test_a_bound_grade_comes_before_the_jobs_of_keys_bound_to_the_host(run_store):
    """The kept sandbox of the grade lives only until keep_until."""
    rollout_id = kept_rollout(run_store, FUTURE)
    run_store.submit(of_group("group-1", "root"), "root")
    run_store.claim("rollout-worker")
    branch = run_store.submit(of_group("group-1", "branch"), "branch")
    grade = {"benchmark": "harbor", "task_dir": TASK_DIR, "in_place_job": rollout_id}
    grade_id = run_store.submit(JobSpec("grade", grade), "bound-grade")["id"]

    claims = [run_store.claim("rollout-worker")["id"], run_store.claim("rollout-worker")["id"]]

    assert claims == [grade_id, branch["id"]]


def blocked_by(cursor: Any) -> int:
    """How many locks wait for the transaction of ``cursor``."""
    cursor.execute(
        """SELECT count(*) AS waiting FROM pg_locks
        WHERE NOT granted AND pg_backend_pid()=ANY(pg_blocking_pids(pid))"""
    )
    return int(cursor.fetchone()["waiting"])


def test_a_claim_that_another_host_beats_to_an_unbound_key_leaves_the_job_queued(run_store):
    """Two hosts that each ran jobs of one key would split the data of its group."""
    registered(run_store, "worker-b", host="host-b")
    queued = run_store.submit(of_group("group-1", "root"), "root")
    claims: list[dict[str, Any] | None] = []
    claim = Thread(target=lambda: claims.append(run_store.claim("worker-b")))

    with run_store.transaction() as cursor:
        # A claim on host-a has bound the key and has not committed yet.
        cursor.execute("INSERT INTO rs_affinity(key,host) VALUES('group-1','host-a')")
        claim.start()
        wait_until(lambda: blocked_by(cursor), timeout_s=4, what="the claim's wait for the binding")
    claim.join(timeout=5)

    assert (claims, run_store.get(queued["id"])["attempt_count"]) == ([None], 0)
