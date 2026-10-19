"""Submitting a job to the Run Store: its schema, idempotency keys, its affinity key and an in-place binding.

These tests use the PostgreSQL test database (``run_store``), and skip without it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg2
import pytest

from sprout.contract.mini_history import NativePoint
from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Conflict, Store
from tests.runstore.builders import (
    TASK_DIR,
    harbor_grade_spec,
    job,
    native_history,
    review_parent,
    review_spec,
    rollout_spec,
    stored_job,
    swerebench_grade_spec,
)
from tests.support.runstore import registered

#: Fixed deadlines, so that no test depends on the clock: one long past, one far ahead.
PAST, FUTURE = "2000-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00"


def kept(keep_until: object) -> dict[str, Any]:
    """The result of a rollout that kept its sandbox until ``keep_until``."""
    return {"status": "completed", "kept_sandbox": {"sandbox_id": "sandbox-1", "keep_until": keep_until}}


def finished_rollout(store: Store, result: dict[str, Any], *, worker: str = "rollout-worker") -> str:
    """The id of a rollout that ``worker`` ran and finished with ``result``."""
    registered(store, worker)
    submitted = store.submit(job("rollout", keep_sandbox=True), "kept-rollout")
    claimed = store.claim(worker)
    store.finish(submitted["id"], claimed["lease_token"], result)
    return submitted["id"]


def in_place_grade(rollout_id: str) -> JobSpec:
    return JobSpec("grade", {"benchmark": "harbor", "task_dir": TASK_DIR, "in_place_job": rollout_id})


def binding(row: dict[str, Any]) -> tuple[object, object]:
    return row["bound_worker"], row["keep_until"]


def projection(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: row[key] for key in keys}


def claimed_by_rollout_worker(store: Store) -> None:
    registered(store, "rollout-worker")
    store.claim("rollout-worker")


#: The step that takes a queued rollout to each state in which it has not finished.
UNFINISHED = {
    "queued": lambda store: None,
    "running": claimed_by_rollout_worker,
}


# Submission


def test_a_submission_queues_the_validated_request(run_store):
    submitted = run_store.submit(job("rollout", model="policy"), "key-1")

    assert projection(submitted, "kind", "state", "phase", "attempt_count", "request") == {
        "kind": "rollout",
        "state": "queued",
        "phase": "queued",
        "attempt_count": 0,
        "request": job("rollout", model="policy").validate(),
    }


def test_a_repeated_submission_under_one_key_returns_the_first_job_after_a_restart(run_store):
    first = run_store.submit(job("rollout"), "key-1")

    assert Store(run_store.dsn).submit(job("rollout"), "key-1") == first


def test_another_request_under_a_used_key_is_a_conflict(run_store):
    run_store.submit(job("rollout"), "key-1")

    with pytest.raises(Conflict, match="Idempotency key already names a different request"):
        run_store.submit(job("rollout", prompt="Another task."), "key-1")


@pytest.mark.parametrize("key", ["k", "k" * 256], ids=["1-character", "256-characters"])
def test_an_idempotency_key_of_1_to_256_characters_is_accepted(run_store, key):
    assert run_store.submit(job("rollout"), key)["idempotency_key"] == key


@pytest.mark.parametrize("key", ["", "k" * 257], ids=["empty", "257-characters"])
def test_an_idempotency_key_outside_1_to_256_characters_is_refused(run_store, key):
    with pytest.raises(ValueError, match="An idempotency key of 1..256 characters is required"):
        run_store.submit(job("rollout"), key)


def test_a_request_that_the_specs_refuse_is_not_stored(run_store):
    with pytest.raises(ValueError, match="RunSpec requires a nonempty prompt"):
        run_store.submit(job("rollout", prompt=" "), "key-1")

    assert run_store.list_jobs() == []


# The schema


def test_initialize_on_an_initialized_schema_keeps_its_jobs(run_store):
    submitted = run_store.submit(job("rollout"), "key-1")

    run_store.initialize()

    assert run_store.get(submitted["id"]) == submitted


def test_initialize_brings_back_the_columns_that_a_database_from_before_them_lacks(run_store):
    """Each column that the schema added later comes back empty, so an upgraded row reads as before."""
    registered(run_store, "worker-1")
    submitted = run_store.submit(job("rollout"), "key-1")
    run_store.claim("worker-1")
    before = run_store.get(submitted["id"])
    with run_store.transaction() as cursor:
        cursor.execute("ALTER TABLE rs_jobs DROP COLUMN stop_requested_at, DROP COLUMN bound_worker")
        cursor.execute("ALTER TABLE rs_jobs DROP COLUMN keep_until")
        cursor.execute("ALTER TABLE rs_attempts DROP COLUMN payload, DROP COLUMN payload_hash")

    run_store.initialize()

    assert (run_store.get(submitted["id"]), run_store.attempts(submitted["id"])[0]["payload"]) == (
        before,
        None,
    )


def test_initialize_refuses_the_quarantined_state_in_a_database_from_before_adr_0006(run_store):
    """No job is quarantined any more, so no reader of a job state needs to know the state."""
    submitted = run_store.submit(job("rollout"), "key-1")
    with run_store.transaction() as cursor:
        cursor.execute("ALTER TABLE rs_jobs DROP CONSTRAINT rs_jobs_state_check")
        cursor.execute(
            """ALTER TABLE rs_jobs ADD CONSTRAINT rs_jobs_state_check CHECK
            (state IN ('queued', 'running', 'quarantined', 'succeeded', 'failed', 'cancelled'))"""
        )

    run_store.initialize()

    with pytest.raises(psycopg2.errors.CheckViolation), run_store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET state='quarantined' WHERE id=%s", (submitted["id"],))


def test_initialize_gives_a_database_from_before_workers_its_table_of_workers(run_store):
    with run_store.transaction() as cursor:
        cursor.execute("DROP TABLE rs_workers")

    run_store.initialize()
    run_store.register_worker("host-1/default", host="host-1", session="session-1")

    assert run_store.worker_heartbeat("host-1/default", "session-1") is True


def test_initialize_removes_the_profile_hash_from_a_request_stored_with_one(run_store):
    """The branch route parses the parent's stored request as a JobSpec, which has no such field."""
    submitted = run_store.submit(job("rollout"), "key-1")
    with run_store.transaction() as cursor:
        cursor.execute(
            "UPDATE rs_jobs SET request=request || %s::jsonb WHERE id=%s",
            ('{"profile_hash": "%s"}' % ("0" * 64), submitted["id"]),
        )

    run_store.initialize()

    assert run_store.get(submitted["id"])["request"] == submitted["request"]


def test_initialize_lets_a_database_from_before_reviews_store_a_review(run_store):
    with run_store.transaction() as cursor:
        cursor.execute("ALTER TABLE rs_jobs DROP CONSTRAINT rs_jobs_kind_check")
        cursor.execute(
            "ALTER TABLE rs_jobs ADD CONSTRAINT rs_jobs_kind_check CHECK (kind IN ('rollout', 'grade'))"
        )

    run_store.initialize()

    assert run_store.submit(stored_job(run_store, "review"), "review-1")["kind"] == "review"


# The binding of an in-place grade


def test_an_in_place_grade_is_bound_to_the_rollout_worker_until_keep_until(run_store):
    rollout_id = finished_rollout(run_store, kept(FUTURE))

    grade = run_store.submit(in_place_grade(rollout_id), "grade-1")

    assert binding(grade) == ("rollout-worker", datetime(2999, 1, 1, tzinfo=UTC))


def test_a_grade_from_a_snapshot_is_bound_to_no_worker(run_store):
    assert binding(run_store.submit(job("grade"), "grade-1")) == (None, None)


@pytest.mark.parametrize("step", UNFINISHED.values(), ids=UNFINISHED.keys())
def test_an_in_place_grade_of_an_unfinished_rollout_is_a_conflict(run_store, step):
    rollout = run_store.submit(job("rollout", keep_sandbox=True), "kept-rollout")
    step(run_store)

    with pytest.raises(Conflict, match="The in_place_job rollout has not finished"):
        run_store.submit(in_place_grade(rollout["id"]), "grade-1")


@pytest.mark.parametrize(
    "result,error",
    [
        ({"status": "completed"}, "The in_place_job rollout kept no sandbox"),
        ({"status": "completed", "kept_sandbox": "sandbox-1"}, "The in_place_job rollout kept no sandbox"),
        (kept("2999-01-01T00:00:00"), "The kept sandbox of the in_place_job rollout has no UTC keep_until"),
        (kept("tomorrow"), "The kept sandbox of the in_place_job rollout has no UTC keep_until"),
        (kept(32503680000), "The kept sandbox of the in_place_job rollout has no UTC keep_until"),
        (
            {"status": "completed", "kept_sandbox": {"sandbox_id": "sandbox-1"}},
            "The kept sandbox of the in_place_job rollout has no UTC keep_until",
        ),
    ],
    ids=["no-kept-sandbox", "kept-sandbox-not-an-object", "naive", "not-a-time", "a-number", "no-keep-until"],
)
def test_an_in_place_grade_needs_a_rollout_that_kept_its_sandbox_until_a_utc_deadline(
    run_store, result, error
):
    rollout_id = finished_rollout(run_store, result)

    with pytest.raises(ValueError, match=error):
        run_store.submit(in_place_grade(rollout_id), "grade-1")


def test_an_in_place_grade_of_a_cancelled_rollout_names_no_kept_sandbox(run_store):
    rollout = run_store.submit(job("rollout", keep_sandbox=True), "kept-rollout")
    run_store.cancel_queued(rollout["id"])

    with pytest.raises(ValueError, match="The in_place_job rollout kept no sandbox"):
        run_store.submit(in_place_grade(rollout["id"]), "grade-1")


def test_an_in_place_grade_of_an_unknown_job_raises_key_error(run_store):
    with pytest.raises(KeyError, match="no-such-job"):
        run_store.submit(in_place_grade("no-such-job"), "grade-1")


# Affinity keys


def affinity(store: Store, job_id: str) -> str:
    return str(store.get(job_id)["affinity"])


def parent_point(store: Store, directory: Path, request: JobSpec) -> str:
    """The id of a recovery point that the job of ``request`` published after one closed turn."""
    registered(store, "worker-1")
    store.submit(request, "parent")
    claimed = store.claim("worker-1")
    events = [{"type": "tool.started", "call_id": "call-1", "name": "bash", "args": {"command": "ls"}}]
    point = NativePoint(1, 1, "snapshot-1", native_history(directory))
    Index(store).project(claimed["id"], claimed["lease_token"], {}, events, [point])
    return str(Index(store).points(claimed["active_attempt"])[0]["id"])


def branch(point_id: str, **fields: Any) -> JobSpec:
    return JobSpec(
        "rollout", rollout_spec(), parent_point=point_id, continuation={"guidance": "none"}, **fields
    )


def review_of(*parent_ids: str, **fields: Any) -> JobSpec:
    parents = [review_parent(name="root-%d" % n, job_id=job_id) for n, job_id in enumerate(parent_ids)]
    return JobSpec("review", review_spec(parents=parents), **fields)


def test_a_job_that_names_no_other_job_takes_its_own_id_as_its_affinity_key(run_store):
    submitted = run_store.submit(job("rollout"), "key-1")

    assert submitted["affinity"] == submitted["id"]


def test_a_job_whose_affinity_is_the_key_of_the_job_that_it_names_keeps_it(run_store, tmp_path):
    point_id = parent_point(run_store, tmp_path, JobSpec("rollout", rollout_spec(), affinity="group-1"))

    submitted = run_store.submit(branch(point_id, affinity="group-1"), "branch")

    assert submitted["affinity"] == "group-1"


def test_a_job_whose_affinity_differs_from_the_key_of_the_job_that_it_names_is_refused(run_store, tmp_path):
    """The other key could bind to another host, where the data of the named job does not exist."""
    point_id = parent_point(run_store, tmp_path, JobSpec("rollout", rollout_spec(), affinity="group-1"))

    with pytest.raises(
        ValueError, match="^The affinity of a request must be the key of the jobs that it names"
    ):
        run_store.submit(branch(point_id, affinity="group-1-r1"), "branch")


def test_a_branch_takes_the_affinity_key_of_the_job_of_its_parent_point(run_store, tmp_path):
    """The branch starts from a snapshot of its parent, which only the parent's host holds."""
    point_id = parent_point(run_store, tmp_path, JobSpec("rollout", rollout_spec(), affinity="group-1"))

    assert run_store.submit(branch(point_id), "branch")["affinity"] == "group-1"


def test_an_in_place_grade_takes_the_affinity_key_of_its_rollout(run_store):
    rollout_id = finished_rollout(run_store, kept(FUTURE))

    grade = run_store.submit(in_place_grade(rollout_id), "grade-1")

    assert grade["affinity"] == affinity(run_store, rollout_id)


def test_a_review_takes_the_affinity_key_that_its_parents_share(run_store):
    first = run_store.submit(JobSpec("rollout", rollout_spec(), affinity="group-1"), "root-1")["id"]
    second = run_store.submit(JobSpec("rollout", rollout_spec(), affinity="group-1"), "root-2")["id"]

    assert run_store.submit(review_of(first, second), "review")["affinity"] == "group-1"


@pytest.mark.parametrize(
    "fields, refusal",
    [
        ({}, "^The jobs that the request names have different affinity keys"),
        ({"affinity": "group-1"}, "^The affinity of a request must be the key of the jobs that it names"),
    ],
    ids=["no key of its own", "a key of its own"],
)
def test_a_review_of_parents_with_two_affinity_keys_is_refused(run_store, fields, refusal):
    """Each key binds to one host, so no host need hold the attempts of both parents."""
    first = run_store.submit(job("rollout"), "root-1")["id"]
    second = run_store.submit(job("rollout"), "root-2")["id"]

    with pytest.raises(ValueError, match=refusal):
        run_store.submit(review_of(first, second, **fields), "review")


@pytest.mark.parametrize(
    "spec", [harbor_grade_spec(), swerebench_grade_spec()], ids=["harbor", "swe-rebench-v2"]
)
def test_a_grade_from_a_snapshot_without_an_affinity_key_is_refused(run_store, spec):
    """It names no job, so no key ties it to the host of the rollout whose snapshot it grades."""
    with pytest.raises(
        ValueError, match="^A grade from a snapshot_id needs the affinity key of its rollout$"
    ):
        run_store.submit(JobSpec("grade", spec), "grade")

    assert run_store.list_jobs() == []


@pytest.mark.parametrize(
    "request_",
    [
        pytest.param(review_of("no-such-job"), id="review-parent"),
        pytest.param(branch("no-such-point"), id="parent-point"),
    ],
)
def test_a_job_that_names_an_unknown_job_or_point_raises_key_error(run_store, request_):
    with pytest.raises(KeyError, match="no-such-"):
        run_store.submit(request_, "job")

    assert run_store.list_jobs() == []


def test_initialize_gives_a_job_from_before_affinity_keys_its_own_id_as_its_key(run_store):
    """The job names no key, so it takes the key that a new job of its request takes."""
    submitted = run_store.submit(job("rollout"), "key-1")
    with run_store.transaction() as cursor:
        cursor.execute("ALTER TABLE rs_jobs DROP COLUMN affinity")
        cursor.execute("DROP TABLE rs_affinity")

    run_store.initialize()

    assert affinity(run_store, submitted["id"]) == submitted["id"]
