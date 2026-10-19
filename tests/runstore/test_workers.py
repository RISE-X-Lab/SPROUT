"""The workers of the Run Store: registration, heartbeats, stops, and the jobs of dead workers and lost hosts.

A worker is live while it has not stopped and its last heartbeat is less than 600 s old (ADR-0006).
These tests use the PostgreSQL test database (``run_store``), and skip without it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Conflict, Store
from tests.runstore.builders import TASK_DIR, job
from tests.support.runstore import registered, stopped

WORKER = "host-1/default"
#: Another worker of host-1.
PEER = "host-1/peer"


def silent_for(store: Store, worker_id: str, seconds: float) -> None:
    """Make the last heartbeat of ``worker_id`` ``seconds`` old, as if the worker had stopped reporting."""
    with store.transaction() as cursor:
        cursor.execute(
            "UPDATE rs_workers SET seen_at=clock_timestamp() - %s * interval '1 second' WHERE worker_id=%s",
            (seconds, worker_id),
        )


def lease_ends_in(store: Store, job_id: str, seconds: float) -> None:
    """Make the lease of ``job_id`` end ``seconds`` from now, in the past for a negative value."""
    with store.transaction() as cursor:
        cursor.execute(
            "UPDATE rs_jobs SET lease_until=clock_timestamp() + %s * interval '1 second' WHERE id=%s",
            (seconds, job_id),
        )


def seconds_from_heartbeat_to_lease_end(store: Store, job_id: str) -> int:
    with store.transaction() as cursor:
        cursor.execute(
            """SELECT j.lease_until - w.seen_at AS gap FROM rs_jobs j
            JOIN rs_workers w ON w.worker_id=j.worker_id WHERE j.id=%s""",
            (job_id,),
        )
        return round(cursor.fetchone()["gap"].total_seconds())


def running_job(store: Store, worker_id: str, key: str) -> dict[str, Any]:
    """A rollout that the live worker ``worker_id`` claimed, with a lease that ends in 10 s."""
    store.submit(job("rollout"), key)
    claimed = store.claim(worker_id)
    assert claimed is not None
    lease_ends_in(store, claimed["id"], 10)
    return claimed


# Registration


def test_a_registered_session_has_its_heartbeats_acknowledged_and_another_session_does_not(run_store):
    run_store.register_worker(WORKER, host="host-1", session="session-1")

    acknowledged = (
        run_store.worker_heartbeat(WORKER, "session-1"),
        run_store.worker_heartbeat(WORKER, "session-2"),
    )

    assert acknowledged == (True, False)


def test_a_worker_id_that_a_session_reported_on_in_the_last_600_s_is_refused_to_another_session(run_store):
    run_store.register_worker(WORKER, host="host-1", session="old")
    silent_for(run_store, WORKER, 590)

    with pytest.raises(Conflict, match="^Worker host-1/default is live in another session$"):
        run_store.register_worker(WORKER, host="host-1", session="new")

    assert run_store.worker_heartbeat(WORKER, "old") is True


def test_a_worker_id_whose_session_is_silent_for_600_s_goes_to_the_next_session(run_store):
    run_store.register_worker(WORKER, host="host-1", session="old")
    silent_for(run_store, WORKER, 610)

    run_store.register_worker(WORKER, host="host-1", session="new")

    holders = (run_store.worker_heartbeat(WORKER, "old"), run_store.worker_heartbeat(WORKER, "new"))
    assert holders == (False, True)


def test_a_session_on_the_host_of_a_live_row_takes_it_at_once_from_the_session_that_it_replaces(run_store):
    """A worker that crashed and starts again names its old session, whose process it found gone."""
    run_store.register_worker(WORKER, host="host-1", session="crashed")

    run_store.register_worker(WORKER, host="host-1", session="new", replaces="crashed")

    holders = (run_store.worker_heartbeat(WORKER, "crashed"), run_store.worker_heartbeat(WORKER, "new"))
    assert holders == (False, True)


@pytest.mark.parametrize(
    "host, replaces",
    [("host-2", "old"), ("host-1", "another")],
    ids=["from another host", "naming a session that does not hold the row"],
)
def test_a_live_row_is_refused_to_a_replacing_session_of_another_host_or_of_another_session(
    run_store, host, replaces
):
    """Only the host of the old session can see that its process is gone."""
    run_store.register_worker(WORKER, host="host-1", session="old")

    with pytest.raises(Conflict, match="^Worker host-1/default is live in another session$"):
        run_store.register_worker(WORKER, host=host, session="new", replaces=replaces)

    assert run_store.worker_heartbeat(WORKER, "old") is True


def test_a_session_that_takes_over_a_worker_id_renews_the_leases_of_the_jobs_left_running_under_it(
    run_store,
):
    """The new session settles those jobs, so their lease tokens must still write."""
    run_store.register_worker(WORKER, host="host-1", session="old")
    left = running_job(run_store, WORKER, "left")
    lease_ends_in(run_store, left["id"], -1)
    silent_for(run_store, WORKER, 610)

    run_store.register_worker(WORKER, host="host-1", session="new")
    run_store.finish(left["id"], left["lease_token"], {"status": "completed"})

    assert run_store.get(left["id"])["state"] == "succeeded"


def test_the_running_jobs_of_a_worker_are_its_own_unfinished_ones_with_their_lease_tokens(run_store):
    """A new session of the worker settles them with these tokens."""
    registered(run_store, WORKER)
    registered(run_store, "host-2/default", host="host-2")
    first, second = running_job(run_store, WORKER, "first"), running_job(run_store, WORKER, "second")
    running_job(run_store, "host-2/default", "another worker's")
    finished = running_job(run_store, WORKER, "finished")
    run_store.finish(finished["id"], finished["lease_token"], {"status": "completed"})

    running = run_store.running_jobs(WORKER)

    assert [(each["id"], each["lease_token"]) for each in running] == [
        (first["id"], first["lease_token"]),
        (second["id"], second["lease_token"]),
    ]


def test_the_lost_attempts_of_a_worker_are_its_own_with_the_child_that_each_recorded(run_store):
    """The worker deletes their local data once it lives again. Another worker's are not its to delete."""
    registered(run_store, WORKER)
    registered(run_store, "host-2/default", host="host-2")
    mine = running_job(run_store, WORKER, "mine")
    run_store.progress(mine["id"], mine["lease_token"], execution={"process": {"pid": 7}})
    other = running_job(run_store, "host-2/default", "other")
    lease_ends_in(run_store, mine["id"], -1)
    lease_ends_in(run_store, other["id"], -1)
    stopped(run_store, WORKER)
    stopped(run_store, "host-2/default")
    run_store.settle_dead_workers()

    lost = run_store.lost_attempts(WORKER)

    assert lost == [{"id": mine["active_attempt"], "job_id": mine["id"], "process": {"pid": 7}}]


def test_a_lost_attempt_whose_data_is_deleted_is_no_longer_one_of_the_lost_attempts(run_store):
    """The worker reads the lost attempts once a minute, so they must not grow with each attempt ever lost."""
    registered(run_store, WORKER)
    deleted, kept = running_job(run_store, WORKER, "deleted"), running_job(run_store, WORKER, "kept")
    lease_ends_in(run_store, deleted["id"], -1)
    lease_ends_in(run_store, kept["id"], -1)
    stopped(run_store, WORKER)
    run_store.settle_dead_workers()

    run_store.note_lost_data_deleted(deleted["active_attempt"], log="")

    assert [attempt["id"] for attempt in run_store.lost_attempts(WORKER)] == [kept["active_attempt"]]


def test_the_note_of_deleted_data_puts_its_log_at_the_end_of_the_error_of_a_lost_attempt_only(run_store):
    registered(run_store, WORKER)
    lost = running_job(run_store, WORKER, "lost")
    lease_ends_in(run_store, lost["id"], -1)
    stopped(run_store, WORKER)
    run_store.settle_dead_workers()
    registered(run_store, WORKER)
    running = running_job(run_store, WORKER, "running")

    run_store.note_lost_data_deleted(lost["active_attempt"], log="; the log")
    run_store.note_lost_data_deleted(running["active_attempt"], log="; the log")

    attempts = [run_store.attempts(job_id)[0] for job_id in (lost["id"], running["id"])]
    assert [(attempt["error"], attempt["data_deleted_at"] is not None) for attempt in attempts] == [
        ("Worker host-1/default stopped reporting heartbeats; the log", True),
        (None, False),
    ]


# Heartbeats


def test_a_heartbeat_renews_each_running_job_of_its_worker_to_600_s_after_it_and_no_other(run_store):
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    registered(run_store, "host-2/default", host="host-2")
    mine = [running_job(run_store, WORKER, "mine-1"), running_job(run_store, WORKER, "mine-2")]
    other = running_job(run_store, "host-2/default", "other")
    before = run_store.get(other["id"])["lease_until"]

    run_store.worker_heartbeat(WORKER, "session-1")

    renewed = [seconds_from_heartbeat_to_lease_end(run_store, claimed["id"]) for claimed in mine]
    assert (renewed, run_store.get(other["id"])["lease_until"]) == ([600, 600], before)


def test_a_heartbeat_of_a_session_that_does_not_hold_the_row_renews_no_lease(run_store):
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    claimed = running_job(run_store, WORKER, "mine")
    before = run_store.get(claimed["id"])["lease_until"]

    acknowledged = run_store.worker_heartbeat(WORKER, "session-2")

    assert (acknowledged, run_store.get(claimed["id"])["lease_until"]) == (False, before)


def test_a_heartbeat_of_a_worker_without_a_row_is_not_acknowledged(run_store):
    assert run_store.worker_heartbeat(WORKER, "session-1") is False


# Stops


def test_a_stopped_row_goes_to_the_next_session_of_its_worker_at_once(run_store):
    """A worker that stops and starts again must not wait 600 s for its own row."""
    run_store.register_worker(WORKER, host="host-1", session="old")

    run_store.worker_stopped(WORKER, "old")
    run_store.register_worker(WORKER, host="host-1", session="new")

    assert run_store.worker_heartbeat(WORKER, "new") is True


def test_a_heartbeat_that_arrives_after_its_worker_stopped_is_not_acknowledged(run_store):
    """The manager stops waiting for a heartbeat after 15 s, so one can arrive after the stop."""
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    run_store.worker_stopped(WORKER, "session-1")

    assert run_store.worker_heartbeat(WORKER, "session-1") is False


def test_a_session_cannot_stop_the_row_of_another_session(run_store):
    run_store.register_worker(WORKER, host="host-1", session="holder")

    run_store.worker_stopped(WORKER, "other")

    assert run_store.worker_heartbeat(WORKER, "holder") is True


# The jobs of dead workers

#: A rerun result of the sweep for a job of WORKER.
DEAD = {
    "status": "error",
    "failure_kind": "infrastructure",
    "error": "Worker host-1/default stopped reporting heartbeats",
}


def sandbox_kept_by(store: Store, worker_id: str, *, keep_until: str = "2999-01-01T00:00:00+00:00") -> str:
    """The id of a finished rollout of ``worker_id`` that keeps its sandbox until ``keep_until``, 2999 by
    default."""
    store.submit(job("rollout", keep_sandbox=True), "kept-rollout")
    claimed = store.claim(worker_id)
    kept = {"sandbox_id": "sandbox-1", "keep_until": keep_until}
    store.finish(claimed["id"], claimed["lease_token"], {"status": "completed", "kept_sandbox": kept})
    return str(claimed["id"])


def in_place_grade(rollout_id: str, *, max_infra_retries: int = 2) -> JobSpec:
    spec = {"benchmark": "harbor", "task_dir": TASK_DIR, "in_place_job": rollout_id}
    return JobSpec("grade", spec, max_infra_retries=max_infra_retries)


def registered_with_a_running_job(store: Store, request: JobSpec | None = None) -> dict[str, Any]:
    """The job of ``request``, a rollout by default, that the session ``session-1`` of WORKER runs.

    PEER, another worker of host-1, stays live, so that host-1 is not lost when WORKER dies.
    """
    registered(store, PEER)
    store.register_worker(WORKER, host="host-1", session="session-1")
    store.submit(request or job("rollout"), "running")
    claimed = store.claim(WORKER)
    assert claimed is not None
    return claimed


def dead_with_an_expired_lease(store: Store, job_id: str) -> None:
    silent_for(store, WORKER, 610)
    lease_ends_in(store, job_id, -1)


#: The step that leaves WORKER in each state: live, silent for 610 s, or stopped just now.
WORKER_STATES: dict[str, Callable[[Store], object]] = {
    "live": lambda store: None,
    "silent": lambda store: silent_for(store, WORKER, 610),
    "stopped": lambda store: store.worker_stopped(WORKER, "session-1"),
}
#: The step that leaves a job's lease in each state.
LEASES: dict[str, Callable[[Store, str], object]] = {
    "live": lambda store, job_id: None,
    "expired": lambda store, job_id: lease_ends_in(store, job_id, -1),
}


@pytest.mark.parametrize(
    "worker, lease, state",
    [
        ("live", "expired", "running"),
        ("live", "live", "running"),
        ("silent", "expired", "queued"),
        ("silent", "live", "running"),
        ("stopped", "expired", "queued"),
        ("stopped", "live", "running"),
    ],
)
def test_the_sweep_takes_a_running_job_only_when_its_worker_is_dead_and_its_lease_expired(
    run_store, worker, lease, state
):
    """A job claimed after the last heartbeat of its worker keeps its lease of 600 s from the claim."""
    claimed = registered_with_a_running_job(run_store)
    WORKER_STATES[worker](run_store)
    LEASES[lease](run_store, claimed["id"])

    run_store.settle_dead_workers()

    assert run_store.get(claimed["id"])["state"] == state


def test_a_job_of_a_dead_worker_runs_again_and_its_attempt_counts_as_lost(run_store):
    claimed = registered_with_a_running_job(run_store)
    run_store.append_events(claimed["id"], claimed["lease_token"], [{"seq": 1, "type": "run.started"}])
    dead_with_an_expired_lease(run_store, claimed["id"])

    settled = run_store.settle_dead_workers()

    queued, [attempt] = run_store.get(claimed["id"]), run_store.attempts(claimed["id"])
    assert (settled, queued["state"], queued["phase"], queued["attempt_count"]) == (
        [claimed["id"]],
        "queued",
        "retry_wait",
        1,
    )
    assert (attempt["state"], attempt["result"], run_store.events(attempt["id"])) == ("lost", DEAD, [])


def test_a_job_of_a_dead_worker_without_an_attempt_left_fails_with_the_error_of_each_attempt(run_store):
    claimed = registered_with_a_running_job(run_store, job("rollout", max_infra_retries=0))
    dead_with_an_expired_lease(run_store, claimed["id"])

    run_store.settle_dead_workers()

    failed = run_store.get(claimed["id"])
    assert (failed["state"], failed["result"], run_store.attempts(claimed["id"])[0]["state"]) == (
        "failed",
        {**DEAD, "attempt_errors": [DEAD["error"]]},
        "lost",
    )


def test_a_job_of_a_dead_worker_that_someone_asked_to_stop_fails_instead_of_running_again(run_store):
    claimed = registered_with_a_running_job(run_store)
    run_store.request_stop(claimed["id"])
    dead_with_an_expired_lease(run_store, claimed["id"])

    run_store.settle_dead_workers()

    assert run_store.get(claimed["id"])["state"] == "failed"


def test_a_running_grade_in_a_kept_sandbox_of_a_dead_worker_runs_again_while_its_host_lives(run_store):
    """It stays bound to the worker id, so the next session of the worker claims it again (ADR-0006)."""
    registered(run_store, PEER)
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    grade = run_store.submit(in_place_grade(sandbox_kept_by(run_store, WORKER)), "grade")
    claimed = run_store.claim(WORKER)
    dead_with_an_expired_lease(run_store, claimed["id"])

    run_store.settle_dead_workers()

    queued, [attempt] = run_store.get(grade["id"]), run_store.attempts(grade["id"])
    assert (queued["state"], queued["phase"], queued["bound_worker"], attempt["state"]) == (
        "queued",
        "retry_wait",
        WORKER,
        "lost",
    )


def test_a_grade_in_a_kept_sandbox_of_a_dead_worker_without_an_attempt_left_scores_0(run_store):
    """The grade has no other sandbox to run in, so its lost sandbox scores 0 (ADR-0004), on a live host."""
    registered(run_store, PEER)
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    rollout_id = sandbox_kept_by(run_store, WORKER)
    grade = run_store.submit(in_place_grade(rollout_id, max_infra_retries=0), "grade")
    claimed = run_store.claim(WORKER)
    dead_with_an_expired_lease(run_store, claimed["id"])

    run_store.settle_dead_workers()

    settled, [attempt] = run_store.get(grade["id"]), run_store.attempts(grade["id"])
    assert (settled["state"], settled["result"], attempt["state"]) == (
        "succeeded",
        {"reward": 0.0, "reason": "sandbox_lost"},
        "lost",
    )


#: The step that ends the session ``session-1`` of WORKER: it dies, or it stops.
SESSION_ENDS: dict[str, Callable[[Store], object]] = {
    "dies": lambda store: silent_for(store, WORKER, 610),
    "stops": lambda store: store.worker_stopped(WORKER, "session-1"),
}


@pytest.mark.parametrize("end", SESSION_ENDS.values(), ids=SESSION_ENDS.keys())
def test_a_queued_grade_bound_to_a_worker_whose_session_ended_goes_to_its_next_session(run_store, end):
    """A worker that starts again on its host still keeps the sandbox, so it can grade in it."""
    registered(run_store, PEER)
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    grade = run_store.submit(in_place_grade(sandbox_kept_by(run_store, WORKER)), "grade")
    end(run_store)
    run_store.settle_dead_workers()

    run_store.register_worker(WORKER, host="host-1", session="session-2")
    claimed = run_store.claim(WORKER)

    assert claimed["id"] == grade["id"]


def test_a_queued_grade_bound_to_a_live_worker_stays_queued(run_store):
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    grade = run_store.submit(in_place_grade(sandbox_kept_by(run_store, WORKER)), "grade")

    run_store.settle_dead_workers()

    assert run_store.get(grade["id"])["state"] == "queued"


def test_the_sweep_skips_a_job_that_another_transaction_holds_and_takes_it_later(run_store):
    """Without the skip, the sweep waits until the statement timeout fails it."""
    claimed = registered_with_a_running_job(run_store)
    dead_with_an_expired_lease(run_store, claimed["id"])

    with run_store.transaction() as cursor:
        cursor.execute("SELECT id FROM rs_jobs WHERE id=%s FOR UPDATE", (claimed["id"],))
        while_held = run_store.settle_dead_workers()

    assert (while_held, run_store.settle_dead_workers()) == ([], [claimed["id"]])


# Lost hosts

#: The result of a job whose affinity key is bound to host-1 once host-1 is lost.
DATA_LOST = {
    "status": "error",
    "failure_kind": "infrastructure",
    "reason": "data_lost",
    "host": "host-1",
    "error": "Host host-1 is lost: none of its workers reported a heartbeat in 600 s",
}


def of_group(name: str) -> JobSpec:
    """A rollout of the affinity key ``group-1``, whose prompt ``name`` makes it a request of its own."""
    return JobSpec("rollout", {"prompt": name}, affinity="group-1")


def stopped_610_s_ago(store: Store, worker_id: str) -> None:
    stopped(store, worker_id)
    silent_for(store, worker_id, 610)


def stopped_after_610_s_of_silence(store: Store, worker_id: str) -> None:
    silent_for(store, worker_id, 610)
    stopped(store, worker_id)


def group_bound_to_host_1(store: Store) -> str:
    """The id of the finished root of ``group-1``, whose claim bound the key to host-1.

    The worker that ran it stopped 610 s ago."""
    registered(store, "host-1/binder")
    root = store.submit(of_group("root"), "root")["id"]
    claimed = store.claim("host-1/binder")
    store.finish(root, claimed["lease_token"], {"status": "completed"})
    stopped_610_s_ago(store, "host-1/binder")
    return str(root)


#: The steps that leave a registered worker row in each state.
ROW_STATES: dict[str, Callable[[Store, str], object]] = {
    "live": lambda store, worker_id: None,
    "dead": lambda store, worker_id: silent_for(store, worker_id, 610),
    "stopped 610 s ago": stopped_610_s_ago,
    "stopped after 610 s of silence": stopped_after_610_s_of_silence,
}


def rows_of_host_1(store: Store, rows: dict[str, str]) -> None:
    """Register each worker of ``rows`` on host-1, and leave its row in its state of ``ROW_STATES``."""
    for worker_id, state in rows.items():
        registered(store, worker_id)
        ROW_STATES[state](store, worker_id)


@pytest.mark.parametrize(
    "rows, state",
    [
        pytest.param({WORKER: "live"}, "queued", id="live"),
        pytest.param({WORKER: "dead"}, "failed", id="dead"),
        pytest.param({WORKER: "dead", PEER: "live"}, "queued", id="dead-and-live"),
        pytest.param({WORKER: "stopped 610 s ago"}, "failed", id="stopped-610-s-ago"),
        pytest.param({WORKER: "stopped after 610 s of silence"}, "queued", id="stopped-after-a-silence"),
    ],
)
def test_a_queued_job_of_a_bound_key_fails_once_no_row_of_its_host_has_a_heartbeat_in_600_s(
    run_store, rows, state
):
    """The stop of a row counts as its last heartbeat, so a worker can start again within 600 s."""
    group_bound_to_host_1(run_store)
    rows_of_host_1(run_store, rows)
    branch = run_store.submit(of_group("branch"), "branch")["id"]

    run_store.settle_dead_workers()

    assert run_store.get(branch)["state"] == state


def test_a_lost_host_fails_no_job_of_a_key_that_another_host_holds(run_store):
    registered(run_store, "host-2/default", host="host-2")
    run_store.submit(JobSpec("rollout", {"prompt": "root"}, affinity="group-2"), "root")
    run_store.claim("host-2/default")
    branch = run_store.submit(JobSpec("rollout", {"prompt": "branch"}, affinity="group-2"), "branch")["id"]
    rows_of_host_1(run_store, {WORKER: "dead"})

    run_store.settle_dead_workers()

    assert run_store.get(branch)["state"] == "queued"


def test_a_queued_job_of_a_lost_host_fails_as_data_lost_and_its_group_keeps_its_rows(run_store):
    """The group runs again under a new key, and the records of its first run stay."""
    root = group_bound_to_host_1(run_store)
    rows_of_host_1(run_store, {WORKER: "dead"})
    branch = run_store.submit(of_group("branch"), "branch")["id"]

    settled = run_store.settle_dead_workers()

    failed = run_store.get(branch)
    assert (settled, failed["state"], failed["phase"], failed["result"], failed["error"]) == (
        [branch],
        "failed",
        "data_lost",
        DATA_LOST,
        DATA_LOST["error"],
    )
    assert (run_store.get(root)["state"], len(run_store.attempts(root))) == ("succeeded", 1)


def test_a_running_job_of_a_lost_host_fails_as_data_lost_once_its_lease_expired(run_store):
    """No worker of host-1 lives, so no worker can run the job again."""
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    run_store.submit(of_group("root"), "root")
    claimed = run_store.claim(WORKER)
    dead_with_an_expired_lease(run_store, claimed["id"])

    run_store.settle_dead_workers()

    failed, [attempt] = run_store.get(claimed["id"]), run_store.attempts(claimed["id"])
    assert (failed["state"], failed["phase"], failed["result"], attempt["state"], attempt["result"]) == (
        "failed",
        "data_lost",
        DATA_LOST,
        "lost",
        DEAD,
    )


#: The step that leaves the grade in the kept sandbox of WORKER queued, or running with an expired lease.
GRADE_STATES: dict[str, Callable[[Store], object]] = {
    "queued": lambda store: None,
    "running": lambda store: lease_ends_in(store, store.claim(WORKER)["id"], -1),
}


@pytest.mark.parametrize("step", GRADE_STATES.values(), ids=GRADE_STATES.keys())
def test_a_grade_in_a_kept_sandbox_of_a_lost_host_fails_as_data_lost_before_its_keep_until(run_store, step):
    """Its search runs again from its roots, as after the loss of the other jobs of its key (ADR-0007)."""
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    grade = run_store.submit(in_place_grade(sandbox_kept_by(run_store, WORKER)), "grade")
    step(run_store)
    silent_for(run_store, WORKER, 610)

    run_store.settle_dead_workers()

    failed = run_store.get(grade["id"])
    assert (failed["state"], failed["phase"], failed["result"]) == ("failed", "data_lost", DATA_LOST)


def test_a_queued_grade_in_a_kept_sandbox_of_a_lost_host_scores_0_once_its_keep_until_passed(run_store):
    """Its sandbox was gone before its host was lost, so it scores 0, though no claim ran first (ADR-0006)."""
    run_store.register_worker(WORKER, host="host-1", session="session-1")
    rollout_id = sandbox_kept_by(run_store, WORKER, keep_until="2000-01-01T00:00:00+00:00")
    grade = run_store.submit(in_place_grade(rollout_id), "grade")
    silent_for(run_store, WORKER, 610)

    run_store.settle_dead_workers()

    settled = run_store.get(grade["id"])
    assert (settled["state"], settled["phase"], settled["result"]) == (
        "succeeded",
        "succeeded",
        {"reward": 0.0, "reason": "sandbox_lost"},
    )


def test_the_key_of_a_lost_host_stays_bound_so_another_host_claims_none_of_its_jobs(run_store):
    """Its group runs again only under a new key, from its roots."""
    group_bound_to_host_1(run_store)
    rows_of_host_1(run_store, {WORKER: "dead"})
    run_store.settle_dead_workers()
    registered(run_store, "host-2/default", host="host-2")
    late = run_store.submit(of_group("late branch"), "late")["id"]

    claim = run_store.claim("host-2/default")
    run_store.settle_dead_workers()

    assert (claim, run_store.get(late)["phase"]) == (None, "data_lost")
