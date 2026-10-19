"""``Worker.run_claimed``: how the worker starts the child of an attempt, watches it and ends the job.

The child is the stand-in of ``tests.worker.attempt_child``, which the worker starts as the ``python``
of the job's profile. The Run Store is the isolated test database. docs/RUNSTORE.md ("Failure
handling") describes the frozen payload and the supervision.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec, digest
from sprout.runstore.store import Fenced, Store
from sprout.worker.attempts import Worker, process_identity, stop_process
from sprout.worker.files import read_json
from tests.support.paths import REPO
from tests.support.runstore import registered
from tests.support.waiting import wait_until
from tests.worker.attempt_child import LAST_WORDS, RECEIVED, STARTED, child_python
from tests.worker.claimed import attempt_directory, claimed, publish_points, rollout, worker_config

GRADE = JobSpec(
    "grade",
    {"benchmark": "harbor", "task_dir": "/path/to/task", "snapshot_id": "snap-1"},
    affinity="group-1",
)


def review(store: Store) -> JobSpec:
    """A review of a rollout in ``store``. A stop request cancelled the rollout, so that no claim takes it."""
    parent = store.submit(rollout(), "review-parent")["id"]
    store.request_stop(parent)
    grade = {"resolved": False}
    return JobSpec(
        "review",
        {
            "problem": "fix the failing test",
            "model": "reviewer",
            "model_endpoint": "http://127.0.0.1:1",
            "parents": [{"name": "root", "job_id": parent, "attempt_id": "attempt", "grade": grade}],
        },
    )


#: A request of each job kind, in the Run Store that it is given.
REQUESTS: dict[str, Callable[[Store], JobSpec]] = {
    "rollout": lambda store: rollout(),
    "grade": lambda store: GRADE,
    "review": review,
}


@pytest.fixture(autouse=True)
def no_child_outlives_its_test(tmp_path: Path) -> Iterator[None]:
    """Stop each stand-in child that a test leaves running, so that a failed test leaks no process."""
    yield
    for started in tmp_path.glob("artifacts/*/*/%s" % STARTED):
        stop_process(read_json(started)["identity"])


def stand_in_worker(store: Store, tmp_path: Path, behavior: str = "complete", **options: Any) -> Worker:
    """A worker whose ``default`` profile starts the stand-in child with ``behavior``."""
    config = worker_config(tmp_path / "artifacts", python=child_python(tmp_path, behavior))
    return Worker(store, config, **options)


def claim_for(worker: Worker, request: JobSpec) -> dict[str, Any]:
    """The job of ``request``, queued and then claimed by ``worker``, which registers first."""
    registered(worker.store, worker.worker_id, host=worker.host)
    worker.store.submit(request, "job")
    job = worker.store.claim(worker.worker_id)
    assert job is not None
    return job


def attempt_in_thread(worker: Worker, job: dict[str, Any], stop: Event | None = None) -> Thread:
    """A started thread that runs the attempt of the claimed ``job``."""
    thread = Thread(target=worker.run_claimed, args=(job,), kwargs={"stop": stop}, daemon=True)
    thread.start()
    return thread


def test_a_claimed_job_finishes_with_the_outcome_of_its_child(run_store, tmp_path):
    worker = stand_in_worker(run_store, tmp_path)
    job = claim_for(worker, rollout())

    finished_id = worker.run_claimed(job)

    finished = run_store.get(job["id"])
    assert (finished_id, finished["state"], finished["result"]) == (
        job["id"],
        "succeeded",
        {"status": "completed", "failure_kind": None},
    )


def test_the_child_gets_its_attempt_directory_the_payload_digest_the_worker_pid_and_the_frozen_payload(
    run_store, tmp_path
):
    """The command line of ``python -m sprout.worker.jobs.rollout``, which checks the payload against it.

    The child stops once the process of the pid is no longer its parent, also before it first looks."""
    worker = stand_in_worker(run_store, tmp_path)
    job = claim_for(worker, rollout())

    worker.run_claimed(job)

    directory = attempt_directory(tmp_path / "artifacts", job)
    frozen = run_store.payload(job["id"], job["active_attempt"])
    assert read_json(directory / STARTED)["argv"] == [
        "-m",
        "sprout.worker.jobs.rollout",
        str(directory),
        digest(frozen),
        str(os.getpid()),
    ]
    assert read_json(directory / RECEIVED) == frozen


def test_the_child_inherits_only_the_host_variables_and_the_ones_its_profile_references(
    run_store, tmp_path, monkeypatch
):
    """The child must run its sandboxes on the podman and store that the worker later inspects."""
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "/path/to/podman")
    monkeypatch.setenv("SPROUT_PODMAN_STORE", "/path/to/store")
    monkeypatch.setenv("FIXTURE_UPSTREAM_KEY", "upstream-key")
    monkeypatch.setenv("SPROUT_RUNSTORE_DSN", "dbname=production")
    monkeypatch.setenv("SPROUT_RUNSTORE_TOKEN", "runstore-token")
    monkeypatch.setenv("FIXTURE_UNRELATED", "unrelated")
    config = worker_config(
        tmp_path / "artifacts",
        python=child_python(tmp_path),
        worker_env={"UPSTREAM_KEY": {"$env": "FIXTURE_UPSTREAM_KEY"}},
    )
    worker = Worker(run_store, config)
    job = claim_for(worker, rollout())

    worker.run_claimed(job)

    env = read_json(attempt_directory(tmp_path / "artifacts", job) / STARTED)["env"]
    names = (
        "SPROUT_PODMAN_BIN",
        "SPROUT_PODMAN_STORE",
        "FIXTURE_UPSTREAM_KEY",
        "PYTHONPATH",
        "SPROUT_RUNSTORE_DSN",
        "SPROUT_RUNSTORE_TOKEN",
        "FIXTURE_UNRELATED",
    )
    assert {name: env.get(name) for name in names} == {
        "SPROUT_PODMAN_BIN": "/path/to/podman",
        "SPROUT_PODMAN_STORE": "/path/to/store",
        "FIXTURE_UPSTREAM_KEY": "upstream-key",
        "PYTHONPATH": "%s/src:%s" % (REPO, REPO),
        "SPROUT_RUNSTORE_DSN": None,
        "SPROUT_RUNSTORE_TOKEN": None,
        "FIXTURE_UNRELATED": None,
    }


def test_the_attempt_records_the_identity_of_its_child(run_store, tmp_path):
    """A worker that takes the job over after a crash stops the child that this identity names."""
    worker = stand_in_worker(run_store, tmp_path)
    job = claim_for(worker, rollout())

    worker.run_claimed(job)

    started = read_json(attempt_directory(tmp_path / "artifacts", job) / STARTED)
    assert run_store.attempts(job["id"])[0]["execution"]["process"] == started["identity"]


class RegistrationFails:
    """The Run Store, but the progress write that records the child's identity fails as for a lost lease.

    It keeps the identity of the child, and whether the child got its payload before the failure.
    """

    def __init__(self, store: Store) -> None:
        self._store = store
        self.child: dict[str, Any] = {}
        self.received = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def progress(self, job_id: str, token: str, **options: Any) -> bool:
        execution = options.get("execution") or {}
        if "process" not in execution:
            return self._store.progress(job_id, token, **options)
        self.child = execution["process"]
        directory = Path(execution["directory"])
        wait_until(lambda: (directory / STARTED).exists(), what="the start of the child")
        # A payload sent before this call reaches the waiting child within this second.
        deadline = time.monotonic() + 1
        while not (directory / RECEIVED).exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.received = (directory / RECEIVED).exists()
        raise Fenced("Attempt no longer owns a live lease")


def test_a_child_whose_identity_was_not_recorded_never_gets_its_payload_and_is_stopped(run_store, tmp_path):
    """Otherwise a worker crash leaves a child that runs the attempt and that nobody can stop.

    The attempt is lost, so the job runs again."""
    store = RegistrationFails(run_store)
    worker = stand_in_worker(store, tmp_path)
    job = claim_for(worker, rollout())

    worker.run_claimed(job)

    [attempt] = run_store.attempts(job["id"])
    assert (run_store.get(job["id"])["state"], attempt["state"], store.received) == ("queued", "lost", False)
    assert process_identity(store.child["pid"]) is None, "the child still runs"


@pytest.mark.parametrize(
    "request_, error",
    [
        (
            JobSpec(
                "rollout", {"prompt": "fix the failing test", "sandbox_image": "image"}, profile="missing"
            ),
            "KeyError: 'missing'",
        ),
        (
            rollout(backend={"backend": "local"}),
            "ValueError: Queued jobs run on an owned sandbox, podman (the default) or microvm, not 'local'",
        ),
        (
            JobSpec("rollout", {"prompt": "fix the failing test"}),
            "ValueError: A queued rollout needs a sandbox_image",
        ),
        (
            rollout(extra={"rollout_contract": {"deadline_at": 1700000000.0}}),
            "ValueError: Rollout group wall-time budget expired before execution",
        ),
    ],
    ids=[
        "unknown profile",
        "backend that the worker does not own",
        "rollout without a sandbox image",
        "group deadline passed in the queue",
    ],
)
def test_a_job_that_cannot_start_fails_without_a_retry_and_starts_no_child(
    run_store, tmp_path, request_, error
):
    worker = stand_in_worker(run_store, tmp_path)
    job = claim_for(worker, request_)

    worker.run_claimed(job)

    finished = run_store.get(job["id"])
    assert (finished["state"], finished["error"], finished["attempt_count"]) == ("failed", error, 1)
    assert list(tmp_path.glob("artifacts/*/*/%s" % STARTED)) == []


def test_a_rollout_whose_group_deadline_comes_first_runs_for_the_seconds_left_and_keeps_its_request(
    run_store, tmp_path
):
    """The time that a job of a group waits in the queue counts against the deadline of the group."""
    worker = stand_in_worker(run_store, tmp_path)
    contract = {"rollout_contract": {"deadline_at": time.time() + 30}}
    job = claim_for(worker, rollout(timeout_s=3600, extra=contract))

    worker.run_claimed(job)

    received = read_json(attempt_directory(tmp_path / "artifacts", job) / RECEIVED)
    assert 20 < received["effective_spec"]["timeout_s"] <= 30
    assert run_store.get(job["id"])["request"]["spec"]["timeout_s"] == 3600


def test_a_child_that_exits_without_an_outcome_is_an_infrastructure_failure_that_runs_again(
    run_store, tmp_path
):
    worker = stand_in_worker(run_store, tmp_path, "exit")
    job = claim_for(worker, GRADE)

    worker.run_claimed(job)

    finished = run_store.get(job["id"])
    assert (finished["state"], finished["error"], finished["result"]["failure_kind"]) == (
        "queued",
        "Attempt process exited without outcome",
        "infrastructure",
    )


def test_a_child_past_its_deadline_is_stopped_and_its_rollout_times_out(run_store, tmp_path):
    worker = stand_in_worker(run_store, tmp_path, "hold")
    job = claim_for(worker, rollout(timeout_s=1))

    worker.run_claimed(job)

    finished = run_store.get(job["id"])
    child = read_json(attempt_directory(tmp_path / "artifacts", job) / STARTED)["identity"]
    assert (finished["state"], finished["result"]["status"], finished["result"]["stop_reason"]) == (
        "failed",
        "timeout",
        "timeout",
    )
    assert process_identity(child["pid"]) is None, "the child still runs"


def test_a_stop_request_stops_the_child_and_ends_the_job_for_good(run_store, tmp_path):
    """A requested stop is nobody's failure: the job is not retried, though it has retries left."""
    worker = stand_in_worker(run_store, tmp_path, "hold", progress_s=0.2)
    job = claim_for(worker, rollout())
    directory = attempt_directory(tmp_path / "artifacts", job)
    attempt = attempt_in_thread(worker, job)
    wait_until(lambda: (directory / RECEIVED).exists(), what="the payload at the child")

    run_store.request_stop(job["id"])
    attempt.join(timeout=15)

    finished = run_store.get(job["id"])
    child = read_json(directory / STARTED)["identity"]
    assert attempt.is_alive() is False, "the attempt still runs after the stop request"
    result = finished["result"]
    assert (finished["state"], finished["attempt_count"], result["failure_kind"], result["stop_reason"]) == (
        "failed",
        1,
        "stopped",
        "stopped",
    )
    assert process_identity(child["pid"]) is None, "the child still runs"


def test_a_worker_shutdown_stops_the_child_of_a_running_attempt(run_store, tmp_path):
    worker = stand_in_worker(run_store, tmp_path, "hold")
    job = claim_for(worker, GRADE)
    directory = attempt_directory(tmp_path / "artifacts", job)
    stop = Event()
    attempt = attempt_in_thread(worker, job, stop)
    wait_until(lambda: (directory / RECEIVED).exists(), what="the payload at the child")
    # The release deletes the attempt directory, so the child is read before the shutdown.
    child = read_json(directory / STARTED)["identity"]

    stop.set()
    attempt.join(timeout=15)

    assert (attempt.is_alive(), process_identity(child["pid"])) == (False, None)


@pytest.mark.parametrize("request_of", REQUESTS.values(), ids=REQUESTS.keys())
def test_a_job_claimed_by_a_worker_that_is_shutting_down_is_released_without_using_an_attempt(
    run_store, tmp_path, request_of
):
    """Its child never started, so the job goes back to the queue as if this worker had not claimed it."""
    worker = stand_in_worker(run_store, tmp_path)
    job = claim_for(worker, request_of(run_store))
    stop = Event()
    stop.set()

    worker.run_claimed(job, stop=stop)

    released = run_store.get(job["id"])
    assert (released["state"], released["phase"], released["attempt_count"]) == ("queued", "released", 0)


def test_a_job_claimed_by_a_worker_that_is_cut_off_is_released_without_using_an_attempt(run_store, tmp_path):
    """Its child never started, so nothing of the attempt is lost."""
    worker = stand_in_worker(run_store, tmp_path)
    job = claim_for(worker, rollout())

    worker.run_claimed(job, cut_off=lambda: True)

    released = run_store.get(job["id"])
    assert (released["state"], released["phase"], released["attempt_count"]) == ("queued", "released", 0)
    assert list(tmp_path.glob("artifacts/*/*/%s" % STARTED)) == []


def test_an_attempt_whose_worker_is_cut_off_while_its_child_runs_is_lost_and_runs_again(run_store, tmp_path):
    """Another worker can run the job once the worker is dead, so the attempt counts as a lost one."""
    worker = stand_in_worker(run_store, tmp_path, "hold")
    job = claim_for(worker, rollout())
    directory = attempt_directory(tmp_path / "artifacts", job)

    worker.run_claimed(job, cut_off=lambda: (directory / RECEIVED).exists())

    rerun = run_store.get(job["id"])
    [attempt] = run_store.attempts(job["id"])
    assert (rerun["state"], rerun["attempt_count"], attempt["state"], attempt["result"]["stop_reason"]) == (
        "queued",
        1,
        "lost",
        "worker_cut_off",
    )


class ShutdownDuringTheAttempt:
    """The Run Store, but the worker's ``stop`` is set at the first progress write after the child's start.

    That write returns once the watchdog has stopped the child, so the attempt ends as a shutdown ends it.
    """

    def __init__(self, store: Store, stop: Event) -> None:
        self._store = store
        self._stop = stop
        self._child: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def progress(self, job_id: str, token: str, **options: Any) -> bool:
        execution = options.get("execution") or {}
        if "process" in execution:
            self._child = execution["process"]
            return self._store.progress(job_id, token, **options)
        self._stop.set()
        wait_until(lambda: process_identity(self._child["pid"]) is None, what="the end of the child")
        return self._store.progress(job_id, token, **options)


def test_a_rollout_that_a_shutdown_stopped_is_released_to_start_again_without_using_an_attempt(
    run_store, tmp_path
):
    """A shutdown is nobody's failure, so the job runs again from its spec, as if nobody had claimed it."""
    stop = Event()
    worker = stand_in_worker(ShutdownDuringTheAttempt(run_store, stop), tmp_path, "hold")
    job = claim_for(worker, rollout())

    worker.run_claimed(job, stop=stop)

    released = run_store.get(job["id"])
    assert (released["state"], released["phase"], released["attempt_count"]) == ("queued", "released", 0)
    assert attempt_directory(tmp_path / "artifacts", job).exists() is False


class ShutdownAfterTheProgressWrite:
    """The Run Store, but the worker's ``stop`` is set by the first progress write after the child's start.

    That write returns at once, before the watchdog of the attempt can see the shutdown.
    """

    def __init__(self, store: Store, stop: Event) -> None:
        self._store = store
        self._stop = stop

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def progress(self, job_id: str, token: str, **options: Any) -> bool:
        if "process" not in (options.get("execution") or {}):
            self._stop.set()
        return self._store.progress(job_id, token, **options)


def test_a_shutdown_that_a_progress_write_sees_first_ends_the_attempt_as_a_shutdown(run_store, tmp_path):
    """The worker's thread and the watchdog's must settle a shutdown the same way, whichever sees it first."""
    stop = Event()
    worker = stand_in_worker(ShutdownAfterTheProgressWrite(run_store, stop), tmp_path, "hold")
    job = claim_for(worker, rollout())

    worker.run_claimed(job, stop=stop)

    [attempt] = run_store.attempts(job["id"])
    assert (attempt["state"], attempt["result"]["stop_reason"]) == ("released", "worker_shutdown")


def test_an_attempt_that_runs_again_leaves_no_directory_no_snapshot_and_no_record_but_its_log(
    run_store, tmp_path, fake_btrfs_store
):
    """The next attempt starts from the spec, so nothing of the lost one may be taken for its work."""
    for snapshot in ("snap-1", "snap-2"):
        (fake_btrfs_store / "snapshots" / snapshot).mkdir(parents=True)
    worker = stand_in_worker(run_store, tmp_path, "crash")
    job = claim_for(
        worker, rollout(backend={"backend": "podman", "podman": {"store": str(fake_btrfs_store)}})
    )

    worker.run_claimed(job)

    [attempt] = run_store.attempts(job["id"])
    log = LAST_WORDS + "\n"
    assert (run_store.get(job["id"])["state"], attempt["state"], attempt["error"]) == (
        "queued",
        "lost",
        "Attempt process exited without outcome\n--- the last %d characters of worker.log ---\n%s"
        % (len(log), log),
    )
    assert (attempt_directory(tmp_path / "artifacts", job).exists(), run_store.events(attempt["id"])) == (
        False,
        [],
    )
    assert list((fake_btrfs_store / "snapshots").iterdir()) == []


def test_a_rollout_whose_attempts_all_failed_on_the_infrastructure_fails_with_the_error_of_each(
    run_store, tmp_path
):
    worker = stand_in_worker(run_store, tmp_path, "exit")
    job = claim_for(worker, JobSpec("rollout", rollout().spec, max_infra_retries=0))

    worker.run_claimed(job)

    finished = run_store.get(job["id"])
    assert (finished["state"], finished["result"]["attempt_errors"]) == (
        "failed",
        ["Attempt process exited without outcome"],
    )


class ProgressSpy:
    """The Run Store, and the phase of each progress write, in order."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self.phases: list[str | None] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def progress(self, job_id: str, token: str, **options: Any) -> bool:
        self.phases.append(options.get("phase"))
        return self._store.progress(job_id, token, **options)


def test_an_attempt_writes_its_progress_at_its_start_and_then_at_most_every_5_s(run_store, tmp_path):
    """The child runs 1.5 s, so after the write of its start, one write of its phase follows."""
    store = ProgressSpy(run_store)
    worker = stand_in_worker(store, tmp_path, "slow")
    job = claim_for(worker, rollout())

    worker.run_claimed(job)

    assert (worker.progress_s, store.phases, run_store.get(job["id"])["state"]) == (
        5.0,
        ["starting", "rollout"],
        "succeeded",
    )


@pytest.mark.parametrize(
    "kind, section, field, value",
    [
        ("rollout", "run_defaults", "model", "profile-model"),
        ("grade", "grade_defaults", "verifier_network", "allow"),
        ("review", "review_defaults", "temperature", 0.5),
    ],
)
def test_each_kind_of_job_takes_the_defaults_of_its_own_profile_section(
    run_store, tmp_path, kind, section, field, value
):
    config = worker_config(tmp_path / "artifacts", python=child_python(tmp_path), **{section: {field: value}})
    worker = Worker(run_store, config)
    job = claim_for(worker, REQUESTS[kind](run_store))

    worker.run_claimed(job)

    received = read_json(attempt_directory(tmp_path / "artifacts", job) / RECEIVED)
    assert received["effective_spec"][field] == value


def test_a_review_is_told_the_artifact_root_where_the_attempts_that_it_reads_live(run_store, tmp_path):
    worker = stand_in_worker(run_store, tmp_path)
    job = claim_for(worker, review(run_store))

    worker.run_claimed(job)

    received = read_json(attempt_directory(tmp_path / "artifacts", job) / RECEIVED)
    assert received["effective_spec"]["artifact_root"] == str((tmp_path / "artifacts").resolve())


def test_each_attempt_of_a_branch_starts_at_the_parent_point_with_the_continuation_of_its_spec(
    run_store, tmp_path
):
    """A rerun starts where the spec starts, never at a point of the lost attempt before it."""
    store_path = tmp_path / "store"
    (store_path / "snapshots" / "snap-1").mkdir(parents=True)
    request = rollout(backend={"backend": "podman", "podman": {"store": str(store_path)}}, timeout_s=1200)
    parent = claimed(run_store, request)
    publish_points(run_store, parent, snapshots=["snap-1"], prefix=tmp_path / "prefix.jsonl")
    [point] = Index(run_store).points(parent["active_attempt"])
    run_store.finish(parent["id"], parent["lease_token"], {"status": "completed"})
    hint = {"guidance": "user-hint", "hint": "Run the tests first."}
    run_store.submit(JobSpec("rollout", request.spec, parent_point=point["id"], continuation=hint), "branch")
    crashing = stand_in_worker(run_store, tmp_path, "exit")
    # The parent ran on host-1 (``claimed``), and its key binds the branch to that host.
    registered(run_store, crashing.worker_id, host="host-1")
    first = run_store.claim(crashing.worker_id)
    crashing.run_claimed(first)
    # The backoff before the rerun is 30 s. The job is made ready now instead of after it.
    with run_store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET ready_at=clock_timestamp() WHERE id=%s", (first["id"],))
    worker = stand_in_worker(run_store, tmp_path)
    second = run_store.claim(worker.worker_id)

    worker.run_claimed(second)

    received = read_json(attempt_directory(tmp_path / "artifacts", second) / RECEIVED)
    started_at = (
        second["attempt_count"],
        received["recovery"]["id"],
        received["continuation"],
        received["effective_spec"]["timeout_s"],
    )
    assert started_at == (2, point["id"], hint, 1200)
