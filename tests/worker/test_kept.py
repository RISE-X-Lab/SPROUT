"""Kept sandboxes (ADR-0004): which rollout keeps its sandbox, how a grade takes it over, when it goes.

A rollout with ``keep_sandbox`` keeps its sandbox for ``KEEP_SANDBOX_S`` after its final snapshot.
The worker records it in ``KeptSandboxes``, a grade in place takes it over, and the reaper removes
it once its ``keep_until`` passes. A grade whose child never started gives the sandbox back.

``KeptSandboxes`` and the worker run against the fake podman, btrfs and findmnt of ``tests/support``.
The attempt child runs in-process on the directory sandboxes of ``run_sandboxes``. The worker tests
use the isolated test database, and the stand-in child of ``tests.worker.attempt_child``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from sprout.contract.run_spec import KEEP_SANDBOX_S
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from sprout.worker import kept as kept_module
from sprout.worker.attempts import Worker
from sprout.worker.files import read_json
from sprout.worker.jobs.rollout import execute
from sprout.worker.kept import LEDGER_NAME, KeptSandboxes
from sprout.worker.run.resources import ResourceLedger
from tests.support.runstore import registered
from tests.support.tools import ManagedContainer
from tests.worker.attempt_child import RECEIVED, child_python
from tests.worker.claimed import attempt_directory, left_running, rollout, worker_config

SANDBOX = "sprout-0000000000aa"
PASSED = "2000-01-01T00:00:00+00:00"
FUTURE = "2999-01-01T00:00:00+00:00"


@pytest.fixture
def podman_store(fake_podman, fake_btrfs_store):
    """A store that the fakes accept, with the sandbox SANDBOX: its container, rootfs and sockets."""
    (fake_btrfs_store / "sandboxes" / SANDBOX).mkdir(parents=True)
    (fake_btrfs_store / "sandboxes" / (SANDBOX + ".json")).write_text("{}")
    (fake_btrfs_store / "sockets" / SANDBOX).mkdir(parents=True)
    sockets = fake_btrfs_store / "sockets" / SANDBOX
    fake_podman.containers_exist(
        ManagedContainer(name=SANDBOX, store=fake_btrfs_store, sockets=sockets, created=0)
    )
    return fake_btrfs_store


def podman_backend(store: Path) -> dict[str, Any]:
    return {"backend": "podman", "podman": {"store": str(store)}}


def rootfs_exists(store: Path) -> bool:
    """Whether the rootfs of SANDBOX is still in ``store``, which a removal of the sandbox deletes last."""
    return (store / "sandboxes" / SANDBOX).exists()


@pytest.fixture
def kept(tmp_path):
    return KeptSandboxes(tmp_path / LEDGER_NAME)


@pytest.mark.parametrize(
    "keep_until, reaped, rootfs, held",
    [(PASSED, [SANDBOX], False, set()), (FUTURE, [], True, {SANDBOX})],
    ids=["passed", "not passed"],
)
def test_the_reaper_removes_a_kept_sandbox_once_its_keep_until_has_passed(
    kept, podman_store, keep_until, reaped, rootfs, held
):
    kept.keep(
        sandbox_id=SANDBOX,
        keep_until=keep_until,
        attempt_id="attempt-1",
        backend=podman_backend(podman_store),
    )

    removed = kept.reap()

    assert (removed, rootfs_exists(podman_store), kept.held()) == (reaped, rootfs, held)


def test_a_kept_sandbox_that_the_reaper_fails_to_remove_stays_recorded_for_the_next_reap(
    kept, podman_store, fake_podman
):
    kept.keep(
        sandbox_id=SANDBOX, keep_until=PASSED, attempt_id="attempt-1", backend=podman_backend(podman_store)
    )
    fake_podman.fail("rm")
    first = kept.reap()
    fake_podman.fail()

    second = kept.reap()

    assert (first, second, rootfs_exists(podman_store)) == ([], [SANDBOX], False)


def test_a_new_record_of_a_kept_sandbox_replaces_its_earlier_record(kept, podman_store):
    backend = podman_backend(podman_store)
    kept.keep(sandbox_id=SANDBOX, keep_until=PASSED, attempt_id="attempt-1", backend=backend)
    kept.keep(sandbox_id=SANDBOX, keep_until=FUTURE, attempt_id="attempt-2", backend=backend)

    reaped = kept.reap()

    assert (reaped, kept.held()) == ([], {SANDBOX})


def test_a_grade_takes_a_kept_sandbox_over_once_and_its_attempt_claims_it(tmp_path, kept):
    kept.keep(
        sandbox_id=SANDBOX,
        keep_until=FUTURE,
        attempt_id="rollout",
        backend=podman_backend(Path("/path/to/store")),
    )
    attempt = ResourceLedger(tmp_path / "resources.jsonl")

    with attempt.run("grader") as claim:
        taken = [kept.take(SANDBOX, claim=claim), kept.take(SANDBOX, claim=claim)]

    claims = [
        (resource.id, resource.backend, resource.store, resource.keep) for resource in attempt.resources()
    ]
    assert (taken, claims, kept.held()) == (
        [True, False],
        [(SANDBOX, "podman", "/path/to/store", False)],
        set(),
    )


def test_a_sandbox_that_a_grade_took_is_left_to_the_grade_after_its_keep_until_passes(
    tmp_path, kept, podman_store, monkeypatch
):
    kept.keep(
        sandbox_id=SANDBOX, keep_until=FUTURE, attempt_id="rollout", backend=podman_backend(podman_store)
    )
    with ResourceLedger(tmp_path / "attempt.jsonl").run("grader") as claim:
        kept.take(SANDBOX, claim=claim)
    monkeypatch.setattr(
        kept_module, "datetime", SimpleNamespace(now=lambda tz: datetime(3000, 1, 1, tzinfo=tz))
    )

    reaped = kept.reap()

    assert (reaped, kept.held(), rootfs_exists(podman_store)) == ([], set(), True)


def test_a_sandbox_that_a_grade_gives_back_is_kept_again_as_its_rollout_kept_it(tmp_path, kept):
    """With its keep_until, so that the reaper removes it in time. The grade's ledger no longer holds it."""
    kept.keep(
        sandbox_id=SANDBOX,
        keep_until=FUTURE,
        attempt_id="rollout",
        backend=podman_backend(Path("/path/to/store")),
    )
    attempt = ResourceLedger(tmp_path / "attempt.jsonl")
    with attempt.run("grader") as claim:
        kept.take(SANDBOX, claim=claim)

    with attempt.run("grader") as claim:
        kept.give_back(SANDBOX, claim=claim)

    [record] = ResourceLedger(tmp_path / LEDGER_NAME).resources()
    [grader] = attempt.resources()
    assert (record.keep_until, record.backend, record.store, kept.held(), grader.released) == (
        FUTURE,
        "podman",
        "/path/to/store",
        {SANDBOX},
        True,
    )


def test_a_kept_sandbox_past_its_keep_until_is_not_handed_to_a_grade(tmp_path, kept):
    """It is the reaper's to remove."""
    kept.keep(
        sandbox_id=SANDBOX,
        keep_until=PASSED,
        attempt_id="rollout",
        backend=podman_backend(Path("/path/to/store")),
    )
    attempt = ResourceLedger(tmp_path / "attempt.jsonl")

    with attempt.run("grader") as claim:
        taken = kept.take(SANDBOX, claim=claim)

    assert (taken, attempt.resources(), kept.held()) == (False, [], {SANDBOX})


def test_a_sandbox_that_no_rollout_keeps_here_is_not_handed_to_a_grade(tmp_path, kept):
    attempt = ResourceLedger(tmp_path / "resources.jsonl")

    with attempt.run("grader") as claim:
        taken = kept.take(SANDBOX, claim=claim)

    assert (taken, attempt.resources()) == (False, [])


@pytest.fixture
def attempt(tmp_path):
    """The attempt directory ``job-1/attempt-1``, which the worker makes before the child starts."""
    (tmp_path / "artifacts" / "job-1" / "attempt-1").mkdir(parents=True)
    return tmp_path / "artifacts" / "job-1" / "attempt-1"


def rollout_request(**spec: Any) -> dict[str, Any]:
    """A rollout request of the nop agent from the image ``base``, with ``spec`` over its effective spec."""
    return {
        "kind": "rollout",
        "attempt_id": "attempt-1",
        "profile_config": {},
        "effective_spec": {"prompt": "Do nothing.", "agent": "nop", "sandbox_image": "base", **spec},
    }


def test_a_kept_rollout_keeps_its_sandbox_for_keep_sandbox_s_after_its_final_snapshot(run_sandboxes, attempt):
    before = datetime.now(UTC)

    result = execute(rollout_request(keep_sandbox=True), attempt)

    after = datetime.now(UTC)
    keep_until = datetime.fromisoformat(result["kept_sandbox"]["keep_until"])
    assert (result["kept_sandbox"]["sandbox_id"], result["final_snapshot_id"], run_sandboxes.live()) == (
        "sandbox-1",
        "snapshot-1",
        ["sandbox-1"],
    )
    assert keep_until.utcoffset() == timedelta(0), "the Run Store accepts only a UTC keep_until"
    assert (
        before + timedelta(seconds=KEEP_SANDBOX_S) <= keep_until <= after + timedelta(seconds=KEEP_SANDBOX_S)
    )


def test_a_rollout_that_does_not_ask_to_keep_its_sandbox_keeps_none(run_sandboxes, attempt):
    result = execute(rollout_request(keep_sandbox=False), attempt)

    assert ("kept_sandbox" in result, run_sandboxes.live()) == (False, [])


def test_a_kept_rollout_without_its_final_snapshot_keeps_no_sandbox_for_a_grade(run_sandboxes, attempt):
    """The sandbox then holds the only copy of the rollout, which the reaper must not remove."""
    run_sandboxes.fail_snapshots()

    result = execute(rollout_request(keep_sandbox=True), attempt)

    assert ("kept_sandbox" in result, result["final_snapshot_id"]) == (False, None)


@pytest.mark.parametrize("keep_sandbox, lease_s", [(False, 1800 + 600), (True, 1800 + KEEP_SANDBOX_S + 600)])
def test_the_lease_of_a_microvm_covers_the_run_and_the_time_that_it_is_kept(
    run_sandboxes, attempt, keep_sandbox, lease_s
):
    """AgentENV removes a microVM when its lease ends. The lease adds 600 s for the start and the removal."""
    backend = {"backend": "microvm", "microvm": {"server_url": "http://127.0.0.1:1"}}

    execute(rollout_request(keep_sandbox=keep_sandbox, timeout_s=1800, backend=backend), attempt)

    [session] = run_sandboxes.sessions
    assert session.backend["microvm"]["sandbox_ttl"] == lease_s


def kept_rollout(store, root: Path, sandbox_store: Path, outcome: dict[str, Any]) -> dict[str, Any]:
    """A keep_sandbox podman rollout, left running, that claimed SANDBOX and wrote ``outcome``."""
    request = rollout(keep_sandbox=True, backend=podman_backend(sandbox_store))
    job = left_running(store, root, request, outcome=outcome)
    with ResourceLedger(attempt_directory(root, job) / "resources.jsonl").run("attempt") as claim:
        claim.sandbox(SANDBOX, keep=True, backend="podman", store=str(sandbox_store))
    return job


def test_a_succeeded_rollout_leaves_its_kept_sandbox_running_and_recorded_for_the_reaper(
    run_store, tmp_path, fake_podman, podman_store
):
    kept = {"sandbox_id": SANDBOX, "keep_until": FUTURE}
    root = tmp_path / "artifacts"
    job = kept_rollout(run_store, root, podman_store, {"status": "completed", "kept_sandbox": kept})

    Worker(run_store, worker_config(root)).settle(job)

    finished = run_store.get(job["id"])
    [record] = ResourceLedger(root / LEDGER_NAME).resources()
    assert (finished["state"], finished["result"]["kept_sandbox"]) == ("succeeded", kept)
    assert (record.id, record.keep_until, record.backend, record.store) == (
        SANDBOX,
        FUTURE,
        "podman",
        str(podman_store.resolve()),
    )
    assert ["rm", "-f", SANDBOX] not in fake_podman.calls()


def test_a_rollout_that_does_not_succeed_keeps_nothing_and_its_sandbox_is_removed(
    run_store, tmp_path, fake_podman, podman_store
):
    """A rollout that failed is not graded in place, so its sandbox goes with the attempt."""
    kept = {"sandbox_id": SANDBOX, "keep_until": FUTURE}
    root = tmp_path / "artifacts"
    outcome = {"status": "timeout", "error": "the agent ran out of time", "kept_sandbox": kept}
    job = kept_rollout(run_store, root, podman_store, outcome)

    Worker(run_store, worker_config(root)).settle(job)

    finished = run_store.get(job["id"])
    assert (finished["state"], "kept_sandbox" in finished["result"]) == ("failed", False)
    assert (ResourceLedger(root / LEDGER_NAME).resources(), ["rm", "-f", SANDBOX] in fake_podman.calls()) == (
        [],
        True,
    )


def grade_in_place(store, worker: Worker, kept: dict[str, Any]) -> dict[str, Any]:
    """A grade of a rollout that ``worker`` ran and that kept ``kept``, claimed by ``worker``."""
    registered(store, worker.worker_id, host=worker.host)
    rollout_job = store.submit(rollout(keep_sandbox=True), "kept-rollout")
    lease = store.claim(worker.worker_id)
    store.finish(rollout_job["id"], lease["lease_token"], {"status": "completed", "kept_sandbox": kept})
    grade = {"benchmark": "harbor", "task_dir": "/path/to/task", "in_place_job": rollout_job["id"]}
    store.submit(JobSpec("grade", grade), "in-place-grade")
    job = store.claim(worker.worker_id)
    assert job is not None and job["kind"] == "grade"
    return job


def in_place_worker(store, tmp_path: Path, sandbox_store: Path) -> Worker:
    """A worker whose grades run on podman in ``sandbox_store``, each in a stand-in child."""
    config = worker_config(
        tmp_path / "artifacts",
        python=child_python(tmp_path),
        grade_defaults={"backend": podman_backend(sandbox_store)},
    )
    return Worker(store, config)


def test_a_grade_in_place_gets_the_sandbox_that_its_worker_keeps_and_its_attempt_removes_it(
    run_store, tmp_path, fake_podman, podman_store
):
    kept = {"sandbox_id": SANDBOX, "keep_until": FUTURE}
    worker = in_place_worker(run_store, tmp_path, podman_store)
    worker.kept.keep(
        sandbox_id=SANDBOX, keep_until=FUTURE, attempt_id="rollout", backend=podman_backend(podman_store)
    )
    job = grade_in_place(run_store, worker, kept)

    worker.run_claimed(job)

    received = read_json(attempt_directory(tmp_path / "artifacts", job) / RECEIVED)
    assert (received["kept_sandbox"], run_store.get(job["id"])["state"]) == (kept, "succeeded")
    assert (["rm", "-f", SANDBOX] in fake_podman.calls(), worker.kept.held()) == (True, set())


def test_a_grade_in_place_gets_no_sandbox_when_its_worker_no_longer_keeps_it(
    run_store, tmp_path, fake_podman, podman_store
):
    """The grade then scores a lost sandbox. Nothing removes a sandbox that this worker does not hold."""
    kept = {"sandbox_id": SANDBOX, "keep_until": FUTURE}
    worker = in_place_worker(run_store, tmp_path, podman_store)
    job = grade_in_place(run_store, worker, kept)

    worker.run_claimed(job)

    received = read_json(attempt_directory(tmp_path / "artifacts", job) / RECEIVED)
    assert (received["kept_sandbox"], ["rm", "-f", SANDBOX] in fake_podman.calls()) == (None, False)


class ShutdownAtFreeze:
    """The Run Store, but the worker's ``stop`` is set when an attempt freezes its payload.

    So the shutdown comes after a grade in place took its sandbox over, and before the grade's child starts.
    """

    def __init__(self, store: Store, stop: Event) -> None:
        self._store = store
        self._stop = stop

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def freeze_payload(self, job_id: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._stop.set()
        return self._store.freeze_payload(job_id, token, payload)


def test_a_grade_in_place_that_a_shutdown_released_gets_its_kept_sandbox_again_on_its_next_attempt(
    run_store, tmp_path, fake_podman, podman_store
):
    """The released attempt gives the sandbox back. Otherwise nothing grades in it, and nothing removes it."""
    kept = {"sandbox_id": SANDBOX, "keep_until": FUTURE}
    stop = Event()
    worker = in_place_worker(ShutdownAtFreeze(run_store, stop), tmp_path, podman_store)
    worker.kept.keep(
        sandbox_id=SANDBOX, keep_until=FUTURE, attempt_id="rollout", backend=podman_backend(podman_store)
    )
    job = grade_in_place(run_store, worker, kept)
    worker.run_claimed(job, stop=stop)
    again = run_store.claim(worker.worker_id)
    assert again is not None, "the released grade is not ready for a claim"

    worker.run_claimed(again)

    received = read_json(attempt_directory(tmp_path / "artifacts", again) / RECEIVED)
    assert (received["kept_sandbox"], run_store.get(job["id"])["state"]) == (kept, "succeeded")
