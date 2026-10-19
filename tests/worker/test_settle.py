"""How a worker settles an attempt from what it left: the next state of the job and its result.

Each test settles, with ``Worker.settle``, a job that an earlier session of the worker left
running. ``settle`` stops the attempt's child, and publishes the attempt's ``outcome.json`` as
the end of a supervised attempt does. The Run Store is the isolated test database.
docs/RUNSTORE.md ("Failure handling") gives the rules.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from sprout.worker.attempts import Worker, process_identity
from sprout.worker.cleanup import LEFTOVERS_NAME
from sprout.worker.files import write_json
from sprout.worker.run.resources import ResourceLedger
from tests.support.runstore import registered
from tests.worker.claimed import (
    OLD_WORKER,
    attempt_directory,
    claimed,
    left_running,
    publish_points,
    rollout,
    worker_config,
)

GRADE = JobSpec(
    "grade",
    {"benchmark": "harbor", "task_dir": "/path/to/task", "snapshot_id": "snap-1"},
    affinity="group-1",
)


def settled(store: Store, root: Path, job: dict[str, Any]) -> dict[str, Any]:
    """The job after a worker with the artifact root ``root`` settled it."""
    Worker(store, worker_config(root)).settle(job)
    return store.get(job["id"])


def test_settle_publishes_the_outcome_that_the_attempt_wrote(run_store, tmp_path):
    job = left_running(run_store, tmp_path, rollout(), outcome={"status": "completed", "final_text": "done"})
    worker = Worker(run_store, {"artifact_root": str(tmp_path), "profiles": {}})

    settled_id = worker.settle(job)

    finished = run_store.get(job["id"])
    assert (settled_id, finished["state"], finished["result"]) == (
        job["id"],
        "succeeded",
        {"status": "completed", "final_text": "done", "failure_kind": None},
    )


@pytest.mark.parametrize(
    "outcome, state, kind",
    [
        ({"status": "completed"}, "succeeded", None),
        (
            {"status": "truncated", "stop_reason": "max_turns_reached", "final_snapshot_id": "snap-9"},
            "succeeded",
            None,
        ),
        ({"status": "error", "error": "No patch produced"}, "failed", "actor"),
        ({"status": "error", "error": "HTTP 502 Bad Gateway"}, "queued", "infrastructure"),
        (
            {
                "status": "error",
                "failure_kind": "infrastructure",
                "stop_reason": "worker_died",
                "error": "Worker process exited",
            },
            "queued",
            "infrastructure",
        ),
    ],
    ids=["completed", "truncated", "actor", "infrastructure", "worker died"],
)
def test_the_kind_of_failure_decides_whether_a_grade_succeeds_fails_or_runs_again(
    run_store, tmp_path, outcome, state, kind
):
    job = left_running(run_store, tmp_path, GRADE, outcome=outcome)

    finished = settled(run_store, tmp_path, job)

    assert (finished["state"], finished["result"]["failure_kind"]) == (state, kind)


#: What an attempt left of its outcome, written into its directory.
UNREADABLE_OUTCOMES = {
    "no outcome": lambda directory: None,
    "malformed outcome": lambda directory: (directory / "outcome.json").write_text('{"status": '),
}


@pytest.mark.parametrize("leave", UNREADABLE_OUTCOMES.values(), ids=UNREADABLE_OUTCOMES.keys())
def test_an_attempt_that_left_no_readable_outcome_is_lost_and_its_job_runs_again(run_store, tmp_path, leave):
    job = left_running(run_store, tmp_path, rollout(), outcome=None)
    directory = attempt_directory(tmp_path, job)
    directory.mkdir(parents=True)
    leave(directory)

    finished = settled(run_store, tmp_path, job)

    assert (finished["state"], finished["error"], run_store.attempts(job["id"])[0]["state"]) == (
        "queued",
        "Attempt left no outcome",
        "lost",
    )


def test_a_job_whose_attempt_froze_no_payload_is_released_without_using_an_attempt(run_store, tmp_path):
    """The payload is frozen before the child starts, so no child ran."""
    registered(run_store, OLD_WORKER)
    job = run_store.submit(rollout(), "unfrozen")
    claim = run_store.claim(OLD_WORKER)

    settled(run_store, tmp_path, claim)

    released = run_store.get(job["id"])
    assert (released["state"], released["phase"], released["attempt_count"]) == ("queued", "released", 0)


@pytest.fixture
def child_left_running() -> Iterator[subprocess.Popen[bytes]]:
    """A live process that leads its own group, as the child of an attempt whose worker was killed."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    yield process
    process.kill()
    process.wait(timeout=5)


def test_settle_stops_the_child_that_the_old_attempt_left_running_before_it_settles_the_job(
    run_store, tmp_path, child_left_running
):
    """Otherwise the old child goes on with the attempt while its job runs again or finishes elsewhere."""
    job = claimed(run_store, rollout())
    identity = process_identity(child_left_running.pid)
    run_store.progress(job["id"], job["lease_token"], execution={"process": identity})

    finished = settled(run_store, tmp_path, job)

    assert (finished["state"], child_left_running.wait(timeout=5)) == ("queued", -signal.SIGTERM)


@pytest.fixture
def process_outside_its_own_group() -> Iterator[subprocess.Popen[bytes]]:
    """A live process in the process group of this test, not in a group that it leads."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield process
    process.kill()
    process.wait(timeout=5)


def test_a_recorded_pid_that_leads_no_group_is_left_running_and_the_job_is_settled(
    run_store, tmp_path, process_outside_its_own_group
):
    """An attempt child leads its own group. A live pid outside one is another process's now."""
    job = claimed(run_store, rollout())
    identity = process_identity(process_outside_its_own_group.pid)
    run_store.progress(job["id"], job["lease_token"], execution={"process": identity})
    write_json(attempt_directory(tmp_path, job) / "outcome.json", {"status": "completed"})

    finished = settled(run_store, tmp_path, job)

    assert (finished["state"], process_outside_its_own_group.poll()) == ("succeeded", None)


def resources(directory: Path, *records: dict[str, Any]) -> None:
    """Write ``records`` as the attempt's resource ledger, in the format of ``worker.run.resources``."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"ts": "2026-10-17T00:00:00Z", **record}) + "\n" for record in records]
    (directory / "resources.jsonl").write_text("".join(lines))


def test_an_attempt_whose_cleanup_is_unconfirmed_is_settled_and_leaves_its_claims_for_the_clean_up(
    run_store, tmp_path
):
    """An unconfirmed cleanup is the worker's problem, not the job's, so the outcome stands."""
    job = left_running(run_store, tmp_path, rollout(), outcome={"status": "completed"})
    resources(
        attempt_directory(tmp_path, job),
        {"event": "allocation_started", "run_id": "attempt", "image": "image"},
        {
            "event": "claim",
            "run_id": "attempt",
            "kind": "sandbox",
            "id": "sprout-1",
            "store": "/path/to/store",
        },
        {"event": "allocation_started", "run_id": "attempt", "image": "snapshot-1"},
        {"event": "allocation_unknown", "run_id": "attempt"},
    )

    finished = settled(run_store, tmp_path, job)

    [left] = ResourceLedger(tmp_path / LEFTOVERS_NAME).resources()
    assert (finished["state"], finished["result"]) == (
        "succeeded",
        {"status": "completed", "failure_kind": None},
    )
    assert (left.id, left.backend, left.store, left.released) == (
        "sprout-1",
        "podman",
        "/path/to/store",
        False,
    )


def test_an_infrastructure_failure_runs_a_rollout_again_from_its_spec_and_forgets_its_points(
    run_store, tmp_path
):
    """The next attempt starts from the spec, so no branch may start from a point of this attempt."""
    store_path = tmp_path / "store"
    (store_path / "snapshots" / "snap-1").mkdir(parents=True)
    job = claimed(run_store, rollout(backend={"backend": "podman", "podman": {"store": str(store_path)}}))
    publish_points(run_store, job, snapshots=["snap-1"], prefix=tmp_path / "prefix.jsonl")
    outcome = {"status": "error", "error": "HTTP 502 Bad Gateway", "elapsed_s": 600}
    write_json(attempt_directory(tmp_path / "artifacts", job) / "outcome.json", outcome)

    finished = settled(run_store, tmp_path / "artifacts", job)

    assert (finished["state"], Index(run_store).points(job["active_attempt"])) == ("queued", [])
