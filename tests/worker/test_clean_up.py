"""What a worker frees and deletes once a minute: leftover sandboxes, and the data of its lost attempts.

``Leftovers`` runs against the fake podman, btrfs and findmnt of ``tests/support``. ``Worker.clean_up``
uses the isolated test database. ADR-0006 records the design, and docs/RUNSTORE.md ("Failure
handling") the behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sprout.runstore.store import Store
from sprout.worker.attempts import Worker, process_identity
from sprout.worker.cleanup import LEFTOVERS_NAME, Leftovers, log_section
from sprout.worker.run.resources import ResourceLedger
from tests.support.runstore import stopped
from tests.support.tools import ManagedContainer
from tests.worker.claimed import attempt_directory, claimed, rollout, worker_config

SANDBOX = "sprout-0000000000bb"


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


def attempt_claiming(directory: Path, sandbox_id: str, *, released: bool = False) -> Path:
    """The attempt ``directory``, whose ledger claims ``sandbox_id``, and releases it with ``released``."""
    directory.mkdir(parents=True)
    records = [{"event": "claim", "run_id": "attempt", "kind": "sandbox", "id": sandbox_id}]
    if released:
        records.append({"event": "release", "run_id": "attempt", "kind": "sandbox", "id": sandbox_id})
    (directory / "resources.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return directory


def podman_backend(store: Path) -> dict[str, Any]:
    return {"backend": "podman", "podman": {"store": str(store)}}


def recorded(tmp_path: Path) -> list[tuple[str, bool]]:
    """The sandboxes that the leftovers ledger in ``tmp_path`` records, each with whether it is freed."""
    return [
        (resource.id, resource.released) for resource in ResourceLedger(tmp_path / LEFTOVERS_NAME).resources()
    ]


# Leftovers


def test_a_left_sandbox_is_removed_by_the_next_free_and_then_recorded_as_freed(tmp_path, podman_store):
    leftovers = Leftovers(tmp_path / LEFTOVERS_NAME)
    leftovers.leave(
        attempt_claiming(tmp_path / "attempt", SANDBOX), backend=podman_backend(podman_store), held=set()
    )

    freed = leftovers.free()

    assert (freed, (podman_store / "sandboxes" / SANDBOX).exists(), recorded(tmp_path)) == (
        [SANDBOX],
        False,
        [(SANDBOX, True)],
    )


def test_an_attempt_leaves_neither_a_released_sandbox_nor_one_that_a_rollout_keeps(tmp_path):
    """A kept sandbox is the kept sandboxes' to remove once its time passes."""
    leftovers = Leftovers(tmp_path / LEFTOVERS_NAME)
    backend = podman_backend(Path("/path/to/store"))

    leftovers.leave(
        attempt_claiming(tmp_path / "released", "sprout-1", released=True), backend=backend, held=set()
    )
    leftovers.leave(attempt_claiming(tmp_path / "kept", "sprout-2"), backend=backend, held={"sprout-2"})

    assert recorded(tmp_path) == []


def test_a_left_sandbox_that_its_backend_no_longer_lists_is_recorded_as_freed(
    tmp_path, fake_podman, fake_btrfs_store
):
    leftovers = Leftovers(tmp_path / LEFTOVERS_NAME)
    backend = podman_backend(fake_btrfs_store)
    leftovers.leave(attempt_claiming(tmp_path / "attempt", "sprout-gone"), backend=backend, held=set())

    freed = leftovers.free()

    assert (freed, recorded(tmp_path)) == ([], [("sprout-gone", True)])


def test_a_left_sandbox_that_cannot_be_removed_stays_recorded_for_the_next_free(
    tmp_path, podman_store, fake_podman
):
    leftovers = Leftovers(tmp_path / LEFTOVERS_NAME)
    leftovers.leave(
        attempt_claiming(tmp_path / "attempt", SANDBOX), backend=podman_backend(podman_store), held=set()
    )
    fake_podman.fail("rm")
    first = leftovers.free()
    fake_podman.fail()

    second = leftovers.free()

    assert (first, second, recorded(tmp_path)) == ([], [SANDBOX], [(SANDBOX, True)])


# The lost attempts of a worker


def lost(store: Store, worker: Worker, process: dict[str, Any] | None = None) -> dict[str, Any]:
    """A job that ``worker`` claimed and that runs again, as the sweep after the death of ``worker`` left it.

    Its attempt recorded the child ``process``, if any. The worker stopped, so it is dead."""
    job = claimed(store, rollout(), by=worker.worker_id)
    if process is not None:
        store.progress(job["id"], job["lease_token"], execution={"process": process})
    stopped(store, worker.worker_id)
    with store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET lease_until=clock_timestamp() - interval '1 second'")
    assert store.settle_dead_workers() == [job["id"]]
    return job


@pytest.fixture
def child_left_running() -> Iterator[subprocess.Popen[bytes]]:
    """A live process that leads its own group, as the child of an attempt whose worker was killed."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    yield process
    process.kill()
    process.wait(timeout=5)


def test_clean_up_stops_the_child_of_a_lost_attempt_and_deletes_its_data_but_keeps_its_log(
    run_store, tmp_path, child_left_running
):
    """Another worker ran the job again, so nothing of this attempt counts any more."""
    worker = Worker(run_store, worker_config(tmp_path / "artifacts"))
    job = lost(run_store, worker, process_identity(child_left_running.pid))
    directory = attempt_directory(tmp_path / "artifacts", job)
    directory.mkdir(parents=True)
    (directory / "worker.log").write_text("the last line of the child\n")

    worker.clean_up()

    [attempt] = run_store.attempts(job["id"])
    assert (directory.exists(), child_left_running.wait(timeout=5) < 0) == (False, True)
    assert attempt["error"] == "Worker %s stopped reporting heartbeats%s" % (
        worker.worker_id,
        log_section("the last line of the child\n"),
    )


def test_clean_up_takes_each_lost_attempt_once_with_its_data_or_without_a_directory(run_store, tmp_path):
    """The worker cleans up once a minute, so the attempts that it reads must not grow with each loss."""
    worker = Worker(run_store, worker_config(tmp_path / "artifacts"))
    left_data = lost(run_store, worker)
    attempt_directory(tmp_path / "artifacts", left_data).mkdir(parents=True)
    # The worker deletes the directory of an attempt before the attempt's job runs again.
    lost(run_store, worker)

    worker.clean_up()

    assert run_store.lost_attempts(worker.worker_id) == []


def test_clean_up_leaves_the_lost_attempts_of_another_worker(run_store, tmp_path):
    """Two workers can share one artifact root. Each one deletes only what it ran."""
    other = Worker(run_store, worker_config(tmp_path / "artifacts"), worker_id="host-1/other")
    job = lost(run_store, other)
    directory = attempt_directory(tmp_path / "artifacts", job)
    directory.mkdir(parents=True)

    Worker(run_store, worker_config(tmp_path / "artifacts"), worker_id="host-1/default").clean_up()

    assert directory.exists() is True
