from collections import deque
from pathlib import Path
import multiprocessing
import subprocess
from threading import Event, Lock, Thread, get_ident
import time
from types import SimpleNamespace

import pytest

from sprout.worker.manager import WorkerManager
from tests.runstore.test_payload import envelope
from tests.runstore.test_store import request, store
from sprout.worker.attempts import Worker, process_identity, stop_process


def eventually(predicate, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert predicate()


class Queue:
    def __init__(self, count: int) -> None:
        self.jobs = deque({"id": str(index)} for index in range(count))
        self.claimers = []

    def expire(self) -> list:
        return []

    def recoverable_jobs(self, **kwargs) -> list:
        return []

    def claim(self, *args, **kwargs) -> dict | None:
        self.claimers.append(get_ident())
        return self.jobs.popleft() if self.jobs else None


def test_one_claim_loop_bounds_concurrency_and_refills_after_failure():
    queue = Queue(30)
    stop = Event()
    lock = Lock()
    first_wave = Event()
    released = Event()
    active = set()
    started = []
    completed = []
    peaks = []

    def execute(job, **kwargs):
        with lock:
            active.add(job["id"])
            started.append(job["id"])
            peaks.append(len(active))
            if len(active) == 4:
                first_wave.set()
        released.wait(10)
        try:
            if job["id"] == "0":
                raise ValueError("One attempt fails without stopping the manager")
            time.sleep(0.02)
        finally:
            with lock:
                active.remove(job["id"])
                completed.append(job["id"])
                if len(completed) == 30:
                    stop.set()

    worker = SimpleNamespace(store=queue, worker_id="one-manager", lease_s=60, run_claimed=execute)
    manager = WorkerManager(worker, concurrency=4, poll_s=0.01)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        assert first_wave.wait(5)
        assert len(started) == 4
        released.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert len(set(started)) == len(completed) == 30
        assert max(peaks) == 4
        assert set(queue.claimers) == {thread.ident}
    finally:
        released.set()
        stop.set()
        thread.join(timeout=5)


def test_manager_backs_off_queue_errors_and_keeps_running():
    queue = Queue(1)
    original = queue.claim
    calls = []

    def flaky(*args, **kwargs):
        calls.append(time.monotonic())
        if len(calls) <= 2:
            raise OSError("Database temporarily unavailable")
        return original(*args, **kwargs)

    queue.claim = flaky
    executed = []
    worker = SimpleNamespace(store=queue, worker_id="fixture", lease_s=60,
                             run_claimed=lambda job, **kwargs: executed.append(job["id"]))
    WorkerManager(worker, poll_s=0.03).run(once=True)
    assert executed == ["0"] and len(calls) == 3
    assert calls[1] - calls[0] >= 0.025
    assert calls[2] - calls[1] >= 0.05


def test_once_claims_at_most_one_new_job_even_with_large_capacity():
    queue = Queue(20)
    executed = []
    worker = SimpleNamespace(store=queue, worker_id="fixture", lease_s=60,
                             run_claimed=lambda job, **kwargs: executed.append(job["id"]))
    WorkerManager(worker, concurrency=24, poll_s=0.01).run(once=True)
    assert executed == ["0"] and len(queue.jobs) == 19


def test_shutdown_does_not_wait_forever_for_blocked_io():
    queue = Queue(2)
    entered, release, stop = Event(), Event(), Event()

    def execute(job, **kwargs):
        entered.set()
        release.wait(10)

    manager = WorkerManager(SimpleNamespace(store=queue, worker_id="fixture", lease_s=60,
                                            run_claimed=execute), poll_s=0.01, shutdown_s=0.1)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        assert entered.wait(5)
        stop.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert len(manager.active) == 1
        assert len(queue.jobs) == 1
    finally:
        release.set()
        stop.set()
        thread.join(timeout=5)
        eventually(lambda: all(future.done() for future in manager.active.values()))


@pytest.fixture
def fixture_children(monkeypatch):
    popen = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        assert command[2] == "sprout.worker.jobs.rollout"
        process = popen([*command[:2], "tests.runstore.payload_child", *command[3:]], **kwargs)
        processes.append((process, process_identity(process.pid)))
        return process

    monkeypatch.setattr(subprocess, "Popen", launch)
    yield processes
    for process, identity in processes:
        stop_process(identity)
        process.wait(timeout=5)


def config(root: Path) -> dict:
    return {"artifact_root": str(root), "profiles": {"default": {"run_defaults": {
        "sandbox_image": "fixture", "backend": {"backend": "microvm"}}}}}


def attempt_directory(store, root: Path, job: dict) -> Path:
    return root / job["id"] / store.get(job["id"])["active_attempt"]


def test_blocked_indexing_does_not_block_sibling_heartbeat_refill_or_deadline(store, tmp_path, fixture_children):
    blocked = store.submit(request(prompt="hold-fixture", timeout_s=2), "blocked")
    sibling = store.submit(request(prompt="gate-fixture", timeout_s=30), "sibling")
    next_job = store.submit(request(), "next")
    worker = Worker(store, config(tmp_path), lease_s=5)
    ingest = worker._ingest
    entered, release, stop = Event(), Event(), Event()

    def blocked_ingest(job, directory, payload):
        if job["id"] == blocked["id"]:
            entered.set()
            release.wait(20)
        return ingest(job, directory, payload)

    worker._ingest = blocked_ingest
    manager = WorkerManager(worker, concurrency=2, poll_s=0.02)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        assert entered.wait(10)
        eventually(lambda: store.get(sibling["id"])["active_attempt"] is not None)
        sibling_directory = attempt_directory(store, tmp_path, sibling)
        eventually(lambda: (sibling_directory / "executed.json").exists())
        assert store.get(next_job["id"])["state"] == "queued"
        sibling_lease = store.get(sibling["id"])["lease_until"]
        blocked_attempt = store.attempts(blocked["id"])[0]
        blocked_identity = blocked_attempt["execution"]["process"]
        eventually(lambda: process_identity(blocked_identity["pid"]) is None, timeout=6)
        assert not release.is_set()
        eventually(lambda: store.get(sibling["id"])["lease_until"] > sibling_lease)
        assert not manager.active[blocked["id"]].done()
        (sibling_directory / "release").touch()
        eventually(lambda: store.get(next_job["id"])["state"] == "succeeded")
        assert store.get(sibling["id"])["state"] == "succeeded"
        release.set()
        eventually(lambda: blocked["id"] not in manager.active)
        assert len(store.attempts(next_job["id"])) == 1
    finally:
        release.set()
        stop.set()
        thread.join(timeout=25)
        assert not thread.is_alive()


def test_malformed_recovery_is_isolated_and_other_jobs_run(store, tmp_path, fixture_children):
    broken = store.submit(request(), "broken-recovery")
    claim = store.claim("dead-manager")
    store.freeze_payload(claim["id"], claim["lease_token"], envelope(claim))
    directory = tmp_path / broken["id"] / claim["active_attempt"]
    directory.mkdir(parents=True)
    (directory / "outcome.json").write_text("malformed outcome")
    with store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                       (broken["id"],))
    healthy = store.submit(request(), "healthy")
    worker = Worker(store, config(tmp_path))
    WorkerManager(worker, poll_s=0.01).run(once=True)
    assert store.get(broken["id"])["state"] == "quarantined"
    assert "JSONDecodeError" in store.get(broken["id"])["error"]
    assert store.get(healthy["id"])["state"] == "succeeded"


def test_attempt_index_exception_does_not_stop_manager_or_siblings(store, tmp_path, fixture_children):
    broken = store.submit(request(prompt="hold-fixture"), "broken-index")
    healthy = store.submit(request(), "healthy")
    worker = Worker(store, config(tmp_path))
    ingest = worker._ingest

    def invalid_index(job, directory, payload):
        if job["id"] == broken["id"]:
            raise ValueError("Malformed trajectory fixture")
        return ingest(job, directory, payload)

    worker._ingest = invalid_index
    stop = Event()
    manager = WorkerManager(worker, concurrency=2, poll_s=0.01)
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        eventually(lambda: store.get(healthy["id"])["state"] == "succeeded")
        eventually(lambda: store.get(broken["id"])["state"] == "quarantined")
        assert "Malformed trajectory" in store.get(broken["id"])["error"]
        assert thread.is_alive()
    finally:
        stop.set()
        thread.join(timeout=25)
        assert not thread.is_alive()


def test_cli_routes_concurrency_and_once_to_manager(tmp_path, monkeypatch):
    import json
    import sys

    from sprout.worker import cli

    configuration = tmp_path / "worker.json"
    configuration.write_text(json.dumps(config(tmp_path / "artifacts")))
    captured = {}

    class Manager:
        def __init__(self, worker, *, concurrency):
            captured["concurrency"] = concurrency

        def run(self, *, stop, once):
            captured["once"] = once
            assert not stop.is_set()

    monkeypatch.setenv("SPROUT_RUNSTORE_DSN", "fixture-not-connected")
    monkeypatch.setattr(sys, "argv", ["worker", "start", "--config", str(configuration),
                                    "--concurrency", "24", "--once"])
    monkeypatch.setattr(cli, "volatile_reason", lambda root: None)
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    monkeypatch.setattr("sprout.worker.manager.WorkerManager", Manager)
    cli.main()
    assert captured == {"concurrency": 24, "once": True}


def test_shutdown_stops_all_owned_children_and_leaves_queued_jobs(store, tmp_path, fixture_children):
    jobs = [store.submit(request(prompt="hold-fixture", timeout_s=30), f"job-{index}") for index in range(3)]
    worker = Worker(store, config(tmp_path))
    manager = WorkerManager(worker, concurrency=2, poll_s=0.02)
    stop = Event()
    thread = Thread(target=lambda: manager.run(stop=stop))
    thread.start()
    try:
        eventually(lambda: len(fixture_children) == 2)
        for job in jobs[:2]:
            directory = attempt_directory(store, tmp_path, job)
            eventually(lambda: (directory / "executed.json").exists())
        stop.set()
        thread.join(timeout=25)
        assert not thread.is_alive()
        assert len(fixture_children) == 2
        assert all(process_identity(identity["pid"]) is None for process, identity in fixture_children)
        assert store.get(jobs[2]["id"])["state"] == "queued"
        assert all(store.get(job["id"])["state"] in {"failed", "quarantined"} for job in jobs[:2])
    finally:
        stop.set()
        thread.join(timeout=25)


def test_expiry_scan_skips_locked_job_instead_of_blocking_dispatch(store):
    locked = store.submit(request(), "locked")
    store.claim("worker")
    with store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                       (locked["id"],))
    other = store.submit(request(), "other")
    with store.transaction() as cursor:
        cursor.execute("SELECT id FROM rs_jobs WHERE id=%s FOR UPDATE", (locked["id"],))
        started = time.monotonic()
        assert store.expire() == []
        assert store.claim("manager")["id"] == other["id"]
        assert time.monotonic() - started < 2
    assert store.expire() == [locked["id"]]


def test_store_sets_sql_and_connection_timeouts(store):
    with store.transaction() as cursor:
        cursor.execute("SHOW statement_timeout")
        assert cursor.fetchone()["statement_timeout"] == "5s"
        settings = cursor.connection.get_dsn_parameters()
        assert settings["connect_timeout"] == "5"
        assert settings["tcp_user_timeout"] == "10000"
        assert settings["keepalives_count"] == "2"


def run_crash_fixture(dsn: str, root: str) -> None:
    from sprout.runstore.store import Store

    popen = subprocess.Popen

    def launch(command, **kwargs):
        return popen([*command[:2], "tests.runstore.payload_child", *command[3:]], **kwargs)

    subprocess.Popen = launch
    WorkerManager(Worker(Store(dsn), config(Path(root))), concurrency=2, poll_s=0.02).run()


def test_manager_sigkill_reconciles_all_attempts_then_refills(store, tmp_path, fixture_children):
    jobs = [store.submit(request(prompt="hold-fixture", timeout_s=30), f"crash-{index}") for index in range(2)]
    following = store.submit(request(), "following")
    controller = multiprocessing.get_context("spawn").Process(target=run_crash_fixture, args=(store.dsn, str(tmp_path)))
    controller.start()
    identities = []
    try:
        for job in jobs:
            eventually(lambda: bool(store.attempts(job["id"])) and
                       bool(store.attempts(job["id"])[0]["execution"].get("process")))
            identities.append(store.attempts(job["id"])[0]["execution"]["process"])
            directory = attempt_directory(store, tmp_path, job)
            eventually(lambda: (directory / "executed.json").exists())
        controller.kill()
        controller.join(timeout=5)
        assert not controller.is_alive() and controller.exitcode == -9
        with store.transaction() as cursor:
            cursor.execute("UPDATE rs_jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=ANY(%s)",
                           ([job["id"] for job in jobs],))
        WorkerManager(Worker(store, config(tmp_path)), concurrency=2, poll_s=0.02).run(once=True)
        assert all(store.get(job["id"])["state"] == "failed" for job in jobs)
        assert store.get(following["id"])["state"] == "succeeded"
        assert all(process_identity(identity["pid"]) is None for identity in identities)
        assert all(len(store.attempts(job["id"])) == 1 for job in [*jobs, following])
    finally:
        if controller.is_alive():
            controller.kill()
        controller.join(timeout=5)
        for identity in identities:
            stop_process(identity)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_rejected(capacity):
    with pytest.raises(ValueError):
        WorkerManager(None, concurrency=capacity)
