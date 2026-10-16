import logging
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sprout.runstore.files import failure_kind
from sprout.runstore.files import read_json, write_json
from sprout.runstore.specs import JobSpec, digest
from tests.runstore.test_payload import envelope
from tests.runstore.test_store import request, store
from sprout.runstore.store import Fenced, Store
from sprout.runstore.worker import Worker, process_identity, stop_process


def run_once(worker: Worker) -> str | None:
    """One synchronous claim cycle -- expire leases, reconcile every
    lease-expired quarantine, claim one job and run it to completion -- and the
    id of the job it ran. The production loop is ``WorkerManager.run``; this is
    the single-threaded shape these tests assert against."""
    worker.store.expire()
    for expired in worker.store.list_jobs("quarantined"):
        if expired["phase"] == "lease_expired":
            try:
                worker.reconcile(expired["id"])
            except Exception:
                logging.getLogger(__name__).exception("Recovery failed for job %s", expired["id"])
    job = worker.store.claim(worker.worker_id, lease_s=worker.lease_s)
    if job is None:
        return None
    return worker.run_claimed(job)


@pytest.mark.parametrize("result,expected", [
    ({"status": "completed", "resolved": False}, None),
    ({"status": "error", "error": "HTTP 502 Bad Gateway"}, "infrastructure"),
    ({"status": "error", "error": "TransportClosedError"}, "infrastructure"),
    ({"status": "timeout", "error": "HTTP 504 during exhausted budget"}, "actor"),
    ({"status": "error", "error": "No patch produced"}, "actor"),
    ({"status": "error", "error": "Frozen dataset checksum changed"}, "configuration"),
])
def test_failure_classification(result, expected):
    assert failure_kind(result) == expected


def test_dead_leader_does_not_leave_active_descendants(tmp_path):
    ready = tmp_path / "ready"
    program = ("import subprocess,time,sys; from pathlib import Path; "
               "subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); "
               f"Path({str(ready)!r}).write_text('ready'); time.sleep(120)")
    process = subprocess.Popen([sys.executable, "-c", program], start_new_session=True)
    identity = process_identity(process.pid)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        process.kill()
        process.wait()
        assert stop_process(identity)
    finally:
        stop_process(identity)
        process.wait()


def test_wrong_process_start_identity_never_kills_another_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    identity = process_identity(process.pid)
    try:
        assert stop_process({**identity, "start_ticks": "wrong"})
        assert process.poll() is None
    finally:
        stop_process(identity)
        process.wait()


def test_bad_job_does_not_prevent_worker_claiming_next_job(store, tmp_path):
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {"default": {}}})
    first = store.submit(request(), "bad-one")
    second = store.submit(request(), "bad-two")
    assert run_once(worker) == first["id"]
    assert store.get(first["id"])["state"] == "failed"
    assert run_once(worker) == second["id"]
    assert store.get(second["id"])["state"] == "failed"


@pytest.mark.parametrize("legacy_file", ["{}", "malformed obsolete request"])
def test_reconcile_does_not_import_legacy_handoff_files(store, tmp_path, legacy_file):
    job = store.submit(request(), "fixture")
    claim = store.claim("old-worker")
    directory = tmp_path / job["id"] / claim["active_attempt"]
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(legacy_file)
    store.finish(job["id"], claim["lease_token"], {"error": "fixture"}, state="quarantined")
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {"default": {}}})
    assert not worker.reconcile(job["id"])
    assert store.get(job["id"])["state"] == "quarantined"
    assert (directory / "request.json").read_text() == legacy_file


def wait_file(path: Path) -> None:
    deadline = time.monotonic() + 15
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert path.exists(), str(path)


@pytest.mark.parametrize("kind", ["rollout", "grade", "review"])
def test_worker_freezes_payload_and_registers_pid_before_child_execution(store, tmp_path, monkeypatch, kind):
    profile = {"run_defaults": {"sandbox_image": "fixture", "backend": {"backend": "microvm"}},
               "review_defaults": {"temperature": 0.5}}
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {"default": profile}})
    if kind == "rollout":
        submitted = request()
    elif kind == "grade":
        submitted = JobSpec("grade", {"benchmark": "swe-rebench-v2", "instance_id": "fixture",
                            "snapshot_id": "snapshot", "dataset_path": "/fixture", "dataset_sha256": "fixture",
                            "grader_revision": "fixture", "parser_path": "/fixture/log_parsers.py",
                            "image": "task-image"})
    else:   # a review runs no sandbox; the worker tells it where the attempts live
        submitted = JobSpec("review", {"problem": "fixture", "model": "fixture", "model_endpoint": "http://127.0.0.1:1",
                            "api_key_env": "FIXTURE_KEY", "parents": [{
                                "name": "root", "job_id": "parent-job", "attempt_id": "parent-attempt",
                                "grade": {"resolved": False}}]})
    job = store.submit(submitted, "handoff")
    monkeypatch.setenv("SPROUT_RUNSTORE_DSN", "must-not-reach-child")
    monkeypatch.setenv("SPROUT_RUNSTORE_TOKEN", "must-not-reach-child")
    popen = subprocess.Popen
    heartbeat = store.heartbeat
    stages = []

    def launch(command, **kwargs):
        attempt = store.attempts(job["id"])[0]
        payload = store.payload(job["id"], attempt["id"])
        assert command[-1] == digest(payload)
        assert command[-2] == str(tmp_path / job["id"] / attempt["id"])
        assert "prompt" not in " ".join(command)
        assert not attempt["execution"].get("process")
        stages.append("frozen-before-spawn")
        return popen([*command[:2], "tests.runstore.payload_child", *command[3:]], **kwargs)

    def register(job_id, token, **kwargs):
        execution = kwargs.get("execution", {})
        if "process" in execution:
            directory = Path(execution["directory"])
            wait_file(directory / "bootstrap-ready")
            assert not (directory / "executed.json").exists()
            assert not (directory / "request.json").exists()
            assert not (directory / "permit.json").exists()
            (directory / "request.json").write_text("invalid old handoff")
            (directory / "permit.json").write_text("invalid old permit")
            assert not store.attempts(job_id)[0]["execution"].get("process")
            stages.append("child-blocked-before-registration")
        heartbeat(job_id, token, **kwargs)

    from sprout.runstore import worker as module
    send = module.send_payload

    def deliver(stream, data, **kwargs):
        attempt = store.attempts(job["id"])[0]
        assert process_identity(attempt["execution"]["process"]["pid"]) == attempt["execution"]["process"]
        assert data == module.encode_payload(attempt["payload"], job["id"], attempt["id"])
        stages.append("registered-before-send")
        send(stream, data, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(store, "heartbeat", register)
    monkeypatch.setattr(module, "send_payload", deliver)
    assert run_once(worker) == job["id"]
    assert store.get(job["id"])["state"] == "succeeded", store.get(job["id"])["error"]
    assert stages == ["frozen-before-spawn", "child-blocked-before-registration", "registered-before-send"]
    attempt = store.attempts(job["id"])[0]
    directory = tmp_path / job["id"] / attempt["id"]
    receipt = read_json(directory / "executed.json")
    assert receipt["payload"] == attempt["payload"]
    assert not {"SPROUT_RUNSTORE_DSN", "SPROUT_RUNSTORE_TOKEN"}.intersection(receipt["env_names"])
    assert (directory / "request.json").read_text() == "invalid old handoff"
    if kind == "review":
        effective = receipt["payload"]["effective_spec"]
        assert effective["artifact_root"] == str(tmp_path), "the worker tells a review where the attempts live"
        assert effective["temperature"] == 0.5, "a review takes the profile's review_defaults"


def test_a_stop_request_ends_a_running_attempt_at_once_and_for_good(store, tmp_path, monkeypatch):
    import threading

    job = store.submit(JobSpec("rollout", {"prompt": "hold-fixture"}, max_infra_retries=2), "stop-me")
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {"default": {
        "run_defaults": {"sandbox_image": "fixture", "backend": {"backend": "microvm"}}}}})
    popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        children.append(popen([*command[:2], "tests.runstore.payload_child", *command[3:]], **kwargs))
        return children[-1]

    monkeypatch.setattr(subprocess, "Popen", launch)
    thread = threading.Thread(target=run_once, args=(worker,))
    thread.start()
    try:
        attempt = None
        deadline = time.monotonic() + 15
        while attempt is None and time.monotonic() < deadline:
            attempt = (store.attempts(job["id"]) or [None])[0]
            time.sleep(0.02)
        wait_file(tmp_path / job["id"] / attempt["id"] / "executed.json")   # the child holds for 120 s
        started = time.monotonic()
        store.request_stop(job["id"])
        thread.join(timeout=15)
        assert not thread.is_alive() and time.monotonic() - started < 10
        assert children[0].poll() is not None, "the worker stopped the child, not this test"
    finally:
        for child in children:
            child.kill()
            child.wait(timeout=5)
        thread.join(timeout=5)
    finished = store.get(job["id"])
    assert finished["state"] == "failed" and finished["attempt_count"] == 1, "stopped, and not retried"
    assert finished["result"]["failure_kind"] == "stopped" and finished["result"]["stop_reason"] == "stopped"


@pytest.mark.parametrize("corrupt", [False, True])
def test_reconcile_uses_db_payload_and_not_files_or_current_profile(store, tmp_path, corrupt):
    job = store.submit(request(), "reconcile")
    claim = store.claim("old-worker")
    directory = tmp_path / job["id"] / claim["active_attempt"]
    payload = envelope(claim)
    store.freeze_payload(job["id"], claim["lease_token"], payload)
    store.finish(job["id"], claim["lease_token"], {"error": "fixture"}, state="quarantined")
    write_json(directory / "outcome.json", {"status": "completed"})
    (directory / "request.json").write_text("malformed obsolete request")
    (directory / "permit.json").write_text("malformed obsolete permit")
    if corrupt:
        with store.transaction() as cursor:
            cursor.execute("UPDATE rs_attempts SET payload_hash='corrupt' WHERE id=%s", (claim["active_attempt"],))
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {}})
    assert worker.reconcile(job["id"]) is not corrupt
    assert store.get(job["id"])["state"] == ("quarantined" if corrupt else "succeeded")
    assert (directory / "request.json").read_text() == "malformed obsolete request"


def crash_worker(dsn: str, root: str, window: str) -> None:
    instance = Store(dsn)
    controller = Worker(instance, {"artifact_root": root, "profiles": {"default": {
        "run_defaults": {"sandbox_image": "fixture", "backend": {"backend": "microvm"}}}}})
    popen = subprocess.Popen
    heartbeat = instance.heartbeat

    def pause() -> None:
        (Path(root) / "kill-worker-now").touch()
        time.sleep(120)

    def launch(command, **kwargs):
        process = popen([*command[:2], "tests.runstore.payload_child", *command[3:]], **kwargs)
        write_json(Path(root) / "child-identity.json", process_identity(process.pid))
        if window == "before-registration":
            wait_file(Path(command[-2]) / "bootstrap-ready")
            pause()
        return process

    def register(job_id, token, **kwargs):
        heartbeat(job_id, token, **kwargs)
        if window == "before-delivery" and "process" in kwargs.get("execution", {}):
            wait_file(Path(kwargs["execution"]["directory"]) / "bootstrap-ready")
            pause()

    subprocess.Popen = launch
    instance.heartbeat = register
    if window == "after-outcome":
        controller._publish = lambda *args: pause()
    run_once(controller)


@pytest.mark.parametrize("window", ["before-registration", "before-delivery", "during-execution", "after-outcome"])
def test_worker_sigkill_reconciles_without_handoff_files(store, tmp_path, window):
    prompt = "hold-fixture" if window == "during-execution" else "fixture"
    job = store.submit(request(prompt=prompt), "worker-crash")
    controller = multiprocessing.get_context("fork").Process(
        target=crash_worker, args=(store.dsn, str(tmp_path), window))
    controller.start()
    identity = None
    try:
        wait_file(tmp_path / "child-identity.json")
        identity = read_json(tmp_path / "child-identity.json")
        attempt = store.attempts(job["id"])[0]
        directory = tmp_path / job["id"] / attempt["id"]
        if window == "during-execution":
            wait_file(directory / "executed.json")
        else:
            wait_file(tmp_path / "kill-worker-now")
        controller.kill()
        controller.join(timeout=5)
        assert not controller.is_alive()
        assert controller.exitcode == -9
        if window in {"before-registration", "before-delivery"}:
            deadline = time.monotonic() + 5
            while process_identity(identity["pid"]) is not None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert process_identity(identity["pid"]) is None
            assert not (directory / "executed.json").exists()
        assert not (directory / "request.json").exists()
        assert not (directory / "permit.json").exists()
        with store.transaction() as cursor:
            cursor.execute("UPDATE rs_jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                           (job["id"],))
        assert store.expire() == [job["id"]]
        replacement = Worker(Store(store.dsn), {"artifact_root": str(tmp_path), "profiles": {}})
        assert replacement.reconcile(job["id"])
        assert store.get(job["id"])["state"] == ("succeeded" if window == "after-outcome" else "failed")
        assert store.payload(job["id"], attempt["id"]) == attempt["payload"]
        assert process_identity(identity["pid"]) is None
    finally:
        if controller.is_alive():
            controller.kill()
        controller.join(timeout=5)
        if identity:
            stop_process(identity)


@pytest.mark.parametrize("mode", ["empty", "truncated", "checksum", "identity"])
def test_child_invalid_stdin_never_executes_or_reads_old_files(tmp_path, mode):
    directory = tmp_path / "job" / "attempt"
    directory.mkdir(parents=True)
    payload = {"version": 1, "job_id": "job", "attempt_id": "attempt", "kind": "rollout",
               "effective_spec": {"prompt": "fixture"}, "profile_config": {}, "recovery": None}
    write_json(directory / "request.json", payload)
    write_json(directory / "permit.json", {"attempt_id": "attempt"})
    from sprout.runstore.specs import canonical

    data = canonical(payload).encode()
    fingerprint = digest(payload)
    if mode == "empty":
        data = b""
    elif mode == "truncated":
        data = data[:-1]
    elif mode == "checksum":
        fingerprint = "wrong"
    else:
        payload["attempt_id"] = "another"
        data = canonical(payload).encode()
        fingerprint = digest(payload)
    result = subprocess.run([sys.executable, "-m", "tests.runstore.payload_child", str(directory), fingerprint],
                            input=data, capture_output=True, timeout=15)
    assert result.returncode == 2, result.stderr.decode()
    assert not (directory / "executed.json").exists()
    assert not (directory / "outcome.json").exists()
    assert not (directory / "resources.jsonl").exists()


def test_failed_process_registration_never_delivers_payload(store, tmp_path, monkeypatch):
    job = store.submit(request(sandbox_image="fixture", backend={"backend": "microvm"}), "fenced-start")
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {"default": {}}})
    popen = subprocess.Popen

    def launch(command, **kwargs):
        process = popen([*command[:2], "tests.runstore.payload_child", *command[3:]], **kwargs)
        wait_file(Path(command[-2]) / "bootstrap-ready")
        return process

    def fenced(*args, **kwargs):
        raise Fenced("Process registration was not committed")

    def refuse_send(*args, **kwargs):
        pytest.fail("Payload delivery before successful process registration")

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(store, "heartbeat", fenced)
    monkeypatch.setattr("sprout.runstore.worker.send_payload", refuse_send)
    assert run_once(worker) == job["id"]
    attempt = store.attempts(job["id"])[0]
    assert store.get(job["id"])["state"] == "quarantined"
    assert attempt["payload"] and not attempt["execution"].get("process")
    assert not (tmp_path / job["id"] / attempt["id"] / "executed.json").exists()


def test_malformed_legacy_request_does_not_block_claiming_next_job(store, tmp_path):
    legacy = store.submit(request(), "legacy")
    claim = store.claim("old-worker")
    directory = tmp_path / legacy["id"] / claim["active_attempt"]
    directory.mkdir(parents=True)
    (directory / "request.json").write_text("malformed obsolete request")
    with store.transaction() as cursor:
        cursor.execute("UPDATE rs_jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE id=%s",
                       (legacy["id"],))
    next_job = store.submit(request(), "next")
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {"default": {}}})
    assert run_once(worker) == next_job["id"]
    assert store.get(legacy["id"])["state"] == "quarantined"
    assert len(store.attempts(next_job["id"])) == 1


def test_example_profiles_only_default_real_runspec_fields():
    """A stale default (a field RunSpec no longer has) would fail every queued
    rollout at submission; the shipped example must stay valid."""
    import json
    from dataclasses import fields
    from pathlib import Path

    from sprout.harness_orchestrator.run import RunSpec

    from sprout.runstore.specs import GradeSpec
    from sprout.runstore.worker import owned_backend

    config = json.loads((Path(__file__).resolve().parents[2] / "resources" / "runstore" / "config.example.json").read_text())
    allowed = {item.name for item in fields(RunSpec)}
    graded = {item.name for item in fields(GradeSpec)}
    for name, profile in config["profiles"].items():
        assert set(profile.get("run_defaults", {})) <= allowed, name
        assert set(profile.get("grade_defaults", {})) <= graded, name
    # podman is the default; every profile names a backend the worker takes.
    assert config["profiles"]["mini"]["run_defaults"]["backend"]["backend"] == "podman"
    assert config["profiles"]["grade"]["grade_defaults"]["backend"]["backend"] == "podman"
    from dataclasses import fields as spec_fields

    from sprout.runstore.specs import ReviewSpec

    reviewable = {item.name for item in spec_fields(ReviewSpec)}
    for name, profile in config["profiles"].items():
        defaults = profile.get("run_defaults") or profile.get("grade_defaults")
        if defaults is None:
            # A review profile runs no sandbox: it may only default the reviewer's settings.
            assert name == "review" and set(profile.get("review_defaults", {})) <= reviewable, name
            continue
        assert owned_backend("grade", defaults) in {"podman", "microvm"}, name
        section = defaults["backend"][defaults["backend"]["backend"]]
        assert section["runtime_bin"], f"{name}: a sandbox needs the runtime binary"
