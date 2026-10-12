"""Real TCP driver -> Run Store API -> isolated PostgreSQL contract checks.

Claims/results are supplied by a controlled worker peer. These checks do not
run agents, create VMs, or establish training-token export/native restore.
"""

from contextlib import contextmanager
import time

import httpx

from sprout.rl_driver.backend import RunStoreClient
from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Ledger
from sprout.rl_driver.server import create_app
from tests.rl_driver.test_driver import request
from tests.serving import serving
from sprout.runstore.api import create_app as runstore_app
from tests.runstore.test_store import store  # isolated schema; never a production database


@contextmanager
def execution_groups(url, token):
    """The driver's debugging interface, over plain HTTP: POST/GET/DELETE /execution-groups."""
    http = httpx.Client(base_url=url, timeout=30, headers={"Authorization": "Bearer " + token})

    class Groups:
        def submit(self, body):
            response = http.post("/execution-groups", json=body)
            response.raise_for_status()
            return response.json()["rollout_job_id"]

        def get(self, group_id):
            response = http.get(f"/execution-groups/{group_id}")
            response.raise_for_status()
            return response.json()

        def wait(self, group_id, *, timeout_s=5, interval_s=0.01):
            deadline = time.monotonic() + timeout_s
            while True:
                result = self.get(group_id)
                if result["ready"] or result["status"] == "quarantined":
                    return result
                if time.monotonic() >= deadline:
                    raise TimeoutError("Stopped waiting; group execution is unchanged")
                time.sleep(interval_s)

        def release(self, group_id):
            response = http.delete(f"/execution-groups/{group_id}")
            response.raise_for_status()
            return response.json()

    try:
        yield Groups()
    finally:
        http.close()


def claim(store, kind):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = store.claim("controlled-worker", kinds=(kind,))
        if job:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Driver did not submit a {kind} job")


class ControlledIndex:
    """Native/snapshot validity is already tested by Run Store's own suite."""
    def __init__(self):
        self.recoveries = []

    def points(self, attempt):
        return [point for point in self.recoveries if point["attempt_id"] == attempt]

    def get_point(self, identifier):
        return next(point for point in self.recoveries if point["id"] == identifier)

    def valid(self, point):
        return point["valid"]

    def tools(self, attempt, after, limit):
        return [{"depth": n, "response": {"output": "ok"}} for n in range(1, 4) if n > after][:limit]

    def add(self, job):
        self.recoveries.append({"id": "point-" + job["id"], "job_id": job["id"],
                               "attempt_id": job["active_attempt"], "valid": True,
                               "snapshot_id": "snapshot-final", "tool_depth": 3,
                               "message_step": 2, "native": {"slot": "mini-swe-agent"}})


def test_real_http_queue_grade_branch_and_driver_restart(store, tmp_path):
    index = ControlledIndex()
    profiles = {"actor": {"run_defaults": {"slot": "mini-swe-agent"}}, "grade": {}}
    with serving(runstore_app(store, "runstore-fixture", index=index, profiles=profiles)) as backend_url:
        backend = RunStoreClient(backend_url, "runstore-fixture")
        path = tmp_path / "driver.sqlite3"
        ledger = Ledger(path)
        ledger.bind(backend_url)
        driver = Driver(backend, ledger)
        try:
            with serving(create_app(driver, "driver-fixture", poll_interval_s=0.01)) as driver_url, \
                    execution_groups(driver_url, "driver-fixture") as groups:
                body = request(grade=True)
                group_id = groups.submit(body)
                actor = claim(store, "rollout")
                assert actor["request"]["context"]["rl_driver"]["sample_slot_id"] == "slot-0"
                index.add(actor)
                store.append_events(actor["id"], actor["lease_token"], [
                    {"seq": 1, "type": "tool.started", "call_id": "call-1"},
                    {"seq": 2, "type": "tool.finished", "call_id": "call-1", "output": "ok"},
                ])
                store.finish(actor["id"], actor["lease_token"], {"status": "completed", "final_text": "patch"})
                grade = claim(store, "grade")
                assert grade["request"]["spec"]["snapshot_id"] == "snapshot-final"
                assert grade["request"]["context"]["rl_driver"]["actor_attempt_id"] == actor["active_attempt"]
                store.finish(grade["id"], grade["lease_token"], {"status": "completed", "resolved": False})
                result = groups.wait(group_id)
                assert result["status"] == "completed"
                assert result["samples"][0]["grade"]["result"]["resolved"] is False
                # Events and recovery points are Run Store's to serve; the driver only names the job.
                actor_job = result["samples"][0]["actor"]["job_id"]
                assert actor_job == actor["id"]
                point = backend.points(actor_job, attempt_id=actor["active_attempt"])[0]
                assert point["available"] is True
                groups.release(group_id)
            restored = Driver(backend, Ledger(path))
            with serving(create_app(restored, "driver-fixture", poll_interval_s=0.01)) as driver_url, \
                    execution_groups(driver_url, "driver-fixture") as groups:
                assert groups.submit(body) == group_id
                assert groups.get(group_id)["acknowledged_at"] is not None
                assert len(store.list_jobs()) == 2
                branch_body = {"rollout_job_id": "branch-group", "prompt_group_id": "prompt", "samples": [{
                    "sample_slot_id": "child-0", "branch": {"job_id": actor["id"], "point_id": point["id"],
                                                              "continuation": {"guidance": "user-hint",
                                                                               "hint": "Try a different fix"}},
                }]}
                groups.submit(branch_body)
                child = claim(store, "rollout")
                assert child["request"]["parent_point"] == point["id"]
                assert child["request"]["continuation"] == {"guidance": "user-hint", "hint": "Try a different fix"}
                assert child["request"]["spec"] == actor["request"]["spec"], "a branch keeps its parent's run"
                store.finish(child["id"], child["lease_token"], {"status": "completed"})
                assert groups.wait("branch-group")["status"] == "completed"
                assert len(store.list_jobs()) == 3
        finally:
            backend.close()


def test_real_http_cancel_waits_for_running_execution(store, tmp_path):
    with serving(runstore_app(store, "fixture")) as url:
        backend = RunStoreClient(url, "fixture")
        driver = Driver(backend, Ledger(tmp_path / "driver.sqlite3"))
        try:
            with serving(create_app(driver, "fixture", poll_interval_s=0.01)) as driver_url, \
                    execution_groups(driver_url, "fixture") as groups:
                group_id = groups.submit(request())
                actor = claim(store, "rollout")
                assert groups.release(group_id)["status"] == "cancelling"
                assert store.get(actor["id"])["state"] == "running"
                assert not groups.get(group_id)["ready"]
                store.finish(actor["id"], actor["lease_token"], {"status": "completed"})
                assert groups.wait(group_id)["status"] == "cancelled"
        finally:
            backend.close()


def test_worker_bounds_child_startup_by_remaining_group_deadline(store, tmp_path, monkeypatch):
    from pathlib import Path
    from sprout.runstore.specs import JobSpec, digest
    from sprout.runstore.worker import Worker
    from tests.runstore.test_worker import run_once

    seen = []
    def before_launch(command, **kwargs):
        directory = Path(command[-2])
        payload = store.payload(directory.parent.name, directory.name)
        assert command[-1] == digest(payload)
        seen.append(payload)
        raise RuntimeError("Controlled probe stops before child execution")
    monkeypatch.setattr("sprout.runstore.worker.subprocess.Popen", before_launch)
    spec = {"prompt": "task", "slot": "mini-swe-agent", "sandbox_image": "fixture",
            "backend": {"backend": "microvm"}, "timeout_s": 3600,
            "extra": {"rollout_contract": {"deadline_at": time.time() + 5}}}
    first = store.submit(JobSpec("rollout", spec), "deadline")
    worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {"default": {}}})
    run_once(worker)
    assert 0 < seen[0]["effective_spec"]["timeout_s"] <= 5
    assert store.get(first["id"])["request"]["spec"]["timeout_s"] == 3600
    spec["extra"]["rollout_contract"]["deadline_at"] = time.time() - 1
    second = store.submit(JobSpec("rollout", spec), "expired")
    run_once(worker)
    assert len(seen) == 1
    assert "expired before execution" in store.get(second["id"])["error"]
