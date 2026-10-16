from copy import deepcopy
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from sprout.rl_driver.backend import RunStoreClient
from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Conflict, Ledger, canonical
from sprout.rl_driver.server import DEFAULT_PORT, create_app
from sprout.rl_driver.specs import validate_request
from sprout.runstore.specs import JobSpec


def request(group="group", *, grade=False):
    sample = {"sample_slot_id": "slot-0", "run": {"profile": "actor", "spec": {
        "prompt": "Fix the parser", "slot": "mini-swe-agent", "sandbox_image": "prepared-image", "timeout_s": 30,
    }}}
    if grade:
        sample["grade"] = {"profile": "grade", "spec": {
            "benchmark": "swe-rebench-v2", "instance_id": "task", "dataset_path": "/worker/task.json",
            "dataset_sha256": "123", "grader_revision": "sha256:abc", "timeout_s": 60,
            "parser_path": "/worker/log_parsers.py", "image": "docker.io/swerebenchv2/task",
        }}
    return {"rollout_job_id": group, "prompt_group_id": "prompt", "samples": [sample]}


class QueueHTTP:
    """An HTTP contract peer; only explicit finish() simulates a worker."""
    def __init__(self):
        self.jobs = {}
        self.keys = {}
        self.requests = []
        self.lose_ack = False
        self.tools = {}
        self.points = {}

    def __call__(self, req):
        body = json.loads(req.content) if req.content else None
        path = req.url.path
        self.requests.append((req.method, path, body, dict(req.url.params)))
        if req.method == "POST" and (path == "/v1/jobs" or path.endswith("/branch")):
            key = req.headers["Idempotency-Key"]
            if key in self.keys:
                job_id, previous = self.keys[key]
                assert previous == canonical(body), "Retry changed the submitted job"
            else:
                job_id = f"job-{len(self.jobs)}"
                kind = JobSpec.from_dict(body).kind if path == "/v1/jobs" else "rollout"
                self.jobs[job_id] = {"id": job_id, "kind": kind, "state": "queued", "phase": "queued",
                                     "idempotency_key": key, "active_attempt": None, "result": None, "error": None}
                self.keys[key] = (job_id, canonical(body))
            if self.lose_ack:
                self.lose_ack = False
                raise httpx.ReadError("Reply lost after durable acceptance", request=req)
            return httpx.Response(202, json={"id": job_id})
        job_id = path.split("/")[3]
        job = self.jobs[job_id]
        if req.method == "GET" and path == f"/v1/jobs/{job_id}":
            return httpx.Response(200, json=job)
        if path.endswith("/stop"):   # Store.request_stop: the worker stops a running job later
            if job["state"] == "queued":
                job["state"] = "cancelled"
            elif job["state"] in {"running", "quarantined"}:
                job["stop_requested"] = True
            return httpx.Response(200, json=job)
        if path.endswith("/tools"):
            after = int(req.url.params.get("after", 0))
            # Small pages ensure the real adapter must follow the cursor.
            rows = [row for row in self.tools.get(job_id, []) if row["depth"] > after][:2]
            return httpx.Response(200, json=rows)
        if path.endswith("/recovery-points"):
            return httpx.Response(200, json=self.points.get(job_id, []))
        raise AssertionError((req.method, path))

    def finish(self, job_id, *, resolved=None, **facts):
        """Settle a job as succeeded. With ``resolved``, a grade: beside its verdict the grader
        reports its progress and its facts about the task's hidden tests (``grading.patch_facts``),
        none by default, each overridable through ``facts``."""
        result = {"status": "completed", "resolved": resolved}
        if resolved is not None:
            result.update(progress=1.0 if resolved else 0.0, test_patch_paths=[], touched_test_paths=[],
                          fail_to_pass=[])
        result.update(facts)
        self.jobs[job_id].update(state="succeeded", active_attempt=f"attempt-{job_id}", result=result)

    def final_point(self, job_id):
        self.tools[job_id] = [{"depth": n, "response": {"output": "ok"}} for n in range(1, 4)]
        self.points[job_id] = [
            {"id": "old", "tool_depth": 2, "message_step": 1, "snapshot_id": "AB", "available": True},
            {"id": "last", "tool_depth": 3, "message_step": 2, "snapshot_id": "ABC", "available": True},
        ]


@pytest.fixture
def peer():
    peer = QueueHTTP()
    client = RunStoreClient("http://runstore", "fixture")
    client.http.close()
    client.http = httpx.Client(base_url="http://runstore", transport=httpx.MockTransport(peer))
    try:
        yield peer, client
    finally:
        client.close()


def test_two_samples_http_polling_out_of_order_and_consumption_survives_restart(tmp_path, peer):
    queue, client = peer
    path = tmp_path / "ledger.sqlite3"
    driver = Driver(client, Ledger(path))
    body = request()
    second = deepcopy(body["samples"][0])
    second["sample_slot_id"] = "slot-1"
    body["samples"].append(second)
    with TestClient(create_app(driver, "driver", background=False)) as http:
        assert http.post("/execution-groups", json=body).status_code == 401
        http.headers["Authorization"] = "Bearer driver"
        assert http.post("/execution-groups", json=body).json()["status"] == "queued"
        driver.tick()
        assert len(queue.jobs) == 2
        queue.finish("job-1")
        driver.tick()
        assert not http.get("/execution-groups/group").json()["ready"]
        queue.finish("job-0")
        driver.tick()
        view = http.get("/execution-groups/group/result").json()
        assert view["status"] == "completed"
        assert [s["actor"]["job_id"] for s in view["samples"]] == ["job-0", "job-1"]
        assert "submission" not in view["samples"][0]["actor"]
        assert view["samples"][0]["actor"]["attempt_id"] == "attempt-job-0"
        assert http.get("/execution-groups/group/samples/slot-0/events").status_code == 404
        assert http.delete("/execution-groups/group").status_code == 200
        changed = deepcopy(body)
        changed["samples"][0]["run"]["spec"]["prompt"] = "different"
        assert http.post("/execution-groups", json=changed).status_code == 409
    restored = Driver(client, Ledger(path))
    assert restored.submit(body)["status"] == "completed"
    assert restored.get("group")["acknowledged_at"] is not None
    restored.tick()
    assert len(queue.jobs) == 2
    assert DEFAULT_PORT == 11001


@pytest.mark.parametrize("branch", [False, True])
def test_lost_submit_reply_restarts_with_same_idempotency_key(tmp_path, peer, branch):
    queue, client = peer
    body = request()
    if branch:
        body["samples"][0].pop("run")
        body["samples"][0]["branch"] = {"job_id": "parent", "point_id": "point",
                                        "continuation": {"guidance": "user-hint", "hint": "retry"},
                                        "budget": {"max_turns": 7}}
    path = tmp_path / "ledger.sqlite3"
    driver = Driver(client, Ledger(path))
    driver.submit(body)
    queue.lose_ack = True
    driver.tick()
    assert len(queue.jobs) == 1
    assert driver.get("group")["samples"][0]["actor"]["state"] == "submitting"
    restored = Driver(client, Ledger(path))
    restored.tick()
    assert len(queue.jobs) == 1
    assert restored.get("group")["samples"][0]["actor"]["job_id"] == "job-0"
    if branch:
        sent = [row[2] for row in queue.requests if row[0] == "POST" and row[1] == "/v1/jobs/parent/branch"]
        assert sent and all(row == {"point_id": "point", "overrides": {}, "budget": {"max_turns": 7},
                                    "continuation": {"guidance": "user-hint", "hint": "retry"}} for row in sent)


@pytest.mark.parametrize("branch, message", [
    ({"job_id": "parent", "point_id": "point"}, "A continuation is an object"),
    ({"job_id": "parent", "point_id": "point", "continuation": {"guidance": "none"},
      "overrides": {"prompt": "Try a different fix"}}, "branch overrides"),
    ({"job_id": "parent", "point_id": "point", "continuation": {"guidance": "assistant-turn"}},
     "needs its assistant_turn"),
    ({"job_id": "parent", "point_id": "point", "continuation": {"guidance": "none"}, "budget": {"turns": 3}},
     "sets max_turns"),
])
def test_a_branch_sample_says_what_the_actor_is_given_next(branch, message):
    body = request()
    body["samples"][0].pop("run")
    body["samples"][0]["branch"] = branch
    with pytest.raises(ValueError, match=message):
        validate_request(body)


def test_cancel_does_not_claim_running_job_stopped_or_launch_grading(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    body = request(grade=True)
    body["samples"].append({"sample_slot_id": "slot-1", "run": deepcopy(body["samples"][0]["run"])})
    driver.submit(body)
    driver.tick()
    queue.jobs["job-0"].update(state="running", active_attempt="attempt-job-0")
    assert driver.release("group")["status"] == "cancelling"
    driver.tick()
    assert queue.jobs["job-0"]["state"] == "running" and queue.jobs["job-0"]["stop_requested"], \
        "a running job is asked to stop, and the group waits for the worker to say it did"
    assert queue.jobs["job-1"]["state"] == "cancelled"
    assert not driver.get("group")["ready"]
    queue.finish("job-0")
    driver.tick()
    assert driver.get("group")["status"] == "cancelled"
    assert driver.get("group")["ready"]
    assert len(queue.jobs) == 2


def test_cancel_before_dispatch_never_submits(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request(grade=True))
    driver.release("group")
    driver.tick()
    assert driver.get("group")["status"] == "cancelled"
    assert queue.requests == []


def test_cancel_lost_ack_does_not_repost_possibly_new_work(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request())
    queue.lose_ack = True
    driver.tick()
    driver.release("group")
    calls = len(queue.requests)
    driver.tick()
    assert len(queue.requests) == calls
    view = driver.get("group")
    assert view["status"] == "cancelling" and not view["ready"]
    assert "acknowledgement unknown" in view["samples"][0]["actor"]["last_error"]
    queue.jobs["wrong-job"] = {**queue.jobs["job-0"], "id": "wrong-job", "idempotency_key": "different"}
    with pytest.raises(ValueError, match="idempotency key"):
        driver.reconcile_submission("group", "slot-0", "actor", "wrong-job")
    driver.reconcile_submission("group", "slot-0", "actor", "job-0")
    driver.tick()
    assert queue.jobs["job-0"]["state"] == "cancelled"
    assert driver.get("group")["ready"]
    assert not [r for r in queue.requests[calls:] if r[0] == "POST" and r[1] == "/v1/jobs"]


def test_cancellation_during_grade_preparation_does_not_submit_grader(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request(grade=True))
    driver.tick()
    queue.finish("job-0")
    queue.final_point("job-0")
    original = client.points

    def points(*args, **kwargs):
        driver.release("group")
        return original(*args, **kwargs)

    client.points = points
    driver.tick()
    assert driver.get("group")["status"] == "cancelled"
    assert len(queue.jobs) == 1


@pytest.mark.parametrize("resolved", [True, False])
def test_grade_uses_final_snapshot_and_unresolved_is_completed(tmp_path, peer, resolved):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request(grade=True))
    driver.tick()
    queue.finish("job-0")
    queue.final_point("job-0")
    driver.tick()
    grades = [body for method, path, body, _ in queue.requests if method == "POST" and body.get("kind") == "grade"]
    assert grades[0]["spec"]["snapshot_id"] == "ABC"
    assert grades[0]["context"]["rl_driver"]["actor_attempt_id"] == "attempt-job-0"
    queue.finish("job-1", resolved=resolved)
    driver.tick()
    view = driver.get("group")
    assert view["status"] == "completed"
    assert view["samples"][0]["grade"]["result"]["resolved"] is resolved
    assert len(queue.jobs) == 2


def test_unavailable_final_message_never_grades_an_older_snapshot(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request(grade=True))
    driver.tick()
    queue.finish("job-0")
    queue.final_point("job-0")
    queue.points["job-0"][-1]["available"] = False
    driver.tick()
    view = driver.get("group")
    assert view["status"] == "failed"
    assert "older snapshot" in view["samples"][0]["grade"]["error"]
    assert len(queue.jobs) == 1


def test_grade_lost_ack_keeps_original_snapshot_on_restart(tmp_path, peer):
    queue, client = peer
    path = tmp_path / "ledger.sqlite3"
    driver = Driver(client, Ledger(path))
    driver.submit(request(grade=True))
    driver.tick()
    queue.finish("job-0")
    queue.final_point("job-0")
    queue.lose_ack = True
    driver.tick()
    queue.points["job-0"] = []
    Driver(client, Ledger(path)).tick()
    assert len(queue.jobs) == 2
    assert driver.get("group")["samples"][0]["grade"]["job_id"] == "job-1"


def test_quarantine_is_visible_and_reconciliation_does_not_rerun_from_driver(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request())
    driver.tick()
    queue.jobs["job-0"]["state"] = "quarantined"
    driver.tick()
    assert driver.get("group")["status"] == "quarantined"
    assert not driver.get("group")["ready"]
    queue.finish("job-0")
    driver.tick()
    assert driver.get("group")["status"] == "completed"
    assert len(queue.jobs) == 1


def test_one_broken_group_does_not_block_another(tmp_path, peer):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    driver.submit(request("first"))
    driver.submit(request("second"))
    queue.lose_ack = True
    driver.tick()
    assert len(queue.jobs) == 2
    assert driver.get("second")["samples"][0]["actor"]["job_id"] == "job-1"


@pytest.mark.parametrize("mutate", [
    lambda body: body.update(protocol_version="anything"),  # no version field any more
    lambda body: body.update(budgets={"max_model_calls": 2}),
    lambda body: body["samples"].append(deepcopy(body["samples"][0])),
    lambda body: body["samples"][0]["run"]["spec"].update(sampling_params={"temperature": 1}),
    lambda body: body["samples"][0]["run"]["spec"].update(api_key="inline-secret"),
])
def test_unsupported_contracts_rejected_before_dispatch(tmp_path, peer, mutate):
    queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger.sqlite3"))
    body = request()
    mutate(body)
    with pytest.raises((ValueError, TypeError)):
        driver.submit(body)
    driver.tick()
    assert not queue.requests


def test_ledger_refuses_second_controller_and_different_backend(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.bind("http://original")
    with pytest.raises(Conflict):
        ledger.bind("http://different")
    with ledger.owner():
        with pytest.raises(Conflict):
            with Ledger(ledger.path).owner():
                pytest.fail("Second controller acquired the ledger")


def test_cli_wires_original_port_and_separate_runstore_url(tmp_path, monkeypatch):
    from sprout.rl_driver import __main__ as command

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"ledger": "state.sqlite3"}))
    monkeypatch.setenv("SPROUT_RUNSTORE_TOKEN", "backend-fixture")
    monkeypatch.setenv("SPROUT_RL_DRIVER_TOKEN", "driver-fixture")
    monkeypatch.setattr("sys.argv", ["rl_driver", "--config", str(config)])
    seen = {}
    monkeypatch.setattr(command.uvicorn, "run", lambda app, **kwargs: seen.update(app=app, **kwargs))
    command.main()
    assert seen["port"] == 11001
    assert seen["host"] == "127.0.0.1"
    assert "/execution-groups" in {route.path for route in seen["app"].routes}
    with Ledger(tmp_path / "state.sqlite3").connection() as db:
        assert db.execute("SELECT value FROM driver_meta WHERE name='backend_url'").fetchone()[0] == "http://127.0.0.1:18110"


def test_the_shipped_config_starts_the_driver_with_its_ledger_outside_the_source_tree(tmp_path, monkeypatch):
    """docs/RL_DRIVER.md starts the driver on the example in place, and a relative
    ledger is resolved against the config's own directory: inside the package."""
    from pathlib import Path
    from sprout.rl_driver import __main__ as command

    example = Path(__file__).resolve().parents[2] / "resources" / "rl_driver" / "config.example.json"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SPROUT_RUNSTORE_TOKEN", "backend-fixture")
    monkeypatch.setattr("sys.argv", ["rl_driver", "--config", str(example)])
    named = []
    # Wherever the example points, the ledger this test opens is in tmp_path.
    monkeypatch.setattr(command, "Ledger", lambda path: named.append(Path(path)) or Ledger(tmp_path / "ledger.sqlite3"))
    seen = {}
    monkeypatch.setattr(command.uvicorn, "run", lambda app, **kwargs: seen.update(app=app, **kwargs))
    command.main()
    assert seen["port"] == DEFAULT_PORT, "the whole example is accepted, its Miles section included"
    (ledger,) = named
    assert ledger.is_absolute() and not ledger.is_relative_to(Path(__file__).resolve().parents[2]), ledger


def test_example_is_valid_and_package_has_no_execution_calls():
    import ast
    from pathlib import Path
    from sprout.rl_driver.specs import validate_request

    directory = Path(__file__).resolve().parents[2] / "src" / "sprout" / "rl_driver"
    example = Path(__file__).resolve().parents[2] / "examples" / "rl_driver" / "execution-group.example.json"
    body = validate_request(json.loads(example.read_text()))
    assert len(body["samples"]) == 2
    # The driver validates requests but never executes: from the harness it may
    # use only the contracts enforced at run time -- the rollout contract
    # (sampling validation) and what a branch adds after its restored history.
    pure_contracts = {"sprout.harness_orchestrator.rollout", "sprout.harness_agents.mini_swe.continuation"}
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module not in pure_contracts:
                assert not (node.module or "").startswith(("sprout.harness", "sprout_sandbox", "swebench", "sprout.runstore.worker", "sprout.runstore.store"))
            if isinstance(node, ast.Import):
                assert not any(alias.name in {"subprocess", "psycopg2", "docker"} for alias in node.names)
