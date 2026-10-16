from copy import deepcopy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from sprout.contract.sampling import parameters
from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Ledger
from sprout.rl_driver.messages import MessageRequest
from sprout.rl_driver.messages import MessageAdapter
from sprout.rl_driver.server import create_app
from tests.rl_driver.test_driver import peer as peer
from sprout.contract.messages import clean_messages, mark_hint
from sprout.worker.jobs.grade import resolved


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "rl_driver" / "request.example.json"
RESULT_EXAMPLE = EXAMPLE.with_name("result.example.json")


def request():
    """The request docs/RL_DRIVER.md points Miles at: the tests drive the shipped example."""
    return json.loads(EXAMPLE.read_text())


def config(body):
    return {
        "profile": "mini", "run_defaults": {"slot": "mini-swe-agent"},
        "image_resources": {"cpu": 2, "memory_mb": 12288},
        "tasks": {body["task_id"]: {"grade": {"profile": "grade", "spec": {
            "benchmark": "swe-rebench-v2", "instance_id": body["task_id"],
            "dataset_path": "/sprout/tasks.jsonl", "dataset_sha256": "fixture",
            "grader_revision": "sha256:fixture", "parser_path": "/sprout/log_parsers.py",
        }}}},
    }


def messages():
    return [
        {"role": "user", "content": "fix task"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call", "type": "function", "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
        }]},
        {"role": "tool", "tool_call_id": "call", "content": "/workspace"},
        {"role": "user", "content": mark_hint("check the parser")},
        {"role": "assistant", "content": "fixed"},
    ]


def exported(**provenance):
    """What a worker's training export attaches to a scratch rollout's result:
    the native history in cleaned form, and which of it no policy sampled."""
    return {"training_messages": clean_messages(messages()),
            "training_provenance": {"prefix_messages": 0, "inserted_messages": [], **provenance}}


def test_real_miles_request_to_queue_grade_and_message_response(tmp_path, peer):
    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, config(body))
    with TestClient(create_app(driver, None, miles=adapter, background=False)) as http:
        assert http.post("/rollout-groups", json=body).status_code == 202
        driver.tick()
        actors = list(queue.jobs)
        for job_id in actors:
            queue.finish(job_id)
            queue.jobs[job_id]["result"].update(exported())
            queue.final_point(job_id)
        driver.tick()
        for job_id, job in queue.jobs.items():
            if job["kind"] == "grade":
                queue.finish(job_id, resolved=True)
        driver.tick()
        result = http.get("/rollout-groups/" + body["rollout_job_id"]).json()
        assert result["status"] == "completed", result
        assert result["actual_samples"] == body["max_samples"]
        # The shipped result example is what Miles's importer is tested against;
        # the adapter must produce exactly that shape for the shipped request.
        assert result == json.loads(RESULT_EXAMPLE.read_text())
        trajectory = result["trajectories"][0]
        assert trajectory["reward"] == 1.0 and trajectory["hints_removed"]
        assert "check the parser" not in json.dumps(trajectory["messages"])
        assert "/workspace" in json.dumps(trajectory["messages"])
        assert "token_ids" not in trajectory
        assert http.post("/rollout-groups", json=body).json()["status"] == "completed"
        deleted = http.delete("/rollout-groups/" + body["rollout_job_id"])
        assert deleted.json()["status"] == "completed"
        assert len(queue.jobs) == len(actors) * 2
    submissions = [row[2] for row in queue.requests if row[0] == "POST" and row[1] == "/v1/jobs"]
    actor = next(row for row in submissions if row["kind"] == "rollout")
    assert actor["spec"]["sandbox_image"] == "docker.io/swerebenchv2/task:base"
    assert actor["spec"]["extra"]["rollout_contract"]["sampling_params"]["top_k"] == -1, "off is forwarded, not dropped"
    assert actor["spec"]["extra"]["rollout_contract"]["message_export"] is True
    assert actor["spec"]["extra"]["rollout_contract"]["max_turns"] == body["max_turns"]


@pytest.mark.parametrize("group_size", [1, 2, 8])
def test_every_queued_trajectory_receives_the_full_turn_limit(tmp_path, peer, group_size):
    queue, client = peer
    body = request()
    body.update(max_samples=group_size, minimum_returned_samples=group_size, max_turns=2)
    body["sample_slots"] = [
        {"sample_slot_id": f"slot-{i}", "sample_index": i} for i in range(group_size)
    ]
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    actors = [row[2] for row in queue.requests if row[0] == "POST" and row[1] == "/v1/jobs"]
    assert len(actors) == group_size
    controls = [actor["spec"]["extra"]["rollout_contract"] for actor in actors]
    assert all(control["max_turns"] == 2 for control in controls)
    assert all(not ({"max_model_calls", "max_tool_calls"} & control.keys()) for control in controls)
    assert len({control["deadline_at"] for control in controls}) == 1
    assert all(actor["spec"]["timeout_s"] == body["budgets"]["max_wall_time_seconds"] for actor in actors)


@pytest.mark.parametrize("reason", ["timeout", "max_turns_reached"])
@pytest.mark.parametrize("resolved", [False, True])
def test_limit_cutoff_is_graded_after_deadline_and_returned_as_truncated(tmp_path, peer, monkeypatch, reason, resolved):
    from sprout.rl_driver.messages import internal_id

    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    (actor_id,) = queue.jobs
    queue.finish(actor_id)
    queue.jobs[actor_id]["result"].update(
        status="truncated", stop_reason=reason, final_snapshot_id="actual-final-state", **exported(),
    )
    row = driver.ledger.get(internal_id(body["rollout_job_id"]))
    document = row["document"]
    # Reaching the execution deadline must not cancel grading.
    execution_deadline = document["execution_deadline_at"]
    assert document["deadline_at"] == execution_deadline + 1800.0
    monkeypatch.setattr("sprout.rl_driver.driver.time.time", lambda: execution_deadline + 1)
    driver.tick()
    grades = [jid for jid, job in queue.jobs.items() if job["kind"] == "grade"]
    assert len(grades) == 1
    submission = next(r[2] for r in queue.requests if r[0] == "POST" and r[1] == "/v1/jobs"
                      and r[2]["kind"] == "grade")
    assert submission["spec"]["snapshot_id"] == "actual-final-state"
    queue.finish(grades[0], resolved=resolved)
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed", result
    trajectory = result["trajectories"][0]
    assert trajectory["status"] == "truncated" and trajectory["stop_reason"] == reason
    assert trajectory["reward"] == float(resolved)
    assert trajectory["metadata"]["graded_snapshot_id"] == "actual-final-state"


@pytest.mark.parametrize("change, reason", [
    # A hint the export left in place is refused, not stripped here: stripping
    # it would shift every index the provenance names.
    (lambda result: result.update(training_messages=messages()), "cleaned training form"),
    (lambda result: result.pop("training_provenance"), "which exported messages the policy sampled"),
])
def test_an_export_that_cannot_be_trained_on_fails_its_trajectory(tmp_path, peer, change, reason):
    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    (actor_id,) = queue.jobs
    queue.finish(actor_id)
    queue.jobs[actor_id]["result"].update(exported())
    change(queue.jobs[actor_id]["result"])
    queue.final_point(actor_id)
    driver.tick()
    (grade_id,) = [job_id for job_id, job in queue.jobs.items() if job["kind"] == "grade"]
    queue.finish(grade_id, resolved=True)
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "failed" and result["trajectories"] == []
    assert reason in result["stop_reason"]


def test_infrastructure_failure_is_not_graded_as_a_timeout(tmp_path, peer):
    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    (actor_id,) = queue.jobs
    queue.jobs[actor_id].update(
        state="failed", active_attempt="a",
        result={"status": "error", "failure_kind": "infrastructure", "error": "execution_uncertain"},
    )
    driver.tick()
    assert all(job["kind"] != "grade" for job in queue.jobs.values())
    assert adapter.get(body["rollout_job_id"])["status"] == "failed"


def graded_group(tmp_path, peer, settle_grade, monkeypatch=None):
    """One root through its actor; ``settle_grade(queue, grade_id)`` decides the grade."""
    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    (actor_id,) = queue.jobs
    queue.finish(actor_id)
    queue.jobs[actor_id]["result"].update(exported())
    queue.final_point(actor_id)
    driver.tick()
    (grade_id,) = [job_id for job_id, job in queue.jobs.items() if job["kind"] == "grade"]
    settle_grade(queue, grade_id, driver)
    driver.tick()
    return adapter.get(body["rollout_job_id"]), queue.jobs[grade_id]


def assert_missing(trajectory, reason):
    """A trajectory returned without a verdict: the wire's 0.0, and nothing a grade would say."""
    assert trajectory["reward"] == 0.0 and trajectory["metadata"]["zero_reason"] == reason
    assert trajectory["metadata"]["progress"] is None, "a missing grade is not a 0"
    assert trajectory["metadata"]["touched_test_paths"] is None


def test_a_failed_grade_marks_its_trajectory_missing_instead_of_failing_the_group(tmp_path, peer):
    def fail(queue, grade_id, driver):
        queue.jobs[grade_id].update(state="failed", active_attempt="g", error="sandbox did not start",
                                    result={"status": "error", "failure_kind": "infrastructure"})
    result, _ = graded_group(tmp_path, peer, fail)
    assert result["status"] == "completed", result
    (trajectory,) = result["trajectories"]
    assert_missing(trajectory, "grade_failed")


def test_a_grade_past_its_wall_time_is_given_up_as_missing_without_waiting(tmp_path, peer, monkeypatch):
    import sprout.rl_driver.driver as driver_module

    def hang(queue, grade_id, driver):
        queue.jobs[grade_id].update(state="running", active_attempt="g")
        driver.tick()   # seen running: its wall time starts now, not at submission
        now = driver_module.time.time()
        monkeypatch.setattr(driver_module.time, "time", lambda: now + 601)
    result, job = graded_group(tmp_path, peer, hang)
    assert job["state"] == "running" and job["stop_requested"], "Run Store is asked to stop it; its verdict is given up"
    assert result["status"] == "completed", result
    (trajectory,) = result["trajectories"]
    assert_missing(trajectory, "grade_expired")


def test_a_grade_that_succeeded_without_a_verdict_is_missing_too(tmp_path, peer):
    result, _ = graded_group(tmp_path, peer, lambda queue, grade_id, driver: queue.finish(grade_id))
    assert result["status"] == "completed", result
    (trajectory,) = result["trajectories"]
    assert_missing(trajectory, "grade_no_verdict")


def test_the_test_files_a_graded_diff_touched_reach_the_trajectory(tmp_path, peer):
    result, _ = graded_group(tmp_path, peer, lambda queue, grade_id, driver: queue.finish(
        grade_id, resolved=True, touched_test_paths=["t.py"]))
    (trajectory,) = result["trajectories"]
    assert trajectory["reward"] == 1.0 and trajectory["metadata"]["zero_reason"] is None
    assert trajectory["metadata"]["touched_test_paths"] == ["t.py"]
    assert list(trajectory["metadata"])[-1] == "touched_test_paths"


def test_a_queued_grade_is_not_charged_its_queue_time(tmp_path, peer, monkeypatch):
    import sprout.rl_driver.driver as driver_module

    def wait(queue, grade_id, driver):
        now = driver_module.time.time()
        monkeypatch.setattr(driver_module.time, "time", lambda: now + 601)
    result, job = graded_group(tmp_path, peer, wait)
    assert job["state"] == "queued" and result["status"] == "running"


def test_the_grade_template_never_retries_and_fits_its_tests_in_the_wall_time(tmp_path, peer):
    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, config(body))
    adapter.submit(body)
    (sample,) = driver.ledger.get(driver.ledger.active()[0])["document"]["samples"]
    template = sample["grade"]["template"]
    assert template["max_infra_retries"] == 0 and template["spec"]["timeout_s"] == 600 - 120
    with pytest.raises(ValueError, match="grade_wall_seconds"):
        MessageAdapter(driver, {**config(body), "grade_wall_seconds": 100})


def test_a_failed_actor_is_named_so_the_trainer_can_count_it(tmp_path, peer):
    queue, client = peer
    body = request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, config(body))
    adapter.submit(body)
    driver.tick()
    (actor_id,) = queue.jobs
    queue.jobs[actor_id].update(state="failed", active_attempt="a",
                                result={"status": "error", "failure_kind": "infrastructure"})
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    (slot,) = body["sample_slots"]
    assert result["failed_samples"] == [{"sample_slot_id": slot["sample_slot_id"], "reason": "Actor did not complete"}]


@pytest.mark.parametrize("mutate", [
    lambda body: body.update(max_turns=None),
    lambda body: body.update(max_turns=0),
    lambda body: body.update(max_turns=-1),
    lambda body: body.update(max_turns=True),
    lambda body: body.update(max_turns=1.5),
    # RL_DRIVER.md: the legacy count budgets are refused with a pointer at max_turns.
    lambda body: body["budgets"].update(max_model_calls=100),
    lambda body: body["budgets"].update(max_tool_calls=100),
])
def test_message_request_rejects_invalid_or_legacy_turn_budgets(mutate):
    body = request()
    mutate(body)
    with pytest.raises(ValueError, match="max_turns"):
        MessageRequest.from_dict(body)


def test_missing_grader_rejected_before_queueing(tmp_path, peer):
    queue, client = peer
    body = request()
    cfg = config(body)
    cfg["tasks"] = {}
    adapter = MessageAdapter(Driver(client, Ledger(tmp_path / "ledger")), cfg)
    with pytest.raises(ValueError, match="grade"):
        adapter.submit(body)
    assert queue.jobs == {}


def test_the_adapter_refuses_any_slot_but_mini(tmp_path, peer):
    _queue, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    with pytest.raises(ValueError, match="mini-swe-agent only"):
        MessageAdapter(driver, {**config(request()), "run_defaults": {"slot": "codex"}})


def test_hint_removal_does_not_delete_assistant_or_tool_text():
    source = messages()
    source[2]["content"] = mark_hint("literal tool output")
    source[-1]["content"] = mark_hint("literal assistant output")
    result = clean_messages(source)
    assert len(result) == len(source) - 1
    assert result[2] == source[2]
    assert result[-1] == source[-1]
    assert source[3]["content"].startswith("<sprout_training_hint>")
    bad = deepcopy(source)
    bad[3]["content"] = "<sprout_training_hint>incomplete"
    with pytest.raises(ValueError, match="Incomplete"):
        clean_messages(bad)


def test_unsupported_controls_and_wire_fields_are_not_silently_accepted():
    with pytest.raises(ValueError, match="Unknown"):
        parameters({"stop_token_ids": [7]})
    body = request()
    body["prompt_token_ids"] = [1, 2]
    with pytest.raises(ValueError, match="Unknown"):
        MessageRequest.from_dict(body)


def test_swe_rebench_rewards_need_all_tests_and_nonempty_expectations():
    task = {"FAIL_TO_PASS": ["new test [10 ms]"], "PASS_TO_PASS": ["old test"]}
    assert resolved({"new test": "PASSED", "old test": "PASSED"}, task)
    assert not resolved({"new test": "PASSED", "old test": "FAILED"}, task)
    assert not resolved({"old test": "PASSED"}, task)
    assert not resolved({}, {"FAIL_TO_PASS": [], "PASS_TO_PASS": []})
