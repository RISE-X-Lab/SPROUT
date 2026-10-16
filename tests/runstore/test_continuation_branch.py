"""A Run Store branch job continues by the same rules a Harbor branch trial does.

The child runs in-process here, through ``child.execute`` exactly as a worker's
child process does, against a filesystem sandbox restored from the parent's
snapshot and a fake model endpoint behind the real gateway.
"""

import json
import time
from types import SimpleNamespace

import pytest

from sprout.contract.continuation import render_branch_note, render_reasoning
from sprout.contract.mini_history import read_entries, read_prefix, training_messages
from sprout.worker.run.journal import read_journal
from sprout.worker.run.runner import Orchestrator
from sprout.worker.jobs import rollout as child
from sprout.contract.messages import clean_messages, mark_hint
from sprout.runstore.specs import JobSpec
from sprout.worker.attempts import apply_branch_budget, starting_point
from tests.harness.test_assistant_turn import assistant_turn
from tests.harness.test_mini_swe import model_server, owned_filesystem, pytestmark, reply
from tests.runstore.test_assistant_turn_branch import restore_files
from tests.runstore.test_mini_native import parent_run
from tests.runstore.test_no_hint_branch import native_messages
from tests.runstore.test_store import store  # isolated schema; never a production database

_ = pytestmark  # these need mini installed, like the harness tests they reuse

HINT = "Check the boundary condition before editing."


def branch_request(url, cut, continuation, *, job_id="branch-job"):
    """What a worker hands its child for a branch's first attempt."""
    point = {"id": "point-1", "job_id": "parent-job", "native": cut.native, "snapshot_id": cut.snapshot_id,
             "tool_depth": cut.tool_depth, "message_step": cut.message_step}
    contract = {"message_export": True, "model_endpoint": url, "model": "fixture",
                "deadline_at": time.time() + 30, "max_turns": 5,
                "sampling_params": {"temperature": .8, "top_k": 20, "max_new_tokens": 300}}
    return {"kind": "rollout", "job_id": job_id, "attempt_id": "attempt-1",
            "effective_spec": {"prompt": "Write the answer.", "slot": "mini-swe-agent", "model": "fixture",
                               "timeout_s": 30, "extra": {"rollout_contract": contract}},
            "profile_config": {}, "recovery": point, "continuation": continuation, "branch_point": point}


def run_branch(tmp_path, monkeypatch, continuation, replies):
    memory, _, points = parent_run(tmp_path / "parent", monkeypatch)
    cut = points[0]
    sandbox = restore_files(memory, cut.snapshot_id, tmp_path / "sandbox")
    # The child wraps the session's re-board; a filesystem sandbox never re-boards.
    sandbox.swap_sandbox = lambda snapshot: False
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(sandbox))
    directory = tmp_path / "attempt"
    directory.mkdir()
    with model_server(replies) as (url, requests):
        result = child.execute(branch_request(url, cut, continuation), directory)
    assert result["status"] == "completed", result.get("error") or result.get("training_export_error")
    return SimpleNamespace(cut=cut, sandbox=sandbox, directory=directory, requests=requests, result=result,
                           prefix=clean_messages(training_messages(read_entries(read_prefix(cut.native)))))


def fork_origin(directory):
    return next(e for e in read_journal(directory / "trajectory.jsonl") if e["type"] == "fork.origin")


def test_an_assistant_turn_runs_once_before_the_actor_and_the_export_marks_it(tmp_path, monkeypatch):
    turn = assistant_turn("printf once >> inserted-count; printf revised > answer; cat answer",
                          identifier="inserted-1")
    run = run_branch(tmp_path, monkeypatch, {"guidance": "assistant-turn", "assistant_turn": turn},
                     [reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")])
    assert (run.sandbox.root / "inserted-count").read_text() == "once"
    retained = native_messages(run.cut.native)
    first = run.requests[0]["messages"]
    assert first[:len(retained)] == retained and first[len(retained)] == turn
    assert json.loads(first[len(retained) + 1]["content"])["output"] == "revised"
    assert "Write the answer." not in json.dumps(first[len(retained):]), "no prompt follows the prefix"

    exported = run.result["training_messages"]
    assert exported[:len(run.prefix)] == run.prefix
    assert exported[len(run.prefix)]["tool_calls"] == turn["tool_calls"]
    assert run.result["training_provenance"] == {"prefix_messages": len(run.prefix),
                                                 "inserted_messages": [len(run.prefix)]}
    assert run.result["training_origin"] == {"job_id": "parent-job", "point_id": "point-1",
                                             "tool_depth": run.cut.tool_depth}
    origin = fork_origin(run.directory)
    assert origin["point_id"] == "point-1" and origin["branch_guidance"] == "assistant-turn"
    assert origin["assistant_turn_source"] == "reviewer"
    assert "assistant_turn_rendering" not in origin


def test_a_reasoning_rendering_reaches_the_model_and_the_export_as_the_turns_reasoning(tmp_path, monkeypatch):
    turn = assistant_turn("printf once >> inserted-count; printf revised > answer; cat answer",
                          identifier="inserted-1")
    rendered = {"role": "assistant", "content": "", "reasoning_content": turn["content"],
                "tool_calls": turn["tool_calls"]}
    run = run_branch(tmp_path, monkeypatch, {"guidance": "assistant-turn", "assistant_turn": turn,
                                             "assistant_turn_rendering": "reasoning"},
                     [reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")])
    assert (run.sandbox.root / "inserted-count").read_text() == "once"
    retained = native_messages(run.cut.native)
    first = run.requests[0]["messages"]
    assert first[:len(retained)] == retained and first[len(retained)] == rendered, "the model is sent the rendering"
    assert json.loads(first[len(retained) + 1]["content"])["output"] == "revised"
    assert len(first) == len(retained) + 2

    exported = run.result["training_messages"]
    assert exported[:len(run.prefix)] == run.prefix and exported[len(run.prefix)] == rendered
    assert run.result["training_provenance"] == {"prefix_messages": len(run.prefix),
                                                 "inserted_messages": [len(run.prefix)]}
    origin = fork_origin(run.directory)
    assert origin["assistant_turn_rendering"] == "reasoning" and origin["assistant_turn_source"] == "reviewer"
    events = read_journal(run.directory / "trajectory.jsonl")
    assert [e["message"] for e in events if e["type"] == "branch.assistant_turn"] == [rendered]
    assert turn["content"] in [e["text"] for e in events if e["type"] == "agent.thinking"]
    assert turn["content"] not in [e["text"] for e in events if e["type"] == "agent.message"]


def test_a_user_hint_is_the_shared_note_marked_for_training_and_stripped_from_the_export(tmp_path, monkeypatch):
    run = run_branch(tmp_path, monkeypatch, {"guidance": "user-hint", "hint": HINT},
                     [reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")])
    retained = native_messages(run.cut.native)
    first = run.requests[0]["messages"]
    assert first[:len(retained)] == retained and len(first) == len(retained) + 1
    assert first[-1]["role"] == "user" and first[-1]["content"] == mark_hint(render_branch_note(HINT))
    assert HINT not in json.dumps(run.result["training_messages"])
    assert run.result["training_provenance"] == {"prefix_messages": len(run.prefix), "inserted_messages": []}
    origin = fork_origin(run.directory)
    assert origin["branch_guidance"] == "user-hint" and origin["actor_hint"] == HINT


def test_a_point_only_branch_adds_nothing_after_the_prefix(tmp_path, monkeypatch):
    run = run_branch(tmp_path, monkeypatch, {"guidance": "none"},
                     [reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")])
    assert run.requests[0]["messages"] == native_messages(run.cut.native)
    assert run.result["training_provenance"] == {"prefix_messages": len(run.prefix), "inserted_messages": []}
    assert fork_origin(run.directory)["branch_guidance"] == "none"


def test_a_branch_without_a_continuation_never_starts(tmp_path, monkeypatch):
    _, _, points = parent_run(tmp_path / "parent", monkeypatch)
    (tmp_path / "attempt").mkdir()
    with pytest.raises(ValueError, match="A continuation is an object"):
        child.execute(branch_request("http://127.0.0.1:1", points[0], None), tmp_path / "attempt")


def test_a_retry_resumes_its_own_history_without_adding_the_branch_again():
    branch = {"parent_point": "point-1",
              "continuation": {"guidance": "assistant-turn", "assistant_turn": assistant_turn()}}
    assert starting_point(branch, None) == ("point-1", branch["continuation"])
    assert starting_point(branch, {"point_id": "own-7", "timeout_s": 10}) == ("own-7", {"guidance": "none"})
    assert starting_point({}, None) == (None, None)
    assert starting_point({}, {"point_id": "own-3", "timeout_s": 10}) == ("own-3", {"guidance": "none"})


def test_only_a_branch_job_carries_a_continuation():
    scratch = JobSpec("rollout", {"prompt": "task"}).validate()
    assert "continuation" not in scratch, "a scratch request keeps the body and hash it had"
    branch = JobSpec("rollout", {"prompt": "task"}, parent_point="point-1",
                     continuation={"guidance": "none"}).validate()
    assert branch["continuation"] == {"guidance": "none"}
    rendered = {"guidance": "assistant-turn", "assistant_turn": assistant_turn(),
                "assistant_turn_rendering": "reasoning"}
    assert JobSpec("rollout", {"prompt": "task"}, parent_point="point-1",
                   continuation=rendered).validate()["continuation"] == rendered
    with pytest.raises(ValueError, match="A continuation is an object"):
        JobSpec("rollout", {"prompt": "task"}, parent_point="point-1").validate()
    with pytest.raises(ValueError, match="belongs to a rollout branch"):
        JobSpec("rollout", {"prompt": "task"}, continuation={"guidance": "none"}).validate()


def test_a_branch_may_set_its_own_turn_and_time_limits():
    """A branch inherits its parent's contract: the group deadline, which may
    already have passed, and a turn limit that restarts from zero. Its budget
    lays its own limits over that contract and touches nothing else."""
    contract = {"message_export": True, "model_endpoint": "http://m:1", "model": "fixture",
                "deadline_at": 1000.0, "max_turns": 40, "sampling_params": {"temperature": 0.8}}
    effective = {"prompt": "task", "extra": {"rollout_contract": contract, "mini": {"environment": {"cwd": "/w"}}}}
    budgeted = apply_branch_budget(effective, {"max_turns": 12, "deadline_at": 2000.0})
    assert budgeted["extra"]["rollout_contract"] == {**contract, "max_turns": 12, "deadline_at": 2000.0}
    assert budgeted["extra"]["mini"] == effective["extra"]["mini"] and budgeted["prompt"] == "task"
    assert effective["extra"]["rollout_contract"]["max_turns"] == 40, "the inherited spec is not edited in place"
    assert apply_branch_budget(effective, None) is effective
    with pytest.raises(ValueError, match="bounds a rollout contract"):
        apply_branch_budget({"prompt": "task", "extra": {}}, {"max_turns": 3})

    base = dict(parent_point="point-1", continuation={"guidance": "none"})
    assert JobSpec("rollout", {"prompt": "task"}, branch_budget={"max_turns": 12}, **base).validate()["branch_budget"] == {"max_turns": 12}
    assert "branch_budget" not in JobSpec("rollout", {"prompt": "task"}, **base).validate()
    for bad, message in [({}, "sets max_turns"), ({"turns": 3}, "sets max_turns"), ({"max_turns": 0}, "positive integer"),
                         ({"max_turns": 2.5}, "positive integer"), ({"deadline_at": -5}, "finite positive")]:
        with pytest.raises(ValueError, match=message):
            JobSpec("rollout", {"prompt": "task"}, branch_budget=bad, **base).validate()
    with pytest.raises(ValueError, match="belongs to a branch"):
        JobSpec("rollout", {"prompt": "task"}, branch_budget={"max_turns": 3}).validate()


class Points:
    """The one recovery point the branch endpoint is asked about."""

    def __init__(self, point):
        self.point = point

    def get_point(self, identifier):
        if identifier != self.point["id"]:
            raise KeyError(identifier)
        return self.point

    def valid(self, point):
        return True


def test_the_branch_endpoint_takes_a_continuation_and_checks_an_inserted_turn(store, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from sprout.runstore.api import create_app

    _, _, points = parent_run(tmp_path / "parent", monkeypatch)
    cut = points[0]
    parent = store.submit(JobSpec("rollout", {"prompt": "task", "slot": "mini-swe-agent"}), "parent")
    index = Points({"id": "point-1", "job_id": parent["id"], "native": cut.native,
                    "snapshot_id": cut.snapshot_id, "tool_depth": cut.tool_depth})
    used = next(call["id"] for message in native_messages(cut.native) for call in message.get("tool_calls") or [])

    def branch(key, body):
        return http.post(f"/v1/jobs/{parent['id']}/branch", json=body, headers={"Idempotency-Key": key})

    with TestClient(create_app(store, "fixture-token", index=index)) as http:
        http.headers["Authorization"] = "Bearer fixture-token"
        accepted = branch("fresh", {"point_id": "point-1", "continuation": {
            "guidance": "assistant-turn", "assistant_turn": assistant_turn(identifier="fresh-call")}})
        assert accepted.status_code == 202, accepted.text
        request = store.get(accepted.json()["id"])["request"]
        assert request["parent_point"] == "point-1" and request["spec"] == {"prompt": "task", "slot": "mini-swe-agent"}
        assert request["continuation"]["assistant_turn"]["tool_calls"][0]["id"] == "fresh-call"
        assert "assistant_turn_rendering" not in request["continuation"], "the default is not written"

        rendered = branch("rendered", {"point_id": "point-1", "continuation": {
            "guidance": "assistant-turn", "assistant_turn": assistant_turn(identifier="rendered-call"),
            "assistant_turn_rendering": "reasoning"}})
        assert rendered.status_code == 202, rendered.text
        assert store.get(rendered.json()["id"])["request"]["continuation"]["assistant_turn_rendering"] == "reasoning"
        prerendered = branch("prerendered", {"point_id": "point-1", "continuation": {
            "guidance": "assistant-turn", "assistant_turn_rendering": "reasoning",
            "assistant_turn": render_reasoning(assistant_turn(identifier="prerendered-call"))}})
        assert prerendered.status_code == 422 and "assistant_turn requires exactly" in prerendered.json()["detail"]
        unknown = branch("unknown-rendering", {"point_id": "point-1", "continuation": {
            "guidance": "assistant-turn", "assistant_turn": assistant_turn(identifier="unknown-call"),
            "assistant_turn_rendering": "think"}})
        assert unknown.status_code == 422 and "content or reasoning" in unknown.json()["detail"]

        reused = branch("reused", {"point_id": "point-1", "continuation": {
            "guidance": "assistant-turn", "assistant_turn": assistant_turn(identifier=used)}})
        assert reused.status_code == 422 and "unique across history" in reused.json()["detail"]
        prompt = branch("prompt", {"point_id": "point-1", "continuation": {"guidance": "none"},
                                   "overrides": {"prompt": "Try a different fix"}})
        assert prompt.status_code == 422 and "continuation" in prompt.json()["detail"]
        missing = branch("missing", {"point_id": "point-1"})
        assert missing.status_code == 422 and "A continuation is an object" in missing.json()["detail"]
        unbounded = branch("unbounded", {"point_id": "point-1", "continuation": {"guidance": "none"},
                                         "budget": {"max_turns": 5}})
        assert unbounded.status_code == 422 and "parent run has none" in unbounded.json()["detail"]

        contract = {"message_export": True, "model_endpoint": "http://m:1", "model": "fixture",
                    "deadline_at": time.time() + 60, "max_turns": 40}
        bounded_parent = store.submit(JobSpec("rollout", {"prompt": "task", "slot": "mini-swe-agent",
                                                          "extra": {"rollout_contract": contract}}), "bounded-parent")
        index.point["job_id"] = bounded_parent["id"]
        accepted = http.post(f"/v1/jobs/{bounded_parent['id']}/branch", headers={"Idempotency-Key": "budgeted"},
                             json={"point_id": "point-1", "continuation": {"guidance": "none"},
                                   "budget": {"max_turns": 5, "deadline_at": contract["deadline_at"] + 30}})
        assert accepted.status_code == 202, accepted.text
        request = store.get(accepted.json()["id"])["request"]
        assert request["branch_budget"] == {"max_turns": 5, "deadline_at": contract["deadline_at"] + 30}
        assert request["spec"]["extra"]["rollout_contract"] == contract, "the budget is applied by the worker, not stored in the spec"
        rejected = http.post(f"/v1/jobs/{bounded_parent['id']}/branch", headers={"Idempotency-Key": "bad-budget"},
                             json={"point_id": "point-1", "continuation": {"guidance": "none"}, "budget": {"max_turns": 0}})
        assert rejected.status_code == 422 and "positive integer" in rejected.json()["detail"]
