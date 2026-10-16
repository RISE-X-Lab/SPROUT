import json
from pathlib import Path

import pytest

from sprout.worker.run.runner import Orchestrator
from sprout.contract.mini_history import read_entries, training_messages
from tests.harness.test_mini_swe import FilesystemSession, model_server, owned_filesystem, reply, spec, pytestmark
from sprout.worker.jobs.export import export_messages
from sprout.contract.messages import mark_hint
from sprout.worker.agents.mini_swe.history import index_native
from sprout.contract.mini_history import materialize, read_prefix
from tests.runstore.test_store import store  # noqa: F401  (fixture)


def parent_run(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("echo early > answer", "cat answer"),
                       reply("echo late > answer"),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, _):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    path = tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl"
    points = index_native(outcome.journal_path, path, "mini-swe-agent", outcome.native_session_id)
    return memory, outcome, points


def test_exact_branch_restores_closed_prefix_without_parent_suffix(tmp_path, monkeypatch):
    memory, parent, points = parent_run(tmp_path / "parent", monkeypatch)
    assert [p.tool_depth for p in points] == [2, 3, 4]
    cut = points[0]
    original = read_prefix(cut.native)
    assert b"echo late" not in original
    child_dir = tmp_path / "child"
    extra = materialize(cut.native, child_dir / "cwd", child_dir / "restoration", child_dir / "native-home")
    child = FilesystemSession(child_dir / "sandbox")
    for name, data in memory.snapshots[cut.snapshot_id].items():
        (child.root / name).write_bytes(data)
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    with model_server([reply("cat answer"), reply("echo child > answer"),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        run_spec = spec(child_dir, url, prompt=mark_hint("private branch direction"))
        run_spec.extra.update(extra)
        outcome = Orchestrator().run(run_spec)
    assert outcome.status == "completed", outcome.error
    assert "private branch direction" in json.dumps(requests[0]["messages"])
    assert "echo late" not in json.dumps(requests)
    assert json.loads(requests[1]["messages"][-1]["content"])["output"] == "early\n"
    messages = export_messages(child_dir, outcome.native_session_id, "mini-swe-agent")
    assert "private branch direction" not in json.dumps(messages)
    assert "echo child" in json.dumps(messages) and "echo late" not in json.dumps(messages)
    child_path = child_dir / "native-home" / f"{outcome.native_session_id}.jsonl"
    parent_entries = read_entries(original)
    child_entries = read_entries(child_path.read_bytes())
    assert parent_entries[0]["workspace"] == child_entries[0]["workspace"]
    parent_messages = training_messages(parent_entries)
    assert requests[0]["messages"][0] == parent_messages[0]
    child_points = index_native(outcome.journal_path, child_path, "mini-swe-agent",
                                outcome.native_session_id, inherited_native=cut.native)
    assert [p.tool_depth for p in child_points] == [1, 2, 3]
    assert read_prefix(cut.native) == original
    assert (memory.root / "answer").read_text() == "late\n"
    assert (child.root / "answer").read_text() == "child\n"


def test_branch_cannot_change_inherited_workspace(tmp_path, monkeypatch):
    _, _, points = parent_run(tmp_path / "parent", monkeypatch)
    child_dir = tmp_path / "child"
    extra = materialize(points[0].native, child_dir / "cwd",
                        child_dir / "restoration", child_dir / "native-home")
    child = FilesystemSession(child_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    with model_server([]) as (url, requests):
        run_spec = spec(child_dir, url, prompt=mark_hint("continue"))
        run_spec.extra.update(extra)
        run_spec.extra["mini"] = {"environment": {"cwd": "/another-repository"}}
        outcome = Orchestrator().run(run_spec)
    assert "differs from its inherited workspace" in outcome.error
    assert requests == []


def test_modified_or_partial_prefix_cannot_be_restored(tmp_path, monkeypatch):
    _, parent, points = parent_run(tmp_path, monkeypatch)
    original = points[0].native
    changed = {**original, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="checksum changed"):
        materialize(changed, tmp_path / "cwd", tmp_path / "recovery", tmp_path / "home")
    path = Path(original["path"])
    entries = path.read_bytes().splitlines(keepends=True)
    # A partial file never publishes the final turn while it is being written.
    path.write_bytes(b"".join(entries[:-2]) + b'{"type":')
    indexed = index_native(parent.journal_path, path, "mini-swe-agent", parent.native_session_id)
    assert all(point.tool_depth < 4 for point in indexed)


def test_branch_point_inspection_reads_each_session_once_without_stale_results(tmp_path, monkeypatch):
    from sprout.worker.agents.mini_swe import history

    _, parent, points = parent_run(tmp_path, monkeypatch)
    original = history.index_mini
    reads = []

    def counted(journal, transcript, session_id, **kwargs):
        reads.append(session_id)
        return original(journal, transcript, session_id, **kwargs)

    monkeypatch.setattr(history, "index_mini", counted)
    assert set(history.available_branch_points(parent.journal_path)) == {2, 3, 4}
    assert reads == [parent.native_session_id]
    # A later inspection must see a truncated file, not a cached set of cuts.
    native = Path(points[0].native["path"])
    native.write_bytes(native.read_bytes()[:points[0].native["byte_length"]] + b'{"type":')
    assert set(history.available_branch_points(parent.journal_path)) == {2}
    assert reads == [parent.native_session_id, parent.native_session_id]


def test_branch_point_inspection_uses_the_latest_session_reference(tmp_path, monkeypatch):
    from sprout.worker.agents.mini_swe.history import available_branch_points

    _, parent, points = parent_run(tmp_path, monkeypatch)
    old = Path(points[0].native["path"])
    moved = old.with_name("moved.jsonl")
    old.rename(moved)
    with Path(parent.journal_path).open("a") as stream:
        stream.write(json.dumps({
            "type": "session.ref", "native_session_id": parent.native_session_id,
            "transcript_path": str(moved),
        }) + "\n")
    assert set(available_branch_points(parent.journal_path)) == {2, 3, 4}


def test_timeout_between_actions_grades_the_matching_closed_prefix(tmp_path, monkeypatch):
    from dataclasses import asdict
    from sprout.worker.agents.mini_swe.sandbox import ToolEnvironment
    from sprout.worker.jobs.export import complete_message_result
    from sprout.contract.messages import is_truncated_result

    original_execute = ToolEnvironment.execute
    def stop_after_partial(env, action, cwd=""):
        output = original_execute(env, action, cwd)
        if "partial" in action["command"]:
            env.control.request_stop("test wall deadline", stop_reason="timeout")
        return output
    monkeypatch.setattr(ToolEnvironment, "execute", stop_after_partial)
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("echo good > answer"),
                       reply("echo partial > answer", "echo never > answer")]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert len(requests) == 2 and outcome.stop_reason == "timeout"
    disk_at_stop = memory.snapshot().id
    result = complete_message_result(
        {**asdict(outcome), "final_snapshot_id": disk_at_stop, "failure_kind": "actor"},
        tmp_path, "mini-swe-agent")
    assert is_truncated_result(result) and not result.get("training_export_error"), result
    assert result["timeout_fallback"]["tool_depth"] == 1
    assert memory.snapshots[result["final_snapshot_id"]]["answer"] == b"good\n"
    assert memory.snapshots[disk_at_stop]["answer"] == b"partial\n"
    assert "echo partial" not in json.dumps(result["training_messages"])
    assert result["rollout_usage"] == {"model_calls": 2, "tool_calls": 2}


def salvage_point(point, identifier="point-last"):
    return {"id": identifier, "snapshot_id": point.snapshot_id, "native": point.native,
            "tool_depth": point.tool_depth, "message_step": point.message_step}


def test_a_failed_attempt_is_cut_back_to_its_last_closed_turn(tmp_path, monkeypatch):
    from dataclasses import asdict
    from sprout.contract.messages import clean_messages, is_truncated_result
    from sprout.worker.jobs.export import salvage_interrupted

    memory, outcome, points = parent_run(tmp_path, monkeypatch)
    last = points[-1]
    failed = {**asdict(outcome), "journal_path": str(outcome.journal_path), "status": "error",
              "failure_kind": "infrastructure", "error": "execution uncertain at step 3 (call x)"}
    result = salvage_interrupted(failed, tmp_path, salvage_point(last))
    assert is_truncated_result(result) and result["stop_reason"] == "interrupted"
    assert result["final_snapshot_id"] == last.snapshot_id, "graded on the disk the messages end at"
    assert result["training_messages"] == clean_messages(training_messages(read_entries(read_prefix(last.native))))
    assert result["training_provenance"] == {"prefix_messages": 0, "inserted_messages": []}
    assert "error" not in result and result["interrupted"]["error"] == failed["error"]
    assert result["interrupted"]["tool_depth"] == last.tool_depth
    assert result["rollout_usage"]["model_calls"] >= 2


def test_the_worker_grades_a_rollout_that_failed_for_good_at_its_last_valid_point(tmp_path, monkeypatch, store):
    from dataclasses import asdict
    from sprout.runstore.specs import JobSpec
    from sprout.worker.attempts import Worker

    memory, outcome, points = parent_run(tmp_path, monkeypatch)
    point = salvage_point(points[-1])
    contract = {"extra": {"rollout_contract": {"message_export": True}}}
    failed = {**asdict(outcome), "journal_path": str(outcome.journal_path), "status": "error",
              "failure_kind": "infrastructure", "error": "TransportClosedError: connection reset"}

    def publish(max_infra_retries, result):
        job = store.submit(JobSpec("rollout", {"prompt": "fixture"}, max_infra_retries=max_infra_retries),
                           f"salvage-{max_infra_retries}-{result['error'][:8]}")
        claimed = store.claim("worker")
        worker = Worker(store, {"artifact_root": str(tmp_path), "profiles": {}})
        monkeypatch.setattr(worker, "_ingest", lambda *args: [])
        monkeypatch.setattr(worker, "_cleanup", lambda *args: True)
        monkeypatch.setattr(worker.index, "points", lambda attempt: [point])
        monkeypatch.setattr(worker.index, "valid", lambda candidate: True)
        worker._publish(claimed, tmp_path, dict(result), {"effective_spec": contract})
        return store.get(job["id"])

    final = publish(0, failed)
    assert final["state"] == "succeeded" and final["result"]["stop_reason"] == "interrupted"
    assert final["result"]["final_snapshot_id"] == point["snapshot_id"]
    retried = publish(2, failed)
    assert retried["state"] == "queued", "an attempt that will run again is not cut short"
    misconfigured = publish(0, {**failed, "error": "Frozen dataset checksum changed"})
    assert misconfigured["state"] == "quarantined", "a configuration fault is not training data"
