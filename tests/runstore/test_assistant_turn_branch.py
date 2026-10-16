from copy import deepcopy
import json

import pytest

from sprout.harness.core.journal import read_journal
from sprout.harness.orchestrator.run import Orchestrator
from sprout.harness.agents.mini_swe.history import read_entries, training_messages
from tests.harness.test_assistant_turn import assistant_turn
from tests.harness.test_mini_swe import FilesystemSession, model_server, owned_filesystem, pytestmark, reply, spec
from sprout.runstore.messages import export_messages
from sprout.harness.agents.mini_swe.history import index_native, materialize, read_prefix
from tests.runstore.test_mini_native import parent_run


def restore_files(memory, snapshot, destination):
    session = FilesystemSession(destination)
    for name, data in memory.snapshots[snapshot].items():
        (session.root / name).parent.mkdir(parents=True, exist_ok=True)
        (session.root / name).write_bytes(data)
    return session


def test_assistant_turn_executes_before_actor_and_resumes_without_reexecution(tmp_path, monkeypatch):
    parent_memory, parent, parent_points = parent_run(tmp_path / "parent", monkeypatch)
    cut = parent_points[0]
    prefix_bytes = read_prefix(cut.native)
    child_dir = tmp_path / "child"
    extra = materialize(cut.native, child_dir / "cwd", child_dir / "restoration", child_dir / "native-home")
    child = restore_files(parent_memory, cut.snapshot_id, child_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    injected = assistant_turn("printf once >> injected-count; printf revised > answer; cat answer")
    original = deepcopy(injected)
    with model_server([reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        run_spec = spec(child_dir, url, prompt="USER_HINT_MUST_NOT_BE_APPENDED")
        run_spec.extra.update(extra, assistant_turn=injected)
        outcome = Orchestrator().run(run_spec)
    assert outcome.status == "completed", outcome.error
    assert injected == original
    assert len(requests) == 2 and outcome.usage["input_tokens"] == 22
    assert (child.root / "injected-count").read_text() == "once"
    first = requests[0]["messages"]
    # Compare the actual native messages sent on the wire, including nullable
    # provider fields; training export deliberately has a narrower schema.
    retained = [{k: v for k, v in e["message"].items() if k != "extra"}
                for e in read_entries(prefix_bytes) if e["type"] == "mini.message"]
    assert first[:len(retained)] == retained
    assert first[len(retained)] == original
    observation = first[len(retained) + 1]
    assert observation["role"] == "tool" and observation["tool_call_id"] == original["tool_calls"][0]["id"]
    assert json.loads(observation["content"])["output"] == "revised"
    assert len(first) == len(retained) + 2
    assert "USER_HINT_MUST_NOT_BE_APPENDED" not in json.dumps(requests)
    assert "echo late" not in json.dumps(requests)
    assert (parent_memory.root / "answer").read_text() == "late\n"
    assert read_prefix(cut.native) == prefix_bytes
    journal = read_journal(outcome.journal_path)
    injections = [e for e in journal if e["type"] == "branch.assistant_turn"]
    assert len(injections) == 1 and injections[0]["message"] == original
    native = child_dir / "native-home" / f"{outcome.native_session_id}.jsonl"
    entries = read_entries(native.read_bytes())
    assert any(e.get("type") == "mini.message" and e["message"].get("extra", {}).get("source") == "reviewer"
               for e in entries)
    points = index_native(outcome.journal_path, native, "mini-swe-agent",
                          outcome.native_session_id, inherited_native=cut.native)
    assert [p.tool_depth for p in points] == [1, 2, 3]
    exported = export_messages(child_dir, outcome.native_session_id, "mini-swe-agent")
    assert original in exported and "USER_HINT_MUST_NOT_BE_APPENDED" not in json.dumps(exported)

    # A further branch from the injected turn inherits its real result and disk.
    grand_dir = tmp_path / "grandchild"
    grand_extra = materialize(points[0].native, grand_dir / "cwd",
                              grand_dir / "restoration", grand_dir / "native-home")
    grand = restore_files(child, points[0].snapshot_id, grand_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(grand))
    with model_server([reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        run_spec = spec(grand_dir, url, prompt="Continue.")
        run_spec.extra.update(grand_extra)
        grand_outcome = Orchestrator().run(run_spec)
    assert grand_outcome.status == "completed", grand_outcome.error
    assert (grand.root / "injected-count").read_text() == "once"
    assert not any(e["type"] == "branch.assistant_turn" for e in read_journal(grand_outcome.journal_path))
    assert sum(m.get("tool_calls") == original["tool_calls"] for m in requests[0]["messages"]) == 1


def test_injection_requires_history_and_cannot_reuse_parent_call_id(tmp_path, monkeypatch):
    memory, parent, points = parent_run(tmp_path / "parent", monkeypatch)
    for label, history in [("no-prefix", None), ("duplicate-id", points[0].native)]:
        child_dir = tmp_path / label
        child = restore_files(memory, points[0].snapshot_id, child_dir / "sandbox")
        monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
        injected = assistant_turn("touch must-not-exist")
        with model_server([]) as (url, requests):
            run_spec = spec(child_dir, url)
            if history:
                run_spec.extra.update(materialize(history, child_dir / "cwd",
                                      child_dir / "restoration", child_dir / "native-home"))
                messages = training_messages(read_entries(read_prefix(history)))
                injected["tool_calls"][0]["id"] = next(m["tool_calls"][0]["id"] for m in messages if m.get("tool_calls"))
            run_spec.extra["assistant_turn"] = injected
            outcome = Orchestrator().run(run_spec)
        assert outcome.status == "error"
        assert ("prefix" if history is None else "unique across history") in outcome.error
        assert requests == [] and not (child.root / "must-not-exist").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_multiple_reviewer_calls_keep_real_errors_and_only_close_complete_turns(tmp_path, monkeypatch, interrupt):
    from sprout.harness_checkpoint.rollback import branch_checkpoints
    from sprout.harness.agents.mini_swe.runtime import ToolEnvironment

    memory, _, points = parent_run(tmp_path / "parent", monkeypatch)
    child_dir = tmp_path / "child"
    child = restore_files(memory, points[0].snapshot_id, child_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    execute = ToolEnvironment.execute

    def checked_execute(env, action, cwd=""):
        output = execute(env, action, cwd)
        if interrupt:
            env.control.request_stop("fixture stopped after first action", stop_reason="timeout")
        return output

    monkeypatch.setattr(ToolEnvironment, "execute", checked_execute)
    turn = assistant_turn("printf diagnostic; exit 7", identifier="review-first")
    turn["tool_calls"].extend(assistant_turn("printf done > marker; cat marker", identifier="review-second")["tool_calls"])
    replies = [] if interrupt else [reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]
    with model_server(replies) as (url, requests):
        run_spec = spec(child_dir, url, prompt="unused hint")
        run_spec.extra.update(materialize(points[0].native, child_dir / "cwd",
                              child_dir / "restoration", child_dir / "native-home"),
                              assistant_turn=turn)
        outcome = Orchestrator().run(run_spec)
    events = read_journal(outcome.journal_path)
    started = [e for e in events if e["type"] == "tool.started"]
    if interrupt:
        assert outcome.stop_reason == "timeout" and requests == []
        assert len(started) == 1 and not (child.root / "marker").exists()
        assert branch_checkpoints(outcome.journal_path) == {}
    else:
        assert outcome.status == "completed", outcome.error
        assert len(requests) == 1
        observations = requests[0]["messages"][-2:]
        assert [m["tool_call_id"] for m in observations] == ["review-first", "review-second"]
        assert [json.loads(m["content"])["returncode"] for m in observations] == [7, 0]
        assert [json.loads(m["content"])["output"] for m in observations] == ["diagnostic", "done"]
        assert set(branch_checkpoints(outcome.journal_path)) == {2, 3}
