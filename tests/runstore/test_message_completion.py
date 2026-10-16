import json
import time
from types import SimpleNamespace

import pytest

from sprout.harness_core.control import RunControl
from sprout.harness_core.slot import SlotResult
from sprout.harness_orchestrator.run import Orchestrator, RunOutcome
from sprout.runstore.child import execute
from sprout.runstore.messages import complete_message_result, is_truncated_result


def write_history(directory):
    home = directory / "native-home"
    home.mkdir(exist_ok=True)
    path = home / "native.jsonl"
    # mini's native history (harness_agents.mini_swe.history).
    items = [
        {"type": "mini.session", "format": "sprout-mini-v1", "mini_version": "2.4.6"},
        {"type": "mini.message", "message": {"role": "user", "content": "fix the task"}},
        {"type": "mini.message", "message": {"role": "assistant", "content": "partial solution"}},
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in items))
    return path


@pytest.mark.parametrize("reason", ["max_turns_reached", "timeout"])
def test_child_captures_final_state_and_exports_native_identity_on_cutoff(tmp_path, monkeypatch, reason):
    directory = tmp_path / "attempt"
    directory.mkdir()
    order = []
    session = SimpleNamespace(
        sandbox_id="vm", supports_snapshot=lambda: True, swap_sandbox=lambda snapshot: False,
        snapshot=lambda **kwargs: order.append(("snapshot", kwargs["disk_only"]))
        or SimpleNamespace(id="final-state"),
    )
    owned = SimpleNamespace(session=session, sandbox_id="vm", keep=False,
                            destroy=lambda: order.append("destroy"))
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned)
    monkeypatch.setattr(Orchestrator, "_wire_gateway", lambda *args: None)
    # The stand-in sandbox has no shell for mini's workspace probe.
    monkeypatch.setattr("sprout.harness_agents.mini_swe.sandbox.resolve_workspace", lambda session, extra, journal: extra)

    class Slot:
        def run(self, task, journal, tools):
            task._rollout_controls.prepare_model_request({})
            write_history(directory)
            journal.emit("session.ref", native_session_id="native")
            journal.emit("rollout.usage", model_calls=1, tool_calls=0)
            if reason == "max_turns_reached":
                task._rollout_controls.prepare_model_request({})
            else:
                until = time.monotonic() + 2
                while task.control.reason is None and time.monotonic() < until:
                    time.sleep(0.01)
                assert task.control.stop_reason == "timeout"
            return SlotResult(status="error", error="stopped")

    monkeypatch.setattr("sprout.harness_orchestrator.run.MiniSweSlot", Slot)
    request = {
        "kind": "rollout", "attempt_id": "attempt", "profile_config": {}, "effective_spec": {
            "prompt": "fix", "slot": "mini-swe-agent", "sandbox_image": "image", "extra": {
                "rollout_contract": {
                    "message_export": True, "max_turns": 1,
                    "deadline_at": time.time() + (0.5 if reason == "timeout" else 30),
                    "model_endpoint": "http://model", "sampling_params": {},
                }
            }
        },
    }
    result = execute(request, directory)
    assert result["status"] == "truncated", result
    assert result["stop_reason"] == reason
    assert result["native_session_id"] == "native"
    assert result["final_snapshot_id"] == "final-state"
    assert result["training_messages"][-1]["content"] == "partial solution"
    # One final disk snapshot, and only then is the VM destroyed.
    assert order == [("snapshot", True), "destroy"]
    assert is_truncated_result(result)


@pytest.mark.parametrize("uncertain", [True, False])
def test_cutoff_does_not_reclassify_uncertain_execution(tmp_path, uncertain):
    write_history(tmp_path)
    # With the event both guards block the relabel; without it only the result's own
    # failure_kind does, the shape the worker and the child leave behind.
    events = [{"type": "execution.uncertain", "kind": "transport_exception"}] if uncertain else []
    (tmp_path / "trajectory.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    result = complete_message_result({
        "status": "error", "stop_reason": "timeout", "error": "execution_uncertain",
        "failure_kind": "infrastructure", "final_snapshot_id": "not-eligible",
        "native_session_id": "native",
    }, tmp_path, "mini-swe-agent")
    assert result["status"] == "error"
    assert not is_truncated_result(result)
    assert "timeout_fallback" not in result


def test_cutoff_exports_complete_records_without_rewriting_partial_native_tail(tmp_path):
    path = write_history(tmp_path)
    path.write_bytes(path.read_bytes() + b'{"unfinished":')
    original = path.read_bytes()
    (tmp_path / "trajectory.jsonl").write_text("")
    result = complete_message_result({
        "status": "error", "stop_reason": "timeout", "error": "rollout wall-time budget exhausted",
        "final_snapshot_id": "final", "native_session_id": "native",
    }, tmp_path, "mini-swe-agent")
    assert result["status"] == "truncated"
    assert result["training_messages"][-1]["role"] == "assistant"
    assert path.read_bytes() == original


def test_limit_reason_does_not_replace_an_earlier_infrastructure_stop():
    control = RunControl()
    control.request_stop("execution_uncertain")
    control.request_stop("deadline", stop_reason="timeout")
    assert control.stop_reason is None


def test_timeout_outcome_records_status_without_claiming_natural_completion():
    result = RunOutcome(run_id="r", journal_path="unused", status="error", stop_reason="timeout")
    assert not result.ok


@pytest.mark.parametrize("cleanup_ok", [True, False])
def test_worker_publishes_scored_limit_episodes_only_after_cleanup(tmp_path, cleanup_ok):
    from sprout.runstore.worker import Worker

    finished = []
    worker = object.__new__(Worker)
    worker.store = SimpleNamespace(finish=lambda *args, **kwargs: finished.append((args, kwargs)))
    worker._ingest = lambda *args: []
    worker._cleanup = lambda *args: cleanup_ok
    worker._publish({"id": "job", "lease_token": "lease"}, tmp_path, {
        "status": "truncated", "stop_reason": "timeout", "final_snapshot_id": "final",
        "error": "rollout wall-time budget exhausted",
    }, {"effective_spec": {}})
    (args, kwargs), = finished
    assert args[2]["status"] == "truncated"
    if cleanup_ok:
        assert kwargs.get("state", "succeeded") == "succeeded"
        assert args[2]["failure_kind"] is None
    else:
        assert kwargs["state"] == "quarantined"
        assert args[2]["failure_kind"] == "infrastructure"
