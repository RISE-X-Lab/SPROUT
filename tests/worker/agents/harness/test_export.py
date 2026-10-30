"""The training export of a harness attempt: a submitted rollout, and an invalid one that ends truncated
with the flaw's stop reason and its malformed turn as the last message."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sprout.contract.messages import is_truncated_result
from sprout.worker.jobs.export import complete_message_result
from tests.worker.agents.harness.rollouts import reply, run_harness, turn
from tests.worker.agents.harness.test_slot import FIX, fixture


def export(
    tmp_path: Path, replies: list[dict[str, object]], *, on_call: Any = None, **harness: object
) -> dict[str, object]:
    rollout = run_harness(fixture(tmp_path), replies, harness=dict(harness), on_call=on_call)
    result = {
        "status": rollout.result.status,
        "stop_reason": rollout.control.stop_reason,
        "final_snapshot_id": rollout.sandbox.snapshots[-1],
        "native_session_id": rollout.result.native_session_id,
    }
    return complete_message_result(result, tmp_path / "attempt", "sprout-harness")


def test_a_submitted_rollout_exports_its_conversation_and_the_three_tools(tmp_path: Path) -> None:
    result = export(tmp_path, list(FIX))

    assert "training_export_error" not in result
    assert result["status"] == "completed"
    messages = result["training_messages"]
    assert isinstance(messages, list) and [m["role"] for m in messages][-2:] == ["assistant", "tool"]
    tools = result["training_tools"]
    assert isinstance(tools, list) and len(tools) == 3
    assert result["training_provenance"] == {"prefix_messages": 0, "inserted_messages": []}


def test_an_invalid_training_rollout_exports_truncated_with_its_malformed_turn_last(tmp_path: Path) -> None:
    malformed = reply(turn("execute_bash", ("command", "cat calc.py")) + "\nDone.")
    result = export(tmp_path, [FIX[0], malformed], on_invalid="terminate")

    assert (result["status"], result["stop_reason"]) == ("truncated", "format_invalid")
    assert is_truncated_result(result)
    messages = result["training_messages"]
    assert isinstance(messages, list)
    assert messages[-1]["role"] == "assistant" and messages[-1]["content"].endswith("\nDone.")
    assert "tool_calls" not in messages[-1]


def test_a_rollout_stopped_during_a_call_exports_up_to_its_last_closed_turn(tmp_path: Path) -> None:
    shells: list[str] = []

    def stop_at_the_command(name: str, args: dict[str, Any], control: Any) -> None:
        # The fix reads with one command and checks with a second: the second is cut off.
        shells.extend([name] if name == "shell" else [])
        if len(shells) == 2:
            control.request_stop("wall time", stop_reason="timeout")

    result = export(tmp_path, list(FIX), on_call=stop_at_the_command)

    assert (result["status"], result["stop_reason"]) == ("truncated", "timeout")
    assert "training_export_error" not in result
    fallback = result["timeout_fallback"]
    assert isinstance(fallback, dict) and fallback["tool_depth"] == 2
    messages = result["training_messages"]
    assert isinstance(messages, list) and [m["role"] for m in messages][-2:] == ["assistant", "tool"]
