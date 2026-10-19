"""The export of an attempt that ended in the middle of a turn: cut back to its last closed turn.

Each test starts from a real mini rollout of two closed turns, run by the attempt child on the
directory sandboxes of ``sandboxes.py`` against a scripted chat server. Its turn limit stopped it
before a third model call. A test then changes how the attempt ended.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import NativePoint
from sprout.worker.agents.mini_swe.history import index_native
from sprout.worker.jobs.export import complete_message_result, salvage_interrupted
from sprout.worker.jobs.rollout import execute
from tests.support.http import chat_completion, chat_server


@pytest.fixture
def attempt(tmp_path):
    (tmp_path / "artifacts" / "job-1" / "attempt-1").mkdir(parents=True)
    return tmp_path / "artifacts" / "job-1" / "attempt-1"


@pytest.fixture
def rollout(run_sandboxes, attempt) -> tuple[dict[str, Any], list[NativePoint]]:
    """The result of a mini rollout of two closed turns in ``attempt``, and the branch points of it."""
    with chat_server(
        [chat_completion("echo first > answer"), chat_completion("echo second > answer")]
    ) as server:
        contract = {
            "message_export": True,
            "model_endpoint": server.url,
            "model": "fixture",
            "deadline_at": time.time() + 60,
            "max_turns": 2,
            "sampling_params": {},
        }
        spec = {"prompt": "Write the answer.", "model": "fixture", "sandbox_image": "base", "timeout_s": 60}
        request = {
            "kind": "rollout",
            "attempt_id": "attempt-1",
            "profile_config": {},
            "effective_spec": {**spec, "extra": {"rollout_contract": contract}},
        }
        result = execute(request, attempt)
    points = index_native(
        attempt / "trajectory.jsonl", history(attempt, result), "mini-swe-agent", result["native_session_id"]
    )
    return result, points


def history(attempt: Path, result: dict[str, Any]) -> Path:
    return attempt / "native-home" / ("%s.jsonl" % result["native_session_id"])


def recovery_point(point: NativePoint, identifier: str) -> dict[str, Any]:
    """The recovery point of ``point`` as the Run Store keeps it."""
    return {
        "id": identifier,
        "snapshot_id": point.snapshot_id,
        "native": point.native,
        "tool_depth": point.tool_depth,
        "message_step": point.message_step,
    }


def test_a_rollout_that_failed_for_good_is_cut_back_to_a_closed_turn_and_graded_there(rollout, attempt):
    """The messages and the snapshot of a point are a pair: the grade sees the disk where the messages end."""
    result, points = rollout
    failed = {
        **result,
        "status": "error",
        "failure_kind": "infrastructure",
        "error": "TransportClosedError: reset",
    }

    salvaged = salvage_interrupted(failed, attempt, recovery_point(points[0], "point-1"))

    assert (salvaged["status"], salvaged["stop_reason"], salvaged["failure_kind"], "error" in salvaged) == (
        "truncated",
        "interrupted",
        None,
        False,
    )
    assert (salvaged["final_snapshot_id"], salvaged["training_messages"]) == (
        points[0].snapshot_id,
        result["training_messages"][:4],
    )
    assert salvaged["interrupted"] == {
        "status": "error",
        "error": "TransportClosedError: reset",
        "failure_kind": "infrastructure",
        "point_id": "point-1",
        "tool_depth": 1,
        "message_step": 1,
    }


def test_a_rollout_whose_point_no_longer_matches_its_history_is_not_salvaged(rollout, attempt):
    result, points = rollout
    path = history(attempt, result)
    path.write_bytes(path.read_bytes().replace(b"echo first", b"echo FIRST"))
    failed = {
        **result,
        "status": "error",
        "failure_kind": "infrastructure",
        "error": "TransportClosedError: reset",
    }

    salvaged = salvage_interrupted(failed, attempt, recovery_point(points[0], "point-1"))

    assert salvaged is None


def test_a_timeout_between_two_actions_of_one_response_grades_the_last_closed_turn(rollout, attempt):
    """Half of the response ran and half did not, so neither the messages nor the disk show a closed turn."""
    result, points = rollout
    partial = [
        {
            "role": "assistant",
            "content": "Two steps.",
            "tool_calls": [bash("p1", "echo partial > answer"), bash("p2", "true")],
        },
        {"role": "tool", "content": "done", "tool_call_id": "p1"},
    ]
    append_messages(history(attempt, result), partial)
    timed_out = {
        "status": "timeout",
        "stop_reason": "timeout",
        "final_snapshot_id": "snapshot-late",
        "native_session_id": result["native_session_id"],
        "failure_kind": "actor",
    }

    exported = complete_message_result(timed_out, attempt, "mini-swe-agent")

    assert (exported["status"], exported["final_snapshot_id"], exported["training_messages"]) == (
        "truncated",
        points[-1].snapshot_id,
        result["training_messages"],
    )
    assert exported["timeout_fallback"] == {
        "tool_depth": 2,
        "message_step": 2,
        "native_prefix": points[-1].native,
        "discarded_final_snapshot_id": "snapshot-late",
    }


def bash(identifier: str, command: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def append_messages(path: Path, messages: list[dict[str, Any]]) -> None:
    """Append each of ``messages`` to mini's native history at ``path``, as mini appends one."""
    with path.open("a") as stream:
        stream.writelines(
            json.dumps({"type": "mini.message", "message": message}) + "\n" for message in messages
        )
