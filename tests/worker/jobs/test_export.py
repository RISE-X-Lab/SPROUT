"""The training export of a mini-swe-agent attempt: its messages, tools, usage and provenance.

Each test writes an attempt directory by hand: mini's native history under ``native-home/`` in the
format of ``sprout.contract.mini_history``, and the journal events that the export reads.
``test_salvage.py`` exports the histories of real mini runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.messages import mark_hint
from sprout.worker.jobs.export import complete_message_result, export_messages

SESSION = {
    "type": "mini.session",
    "format": "sprout-mini-v1",
    "mini_version": "2.4.6",
    "session_id": "native",
}
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run one command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def message(role: str, content: str, **fields: Any) -> dict[str, Any]:
    """A ``mini.message`` entry of the native history."""
    return {"type": "mini.message", "message": {"role": role, "content": content, **fields}}


def call(identifier: str, command: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


#: A history of one closed turn: the task, one bash call and its result, and the answer.
HISTORY = [
    message("system", "You fix bugs."),
    message("user", "Fix the task."),
    message("assistant", "Look first.", tool_calls=[call("a1", "cat answer")]),
    message("tool", "0", tool_call_id="a1"),
    message("assistant", "Done."),
]


def lines(entries: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(entry).encode() + b"\n" for entry in entries)


def attempt_with(
    directory: Path, history: list[dict[str, Any]], events: list[dict[str, Any]], *, tail: bytes = b""
) -> Path:
    """``directory`` with mini's native history of the session ``native`` and the journal ``events``.

    ``tail`` follows the last complete record of the history. Returns the path of the history."""
    (directory / "native-home").mkdir(parents=True)
    path = directory / "native-home" / "native.jsonl"
    path.write_bytes(lines([SESSION, *history]) + tail)
    (directory / "trajectory.jsonl").write_bytes(lines(events))
    return path


def write_histories(home: Path, names: list[str]) -> None:
    """A copy of ``HISTORY`` at each of ``names`` under ``home``."""
    for name in names:
        (home / name).parent.mkdir(parents=True, exist_ok=True)
        (home / name).write_bytes(lines([SESSION, *HISTORY]))


def prefix_reference(path: Path, entries: int) -> dict[str, Any]:
    """The native reference of the first ``entries`` records of the history at ``path``, its header first."""
    data = b"".join(path.read_bytes().splitlines(keepends=True)[:entries])
    return {
        "slot": "mini-swe-agent",
        "path": str(path),
        "byte_length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


#: The result of a finished attempt of the session ``native``.
FINISHED = {"status": "completed", "native_session_id": "native"}


def cut_off(**changes: Any) -> dict[str, Any]:
    """The result of an attempt that its turn limit stopped after its final snapshot, with ``changes``."""
    return {
        "status": "timeout",
        "stop_reason": "max_turns_reached",
        "final_snapshot_id": "final",
        "native_session_id": "native",
        "failure_kind": "actor",
        **changes,
    }


def test_the_export_holds_the_native_messages_without_the_marked_hints(tmp_path):
    """The actor reads a branch's hint, and training does not."""
    history = [
        message("user", "Fix the task."),
        message("assistant", "parent attempt"),
        message("user", mark_hint("private guidance")),
        message("user", "Continue. " + mark_hint("more guidance")),
        message("assistant", "branch patch"),
    ]
    attempt_with(tmp_path, history, [])

    messages = export_messages(tmp_path, "native", "mini-swe-agent")

    assert messages == [
        {"role": "user", "content": "Fix the task."},
        {"role": "assistant", "content": "parent attempt"},
        {"role": "user", "content": "Continue. "},
        {"role": "assistant", "content": "branch patch"},
    ]


@pytest.mark.parametrize(
    "session_id, slot, histories, message_text",
    [
        (None, "mini-swe-agent", ["native.jsonl"], "Missing or unsupported native session identity"),
        ("native", "another-slot", ["native.jsonl"], "Missing or unsupported native session identity"),
        ("native", "mini-swe-agent", [], "Expected one native history for the attempt"),
        ("native", "mini-swe-agent", ["one/native.jsonl", "two/native.jsonl"], "Expected one native history"),
    ],
)
def test_an_export_needs_the_one_native_history_of_a_mini_session(
    tmp_path, session_id, slot, histories, message_text
):
    write_histories(tmp_path / "native-home", histories)

    with pytest.raises(ValueError, match=message_text):
        export_messages(tmp_path, session_id, slot)


#: The start of a record that mini had not finished writing when the attempt stopped.
UNFINISHED = b'{"type": "mini.message", "message": {"role": "user", "con'


def test_an_export_refuses_a_history_whose_last_record_is_unfinished(tmp_path):
    attempt_with(tmp_path, HISTORY, [], tail=UNFINISHED)

    with pytest.raises(ValueError, match="^Native history has an incomplete final record$"):
        export_messages(tmp_path, "native", "mini-swe-agent")


def test_an_export_that_may_drop_an_unfinished_last_record_ends_at_the_last_complete_one(tmp_path):
    attempt_with(tmp_path, HISTORY, [], tail=UNFINISHED)

    messages = export_messages(tmp_path, "native", "mini-swe-agent", allow_incomplete_tail=True)

    assert messages[-1] == {"role": "assistant", "content": "Done."}


@pytest.mark.parametrize(
    "result, events, labels",
    [
        pytest.param(cut_off(), [], ("truncated", "timeout", None), id="turn-limit"),
        pytest.param(
            cut_off(stop_reason="timeout"), [], ("truncated", "timeout", None), id="wall-time-limit"
        ),
        pytest.param(cut_off(stop_reason=None), [], ("timeout", None, "actor"), id="no-limit"),
        pytest.param(cut_off(final_snapshot_id=None), [], ("timeout", None, "actor"), id="no-final-snapshot"),
        pytest.param(
            cut_off(),
            [{"type": "execution.uncertain", "kind": "transport_exception"}],
            ("timeout", None, "actor"),
            id="uncertain-execution",
        ),
        pytest.param(
            cut_off(
                status="error", error="TransportClosedError: connection reset", failure_kind="infrastructure"
            ),
            [],
            ("error", None, "infrastructure"),
            id="infrastructure-fault",
        ),
    ],
)
def test_only_a_safe_cutoff_by_a_limit_after_the_final_snapshot_becomes_a_truncated_episode(
    tmp_path, result, events, labels
):
    """A truncated episode is graded on its final snapshot. ``agent_status`` keeps the status of the run."""
    attempt_with(tmp_path, HISTORY, events)

    exported = complete_message_result(result, tmp_path, "mini-swe-agent")

    assert (exported["status"], exported.get("agent_status"), exported["failure_kind"]) == labels


def test_a_truncated_export_keeps_the_complete_records_and_leaves_the_partial_tail_on_disk(tmp_path):
    path = attempt_with(tmp_path, HISTORY, [], tail=b'{"unfinished":')
    on_disk = path.read_bytes()

    exported = complete_message_result(cut_off(), tmp_path, "mini-swe-agent")

    assert (exported["status"], exported["training_messages"][-1], "training_export_error" in exported) == (
        "truncated",
        {"role": "assistant", "content": "Done."},
        False,
    )
    assert path.read_bytes() == on_disk


def test_a_finished_rollout_whose_history_ends_unfinished_has_no_export_and_says_why(tmp_path):
    attempt_with(tmp_path, HISTORY, [], tail=b'{"unfinished":')

    exported = complete_message_result(FINISHED, tmp_path, "mini-swe-agent")

    assert (exported["training_export_error"], "training_messages" in exported) == (
        "Native history has an incomplete final record",
        False,
    )


def test_the_export_names_the_latest_session_the_tools_offered_and_the_last_usage(tmp_path):
    events = [
        {"type": "session.ref", "native_session_id": "older"},
        {"type": "rollout.model_tools", "shape": "chat/completions", "tools": [BASH]},
        {"type": "rollout.usage", "model_calls": 1, "tool_calls": 0},
        {"type": "session.ref", "native_session_id": "native"},
        {"type": "rollout.model_tools", "shape": "chat/completions", "tools": [BASH]},
        {"type": "rollout.usage", "model_calls": 2, "tool_calls": 1},
    ]
    attempt_with(tmp_path, HISTORY, events)

    exported = complete_message_result({"status": "completed"}, tmp_path, "mini-swe-agent")

    assert exported == {
        "status": "completed",
        "native_session_id": "native",
        "training_messages": [
            {"role": "system", "content": "You fix bugs."},
            {"role": "user", "content": "Fix the task."},
            {"role": "assistant", "content": "Look first.", "tool_calls": [call("a1", "cat answer")]},
            {"role": "tool", "content": "0", "tool_call_id": "a1"},
            {"role": "assistant", "content": "Done."},
        ],
        "training_tools": [BASH],
        "rollout_usage": {"model_calls": 2, "tool_calls": 1},
        "training_provenance": {"prefix_messages": 0, "inserted_messages": []},
    }


def test_a_branch_export_counts_the_messages_restored_from_its_point_and_names_its_origin(tmp_path):
    path = attempt_with(tmp_path, HISTORY, [])
    point = {"job_id": "parent-job", "id": "point-1", "tool_depth": 1, "native": prefix_reference(path, 5)}

    exported = complete_message_result(FINISHED, tmp_path, "mini-swe-agent", recovery=point)

    assert (exported["training_provenance"], exported["training_origin"]) == (
        {"prefix_messages": 4, "inserted_messages": []},
        {"job_id": "parent-job", "point_id": "point-1", "tool_depth": 1},
    )


def test_a_branch_whose_history_does_not_begin_with_its_point_has_no_export(tmp_path):
    """Messages whose authorship cannot be told apart are not training data."""
    parent = attempt_with(tmp_path / "parent", [message("user", "Another task."), *HISTORY[1:]], [])
    attempt_with(tmp_path / "branch", HISTORY, [])
    point = {"job_id": "parent-job", "id": "point-1", "tool_depth": 1, "native": prefix_reference(parent, 5)}

    exported = complete_message_result(FINISHED, tmp_path / "branch", "mini-swe-agent", recovery=point)

    assert exported["training_export_error"] == (
        "training provenance: The exported history does not begin with the branch point's prefix"
    )


def test_the_export_indexes_each_assistant_turn_that_a_reviewer_wrote(tmp_path):
    """Its tool results are real, but the policy did not sample the turn."""
    history = [
        message("user", "Fix the task."),
        message("assistant", "Inserted.", tool_calls=[call("r1", "ls")], extra={"source": "reviewer"}),
        message("tool", "answer", tool_call_id="r1"),
        message("assistant", "Sampled.", tool_calls=[call("a1", "cat answer")]),
        message("tool", "0", tool_call_id="a1"),
    ]
    attempt_with(tmp_path, history, [])

    exported = complete_message_result(FINISHED, tmp_path, "mini-swe-agent")

    assert exported["training_provenance"] == {"prefix_messages": 0, "inserted_messages": [1]}


def test_a_rollout_without_a_native_history_keeps_its_result_and_says_why_it_has_no_export(tmp_path):
    (tmp_path / "trajectory.jsonl").write_text("")

    exported = complete_message_result(FINISHED, tmp_path, "mini-swe-agent")

    assert exported == {
        "status": "completed",
        "native_session_id": "native",
        "training_export_error": "Expected one native history for the attempt",
    }
