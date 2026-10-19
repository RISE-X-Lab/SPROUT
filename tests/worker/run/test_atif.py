"""The export of a journal as an ATIF v1.8 trajectory (Harbor RFC 0001)."""

from __future__ import annotations

from typing import Any

import pytest

from sprout.worker.run.atif import journal_to_atif


def record(seq: int, event_type: str, **payload: Any) -> dict[str, Any]:
    """A journal record of run ``run-1``, written at second ``seq`` of 2030."""
    envelope = {"v": 2, "type": event_type, "ts": "2030-01-01T00:00:%02d.000Z" % seq, "seq": seq}
    return {**envelope, "run_id": "run-1", "agent_id": "agent", "sandbox_id": "sandbox-1", **payload}


#: Two turns of one tool call each. The second turn says two messages.
TWO_TURNS = [
    record(1, "run.started", slot="mini-swe-agent", slot_version="2.4.6", model="model-1", task_prompt="Fix"),
    record(2, "agent.thinking", text="Read the file first."),
    record(3, "tool.started", call_id="call-1", name="shell", args={"command": "cat a.txt"}, step=1),
    record(4, "tool.finished", call_id="call-1", status="ok", output="contents"),
    record(5, "turn.completed", usage={"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 2,
                                       "cost_usd": 0.25}),
    record(6, "agent.message", text="Now fix it."),
    record(7, "checkpoint.captured", step=1, snapshot_id="snapshot-1", session_ckpt="session-1"),
    record(8, "agent.message", text="Then submit."),
    record(9, "tool.started", call_id="call-2", name="shell", args={"command": "echo fixed > a.txt"}, step=2),
    record(10, "tool.finished", call_id="call-2", status="ok", output="fixed"),
    record(11, "checkpoint.captured", step=2, snapshot_id="snapshot-2", session_ckpt="session-1", final=True),
    record(12, "turn.completed", usage={"input_tokens": 20, "output_tokens": 7, "cost_usd": 0.5}),
    record(13, "session.ref", native_session_id="session-1"),
    record(14, "run.finished", status="completed", usage={"input_tokens": 30, "output_tokens": 12}),
]  # fmt: skip


def test_a_journal_exports_as_an_atif_document_with_one_step_for_each_turn():
    document = journal_to_atif(TWO_TURNS, trajectory_id="trajectory-1", continued_from="trajectory-0")

    assert document == {
        "schema_version": "ATIF-v1.8",
        "session_id": "session-1",
        "trajectory_id": "trajectory-1",
        "agent": {"name": "mini-swe-agent", "version": "2.4.6", "model": "model-1"},
        "steps": [
            {
                "step_id": 1,
                "timestamp": "2030-01-01T00:00:02.000Z",
                "source": "agent",
                "model_name": "model-1",
                "message": None,
                "reasoning_content": "Read the file first.",
                "tool_calls": [{"id": "call-1", "name": "shell", "arguments": {"command": "cat a.txt"}}],
                "observation": "contents",
                "metrics": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 2,
                    "cost_usd": 0.25,
                },
                "is_copied_context": False,
            },
            {
                "step_id": 2,
                "timestamp": "2030-01-01T00:00:06.000Z",
                "source": "agent",
                "model_name": "model-1",
                "message": "Now fix it.\nThen submit.",
                "reasoning_content": None,
                "tool_calls": [
                    {"id": "call-2", "name": "shell", "arguments": {"command": "echo fixed > a.txt"}}
                ],
                "observation": "fixed",
                "metrics": {"prompt_tokens": 20, "completion_tokens": 7, "cached_tokens": 0, "cost_usd": 0.5},
                "is_copied_context": False,
            },
        ],
        "final_metrics": {"prompt_tokens": 30, "completion_tokens": 12, "cached_tokens": 2, "cost_usd": 0.75},
        "extra": {
            "harness": "sprout",
            "journal_schema_version": 2,
            "status": "completed",
            "usage": {"input_tokens": 30, "output_tokens": 12},
            "checkpoints": [
                {"step": 1, "seq": 7, "snapshot_id": "snapshot-1", "session_ckpt": "session-1"},
                {"step": 2, "seq": 11, "snapshot_id": "snapshot-2", "session_ckpt": "session-1"},
            ],
        },
        "continued_trajectory_ref": "trajectory-0",
    }


def test_a_journal_without_steps_exports_one_system_step_with_the_task():
    """ATIF requires at least one step, also for a run that failed before its first turn."""
    records = [
        record(
            1,
            "run.started",
            slot="mini-swe-agent",
            slot_version="2.4.6",
            model="model-1",
            task_prompt="Fix",
        ),
        record(2, "run.finished", status="error", error="auth failed"),
    ]

    document = journal_to_atif(records)

    assert document == {
        "schema_version": "ATIF-v1.8",
        "session_id": "run-1",
        "trajectory_id": "run-1",
        "agent": {"name": "mini-swe-agent", "version": "2.4.6", "model": "model-1"},
        "steps": [
            {
                "step_id": 1,
                "source": "system",
                "message": "Fix",
                "model_name": "model-1",
                "tool_calls": [],
                "metrics": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0},
                "is_copied_context": False,
            }
        ],
        "final_metrics": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0},
        "extra": {
            "harness": "sprout",
            "journal_schema_version": 2,
            "status": "error",
            "usage": None,
            "checkpoints": [],
        },
    }


#: A branch's own journal: its seq numbers start at 1 again.
CHILD = [
    record(
        1, "run.started", slot="mini-swe-agent", slot_version="2.4.6", model="model-1", task_prompt="Branch."
    ),
    record(2, "agent.message", text="child work"),
    record(3, "turn.completed", usage={"input_tokens": 3}),
]


@pytest.mark.parametrize(
    "parent_records,expected",
    [
        pytest.param(
            TWO_TURNS[:5], [(True, None), (False, "child work")], id="the-fork-is-at-the-end-of-a-turn"
        ),
        # The fork is after the tool result of the second turn, before its turn.completed.
        pytest.param(
            TWO_TURNS[:10],
            [(True, None), (True, "Now fix it.\nThen submit."), (False, "child work")],
            id="the-fork-is-inside-a-turn",
        ),
    ],
)
def test_the_steps_inherited_from_the_parent_are_marked_copied_and_the_branchs_own_steps_are_not(
    parent_records, expected
):
    """A consumer drops the copied steps, so the branch's first step must not merge into the inherited one."""
    document = journal_to_atif(CHILD, inherited=parent_records)

    assert [(step["is_copied_context"], step["message"]) for step in document["steps"]] == expected


def test_a_rejected_turn_and_a_failed_tool_call_are_marked_in_their_steps():
    records = [
        record(1, "agent.message", text="A malformed call."),
        record(2, "turn.completed", rejected=True),
        record(3, "tool.started", call_id="call-1", name="shell", args={"command": "false"}, step=1),
        record(4, "tool.finished", call_id="call-1", status="error", output="exit 1"),
        record(5, "turn.completed"),
    ]

    steps = journal_to_atif(records)["steps"]

    assert [(step["tool_calls"], step.get("extra")) for step in steps] == [
        ([], {"rejected": True}),
        ([{"id": "call-1", "name": "shell", "arguments": {"command": "false"}}], {"tool_error": True}),
    ]
