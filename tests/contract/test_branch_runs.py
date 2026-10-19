"""A branch attempt from a parent's point: the restored history, then what the continuation says.

Each test runs a parent rollout, indexes its points, and runs one attempt of a branch job through the
attempt process (``sprout.worker.jobs.rollout.execute``), as a Run Store worker does. The model replies are
scripted, and the sandboxes are the fake backend of ``fake_sandbox.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.continuation import render_branch_note
from sprout.contract.messages import mark_hint
from sprout.contract.mini_history import NativePoint, read_entries, read_prefix
from sprout.worker.agents.mini_swe.history import index_native
from sprout.worker.jobs.rollout import execute
from sprout.worker.run.journal import read_journal
from sprout.worker.run.runner import Runner
from tests.contract.builders import REVIEWER_TEXT, assistant_turn
from tests.contract.mini_runs import NEEDS_MINI, SUBMIT, mini_spec, rollout_contract
from tests.support.http import chat_completion, chat_server

pytestmark = NEEDS_MINI

HINT = "Check the boundary condition before editing."
#: The turn of a reviewer, which counts its own runs in ``inserted-count``.
TURN = assistant_turn("printf once >> inserted-count; printf revised > answer; cat answer", "inserted-1")
#: The journal fields that every event has.
ENVELOPE = {"v", "type", "ts", "seq", "run_id", "agent_id", "sandbox_id"}


def parent_point(directory: Path) -> NativePoint:
    """The first branch point of a parent rollout that writes ``early`` to ``answer``, then ``late``."""
    replies = [
        chat_completion("echo early > answer", "cat answer", ids=("parent-1", "parent-2")),
        chat_completion("echo late > answer", ids=("parent-3",)),
        chat_completion(SUBMIT, ids=("parent-4",)),
    ]
    with chat_server(replies) as server:
        outcome = Runner(out_dir=directory).run(mini_spec(directory, server.url))
    native = directory / "native-home" / ("%s.jsonl" % outcome.native_session_id)
    return index_native(outcome.journal_path, native, "mini-swe-agent", outcome.native_session_id)[0]


def branch_request(url: str, point: NativePoint, continuation: object) -> dict[str, Any]:
    """The first attempt of a branch job from ``point``, as a worker hands it to the attempt process."""
    recovery = {
        "id": "point-1",
        "job_id": "parent-job",
        "native": point.native,
        "snapshot_id": point.snapshot_id,
        "tool_depth": point.tool_depth,
        "message_step": point.message_step,
    }
    spec = {
        "prompt": "Write the answer.",
        "slot": "mini-swe-agent",
        "model": "fixture",
        "timeout_s": 60,
        "extra": {"rollout_contract": rollout_contract(url)},
    }
    return {
        "kind": "rollout",
        "job_id": "branch-job",
        "attempt_id": "attempt-1",
        "effective_spec": spec,
        "profile_config": {},
        "recovery": recovery,
        "continuation": continuation,
    }


def actor_replies() -> list[dict[str, Any]]:
    """The actor's replies: read ``answer``, then submit, with call ids that are new to the history."""
    return [chat_completion("cat answer", ids=("branch-1",)), chat_completion(SUBMIT, ids=("branch-2",))]


def run_branch(tmp_path: Path, continuation: object) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """The result and the model requests of a branch attempt, and the number of messages that it restored."""
    point = parent_point(tmp_path / "parent")
    (tmp_path / "attempt").mkdir()
    with chat_server(actor_replies()) as server:
        result = execute(branch_request(server.url, point, continuation), tmp_path / "attempt")
    return result, server.requests, len(restored_messages(point))


def restored_messages(point: NativePoint) -> list[dict[str, Any]]:
    """The chat messages of the history up to ``point``, without mini's ``extra``, as a model gets them."""
    entries = read_entries(read_prefix(point.native))
    return [
        {key: value for key, value in entry["message"].items() if key != "extra"}
        for entry in entries
        if entry["type"] == "mini.message"
    ]


def observation(message: dict[str, Any]) -> dict[str, Any]:
    """What a tool message tells the model."""
    return json.loads(message["content"])


def test_a_branch_starts_from_the_snapshot_of_its_point_and_its_files(tmp_path, sandboxes):
    point = parent_point(tmp_path / "parent")
    (tmp_path / "attempt").mkdir()

    with chat_server(actor_replies()) as server:
        execute(branch_request(server.url, point, {"guidance": "none"}), tmp_path / "attempt")

    assert (sandboxes.sessions[-1].image, observation(server.requests[1]["messages"][-1])) == (
        point.snapshot_id,
        {"returncode": 0, "output": "early\n"},
    )


def test_an_assistant_turn_runs_once_before_the_first_request_of_the_actor(tmp_path, sandboxes):
    """Its real result joins the history, and no prompt follows the restored history."""
    _, requests, restored = run_branch(tmp_path, {"guidance": "assistant-turn", "assistant_turn": TURN})
    first = requests[0]["messages"]

    assert (len(first), first[restored], observation(first[restored + 1])) == (
        restored + 2,
        TURN,
        {"returncode": 0, "output": "revised"},
    )
    assert (sandboxes.sessions[-1].root / "inserted-count").read_text() == "once"


def test_the_export_of_an_assistant_turn_branch_marks_the_turn_that_no_policy_sampled(tmp_path, sandboxes):
    result, _, restored = run_branch(tmp_path, {"guidance": "assistant-turn", "assistant_turn": TURN})

    assert (result["training_provenance"], result["training_messages"][restored]["tool_calls"]) == (
        {"prefix_messages": restored, "inserted_messages": [restored]},
        TURN["tool_calls"],
    )


def test_a_reasoning_rendering_is_the_turn_that_the_actor_and_training_read(tmp_path, sandboxes):
    rendered = {
        "role": "assistant",
        "content": "",
        "reasoning_content": REVIEWER_TEXT,
        "tool_calls": TURN["tool_calls"],
    }
    continuation = {
        "guidance": "assistant-turn",
        "assistant_turn": TURN,
        "assistant_turn_rendering": "reasoning",
    }

    result, requests, restored = run_branch(tmp_path, continuation)

    assert (requests[0]["messages"][restored], result["training_messages"][restored]) == (rendered, rendered)


def test_the_journal_records_the_text_of_a_rendered_turn_as_the_thinking_of_its_turn(tmp_path, sandboxes):
    continuation = {
        "guidance": "assistant-turn",
        "assistant_turn": TURN,
        "assistant_turn_rendering": "reasoning",
    }
    rendered = {
        "role": "assistant",
        "content": "",
        "reasoning_content": REVIEWER_TEXT,
        "tool_calls": TURN["tool_calls"],
    }

    run_branch(tmp_path, continuation)

    journal = read_journal(tmp_path / "attempt" / "trajectory.jsonl")
    assert (
        [event["message"] for event in journal if event["type"] == "branch.assistant_turn"],
        REVIEWER_TEXT in [event["text"] for event in journal if event["type"] == "agent.thinking"],
        REVIEWER_TEXT in [event["text"] for event in journal if event["type"] == "agent.message"],
    ) == ([rendered], True, False)


def test_the_actor_of_a_user_hint_branch_reads_the_marked_branch_note_after_the_restored_history(
    tmp_path, sandboxes
):
    _, requests, restored = run_branch(tmp_path, {"guidance": "user-hint", "hint": HINT})
    first = requests[0]["messages"]

    assert (len(first), first[restored]) == (
        restored + 1,
        {"role": "user", "content": mark_hint(render_branch_note(HINT))},
    )


def test_the_export_of_a_user_hint_branch_holds_none_of_the_hint(tmp_path, sandboxes):
    result, _, restored = run_branch(tmp_path, {"guidance": "user-hint", "hint": HINT})

    assert (HINT in json.dumps(result["training_messages"]), result["training_provenance"]) == (
        False,
        {"prefix_messages": restored, "inserted_messages": []},
    )


def test_the_actor_of_a_branch_without_guidance_gets_the_restored_history_alone(tmp_path, sandboxes):
    point = parent_point(tmp_path / "parent")
    (tmp_path / "attempt").mkdir()

    with chat_server(actor_replies()) as server:
        execute(branch_request(server.url, point, {"guidance": "none"}), tmp_path / "attempt")

    assert server.requests[0]["messages"] == restored_messages(point)


def test_the_journal_of_a_branch_opens_with_its_point_and_its_guidance(tmp_path, sandboxes):
    point = parent_point(tmp_path / "parent")
    (tmp_path / "attempt").mkdir()

    with chat_server(actor_replies()) as server:
        execute(
            branch_request(server.url, point, {"guidance": "user-hint", "hint": HINT}), tmp_path / "attempt"
        )

    first = read_journal(tmp_path / "attempt" / "trajectory.jsonl")[0]
    assert (first["type"], {key: value for key, value in first.items() if key not in ENVELOPE}) == (
        "fork.origin",
        {
            "point_id": "point-1",
            "snapshot_id": point.snapshot_id,
            "tool_depth": 2,
            "message_step": 1,
            "branch_guidance": "user-hint",
            "actor_hint": HINT,
            "hint_delivery": "reviewer-direct",
        },
    )


def test_a_branch_without_a_continuation_never_starts_a_sandbox(tmp_path, sandboxes):
    point = parent_point(tmp_path / "parent")
    (tmp_path / "attempt").mkdir()

    with pytest.raises(ValueError, match="^A continuation is an object with guidance"):
        execute(branch_request("http://127.0.0.1:9", point, None), tmp_path / "attempt")

    assert [session.image for session in sandboxes.sessions] == ["fixture-image"]
