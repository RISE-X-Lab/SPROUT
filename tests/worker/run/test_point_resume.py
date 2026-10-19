"""A branch that resumes from a recovery point without a new user message: it continues the exact history.

The parent runs mini on a ``DirectoryBackend`` sandbox. The branch starts its sandbox from the snapshot
of the point, and its history from the native prefix of the point, as an attempt of a branch job does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import NativePoint, materialize, read_entries, read_prefix
from sprout.contract.run_spec import RunSpec
from sprout.worker.agents.mini_swe.history import index_native
from sprout.worker.run.runner import Runner, RunOutcome
from tests.support.http import chat_completion, chat_server
from tests.worker.run.directory_session import DirectoryBackend
from tests.worker.run.rollouts import SUBMIT, events, needs_mini, rollout_spec, transcript
from tests.worker.run.upstream import provider

pytestmark = needs_mini


def run_parent(directory: Path, replies: list[dict[str, Any]]) -> list[NativePoint]:
    """The recovery points of a completed root rollout on ``replies``, with its files in ``directory``."""
    with provider(replies) as upstream:
        outcome = Runner().run(rollout_spec(directory, upstream.url))
    assert outcome.status == "completed", outcome.error
    return points_of(directory, outcome)


def points_of(
    directory: Path, outcome: RunOutcome, inherited: dict[str, Any] | None = None
) -> list[NativePoint]:
    session = outcome.native_session_id
    return index_native(
        outcome.journal_path,
        transcript(directory, outcome),
        "mini-swe-agent",
        session,
        inherited_native=inherited,
    )


def resume_spec(directory: Path, url: str, point: NativePoint, **extra: Any) -> RunSpec:
    """The spec of a branch from ``point`` that adds no user message, with its files in ``directory``."""
    spec = rollout_spec(directory, url, prompt="NO_NEW_USER_MESSAGE", sandbox_image=point.snapshot_id)
    restored = materialize(
        point.native, directory / "cwd", directory / "restoration", directory / "native-home"
    )
    spec.extra.update(restored, resume_without_hint=True, **extra)
    return spec


def prefix_messages(point: NativePoint) -> list[dict[str, Any]]:
    """The chat messages of the native prefix of ``point``, as mini sends them to the model."""
    entries = read_entries(read_prefix(point.native))
    return [
        {key: value for key, value in entry["message"].items() if key != "extra"}
        for entry in entries
        if entry["type"] == "mini.message"
    ]


def observed_output(request: dict[str, Any]) -> str:
    """The output of the command whose result is the last message of ``request``."""
    return json.loads(request["messages"][-1]["content"])["output"]


PARENT = [
    chat_completion("echo early > answer", "cat answer", ids=("parent-1", "parent-2")),
    chat_completion("echo late > answer", ids=("parent-3",)),
    chat_completion(SUBMIT, ids=("parent-4",)),
]


def test_a_branch_from_a_point_sends_the_history_of_the_point_unchanged_and_no_new_message(
    tmp_path, sandboxes: DirectoryBackend
):
    [point, *_] = run_parent(tmp_path / "parent", PARENT)
    replies = [chat_completion("cat answer", ids=("child-1",)), chat_completion(SUBMIT, ids=("child-2",))]

    with provider(replies) as upstream:
        outcome = Runner().run(resume_spec(tmp_path / "child", upstream.url, point))

    requests = [request.body for request in upstream.requests]
    assert (outcome.status, requests[0]["messages"]) == ("completed", prefix_messages(point))
    assert ("NO_NEW_USER_MESSAGE" in json.dumps(requests), "echo late" in json.dumps(requests)) == (
        False,
        False,
    )
    assert (observed_output(requests[1]), events(outcome, "branch.assistant_turn")) == ("early\n", [])


def test_a_branch_from_a_point_has_recovery_points_of_its_own_and_leaves_the_parent_prefix_unchanged(
    tmp_path, sandboxes: DirectoryBackend
):
    [point, *_] = run_parent(tmp_path / "parent", PARENT)
    before = read_prefix(point.native)
    replies = [chat_completion("cat answer", ids=("child-1",)), chat_completion(SUBMIT, ids=("child-2",))]

    with provider(replies) as upstream:
        outcome = Runner().run(resume_spec(tmp_path / "child", upstream.url, point))

    points = points_of(tmp_path / "child", outcome, inherited=point.native)
    assert ([p.tool_depth for p in points], read_prefix(point.native)) == ([1, 2], before)


def test_a_branch_of_a_branch_resumes_from_the_history_of_the_first_branch(
    tmp_path, sandboxes: DirectoryBackend
):
    [point, *_] = run_parent(tmp_path / "parent", PARENT)
    replies = [chat_completion("cat answer", ids=("child-1",)), chat_completion(SUBMIT, ids=("child-2",))]
    with provider(replies) as upstream:
        child = Runner().run(resume_spec(tmp_path / "child", upstream.url, point))
    [child_point, *_] = points_of(tmp_path / "child", child, inherited=point.native)
    replies = [
        chat_completion("cat answer", ids=("grandchild-1",)),
        chat_completion(SUBMIT, ids=("grandchild-2",)),
    ]

    with provider(replies) as upstream:
        outcome = Runner().run(resume_spec(tmp_path / "grandchild", upstream.url, child_point))

    requests = [request.body for request in upstream.requests]
    assert (outcome.status, requests[0]["messages"], observed_output(requests[1])) == (
        "completed",
        prefix_messages(child_point),
        "early\n",
    )


def test_a_branch_from_the_point_after_a_rejected_response_resumes_with_the_rejection_in_its_history(
    tmp_path, sandboxes: DirectoryBackend
):
    bad = chat_completion("touch must-not-execute", ids=("parent-bad",))
    bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"command": 123}'
    parent = [
        bad,
        chat_completion("printf fixed > answer", ids=("parent-1",)),
        chat_completion("printf future > answer", ids=("parent-2",)),
        chat_completion(SUBMIT, ids=("parent-3",)),
    ]
    [point, *_] = run_parent(tmp_path / "parent", parent)
    replies = [chat_completion("cat answer", ids=("child-1",)), chat_completion(SUBMIT, ids=("child-2",))]

    with provider(replies) as upstream:
        outcome = Runner().run(resume_spec(tmp_path / "child", upstream.url, point))

    requests = [request.body for request in upstream.requests]
    assert [message["role"] for message in requests[0]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert (
        requests[0]["messages"],
        observed_output(requests[1]),
        "printf future" in json.dumps(requests),
    ) == (
        prefix_messages(point),
        "fixed",
        False,
    )
    assert [p.tool_depth for p in points_of(tmp_path / "child", outcome, inherited=point.native)] == [1, 2]


def test_a_branch_from_a_reply_that_submitted_before_its_last_call_resumes_with_its_not_executed_result(
    tmp_path, sandboxes: DirectoryBackend
):
    """The point is the final snapshot: the disk after the submitting call, with the later call unrun."""
    submitting = chat_completion(
        "printf submitted > answer && " + SUBMIT, "printf late > answer", ids=("parent-1", "parent-2")
    )
    [point] = run_parent(tmp_path / "parent", [submitting])
    replies = [chat_completion("cat answer", ids=("child-1",)), chat_completion(SUBMIT, ids=("child-2",))]

    with provider(replies) as upstream:
        outcome = Runner().run(resume_spec(tmp_path / "child", upstream.url, point))

    requests = [request.body for request in upstream.requests]
    assert (outcome.status, requests[0]["messages"], observed_output(requests[1])) == (
        "completed",
        prefix_messages(point),
        "submitted",
    )
    assert requests[0]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "parent-2",
        "content": "Not executed: the task was already submitted by an earlier call in this response.",
    }


@pytest.mark.parametrize(
    "extra,error",
    [
        pytest.param(
            {"resume_without_hint": True},
            "ValueError: This branch mode requires an exact mini history prefix",
            id="no-prefix",
        ),
        pytest.param(
            {"resume_without_hint": "true"},
            "ValueError: resume_without_hint must be a boolean",
            id="not-a-bool",
        ),
        pytest.param(
            {
                "resume_without_hint": True,
                "assistant_turn": {"role": "assistant", "content": "Check the boundary.", "tool_calls": []},
            },
            "ValueError: Point-only continuation cannot include an assistant_turn",
            id="with-an-assistant-turn",
        ),
    ],
)
def test_a_point_only_resume_that_cannot_apply_stops_before_any_model_request_or_tool_call(
    tmp_path, sandboxes: DirectoryBackend, extra, error
):
    with chat_server([]) as server:
        spec = rollout_spec(tmp_path, server.url)
        spec.extra.update(extra)
        outcome = Runner().run(spec)

    assert (outcome.status, outcome.error, server.requests, events(outcome, "tool.started")) == (
        "error",
        error,
        [],
        [],
    )
