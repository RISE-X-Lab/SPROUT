"""A branch run continues an exact prefix of its parent's native history.

The parent's first turn runs two commands, and its second turn submits. Each branch starts at the
end of the first turn, so its first request must repeat the parent's second request. Each reply of
a branch names its call ids, because the ids of the prefix may not come again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import materialize
from sprout.worker.agents.mini_swe.history import index_native, reference_at
from sprout.worker.checkpoint.rollback import branch_checkpoints
from sprout.worker.run.control import RunControl
from tests.support.http import chat_completion
from tests.worker.agents.mini_swe.rollouts import (
    SUBMIT,
    Rollout,
    ShellSandbox,
    needs_mini,
    run_mini,
    stop_at,
)

pytestmark = needs_mini

PARENT = [chat_completion("printf early > answer", "cat answer"), chat_completion(SUBMIT)]


def reviewer_turn(command: str, *, identifier: str = "reviewer-1") -> dict[str, Any]:
    """An assistant turn as a reviewer writes it: text and one bash call."""
    arguments = json.dumps({"command": command})
    return {
        "role": "assistant",
        "content": "Check the boundary first.",
        "tool_calls": [
            {"id": identifier, "type": "function", "function": {"name": "bash", "arguments": arguments}}
        ],
    }


def two_call_turn(first: str, second: str) -> dict[str, Any]:
    """A reviewer turn with two bash calls, ``reviewer-1`` and then ``reviewer-2``."""
    turn = reviewer_turn(first)
    turn["tool_calls"] += reviewer_turn(second, identifier="reviewer-2")["tool_calls"]
    return turn


def branch_extra(tmp_path: Path) -> tuple[Rollout, dict[str, Any]]:
    """A parent run, and the ``extra`` of a branch from the end of its first turn."""
    parent = run_mini(tmp_path / "parent", PARENT)
    point = index_native(
        parent.journal_path, parent.transcript, "mini-swe-agent", parent.result.native_session_id
    )[0]
    extra = materialize(point.native, tmp_path / "cwd", tmp_path / "restoration", tmp_path / "child-home")
    return parent, extra


def reviewer_sources(rollout: Rollout) -> list[tuple[list[str], str | None]]:
    """The call ids and the ``source`` of each assistant message in the native history."""
    return [
        (
            [call["id"] for call in entry["message"]["tool_calls"]],
            entry["message"].get("extra", {}).get("source"),
        )
        for entry in rollout.entries()
        if entry["type"] == "mini.message" and entry["message"]["role"] == "assistant"
    ]


def test_a_branch_continues_the_parent_prefix_with_its_prompt_as_a_hint(tmp_path):
    parent, extra = branch_extra(tmp_path)

    child = run_mini(
        tmp_path / "child", [chat_completion(SUBMIT, ids=("child-1",))], extra=extra, prompt="Try again."
    )

    assert child.requests[0]["messages"] == parent.requests[1]["messages"] + [
        {"role": "user", "content": "Try again."}
    ]


def test_a_branch_without_a_hint_continues_the_prefix_with_no_new_message(tmp_path):
    parent, extra = branch_extra(tmp_path)

    child = run_mini(
        tmp_path / "child",
        [chat_completion(SUBMIT, ids=("child-1",))],
        extra={**extra, "resume_without_hint": True},
    )

    assert child.requests[0]["messages"] == parent.requests[1]["messages"]


def test_a_reviewer_turn_runs_before_the_first_model_request_and_its_results_join_the_prefix(tmp_path):
    parent, extra = branch_extra(tmp_path)
    turn = reviewer_turn("printf reviewed")

    child = run_mini(
        tmp_path / "child",
        [chat_completion(SUBMIT, ids=("child-1",))],
        extra={**extra, "assistant_turn": turn},
        prompt="This hint is not sent.",
    )

    assert child.sandbox.commands == ["printf reviewed", SUBMIT]
    assert child.requests[0]["messages"] == parent.requests[1]["messages"] + [
        turn,
        {
            "role": "tool",
            "tool_call_id": "reviewer-1",
            "content": '{\n  "returncode": 0,\n  "output": "reviewed"\n}',
        },
    ]


def test_each_call_of_a_reviewer_turn_runs_in_order_and_a_failed_call_reaches_the_model_as_it_failed(
    tmp_path,
):
    """A reviewer's diagnostic command can exit non-zero on purpose. The model must see that exit."""
    _, extra = branch_extra(tmp_path)

    child = run_mini(
        tmp_path / "child",
        [chat_completion(SUBMIT, ids=("child-1",))],
        extra={**extra, "assistant_turn": two_call_turn("printf diagnostic; exit 7", "printf done")},
    )

    assert child.sandbox.commands == ["printf diagnostic; exit 7", "printf done", SUBMIT]
    assert child.requests[0]["messages"][-2:] == [
        {
            "role": "tool",
            "tool_call_id": "reviewer-1",
            "content": '{\n  "returncode": 7,\n  "output": "diagnostic"\n}',
        },
        {
            "role": "tool",
            "tool_call_id": "reviewer-2",
            "content": '{\n  "returncode": 0,\n  "output": "done"\n}',
        },
    ]


def test_a_stop_inside_a_reviewer_turn_runs_none_of_its_later_calls_and_leaves_no_branch_point(tmp_path):
    """Half of a reviewer turn is no closed turn, so no branch may start from it."""
    _, extra = branch_extra(tmp_path)
    control = RunControl()
    sandbox = ShellSandbox(tmp_path / "child" / "sandbox", on_call=stop_at(control, "printf first"))

    child = run_mini(
        tmp_path / "child",
        [],
        sandbox=sandbox,
        control=control,
        extra={**extra, "assistant_turn": two_call_turn("printf first", "printf second")},
    )

    assert (child.sandbox.commands, child.requests) == (["printf first"], [])
    assert branch_checkpoints(child.journal_path) == {}


def test_a_branch_from_a_reviewer_turn_replays_the_turn_and_does_not_run_it_again(tmp_path):
    _, extra = branch_extra(tmp_path)
    child = run_mini(
        tmp_path / "child",
        [chat_completion(SUBMIT, ids=("child-1",))],
        extra={**extra, "assistant_turn": reviewer_turn("printf reviewed")},
    )
    reviewed = reference_at(child.journal_path, 1, child.result.native_session_id)
    grand_extra = materialize(reviewed, tmp_path / "cwd2", tmp_path / "restoration2", tmp_path / "grand-home")

    grandchild = run_mini(
        tmp_path / "grandchild",
        [chat_completion(SUBMIT, ids=("grand-1",))],
        extra=grand_extra,
        prompt="Go on.",
    )

    assert grandchild.sandbox.commands == [SUBMIT]
    assert grandchild.requests[0]["messages"] == child.requests[0]["messages"] + [
        {"role": "user", "content": "Go on."}
    ]


def test_the_native_history_tells_the_reviewers_turn_from_the_actors(tmp_path):
    """A training export must not count a turn that no policy sampled as the policy's."""
    _, extra = branch_extra(tmp_path)

    child = run_mini(
        tmp_path / "child",
        [chat_completion(SUBMIT, ids=("child-1",))],
        extra={**extra, "assistant_turn": reviewer_turn("true")},
    )

    assert reviewer_sources(child) == [
        (["call-1", "call-2"], None),
        (["reviewer-1"], "reviewer"),
        (["child-1"], None),
    ]


def test_a_reviewer_turn_that_reuses_a_call_id_of_the_prefix_is_refused_before_any_command(tmp_path):
    _, extra = branch_extra(tmp_path)
    sandbox = ShellSandbox(tmp_path / "child" / "sandbox")

    with pytest.raises(
        ValueError, match=re.escape("tool call ids must be non-empty and unique across history")
    ):
        run_mini(
            tmp_path / "child",
            [chat_completion(SUBMIT, ids=("child-1",))],
            sandbox=sandbox,
            extra={**extra, "assistant_turn": reviewer_turn("touch reused", identifier="call-1")},
        )

    assert sandbox.calls == []
