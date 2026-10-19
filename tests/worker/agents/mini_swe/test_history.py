"""The branch points of a mini run: its closed turns that have a snapshot, read from its native history.

The parent run of most tests has two turns. The first runs two commands (tool steps 1 and 2), and
the second submits (step 3). The sandbox numbers its snapshots from ``snapshot-1``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import NativePoint, materialize, read_entries, read_prefix
from sprout.worker.agents.mini_swe.history import (
    actor_tools_at,
    available_branch_points,
    conversation_restore,
    index_mini,
    index_native,
    reference_at,
)
from sprout.worker.checkpoint.rollback import branch_checkpoints
from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter
from tests.support.http import chat_completion
from tests.worker.agents.mini_swe.rollouts import (
    BASH_TOOL,
    SUBMIT,
    Rollout,
    ShellSandbox,
    needs_mini,
    outline,
    run_mini,
    stop_at,
    with_tool_name,
)

#: A first turn of two commands, then a turn that submits.
PARENT = [chat_completion("printf early > answer", "cat answer"), chat_completion(SUBMIT)]


def points_of(rollout: Rollout, *, inherited: dict[str, Any] | None = None) -> list[NativePoint]:
    return index_native(
        rollout.journal_path,
        rollout.transcript,
        "mini-swe-agent",
        rollout.result.native_session_id,
        inherited_native=inherited,
    )


def summary(points: list[NativePoint]) -> list[tuple[int, int, str, str]]:
    """Each point as its tool depth, its message step, its snapshot and the turn that it cuts at."""
    return [
        (point.tool_depth, point.message_step, point.snapshot_id, point.native["cut"]) for point in points
    ]


def turn_ids(rollout: Rollout) -> list[str]:
    """The id of each model turn that the journal closed, in order."""
    return [event["turn_id"] for event in rollout.events("model.turn.completed")]


def transcript_with(path: Path, *entries: dict[str, Any]) -> Path:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return path


def journal_with(path: Path, events: list[tuple[str, dict[str, Any]]]) -> None:
    with JournalWriter(path) as journal:
        for kind, payload in events:
            journal.emit(kind, **payload)


def header(**changes: Any) -> dict[str, Any]:
    return {
        "type": "mini.session",
        "format": "sprout-mini-v1",
        "mini_version": "2.4.6",
        "session_id": "s",
        **changes,
    }


# --- which turns are branch points --------------------------------------------------------


@needs_mini
def test_the_branch_points_are_the_closed_turns_that_have_a_snapshot(tmp_path):
    rollout = run_mini(tmp_path, PARENT)

    first, second = turn_ids(rollout)
    assert summary(points_of(rollout)) == [(2, 1, "snapshot-1", first), (3, 2, "snapshot-2", second)]


@needs_mini
def test_a_branch_point_names_the_native_history_up_to_the_end_of_its_turn(tmp_path):
    rollout = run_mini(tmp_path, PARENT)

    native = points_of(rollout)[0].native
    assert (native["slot"], native["session_id"], native["path"]) == (
        "mini-swe-agent",
        rollout.result.native_session_id,
        str(rollout.transcript.resolve()),
    )
    assert outline(read_entries(read_prefix(native))) == [
        ("mini.session",),
        ("message", "system"),
        ("message", "user"),
        ("message", "assistant", ["call-1", "call-2"]),
        ("message", "tool", "call-1"),
        ("message", "tool", "call-2"),
        ("turn", ["call-1", "call-2"]),
    ]


@needs_mini
def test_a_turn_that_a_stop_cuts_short_is_no_branch_point(tmp_path):
    control = RunControl()
    sandbox = ShellSandbox(tmp_path / "sandbox", on_call=stop_at(control, "printf a"))
    replies = [chat_completion("true"), chat_completion("printf a", "printf b")]

    rollout = run_mini(tmp_path, replies, sandbox=sandbox, control=control)

    assert summary(points_of(rollout)) == [(1, 1, "snapshot-1", turn_ids(rollout)[0])]


@needs_mini
def test_a_rejected_turn_counts_as_a_message_step_but_is_no_branch_point(tmp_path):
    rejected = with_tool_name(chat_completion("ls"), index=0, name="shell")

    rollout = run_mini(tmp_path, [chat_completion("printf a > answer"), rejected, chat_completion(SUBMIT)])

    first, last = turn_ids(rollout)
    assert summary(points_of(rollout)) == [(1, 1, "snapshot-1", first), (2, 3, "snapshot-2", last)]


@needs_mini
def test_the_final_snapshot_of_a_reply_that_submits_before_its_last_call_is_a_branch_point(tmp_path):
    """The later call never runs, and its not-executed result settles it."""
    rollout = run_mini(tmp_path, [chat_completion(SUBMIT, "touch late")])

    [turn] = turn_ids(rollout)
    checkpoints = branch_checkpoints(rollout.journal_path)
    assert [(step, point.snapshot_id, point.final) for step, point in checkpoints.items()] == [
        (1, "snapshot-1", True)
    ]
    assert summary(points_of(rollout)) == [(1, 1, "snapshot-1", turn)]


@needs_mini
def test_indexing_stops_at_a_torn_last_line(tmp_path):
    """A history that is being written never publishes a turn whose record is incomplete."""
    rollout = run_mini(tmp_path, PARENT)
    first = points_of(rollout)[0]
    data = rollout.transcript.read_bytes()

    rollout.transcript.write_bytes(data[: first.native["byte_length"]] + b'{"type":')

    assert points_of(rollout) == [first]


@needs_mini
def test_a_branch_indexes_its_own_tool_steps_after_the_prefix_that_it_inherited(tmp_path):
    parent = run_mini(tmp_path / "parent", PARENT)
    point = points_of(parent)[0]
    extra = materialize(point.native, tmp_path / "cwd", tmp_path / "restoration", tmp_path / "child-home")
    replies = [chat_completion("true", ids=("child-1",)), chat_completion(SUBMIT, ids=("child-2",))]

    child = run_mini(tmp_path / "child", replies, extra=extra, prompt="Try again.")

    first, second = turn_ids(child)
    assert summary(points_of(child, inherited=point.native)) == [
        (1, 2, "snapshot-1", first),
        (2, 3, "snapshot-2", second),
    ]


# --- what a stored history may hold -----------------------------------------------------------


@pytest.mark.parametrize("format_name", ["sprout-mini-v1", "ash-mini-v1"])
def test_a_history_in_the_current_or_the_older_format_name_is_read(tmp_path, format_name):
    transcript = transcript_with(tmp_path / "history.jsonl", header(format=format_name))

    assert index_mini(tmp_path / "journal.jsonl", transcript, "s", events=[]) == []


@pytest.mark.parametrize(
    "changes", [{"format": "other-v1"}, {"mini_version": "2.4.5"}], ids=["other-format", "other-mini"]
)
def test_a_history_of_another_format_or_mini_version_is_refused(tmp_path, changes):
    transcript = transcript_with(tmp_path / "history.jsonl", header(**changes))

    with pytest.raises(ValueError, match="^Unsupported mini native history version$"):
        index_mini(tmp_path / "journal.jsonl", transcript, "s", events=[])


def test_a_history_with_an_unknown_record_is_refused(tmp_path):
    transcript = transcript_with(tmp_path / "history.jsonl", header(), {"type": "mini.checkpoint"})

    with pytest.raises(ValueError, match="^Unknown mini native record$"):
        index_mini(tmp_path / "journal.jsonl", transcript, "s", events=[])


def test_only_mini_histories_are_indexed(tmp_path):
    transcript = transcript_with(tmp_path / "history.jsonl", header())

    with pytest.raises(ValueError, match="^Unsupported native history slot$"):
        index_native(tmp_path / "journal.jsonl", transcript, "claude-code", "s", events=[])


# --- the readers of a run's journal ---------------------------------------------------------


@needs_mini
def test_the_reference_at_a_step_is_the_branch_point_whose_turn_ends_there(tmp_path):
    rollout = run_mini(tmp_path, PARENT)
    session = rollout.result.native_session_id

    within = reference_at(rollout.journal_path, 1, session)
    first_end = reference_at(rollout.journal_path, 2, session)
    second_end = reference_at(rollout.journal_path, 3, session)

    assert (within, first_end["cut"], second_end["cut"]) == (None, *turn_ids(rollout))


@needs_mini
def test_the_readers_follow_the_last_session_ref_of_a_moved_history(tmp_path):
    rollout = run_mini(tmp_path, PARENT)
    session = rollout.result.native_session_id
    moved = rollout.transcript.rename(tmp_path / "moved.jsonl")
    with JournalWriter(rollout.journal_path) as journal:
        journal.emit("session.ref", native_session_id=session, transcript_path=str(moved))

    reference = reference_at(rollout.journal_path, 2, session)

    assert (reference["path"], sorted(available_branch_points(rollout.journal_path))) == (
        str(moved.resolve()),
        [2, 3],
    )


@needs_mini
def test_a_branch_journal_finds_its_points_from_the_prefix_that_it_records(tmp_path):
    """The child's journal names its inherited prefix, so no caller has to pass it."""
    parent = run_mini(tmp_path / "parent", PARENT)
    point = points_of(parent)[0]
    extra = materialize(point.native, tmp_path / "cwd", tmp_path / "restoration", tmp_path / "child-home")
    replies = [chat_completion("true", ids=("child-1",)), chat_completion(SUBMIT, ids=("child-2",))]
    child = run_mini(tmp_path / "child", replies, extra=extra, prompt="Try again.")

    reference = reference_at(child.journal_path, 1, child.result.native_session_id)

    assert reference == points_of(child, inherited=point.native)[0].native


@needs_mini
def test_the_conversation_restore_point_of_a_step_is_the_turn_that_ends_there(tmp_path):
    rollout = run_mini(tmp_path, PARENT)

    cut = conversation_restore(rollout.journal_path, 2, rollout.result.native_session_id)

    assert cut == turn_ids(rollout)[0]


@needs_mini
def test_the_available_branch_points_are_the_checkpoints_whose_turn_the_history_holds(tmp_path):
    rollout = run_mini(tmp_path, PARENT)

    points = available_branch_points(rollout.journal_path)

    assert points == branch_checkpoints(rollout.journal_path)
    assert sorted(points) == [2, 3]


@needs_mini
def test_the_available_branch_points_reflect_the_history_as_it_is_now(tmp_path):
    """Nothing is cached between two calls."""
    rollout = run_mini(tmp_path, PARENT)
    before = sorted(available_branch_points(rollout.journal_path))
    first = points_of(rollout)[0]

    rollout.transcript.write_bytes(
        rollout.transcript.read_bytes()[: first.native["byte_length"]] + b'{"type":'
    )

    assert (before, sorted(available_branch_points(rollout.journal_path))) == ([2, 3], [2])


@needs_mini
def test_the_actor_tools_at_a_step_are_the_bash_tool_that_the_run_declared(tmp_path):
    rollout = run_mini(tmp_path, PARENT)

    assert actor_tools_at(rollout.journal_path, 2) == [BASH_TOOL]


@pytest.mark.parametrize(
    "events, message",
    [
        ([], "No completed turn for the selected actor tool schema"),
        ([("model.turn.completed", {"step": 2})], "No recorded actor tool schema at the selected checkpoint"),
    ],
    ids=["no-turn-at-the-step", "no-declaration-before-the-turn"],
)
def test_the_actor_tools_need_a_completed_turn_and_a_declaration_before_it(tmp_path, events, message):
    journal_with(tmp_path / "journal.jsonl", events)

    with pytest.raises(ValueError, match="^%s$" % message):
        actor_tools_at(tmp_path / "journal.jsonl", 2)
